#include "fft_executor.h"

#include <mutex>
#include <numeric>
#include <stdexcept>
#include <iostream>

using namespace std;

namespace {

std::mutex& fftw_planner_mutex() {
    static std::mutex mtx;
    return mtx;
}

std::once_flag& fftw_threads_once_flag() {
    static std::once_flag flag;
    return flag;
}

bool& fftw_threads_ready() {
    static bool ready = false;
    return ready;
}

}

/* =========================
   INTERNAL
   ========================= */

static int to_fftw_sign(FFT_Direction dir) {
    return (dir == FFT_Direction::Forward)
           ? FFTW_FORWARD
           : FFTW_BACKWARD;
}

/* =========================
   CTOR / DTOR
   ========================= */

FFT_Executor::FFT_Executor() = default;

FFT_Executor::~FFT_Executor() {
    destroy_plan();
}

/* =========================
   THREAD INIT
   ========================= */

void FFT_Executor::init_threads(int nthreads) {
    const int requested_threads = nthreads > 0 ? nthreads : 1;

    if (requested_threads > 1) {
        std::call_once(fftw_threads_once_flag(), []() {
            fftw_threads_ready() = (fftw_init_threads() != 0);
        });

        if (!fftw_threads_ready()) {
            throw std::runtime_error("FFTW thread init failed");
        }

        fftw_plan_with_nthreads(requested_threads);
    } else if (fftw_threads_ready()) {
        fftw_plan_with_nthreads(1);
    }
}

/* =========================
   VALIDATION
   ========================= */

void FFT_Executor::validate(const FFT_Config& cfg) const {
    if ((int)cfg.shape.size() != cfg.dim)
        throw std::invalid_argument("shape.size() != dim");

    if (cfg.dim >= 4 && cfg.type != FFT_Type::C2C)
        throw std::invalid_argument("dim >= 4 only supports C2C");

    if (cfg.inplace && cfg.type != FFT_Type::C2C)
        throw std::invalid_argument("inplace only allowed for C2C");
}

/* =========================
   PLAN CREATION
   ========================= */

void FFT_Executor::create_plan(const FFT_Config& cfg,
                              void* input,
                              void* output)
{
    validate(cfg);

    std::lock_guard<std::mutex> lock(fftw_planner_mutex());
    init_threads(cfg.nthreads);

    if (plan_) {
        fftw_destroy_plan(plan_);
        plan_ = nullptr;
    }

    this->current_config_ = cfg;

    const int* n = cfg.shape.data();
    unsigned flags = static_cast<unsigned>(cfg.flags);
    int sign = to_fftw_sign(cfg.direction);

    if (cfg.inplace) {
        output = input;
    }

    switch (cfg.type) {

    case FFT_Type::R2C:
        if (cfg.dim == 1)
            plan_ = fftw_plan_dft_r2c_1d(
                n[0], (double*)input, (fftw_complex*)output, flags);
        else if (cfg.dim == 2)
            plan_ = fftw_plan_dft_r2c_2d(
                n[0], n[1],
                (double*)input, (fftw_complex*)output, flags);
        else if (cfg.dim == 3)
            plan_ = fftw_plan_dft_r2c_3d(
                n[0], n[1], n[2],
                (double*)input, (fftw_complex*)output, flags);
        break;

    case FFT_Type::C2R:
        if (cfg.dim == 1)
            plan_ = fftw_plan_dft_c2r_1d(
                n[0], (fftw_complex*)input, (double*)output, flags);
        else if (cfg.dim == 2)
            plan_ = fftw_plan_dft_c2r_2d(
                n[0], n[1],
                (fftw_complex*)input, (double*)output, flags);
        else if (cfg.dim == 3)
            plan_ = fftw_plan_dft_c2r_3d(
                n[0], n[1], n[2],
                (fftw_complex*)input, (double*)output, flags);
        break;

    case FFT_Type::C2C:
        plan_ = fftw_plan_dft(
            cfg.dim, n,
            (fftw_complex*)input,
            (fftw_complex*)output,
            sign, flags);
        break;
    }

    if (!plan_)
        throw std::runtime_error("Failed to create FFTW plan");
}

/* =========================
   EXECUTE
   ========================= */

void FFT_Executor::execute_plan(const FFT_Config& cfg,
                                void* input,
                                void* output)
{
    if (!plan_)
        throw std::runtime_error("Plan not created");

    if (cfg.denormalize) {
        denormalize(cfg, input);
    }

    // Use new-array execute interface to handle actual input/output pointers
    if (cfg.type == FFT_Type::C2C) {
        fftw_execute_dft(plan_,
                         (fftw_complex*)input,
                         (fftw_complex*)output);
    } else if (cfg.type == FFT_Type::R2C) {
        fftw_execute_dft_r2c(plan_,
                             (double*)input,
                             (fftw_complex*)output);
    } else if (cfg.type == FFT_Type::C2R) {
        fftw_execute_dft_c2r(plan_,
                             (fftw_complex*)input,
                             (double*)output);
    }

    if (cfg.normalize) {
        normalize(cfg, cfg.inplace ? input : output);
    }
}

void FFT_Executor::execute_plan(void* input, void* output)
{
    execute_plan(this->current_config_, input, output);
}

/* =========================
   DESTROY
   ========================= */

void FFT_Executor::destroy_plan() {
    std::lock_guard<std::mutex> lock(fftw_planner_mutex());
    if (plan_) {
        fftw_destroy_plan(plan_);
        plan_ = nullptr;
    }
}

/* =========================
   SCALING
   ========================= */

size_t FFT_Executor::total_size(const FFT_Config& cfg) {
    return std::accumulate(
        cfg.shape.begin(),
        cfg.shape.end(),
        size_t(1),
        std::multiplies<>());
}

void FFT_Executor::normalize(const FFT_Config& cfg, void* data) {
    double factor = 1.0 / total_size(cfg);
    //cout<< "Total size for normalization: " << total_size(cfg) << endl;
    //cout<< "Normalizing with factor: " << factor << endl;

    size_t N = total_size(cfg);

    if (cfg.type == FFT_Type::C2C || cfg.type == FFT_Type::R2C) {
        auto* z = (fftw_complex*)data;
#pragma omp parallel for num_threads(cfg.nthreads > 0 ? cfg.nthreads : 1)
        for (size_t i = 0; i < N; ++i) {
            z[i][0] *= factor;
            z[i][1] *= factor;
        }
    } else {
        auto* x = (double*)data;
#pragma omp parallel for num_threads(cfg.nthreads > 0 ? cfg.nthreads : 1)
        for (size_t i = 0; i < N; ++i)
            x[i] *= factor;
    }
}

void FFT_Executor::denormalize(const FFT_Config& cfg, void* data) {
    double factor = total_size(cfg);
    size_t N = total_size(cfg);

    if (cfg.type == FFT_Type::C2C || cfg.type == FFT_Type::R2C) {
        auto* z = (fftw_complex*)data;
#pragma omp parallel for num_threads(cfg.nthreads > 0 ? cfg.nthreads : 1)
        for (size_t i = 0; i < N; ++i) {
            z[i][0] *= factor;
            z[i][1] *= factor;
        }
    } else {
        auto* x = (double*)data;
#pragma omp parallel for num_threads(cfg.nthreads > 0 ? cfg.nthreads : 1)
        for (size_t i = 0; i < N; ++i)
            x[i] *= factor;
    }
}
