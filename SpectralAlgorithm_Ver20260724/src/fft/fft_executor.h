#pragma once

#include <vector>
#include <fftw3.h>

/* =========================
   ENUMS
   ========================= */

enum class FFT_Type {
    R2C,
    C2R,
    C2C
};

enum class FFT_Direction {
    Forward,
    Backward
};

enum class FFT_Flag : unsigned {
    Estimate    = FFTW_ESTIMATE,
    Measure     = FFTW_MEASURE,
    Patient     = FFTW_PATIENT,
    Exhaustive  = FFTW_EXHAUSTIVE
};

inline unsigned operator|(FFT_Flag a, FFT_Flag b) {
    return static_cast<unsigned>(a) |
           static_cast<unsigned>(b);
}

/* =========================
   CONFIG
   ========================= */

struct FFT_Config {
    int dim;
    std::vector<int> shape;

    FFT_Type type;
    FFT_Direction direction;

    bool inplace   = false;
    bool normalize = false;     // normalize after FFT
    bool denormalize = false;   // multiply before IFFT

    int nthreads  = 1;
    FFT_Flag flags = FFT_Flag::Measure;
};

/* =========================
   FFT EXECUTOR
   ========================= */

class FFT_Executor {
public:
    FFT_Executor();
    ~FFT_Executor();

    /* ---- plan lifecycle ---- */
    void create_plan(const FFT_Config& config,
                     void* input,
                     void* output);

    void execute_plan(const FFT_Config& config,
                      void* input,
                      void* output);

    void execute_plan(void* input, void* output);

    void destroy_plan();

    /* ---- scaling ---- */
    void normalize(const FFT_Config& config, void* data);
    void denormalize(const FFT_Config& config, void* data);

private:
    fftw_plan plan_ = nullptr;
    bool threads_initialized_ = false;

    FFT_Config current_config_;

    void init_threads(int nthreads);
    void validate(const FFT_Config& cfg) const;

    static size_t total_size(const FFT_Config& cfg);
};
