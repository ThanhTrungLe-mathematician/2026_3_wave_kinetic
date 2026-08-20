#include "solver_2D.h"
#include "fft_executor.h"
#include "utilities.h"
#include "tensor_free_function.h"
using namespace std;

Solver_2D::Solver_2D(const ModelParameters &model_params, SimulationParameters &sim_params, const SpectralStabilizationConfig &spectral_stabilization_config)
    : Solver_Base_2D(model_params, sim_params, spectral_stabilization_config) {}

Solver_2D::~Solver_2D()
{
    k_grid_.Destroy();
    W_tensor_.Destroy();
    W_hat_tensor_.Destroy();
    f_initial_.Destroy();
    f_solution_.Destroy();
    f_solution_time_.Destroy();
    f_hat_previous_.Destroy();
    f_hat_current_.Destroy();
    f_hat_time_.Destroy();

    fft_idx_.clear();
    t_save_points_.Destroy();
}

void Solver_2D::create_objects()
{
    size_t grid_size = static_cast<size_t>(2 * this->sim_params_.N);
    size_t N_time = static_cast<size_t>(this->N_time_);

    k_grid_.resize(grid_size);
    W_tensor_.resize(grid_size, grid_size, grid_size, grid_size);
    W_hat_tensor_.resize(grid_size, grid_size, grid_size, grid_size);
    f_initial_.resize(grid_size, grid_size);
    f_solution_.resize(grid_size, grid_size);
    f_solution_time_.resize(N_time, grid_size, grid_size);
    f_hat_previous_.resize(grid_size, grid_size);
    f_hat_current_.resize(grid_size, grid_size);
    f_hat_time_.resize(N_time, grid_size, grid_size);

    fft_idx_.resize(grid_size);
    t_save_points_.resize(2, N_time);
}

void Solver_2D::compute_solution()
{
    if (this->is_parallel_)
    {
        this->nthreads_FFTW_ = this->sim_params_.num_threads;
        // Parallel computation implementation
        cout << "Parallel computation is implemented." << endl;
    }
    else
    {
        this->nthreads_FFTW_ = 1;
        this->sim_params_.num_threads = 1;
        // Parallel computation implementation
        cout << "Parallel computation is not yet implemented." << endl;
    }

    // Initialize variables
    int grid_size = 2 * this->sim_params_.N;
    int N = this->sim_params_.N;
    int nthreads = this->sim_params_.num_threads;
    int nthreads_FFTW = this->nthreads_FFTW_;
    double dt = this->sim_params_.dt;
    double C = this->physical_model_->C_;
    double R = this->physical_model_->model_params_.R;

    this->construct_k_grid();

    this->make_saved_times_points();
    int save_time_index = 1; // start from 1 since 0 is already saved

    // Precompute FFTW indices
    this->compute_fftw_indices();

    this->construct_f_initial_tensor();
    this->f_solution_time_.insert_slice(0, 0, this->f_initial_);

    this->compute_f_initial_hat();
    this->f_hat_time_.insert_slice(0, 0, this->f_hat_previous_);

    this->construct_W_tensor();
    this->compute_W_hat();

    // Create FFT plan C2C for reduction f_solution at each time step
    FFT_Config config_c2c_2d_backward = {
        .dim = 2,
        .shape = {grid_size, grid_size},
        .type = FFT_Type::C2C,
        .direction = FFT_Direction::Backward,
        .inplace = false,
        .normalize = false,
        .denormalize = false,
        .nthreads = this->nthreads_FFTW_,
        .flags = FFT_Flag::Estimate}; // use Estimate to avoid overwriting f_hat data

    FFT_Executor fft_c2c_2d_backward;
    fft_c2c_2d_backward.create_plan(
        config_c2c_2d_backward,
        this->f_hat_current_.data(),
        this->f_solution_.data());

    for (int t_step = 1; t_step < this->N_time_; ++t_step)
    {
        // Time-stepping loop implementation
        // cout << "Time step " << t_step + 1 << " / " << this->N_time_ << endl;

#pragma omp parallel for collapse(2) num_threads(nthreads)
        for (int n1 = -N; n1 < N; ++n1)
        {
            for (int n2 = -N; n2 < N; ++n2)
            {
                size_t idx_n1 = this->fft_idx_[n1 + N];
                size_t idx_n2 = this->fft_idx_[n2 + N];

                complex<double> f_hat_f_hat_K_npm = complex<double>(0.0, 0.0);
                complex<double> f_hat_H_np = complex<double>(0.0, 0.0);

                for (int p1 = -N; p1 < N; ++p1)
                {
                    for (int p2 = -N; p2 < N; ++p2)
                    {
                        size_t idx_p1 = this->fft_idx_[p1 + N];
                        size_t idx_p2 = this->fft_idx_[p2 + N];

                        // Cache f_hat_p once per (p1,p2) iteration
                        complex<double> f_hat_p = f_hat_previous_(idx_p1, idx_p2);

                        // Compute f_hat_H_np
                        f_hat_H_np += f_hat_p * this->compute_H_np(n1, n2, p1, p2, R);
                        // f_hat_H_np += this->compute_H_np(n1, n2, p1, p2);

                        for (int m1 = -N; m1 < N; ++m1)
                        {
                            for (int m2 = -N; m2 < N; ++m2)
                            {
                                // Compute f_hat_f_hat_K_nmp
                                size_t idx_m1 = this->fft_idx_[m1 + N];
                                size_t idx_m2 = this->fft_idx_[m2 + N];

                                // Cache f_hat_m once per (m1,m2) iteration
                                complex<double> f_hat_m = f_hat_previous_(idx_m1, idx_m2);

                                f_hat_f_hat_K_npm += f_hat_p * f_hat_m * this->compute_K_npm(n1, n2, p1, p2, m1, m2, R);
                                // f_hat_f_hat_K_npm += this->compute_K_npm(n1, n2, p1, p2, m1, m2);
                            }
                        }
                    }
                }

                // compute f_hat at current time step

                this->f_hat_current_(idx_n1, idx_n2) =
                    this->f_hat_previous_(idx_n1, idx_n2) +
                    dt * f_hat_f_hat_K_npm +
                    C * dt * f_hat_H_np;
            }
        }

        // Apply spectral stabilization to f_hat_current_
        this->block_apply_spectral_stabilization_f_hat();

        // Update previous time step solution
        this->block_update_and_store_solution_for_the_next_time_step(fft_c2c_2d_backward, save_time_index, t_step);
    }
}

void Solver_2D::compute_solution_manual()
{
    if (this->is_parallel_)
    {
        this->nthreads_FFTW_ = this->sim_params_.num_threads;
        // Parallel computation implementation
        cout << "Parallel computation is implemented." << endl;
    }
    else
    {
        this->nthreads_FFTW_ = 1;
        this->sim_params_.num_threads = 1;
        // Parallel computation implementation
        cout << "Parallel computation is not yet implemented." << endl;
    }

    // Initialize variables
    int grid_size = 2 * this->sim_params_.N;
    int N = this->sim_params_.N;
    int nthreads = this->sim_params_.num_threads;
    int nthreads_FFTW = this->nthreads_FFTW_;
    double dt = this->sim_params_.dt;
    double C = this->physical_model_->C_;
    double R = this->physical_model_->model_params_.R;

    this->construct_k_grid();

    this->construct_f_initial_tensor();
    this->f_solution_time_.insert_slice(0, 0, this->f_initial_);

    this->compute_f_initial_hat();
    this->f_hat_time_.insert_slice(0, 0, this->f_hat_previous_);

    cout << "Start computation for W_hat tensor at " + get_local_time();
    this->compute_W_hat_manual();
    cout << "End computation for W_hat tensor at " + get_local_time();

    for (int t_step = 1; t_step < this->N_time_; ++t_step)
    {
        // Time-stepping loop implementation
        // cout << "Time step " << t_step + 1 << " / " << this->N_time_ << endl;

#pragma omp parallel for collapse(2) num_threads(this->sim_params_.num_threads)
        for (int n1 = -N; n1 < N; ++n1)
        {
            for (int n2 = -N; n2 < N; ++n2)
            {
                size_t idx_n1 = this->get_fft_index_from_math(n1);
                size_t idx_n2 = this->get_fft_index_from_math(n2);

                complex<double> f_hat_f_hat_K_npm = complex<double>(0.0, 0.0);
                complex<double> f_hat_H_np = complex<double>(0.0, 0.0);

                for (int p1 = -N; p1 < N; ++p1)
                {
                    for (int p2 = -N; p2 < N; ++p2)
                    {
                        size_t idx_p1 = this->get_fft_index_from_math(p1);
                        size_t idx_p2 = this->get_fft_index_from_math(p2);
                        // Compute f_hat_H_np
                        f_hat_H_np += this->f_hat_previous_(idx_p1, idx_p2) *
                                      this->compute_H_np(n1, n2, p1, p2, R);
                        // f_hat_H_np += this->compute_H_np(n1, n2, p1, p2);

                        for (int m1 = -N; m1 < N; ++m1)
                        {
                            for (int m2 = -N; m2 < N; ++m2)
                            {
                                // Compute f_hat_f_hat_K_nmp
                                size_t idx_m1 = this->get_fft_index_from_math(m1);
                                size_t idx_m2 = this->get_fft_index_from_math(m2);
                                f_hat_f_hat_K_npm +=
                                    this->f_hat_previous_(idx_p1, idx_p2) *
                                    this->f_hat_previous_(idx_m1, idx_m2) *
                                    this->compute_K_npm(n1, n2, p1, p2, m1, m2, R);
                                // f_hat_f_hat_K_npm += this->compute_K_npm(n1, n2, p1, p2, m1, m2);
                            }
                        }
                    }
                }

                // compute f_hat at current time step

                this->f_hat_current_(idx_n1, idx_n2) =
                    this->f_hat_previous_(idx_n1, idx_n2) +
                    dt * f_hat_f_hat_K_npm +
                    C * dt * f_hat_H_np;

                // if (t_step == 1)
                // {
                //     cout << n1 << "," << n2 << ": " << endl;
                //     cout << this->comp_2D_->f_hat_current_(idx_n1, idx_n2) << endl;
                //     cout << this->comp_2D_->f_hat_previous_(idx_n1, idx_n2) << endl;
                //     cout << f_hat_f_hat_K_npm << endl;
                //     cout << f_hat_H_np << endl;
                // }
            }
        }

        // Update previous time step solution
        copy_tensor(this->f_hat_previous_, this->f_hat_current_, nthreads);

        // if (t_step == 1)
        // {
        //     //this->comp_2D_->W_hat_tensor_.Print();
        //     //this->comp_2D_->f_hat_current_.Print();
        //     //this->comp_2D_->f_hat_previous_.Print();
        //     this->f_solution_.Print();
        //     cout <<this->f_solution_.get_L2_norm() << endl;
        //     cout <<this->f_solution_.get_max_by_modulus() << endl;
        //     cout <<this->f_solution_.get_min_by_modulus() << endl;
        //     cout <<this->f_solution_.get_max_real_part() << endl;
        //     cout <<this->f_solution_.get_min_real_part() << endl;
        //     cout <<this->f_solution_.get_max_imag_part() << endl;
        //     cout <<this->f_solution_.get_min_imag_part() << endl;
        //     break;
        // }

        this->recontruct_solution_manual();
        this->f_solution_time_.insert_slice(0, t_step, this->f_solution_, nthreads);
        this->f_hat_time_.insert_slice(0, t_step, this->f_hat_current_, nthreads);

        this->print_infomation_each_timestep(t_step);
    }
}

void Solver_2D::compute_f_initial_hat_manual()
{
    // Implementation of manual initial condition computation if needed
    double R = this->physical_model_->model_params_.R;
    int N = this->sim_params_.N;
    double hx = this->manual_step_size_;
    double hy = this->manual_step_size_;

    int Nx = static_cast<int>(2.0 * R / hx);
    int Ny = static_cast<int>(2.0 * R / hy);

    // cout<<"Manual initial condition computation: Nx = " << Nx << ", Ny = " << Ny << endl;

#pragma omp parallel for collapse(2) num_threads(this->sim_params_.num_threads)
    for (int nx = -N; nx < N; ++nx)
    {
        for (int ny = -N; ny < N; ++ny)
        {
            size_t idx = this->get_fft_index_from_math(nx);
            size_t idy = this->get_fft_index_from_math(ny);

            complex<double> sum = complex<double>(0.0, 0.0);

            for (int p = 0; p < Nx; ++p)
            {
                double kx_mid = -R + (p + 0.5) * hx;

                for (int q = 0; q < Ny; ++q)
                {
                    double ky_mid = -R + (q + 0.5) * hy;

                    double phase =
                        -constants::PI / R * (nx * kx_mid + ny * ky_mid);

                    sum += this->physical_model_->func_initial_condition({kx_mid, ky_mid}) * exp(complex<double>(0, phase));
                }
            }

            complex<double> f_hat_manual = sum * (hx * hy) / ((2.0 * R) * (2.0 * R));

            this->f_hat_previous_(idx, idy) = f_hat_manual;
        }
    }

    this->f_hat_previous_.WriteToHDF5(this->sim_params_.output_directory + "/f_hat_initial.h5");
    this->f_hat_previous_.WriteToText(this->sim_params_.output_directory + "/f_hat_initial.txt");
}

void Solver_2D::compute_W_hat()
{

    // FFTW C2C for W_tensor_
    int grid_size = 2 * this->sim_params_.N;
    int Nx = grid_size;
    int Ny = grid_size;
    int Nz = grid_size;
    int Nw = grid_size;

    FFT_Config config_c2c_4d = {
        .dim = 4,
        .shape = {Nx, Ny, Nz, Nw},
        .type = FFT_Type::C2C,
        .direction = FFT_Direction::Forward,
        .inplace = false,
        .normalize = true,
        .denormalize = false,
        .nthreads = this->nthreads_FFTW_,
        .flags = FFT_Flag::Estimate};

    FFT_Executor fft_c2c_4d;
    fft_c2c_4d.create_plan(
        config_c2c_4d,
        this->W_tensor_.data(),
        this->W_hat_tensor_.data());

    fft_c2c_4d.execute_plan(
        config_c2c_4d,
        this->W_tensor_.data(),
        this->W_hat_tensor_.data());

    fft_c2c_4d.destroy_plan();

    // this->W_hat_tensor_.WriteToHDF5(this->sim_params_.output_directory + "/W_hat_tensor.h5");
    // this->W_hat_tensor_.WriteToText(this->sim_params_.output_directory + "/W_hat_tensor.txt");
}

void Solver_2D::compute_W_hat_manual()
{
    double R = this->physical_model_->model_params_.R;
    int N = this->sim_params_.N;
    double h = this->manual_step_size_;

    int M = static_cast<int>(2.0 * R / h);

#pragma omp parallel for collapse(4) num_threads(this->sim_params_.num_threads)
    for (int n1 = -N; n1 < N; ++n1)
        for (int n2 = -N; n2 < N; ++n2)
            for (int n3 = -N; n3 < N; ++n3)
                for (int n4 = -N; n4 < N; ++n4)
                {
                    complex<double> sum = complex<double>(0.0, 0.0);

                    for (int p1 = 0; p1 < M; ++p1)
                    {
                        double x1 = -R + (p1 + 0.5) * h;
                        for (int p2 = 0; p2 < M; ++p2)
                        {
                            double x2 = -R + (p2 + 0.5) * h;
                            for (int p3 = 0; p3 < M; ++p3)
                            {
                                double x3 = -R + (p3 + 0.5) * h;
                                for (int p4 = 0; p4 < M; ++p4)
                                {
                                    double x4 = -R + (p4 + 0.5) * h;

                                    double phase =
                                        -constants::PI / R * (n1 * x1 + n2 * x2 + n3 * x3 + n4 * x4);

                                    sum += this->physical_model_->func_W(
                                               {x1, x2, x3, x4}) *
                                           std::exp(std::complex<double>(0.0, phase));
                                }
                            }
                        }
                    }
                    // cout<<sum<<endl;
                    complex<double> W_hat =
                        sum * pow(h, 4) / pow(2.0 * R, 4);

                    size_t i1 = this->get_fft_index_from_math(n1);
                    size_t i2 = this->get_fft_index_from_math(n2);
                    size_t i3 = this->get_fft_index_from_math(n3);
                    size_t i4 = this->get_fft_index_from_math(n4);

                    // cout << "W_hat(" << n1 << "," << n2 << "," << n3 << "," << n4 << ") = " << W_hat << endl;
                    this->W_hat_tensor_(i1, i2, i3, i4) = W_hat;
                }
    this->W_hat_tensor_.WriteToHDF5(this->sim_params_.output_directory + "/W_hat_tensor.h5");
    this->W_hat_tensor_.WriteToText(this->sim_params_.output_directory + "/W_hat_tensor.txt");
}

// Ở solver_2D.cpp, thêm:
complex<double> Solver_2D::compute_K_npm(int n1, int n2, int p1, int p2, int m1, int m2, double R)
{
    size_t idx_n1_minus_p1 = this->get_fft_index_from_math(n1 - p1);
    size_t idx_n2_minus_p2 = this->get_fft_index_from_math(n2 - p2);
    size_t idx_n1_minus_m1 = this->get_fft_index_from_math(n1 - m1);
    size_t idx_n2_minus_m2 = this->get_fft_index_from_math(n2 - m2);
    size_t idx_n1_minus_p1_m1 = this->get_fft_index_from_math(n1 - p1 - m1);
    size_t idx_n2_minus_p2_m2 = this->get_fft_index_from_math(n2 - p2 - m2);
    size_t idx_minus_p1 = this->get_fft_index_from_math(-p1);
    size_t idx_minus_p2 = this->get_fft_index_from_math(-p2);
    size_t idx_minus_m1 = this->get_fft_index_from_math(-m1);
    size_t idx_minus_m2 = this->get_fft_index_from_math(-m2);
    size_t idx_minus_p1_m1 = this->get_fft_index_from_math(-p1 - m1);
    size_t idx_minus_p2_m2 = this->get_fft_index_from_math(-p2 - m2);

    // K_npm_1: W(n-p, n-m) [+]
    complex<double> K_npm_1 = this->W_hat_tensor_(idx_n1_minus_p1, idx_n2_minus_p2, idx_n1_minus_m1, idx_n2_minus_m2);
    // K_npm_2: W(n-p, -m) [-]
    complex<double> K_npm_2 = this->W_hat_tensor_(idx_n1_minus_p1, idx_n2_minus_p2, idx_minus_m1, idx_minus_m2);
    // K_npm_3: W(-p, n-m) [-]
    complex<double> K_npm_3 = this->W_hat_tensor_(idx_minus_p1, idx_minus_p2, idx_n1_minus_m1, idx_n2_minus_m2);
    // K_npm_4: W(n-p-m, n-p) [-]
    complex<double> K_npm_4 = this->W_hat_tensor_(idx_n1_minus_p1_m1, idx_n2_minus_p2_m2, idx_n1_minus_p1, idx_n2_minus_p2);
    // K_npm_5: W(n-p-m, -p) [+]
    complex<double> K_npm_5 = this->W_hat_tensor_(idx_n1_minus_p1_m1, idx_n2_minus_p2_m2, idx_minus_p1, idx_minus_p2);
    // K_npm_6: W(-p-m, n-p) [+]
    complex<double> K_npm_6 = this->W_hat_tensor_(idx_minus_p1_m1, idx_minus_p2_m2, idx_n1_minus_p1, idx_n2_minus_p2);
    // K_npm_7: W(n-p, n-p-m) [-]
    complex<double> K_npm_7 = this->W_hat_tensor_(idx_n1_minus_p1, idx_n2_minus_p2, idx_n1_minus_p1_m1, idx_n2_minus_p2_m2);
    // K_npm_8: W(n-p, -p-m) [+]
    complex<double> K_npm_8 = this->W_hat_tensor_(idx_n1_minus_p1, idx_n2_minus_p2, idx_minus_p1_m1, idx_minus_p2_m2);
    // K_npm_9: W(-p, n-p-m) [+]
    complex<double> K_npm_9 = this->W_hat_tensor_(idx_minus_p1, idx_minus_p2, idx_n1_minus_p1_m1, idx_n2_minus_p2_m2);

    double R2 = 4 * R * R;
    // double R2 = 1;

    // cout<<"K_npm components: " << K_npm_1 << ", " << K_npm_2 << ", " << K_npm_3 << ", " << K_npm_4 << ", " << K_npm_5 << ", " << K_npm_6 << ", " << K_npm_7 << ", " << K_npm_8 << ", " << K_npm_9 << endl;
    // cout<<"K_npm at (" << n1 << ", " << n2 << ", " << p1 << ", " << p2 << ", " << m1 << ", " << m2 << ") = " << R2 * (K_npm_1 - K_npm_2 - K_npm_3 - K_npm_4 + K_npm_5 + K_npm_6 - K_npm_7 + K_npm_8 + K_npm_9) << endl;
    return R2 * (K_npm_1 - K_npm_2 - K_npm_3 - K_npm_4 + K_npm_5 + K_npm_6 - K_npm_7 + K_npm_8 + K_npm_9);
}

complex<double> Solver_2D::compute_H_np(int n1, int n2, int p1, int p2, double R)
{
    size_t idx_n1_minus_p1 = this->get_fft_index_from_math(n1 - p1);
    size_t idx_n2_minus_p2 = this->get_fft_index_from_math(n2 - p2);
    size_t idx_minus_p1 = this->get_fft_index_from_math(-p1);
    size_t idx_minus_p2 = this->get_fft_index_from_math(-p2);
    complex<double> H_np_1 =
        this->W_hat_tensor_(
            idx_n1_minus_p1,
            idx_n2_minus_p2,
            idx_n1_minus_p1,
            idx_n2_minus_p2);

    complex<double> H_np_2 =
        this->W_hat_tensor_(idx_n1_minus_p1,
                            idx_n2_minus_p2,
                            idx_minus_p1,
                            idx_minus_p2);
    complex<double> H_np_3 =
        this->W_hat_tensor_(idx_minus_p1,
                            idx_minus_p2,
                            idx_n1_minus_p1,
                            idx_n2_minus_p2);

    // cout<<"H_np components: " << H_np_1 << ", " << H_np_2 << ", " << H_np_3 << endl;
    double R2 = 4 * R * R;
    // double R2 = 1;
    // cout<<"H_np at (" << n1 << ", " << n2 << ", " << p1 << ", " << p2 << ") = " << - R2 *(H_np_1 - H_np_2 - H_np_3) << endl;
    return -R2 * (H_np_1 - H_np_2 - H_np_3);
}

void Solver_2D::recontruct_solution_manual()
{
    // Implementation of manual reconstruction of solution if needed

    vector<double> grid_vector = this->k_grid_.get_data_vector();

    int N = this->sim_params_.N;
    double R = this->physical_model_->model_params_.R;

#pragma omp parallel for collapse(2) num_threads(this->sim_params_.num_threads)
    for (int i = -N; i < N; ++i)
    {
        for (int j = -N; j < N; ++j)
        {
            size_t i_index = this->get_fft_index_from_math(i);
            double x = grid_vector[i_index];
            size_t j_index = this->get_fft_index_from_math(j);
            double y = grid_vector[j_index];

            complex<double> sum = 0.0;
            for (int n1 = -N; n1 < N; ++n1)
            {
                size_t idx = this->get_fft_index_from_math(n1);
                for (int n2 = -N; n2 < N; ++n2)
                {
                    size_t idx2 = this->get_fft_index_from_math(n2);

                    complex<double> f_hat_n = this->f_hat_current_(idx, idx2);

                    double phase = constants::PI / R * (n1 * x + n2 * y);

                    sum += f_hat_n * exp(complex<double>(0, phase));
                }
            }
            complex<double> f_manual = sum;
            this->f_solution_(i_index, j_index) = f_manual;
        }
    }
}