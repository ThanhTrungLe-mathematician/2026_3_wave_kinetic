#include "solver_2D_new.h"
#include "fft_executor.h"
#include "utilities.h"
#include "tensor_free_function.h"
#include <complex>
#include <cmath>
using namespace std;

Solver_2D_new::Solver_2D_new(const ModelParameters &model_params, SimulationParameters &sim_params, const SpectralStabilizationConfig &spectral_stabilization_config)
    : Solver_Base_2D(model_params, sim_params, spectral_stabilization_config) {}

Solver_2D_new::~Solver_2D_new()
{
    k_grid_.Destroy();
    W_tensor_.Destroy();
    f_initial_.Destroy();
    f_solution_.Destroy();
    f_solution_time_.Destroy();
    f_hat_previous_.Destroy();
    f_hat_current_.Destroy();
    f_hat_time_.Destroy();
    T_.Destroy();
    // T_hat_.Destroy();
    Y_.Destroy();
    // Y_hat_.Destroy();
    U_.Destroy();
    // U_hat_.Destroy();
    I_.Destroy();
    // I_hat_.Destroy();

    fft_idx_.clear();

    t_save_points_.Destroy();

    Q_hat_temp_1_.Destroy();
    Q_hat_temp_2_.Destroy();
    Q_hat_temp_3_.Destroy();
    Q_hat_temp_4_.Destroy();
}

void Solver_2D_new::create_objects()
{
    double estimated_memory_GB = 0.0;
    size_t grid_size = static_cast<size_t>(2 * this->sim_params_.N);
    size_t N_time = static_cast<size_t>(this->sim_params_.total_number_of_times_saves);

    estimated_memory_GB += 2 * estimate_memory_usage_GB<complex<double>>({grid_size});
    estimated_memory_GB += 4 * estimate_memory_usage_GB<complex<double>>({grid_size, grid_size});
    estimated_memory_GB += 2 * estimate_memory_usage_GB<complex<double>>({N_time, grid_size, grid_size});
    estimated_memory_GB += 5 * estimate_memory_usage_GB<complex<double>>({grid_size, grid_size, grid_size, grid_size});
    estimated_memory_GB += 1 * estimate_memory_usage_GB<complex<double>>({grid_size, grid_size});

    if (estimated_memory_GB > this->sim_params_.max_memory_GB)
    {
        throw runtime_error("Estimated memory usage " + to_string(estimated_memory_GB) + " GB exceeds the limit of " + to_string(this->sim_params_.max_memory_GB) + " GB. Please increase max_memory_GB or reduce grid size N.");
    }
    else
    {
        cout << "Estimated memory usage: " << estimated_memory_GB << " GB" << endl;

        k_grid_.resize(grid_size);
        W_tensor_.resize(grid_size, grid_size, grid_size, grid_size);
        f_initial_.resize(grid_size, grid_size);
        f_solution_time_.resize(N_time, grid_size, grid_size);
        f_hat_previous_.resize(grid_size, grid_size);
        f_hat_current_.resize(grid_size, grid_size);
        f_solution_.resize(grid_size, grid_size);
        f_hat_time_.resize(N_time, grid_size, grid_size);

        T_.resize(grid_size, grid_size, grid_size, grid_size);
        // T_hat_.resize(grid_size, grid_size, grid_size, grid_size);
        Y_.resize(grid_size, grid_size, grid_size, grid_size);
        // Y_hat_.resize(grid_size, grid_size, grid_size, grid_size);
        U_.resize(grid_size, grid_size, grid_size, grid_size);
        // U_hat_.resize(grid_size, grid_size, grid_size, grid_size);
        I_.resize(grid_size, grid_size, grid_size, grid_size);
        // I_hat_.resize(grid_size, grid_size, grid_size, grid_size);

        fft_idx_.resize(grid_size);

        t_save_points_.resize(2, N_time);

        if (this->sim_params_.ode_solver == ODE_solver_type::RK2 ||
            this->sim_params_.ode_solver == ODE_solver_type::RK3 ||
            this->sim_params_.ode_solver == ODE_solver_type::SSP_RK3 ||
            this->sim_params_.ode_solver == ODE_solver_type::RK4)
        {
            Q_hat_temp_1_.resize(grid_size, grid_size);
            Q_hat_temp_2_.resize(grid_size, grid_size);
            Q_hat_temp_3_.resize(grid_size, grid_size);
            Q_hat_temp_4_.resize(grid_size, grid_size);
        }
    }
}

void Solver_2D_new::compute_solution()
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
    const int grid_size = 2 * this->sim_params_.N;
    const int N = this->sim_params_.N;
    const int nthreads = this->sim_params_.num_threads;
    const int nthreads_FFTW = this->nthreads_FFTW_;

    this->construct_k_grid();

    this->make_saved_times_points();

    // Precompute FFTW indices
    this->compute_fftw_indices();

    this->construct_f_initial_tensor();
    this->f_solution_time_.insert_slice(0, 0, this->f_initial_, nthreads);
    copy_tensor(this->f_solution_, this->f_initial_, nthreads);

    this->compute_f_initial_hat();
    this->f_hat_time_.insert_slice(0, 0, this->f_hat_previous_, nthreads);

    this->construct_W_tensor();

    // Create FFT executors and plans
    FFT_Executor fft_c2c_4d_forward;
    FFT_Executor fft_c2c_2d_backward;
    this->block_create_fftw_plans(fft_c2c_4d_forward, fft_c2c_2d_backward, grid_size, nthreads_FFTW);

    // Time-stepping loop implementation
    if (this->sim_params_.ode_solver == ODE_solver_type::Euler)
    {
        this->apply_ODE_solver_euler(fft_c2c_4d_forward, fft_c2c_2d_backward);
    }
    else if (this->sim_params_.ode_solver == ODE_solver_type::RK2)
    {
        this->apply_ODE_solver_RK2(fft_c2c_4d_forward, fft_c2c_2d_backward);
    }
    else if (this->sim_params_.ode_solver == ODE_solver_type::RK3)
    {
        this->apply_ODE_solver_RK3(fft_c2c_4d_forward, fft_c2c_2d_backward);
    }
    else if (this->sim_params_.ode_solver == ODE_solver_type::SSP_RK3)
    {
        this->apply_ODE_solver_SSP_RK3(fft_c2c_4d_forward, fft_c2c_2d_backward);
    }
    else if (this->sim_params_.ode_solver == ODE_solver_type::RK4)
    {
        this->apply_ODE_solver_RK4(fft_c2c_4d_forward, fft_c2c_2d_backward);
    }
    else
    {
        throw runtime_error("ODE solver not implemented yet.");
    }

    // Destroy FFT plans
    fft_c2c_2d_backward.destroy_plan();
    fft_c2c_4d_forward.destroy_plan();
}

void Solver_2D_new::compute_T_Y_U_I_tensors()
{
    size_t grid_size = static_cast<size_t>(2 * this->sim_params_.N);

#pragma omp parallel for collapse(2) num_threads(this->sim_params_.num_threads)
    for (size_t i1 = 0; i1 < grid_size; ++i1)
    {
        for (size_t i2 = 0; i2 < grid_size; ++i2)
        {
            complex<double> f_k2 = this->f_solution_(i1, i2);

            for (size_t i3 = 0; i3 < grid_size; ++i3)
            {
                for (size_t i4 = 0; i4 < grid_size; ++i4)
                {
                    // k2 + k3
                    size_t idx_sum1 = (i1 + i3) % grid_size;
                    size_t idx_sum2 = (i2 + i4) % grid_size;

                    complex<double> W_val = this->W_tensor_(i1, i2, i3, i4);
                    complex<double> f_k3 = this->f_solution_(i3, i4);
                    complex<double> f_sum = this->f_solution_(idx_sum1, idx_sum2);

                    this->T_(i1, i2, i3, i4) = W_val * f_k2 * f_k3;
                    this->Y_(i1, i2, i3, i4) = W_val * f_sum * f_k2;
                    this->U_(i1, i2, i3, i4) = W_val * f_sum * f_k3;
                    this->I_(i1, i2, i3, i4) = W_val * f_sum;
                }
            }
        }
    }
}

void Solver_2D_new::apply_de_aliasing_two_thirds_T_Y_U_I_hat()
{
    const int N = this->sim_params_.N;
    const double cutoff_fraction = this->spectral_config_.two_thirds_dealiasing.cutoff_fraction;
    const int cutoff = static_cast<int>(std::floor(cutoff_fraction * N));
    const int cutoff_squared = cutoff * cutoff;

    if (this->spectral_config_.two_thirds_dealiasing.use_L2)
    {
#pragma omp parallel for collapse(4) num_threads(this->sim_params_.num_threads) schedule(guided)
        for (int n1 = -N; n1 < N; ++n1)
        {
            for (int n2 = -N; n2 < N; ++n2)
            {
                for (int n3 = -N; n3 < N; ++n3)
                {
                    for (int n4 = -N; n4 < N; ++n4)
                    {
                        if ((n1 * n1 + n2 * n2 + n3 * n3 + n4 * n4) > cutoff_squared)
                        {
                            size_t idx_n1 = this->fft_idx_[n1 + N];
                            size_t idx_n2 = this->fft_idx_[n2 + N];
                            size_t idx_n3 = this->fft_idx_[n3 + N];
                            size_t idx_n4 = this->fft_idx_[n4 + N];

                            this->T_(idx_n1, idx_n2, idx_n3, idx_n4) = complex<double>(0.0, 0.0);
                            this->Y_(idx_n1, idx_n2, idx_n3, idx_n4) = complex<double>(0.0, 0.0);
                            this->U_(idx_n1, idx_n2, idx_n3, idx_n4) = complex<double>(0.0, 0.0);
                            this->I_(idx_n1, idx_n2, idx_n3, idx_n4) = complex<double>(0.0, 0.0);
                        }
                    }
                }
            }
        }
    }
    else
    {
#pragma omp parallel for collapse(4) num_threads(this->sim_params_.num_threads)
        for (int n1 = -N; n1 < N; ++n1)
        {
            for (int n2 = -N; n2 < N; ++n2)
            {
                for (int n3 = -N; n3 < N; ++n3)
                {
                    for (int n4 = -N; n4 < N; ++n4)
                    {
                        if (std::abs(n1) > cutoff || std::abs(n2) > cutoff || std::abs(n3) > cutoff || std::abs(n4) > cutoff)
                        {
                            size_t idx_n1 = this->fft_idx_[n1 + N];
                            size_t idx_n2 = this->fft_idx_[n2 + N];
                            size_t idx_n3 = this->fft_idx_[n3 + N];
                            size_t idx_n4 = this->fft_idx_[n4 + N];

                            this->T_(idx_n1, idx_n2, idx_n3, idx_n4) = complex<double>(0.0, 0.0);
                            this->Y_(idx_n1, idx_n2, idx_n3, idx_n4) = complex<double>(0.0, 0.0);
                            this->U_(idx_n1, idx_n2, idx_n3, idx_n4) = complex<double>(0.0, 0.0);
                            this->I_(idx_n1, idx_n2, idx_n3, idx_n4) = complex<double>(0.0, 0.0);
                        }
                    }
                }
            }
        }
    }
}

void Solver_2D_new::apply_early_de_aliasing()
{
    const int N = this->sim_params_.N;
    const double cutoff_fraction = this->spectral_config_.early_dealiasing.cutoff_fraction;
    const int cutoff = static_cast<int>(std::floor(cutoff_fraction * N));
    const int cutoff_squared = cutoff * cutoff;
    const size_t grid_size = static_cast<size_t>(2 * N);

    if (this->spectral_config_.early_dealiasing.use_L2)
    {
        // cout << "Applying early de-aliasing by L2 norm..." << endl;

#pragma omp parallel for collapse(2) num_threads(this->sim_params_.num_threads) schedule(guided)
        for (int n1 = -N; n1 < N; ++n1)
        {
            for (int n2 = -N; n2 < N; ++n2)
            {
                size_t idx_n1 = this->fft_idx_[n1 + N];
                size_t idx_n2 = this->fft_idx_[n2 + N];
                complex<double> f_k2 = this->f_solution_(idx_n1, idx_n2);

                for (int n3 = -N; n3 < N; ++n3)
                {
                    size_t idx_n3 = this->fft_idx_[n3 + N];
                    for (int n4 = -N; n4 < N; ++n4)
                    {
                        size_t idx_n4 = this->fft_idx_[n4 + N];

                        int sum1 = (n1 + n3);
                        int sum2 = (n2 + n4);

                        if ((n1 * n1 + n2 * n2 + n3 * n3 + n4 * n4) > cutoff_squared)
                        {
                            this->T_(idx_n1, idx_n2, idx_n3, idx_n4) = complex<double>(0.0, 0.0);
                            this->Y_(idx_n1, idx_n2, idx_n3, idx_n4) = complex<double>(0.0, 0.0);
                            this->U_(idx_n1, idx_n2, idx_n3, idx_n4) = complex<double>(0.0, 0.0);
                            this->I_(idx_n1, idx_n2, idx_n3, idx_n4) = complex<double>(0.0, 0.0);
                            continue;
                        }

                        complex<double> W_val = this->W_tensor_(idx_n1, idx_n2, idx_n3, idx_n4);
                        complex<double> f_k3 = this->f_solution_(idx_n3, idx_n4);

                        this->T_(idx_n1, idx_n2, idx_n3, idx_n4) = W_val * f_k2 * f_k3;

                        if ((sum1 * sum1 + sum2 * sum2) > cutoff_squared)
                        {
                            this->Y_(idx_n1, idx_n2, idx_n3, idx_n4) = complex<double>(0.0, 0.0);
                            this->U_(idx_n1, idx_n2, idx_n3, idx_n4) = complex<double>(0.0, 0.0);
                            this->I_(idx_n1, idx_n2, idx_n3, idx_n4) = complex<double>(0.0, 0.0);
                            continue;
                        }

                        size_t idx_sum1 = this->fft_idx_[sum1 + N];
                        size_t idx_sum2 = this->fft_idx_[sum2 + N];
                        complex<double> f_sum = this->f_solution_(idx_sum1, idx_sum2);

                        this->Y_(idx_n1, idx_n2, idx_n3, idx_n4) = W_val * f_sum * f_k2;
                        this->U_(idx_n1, idx_n2, idx_n3, idx_n4) = W_val * f_sum * f_k3;
                        this->I_(idx_n1, idx_n2, idx_n3, idx_n4) = W_val * f_sum;
                    }
                }
            }
        }
    }
    else
    {
        // cout << "Applying early de-aliasing by maximum mode..." << endl;
#pragma omp parallel for collapse(2) num_threads(this->sim_params_.num_threads)
        for (int n1 = -N; n1 < N; ++n1)
        {
            for (int n2 = -N; n2 < N; ++n2)
            {
                size_t idx_n1 = this->fft_idx_[n1 + N];
                size_t idx_n2 = this->fft_idx_[n2 + N];
                complex<double> f_k2 = this->f_solution_(idx_n1, idx_n2);

                for (int n3 = -N; n3 < N; ++n3)
                {
                    size_t idx_n3 = this->fft_idx_[n3 + N];
                    for (int n4 = -N; n4 < N; ++n4)
                    {
                        size_t idx_n4 = this->fft_idx_[n4 + N];

                        int sum1 = (n1 + n3);
                        int sum2 = (n2 + n4);

                        if (std::abs(n1) > cutoff || std::abs(n2) > cutoff || std::abs(n3) > cutoff || std::abs(n4) > cutoff)
                        {
                            this->T_(idx_n1, idx_n2, idx_n3, idx_n4) = complex<double>(0.0, 0.0);
                            this->Y_(idx_n1, idx_n2, idx_n3, idx_n4) = complex<double>(0.0, 0.0);
                            this->U_(idx_n1, idx_n2, idx_n3, idx_n4) = complex<double>(0.0, 0.0);
                            this->I_(idx_n1, idx_n2, idx_n3, idx_n4) = complex<double>(0.0, 0.0);
                            continue;
                        }

                        complex<double> W_val = this->W_tensor_(idx_n1, idx_n2, idx_n3, idx_n4);
                        complex<double> f_k3 = this->f_solution_(idx_n3, idx_n4);

                        this->T_(idx_n1, idx_n2, idx_n3, idx_n4) = W_val * f_k2 * f_k3;

                        if (std::abs(sum1) > cutoff || std::abs(sum2) > cutoff)
                        {
                            this->Y_(idx_n1, idx_n2, idx_n3, idx_n4) = complex<double>(0.0, 0.0);
                            this->U_(idx_n1, idx_n2, idx_n3, idx_n4) = complex<double>(0.0, 0.0);
                            this->I_(idx_n1, idx_n2, idx_n3, idx_n4) = complex<double>(0.0, 0.0);
                            continue;
                        }

                        size_t idx_sum1 = this->fft_idx_[sum1 + N];
                        size_t idx_sum2 = this->fft_idx_[sum2 + N];
                        complex<double> f_sum = this->f_solution_(idx_sum1, idx_sum2);

                        this->Y_(idx_n1, idx_n2, idx_n3, idx_n4) = W_val * f_sum * f_k2;
                        this->U_(idx_n1, idx_n2, idx_n3, idx_n4) = W_val * f_sum * f_k3;
                        this->I_(idx_n1, idx_n2, idx_n3, idx_n4) = W_val * f_sum;
                    }
                }
            }
        }
    }
}

void Solver_2D_new::apply_ODE_solver_euler(FFT_Executor &fft_c2c_4d_forward,
                                           FFT_Executor &fft_c2c_2d_backward)
{
    int save_time_index = 1; // start from 1 since 0 is already saved
    const int nthreads = this->sim_params_.num_threads;
    const int grid_size = 2 * this->sim_params_.N;
    const double dt = this->sim_params_.dt;
    const double C = this->physical_model_->C_;

    const double R2_RHS = 4 * this->physical_model_->model_params_.R * this->physical_model_->model_params_.R; // precompute 4*R^2 for RHS computation

    for (int t_step = 1; t_step < this->N_time_; ++t_step)
    {
        // Compute T_hat, Y_hat, U_hat, I_hat
        this->block_compute_T_Y_U_I_hat(fft_c2c_4d_forward);

#pragma omp parallel for num_threads(nthreads)
        for (size_t idx_n1 = 0; idx_n1 < static_cast<size_t>(grid_size); ++idx_n1)
        {
            for (size_t idx_n2 = 0; idx_n2 < static_cast<size_t>(grid_size); ++idx_n2)
            {
                complex<double> rhs = this->compute_RHS(idx_n1, idx_n2, R2_RHS, C);

                this->f_hat_current_(idx_n1, idx_n2) = this->f_hat_previous_(idx_n1, idx_n2) + dt * rhs;
            }
        }

        // Apply spectral stabilization if enabled
        this->block_apply_spectral_stabilization_f_hat();

        // Update and store solution for the next time step
        this->block_update_and_store_solution_for_the_next_time_step(fft_c2c_2d_backward, save_time_index, t_step);
    }
}

void Solver_2D_new::apply_ODE_solver_RK2(FFT_Executor &fft_c2c_4d_forward,
                                         FFT_Executor &fft_c2c_2d_backward)
{
    int save_time_index = 1; // start from 1 since 0 is already saved
    const int nthreads = this->sim_params_.num_threads;
    const int grid_size = 2 * this->sim_params_.N;
    const double dt = this->sim_params_.dt;
    const double C = this->physical_model_->C_;

    const double R2_RHS = 4 * this->physical_model_->model_params_.R * this->physical_model_->model_params_.R; // precompute 4*R^2 for RHS computation

    for (int t_step = 1; t_step < this->N_time_; ++t_step)
    {
        // Stage 1 for RK2
        this->block_compute_T_Y_U_I_hat(fft_c2c_4d_forward);

#pragma omp parallel for num_threads(nthreads)
        for (size_t idx_n1 = 0; idx_n1 < static_cast<size_t>(grid_size); ++idx_n1)
        {
            for (size_t idx_n2 = 0; idx_n2 < static_cast<size_t>(grid_size); ++idx_n2)
            {
                Q_hat_temp_1_(idx_n1, idx_n2) = this->compute_RHS(idx_n1, idx_n2, R2_RHS, C);
                this->f_hat_current_(idx_n1, idx_n2) = this->f_hat_previous_(idx_n1, idx_n2) + dt * Q_hat_temp_1_(idx_n1, idx_n2);
            }
        }

        this->block_apply_spectral_stabilization_f_hat();

        // Stage 2 for RK2
        fft_c2c_2d_backward.execute_plan(this->f_hat_current_.data(), this->f_solution_.data());
        this->block_compute_T_Y_U_I_hat(fft_c2c_4d_forward);

#pragma omp parallel for num_threads(nthreads)
        for (size_t idx_n1 = 0; idx_n1 < static_cast<size_t>(grid_size); ++idx_n1)
        {
            for (size_t idx_n2 = 0; idx_n2 < static_cast<size_t>(grid_size); ++idx_n2)
            {
                Q_hat_temp_2_(idx_n1, idx_n2) = this->compute_RHS(idx_n1, idx_n2, R2_RHS, C);
                this->f_hat_current_(idx_n1, idx_n2) = this->f_hat_previous_(idx_n1, idx_n2) +
                                                       0.5 * dt * (Q_hat_temp_1_(idx_n1, idx_n2) + Q_hat_temp_2_(idx_n1, idx_n2));
            }
        }

        this->block_apply_spectral_stabilization_f_hat();

        // Update and store solution
        this->block_update_and_store_solution_for_the_next_time_step(fft_c2c_2d_backward, save_time_index, t_step);
    }
}

void Solver_2D_new::apply_ODE_solver_RK3(FFT_Executor &fft_c2c_4d_forward,
                                         FFT_Executor &fft_c2c_2d_backward)
{
    // Implementation of RK3 solver
    int save_time_index = 1; // start from 1 since 0 is already saved
    const int nthreads = this->sim_params_.num_threads;
    const int grid_size = 2 * this->sim_params_.N;
    const double dt = this->sim_params_.dt;
    const double C = this->physical_model_->C_;

    const double R2_RHS = 4 * this->physical_model_->model_params_.R * this->physical_model_->model_params_.R; // precompute 4*R^2 for RHS computation

    for (int t_step = 1; t_step < this->N_time_; ++t_step)
    {
        // Perform FFT on T, Y, U, I tensors to get T_hat, Y_hat, U_hat, I_hat
        this->block_compute_T_Y_U_I_hat(fft_c2c_4d_forward);

        // Stage 1 for RK3
#pragma omp parallel for num_threads(nthreads)
        for (size_t idx_n1 = 0; idx_n1 < static_cast<size_t>(grid_size); ++idx_n1)
        {
            for (size_t idx_n2 = 0; idx_n2 < static_cast<size_t>(grid_size); ++idx_n2)
            {
                // Compute f_hat at current time step

                Q_hat_temp_1_(idx_n1, idx_n2) = compute_RHS(idx_n1, idx_n2, R2_RHS, C);

                this->f_hat_current_(idx_n1, idx_n2) =
                    this->f_hat_previous_(idx_n1, idx_n2) + 0.5 * dt * Q_hat_temp_1_(idx_n1, idx_n2);
            }
        }

        // Apply spectral stabilization if enabled
        this->block_apply_spectral_stabilization_f_hat();
        // Finish Stage 1 for RK3

        //////////////////////////////////////////////////////////////////////////

        // Stage 2 for RK3
        fft_c2c_2d_backward.execute_plan(this->f_hat_current_.data(), this->f_solution_.data());

        // Perform FFT on T, Y, U, I tensors to get T_hat, Y_hat, U_hat, I_hat
        this->block_compute_T_Y_U_I_hat(fft_c2c_4d_forward);

#pragma omp parallel for num_threads(nthreads)
        for (size_t idx_n1 = 0; idx_n1 < static_cast<size_t>(grid_size); ++idx_n1)
        {
            for (size_t idx_n2 = 0; idx_n2 < static_cast<size_t>(grid_size); ++idx_n2)
            {
                // Compute f_hat at current time step
                Q_hat_temp_2_(idx_n1, idx_n2) = compute_RHS(idx_n1, idx_n2, R2_RHS, C);

                this->f_hat_current_(idx_n1, idx_n2) =
                    this->f_hat_previous_(idx_n1, idx_n2) - dt * Q_hat_temp_1_(idx_n1, idx_n2) + 2 * dt * Q_hat_temp_2_(idx_n1, idx_n2);
            }
        }

        // Apply spectral stabilization if enabled
        this->block_apply_spectral_stabilization_f_hat();

        // Finish Stage 2 for RK3

        //////////////////////////////////////////////////////////////////////////

        // Stage 3 for RK3
        fft_c2c_2d_backward.execute_plan(this->f_hat_current_.data(), this->f_solution_.data());

        // Perform FFT on T, Y, U, I tensors to get T_hat, Y_hat, U_hat, I_hat
        this->block_compute_T_Y_U_I_hat(fft_c2c_4d_forward);

#pragma omp parallel for num_threads(nthreads)
        for (size_t idx_n1 = 0; idx_n1 < static_cast<size_t>(grid_size); ++idx_n1)
        {
            for (size_t idx_n2 = 0; idx_n2 < static_cast<size_t>(grid_size); ++idx_n2)
            {
                // Compute f_hat at current time step
                Q_hat_temp_3_(idx_n1, idx_n2) = compute_RHS(idx_n1, idx_n2, R2_RHS, C);

                this->f_hat_current_(idx_n1, idx_n2) =
                    this->f_hat_previous_(idx_n1, idx_n2) + (dt / 6.0) * (Q_hat_temp_1_(idx_n1, idx_n2) + 4.0 * Q_hat_temp_2_(idx_n1, idx_n2) + Q_hat_temp_3_(idx_n1, idx_n2));
            }
        }

        // Apply spectral stabilization if enabled
        this->block_apply_spectral_stabilization_f_hat();

        // Finish Stage 3 for RK3

        //////////////////////////////////////////////////////////////////////////

        // Update previous time step solution
        this->block_update_and_store_solution_for_the_next_time_step(fft_c2c_2d_backward, save_time_index, t_step);
    }
}

void Solver_2D_new::apply_ODE_solver_SSP_RK3(FFT_Executor &fft_c2c_4d_forward,
                                             FFT_Executor &fft_c2c_2d_backward)
{
    // Implementation of SSP RK3 solver
    int save_time_index = 1; // start from 1 since 0 is already saved
    const int nthreads = this->sim_params_.num_threads;
    const int grid_size = 2 * this->sim_params_.N;
    const double dt = this->sim_params_.dt;
    const double C = this->physical_model_->C_;

    const double R2_RHS = 4 * this->physical_model_->model_params_.R * this->physical_model_->model_params_.R; // precompute 4*R^2 for RHS computation

    for (int t_step = 1; t_step < this->N_time_; ++t_step)
    {

        // Perform FFT on T, Y, U, I tensors to get T_hat, Y_hat, U_hat, I_hat
        this->block_compute_T_Y_U_I_hat(fft_c2c_4d_forward);

        // Stage 1 for SSP RK3
#pragma omp parallel for num_threads(nthreads)
        for (size_t idx_n1 = 0; idx_n1 < static_cast<size_t>(grid_size); ++idx_n1)
        {
            for (size_t idx_n2 = 0; idx_n2 < static_cast<size_t>(grid_size); ++idx_n2)
            {
                // Compute f_hat at current time step
                Q_hat_temp_1_(idx_n1, idx_n2) = compute_RHS(idx_n1, idx_n2, R2_RHS, C);

                this->f_hat_current_(idx_n1, idx_n2) =
                    this->f_hat_previous_(idx_n1, idx_n2) + dt * Q_hat_temp_1_(idx_n1, idx_n2);
            }
        }

        // Apply spectral stabilization if enabled
        this->block_apply_spectral_stabilization_f_hat();
        // Finish Stage 1 for RK3

        //////////////////////////////////////////////////////////////////////////

        // Stage 2 for RK3
        fft_c2c_2d_backward.execute_plan(this->f_hat_current_.data(), this->f_solution_.data());

        // Perform FFT on T, Y, U, I tensors to get T_hat, Y_hat, U_hat, I_hat
        this->block_compute_T_Y_U_I_hat(fft_c2c_4d_forward);

#pragma omp parallel for num_threads(nthreads)
        for (size_t idx_n1 = 0; idx_n1 < static_cast<size_t>(grid_size); ++idx_n1)
        {
            for (size_t idx_n2 = 0; idx_n2 < static_cast<size_t>(grid_size); ++idx_n2)
            {
                // Compute f_hat at current time step
                Q_hat_temp_2_(idx_n1, idx_n2) = compute_RHS(idx_n1, idx_n2, R2_RHS, C);

                this->f_hat_current_(idx_n1, idx_n2) = 0.25 * this->f_hat_current_(idx_n1, idx_n2) + 0.75 * this->f_hat_previous_(idx_n1, idx_n2) + 0.25 * dt * Q_hat_temp_2_(idx_n1, idx_n2);
            }
        }

        // Apply spectral stabilization if enabled
        this->block_apply_spectral_stabilization_f_hat();

        // Finish Stage 2 for RK3

        //////////////////////////////////////////////////////////////////////////

        // Stage 3 for RK3
        fft_c2c_2d_backward.execute_plan(this->f_hat_current_.data(), this->f_solution_.data());

        // Perform FFT on T, Y, U, I tensors to get T_hat, Y_hat, U_hat, I_hat
        this->block_compute_T_Y_U_I_hat(fft_c2c_4d_forward);

#pragma omp parallel for num_threads(nthreads)
        for (size_t idx_n1 = 0; idx_n1 < static_cast<size_t>(grid_size); ++idx_n1)
        {
            for (size_t idx_n2 = 0; idx_n2 < static_cast<size_t>(grid_size); ++idx_n2)
            {
                // Compute f_hat at current time step
                Q_hat_temp_3_(idx_n1, idx_n2) = compute_RHS(idx_n1, idx_n2, R2_RHS, C);

                this->f_hat_current_(idx_n1, idx_n2) = (2.0) / (3.0) * this->f_hat_current_(idx_n1, idx_n2) + (1.0) / (3.0) * this->f_hat_previous_(idx_n1, idx_n2) + (2.0) / (3.0) * dt * Q_hat_temp_3_(idx_n1, idx_n2);
            }
        }

        // Apply spectral stabilization if enabled
        this->block_apply_spectral_stabilization_f_hat();

        // Finish Stage 3 for RK3

        //////////////////////////////////////////////////////////////////////////

        // Update previous time step solution
        this->block_update_and_store_solution_for_the_next_time_step(fft_c2c_2d_backward, save_time_index, t_step);
    }
}

void Solver_2D_new::apply_ODE_solver_RK4(FFT_Executor &fft_c2c_4d_forward,
                                         FFT_Executor &fft_c2c_2d_backward)
{
    // Implementation of SSP RK4 solver
    int save_time_index = 1; // start from 1 since 0 is already saved
    const int nthreads = this->sim_params_.num_threads;
    const int grid_size = 2 * this->sim_params_.N;
    const double dt = this->sim_params_.dt;
    const double C = this->physical_model_->C_;

    const double R2_RHS = 4 * this->physical_model_->model_params_.R * this->physical_model_->model_params_.R; // precompute 4*R^2 for RHS computation

    for (int t_step = 1; t_step < this->N_time_; ++t_step)
    {
        // Stage 1 for RK4

        // Perform FFT on T, Y, U, I tensors to get T_hat, Y_hat, U_hat, I_hat
        this->block_compute_T_Y_U_I_hat(fft_c2c_4d_forward);

#pragma omp parallel for num_threads(nthreads)
        for (size_t idx_n1 = 0; idx_n1 < static_cast<size_t>(grid_size); ++idx_n1)
        {
            for (size_t idx_n2 = 0; idx_n2 < static_cast<size_t>(grid_size); ++idx_n2)
            {
                // Compute f_hat at current time step
                Q_hat_temp_1_(idx_n1, idx_n2) = compute_RHS(idx_n1, idx_n2, R2_RHS, C);

                this->f_hat_current_(idx_n1, idx_n2) =
                    this->f_hat_previous_(idx_n1, idx_n2) + 0.5 * dt * Q_hat_temp_1_(idx_n1, idx_n2);
            }
        }

        // Apply spectral stabilization if enabled
        this->block_apply_spectral_stabilization_f_hat();
        // Finish Stage 1 for RK4

        //////////////////////////////////////////////////////////////////////////

        // Stage 2 for RK4
        fft_c2c_2d_backward.execute_plan(this->f_hat_current_.data(), this->f_solution_.data());

        // Perform FFT on T, Y, U, I tensors to get T_hat, Y_hat, U_hat, I_hat
        this->block_compute_T_Y_U_I_hat(fft_c2c_4d_forward);

#pragma omp parallel for num_threads(nthreads)
        for (size_t idx_n1 = 0; idx_n1 < static_cast<size_t>(grid_size); ++idx_n1)
        {
            for (size_t idx_n2 = 0; idx_n2 < static_cast<size_t>(grid_size); ++idx_n2)
            {
                // Compute f_hat at current time step
                Q_hat_temp_2_(idx_n1, idx_n2) = compute_RHS(idx_n1, idx_n2, R2_RHS, C);

                this->f_hat_current_(idx_n1, idx_n2) = this->f_hat_previous_(idx_n1, idx_n2) + 0.5 * dt * Q_hat_temp_2_(idx_n1, idx_n2);
            }
        }

        // Apply spectral stabilization if enabled
        this->block_apply_spectral_stabilization_f_hat();

        // Finish Stage 2 for RK4

        //////////////////////////////////////////////////////////////////////////

        // Stage 3 for RK4
        fft_c2c_2d_backward.execute_plan(this->f_hat_current_.data(), this->f_solution_.data());

        // Perform FFT on T, Y, U, I tensors to get T_hat, Y_hat, U_hat, I_hat
        this->block_compute_T_Y_U_I_hat(fft_c2c_4d_forward);

#pragma omp parallel for num_threads(nthreads)
        for (size_t idx_n1 = 0; idx_n1 < static_cast<size_t>(grid_size); ++idx_n1)
        {
            for (size_t idx_n2 = 0; idx_n2 < static_cast<size_t>(grid_size); ++idx_n2)
            {
                // Compute f_hat at current time step
                Q_hat_temp_3_(idx_n1, idx_n2) = compute_RHS(idx_n1, idx_n2, R2_RHS, C);

                this->f_hat_current_(idx_n1, idx_n2) = this->f_hat_previous_(idx_n1, idx_n2) + dt * Q_hat_temp_3_(idx_n1, idx_n2);
            }
        }

        // Apply spectral stabilization if enabled
        this->block_apply_spectral_stabilization_f_hat();

        // Finish Stage 3 for RK4

        //////////////////////////////////////////////////////////////////////////

        // Stage 4 for RK4
        fft_c2c_2d_backward.execute_plan(this->f_hat_current_.data(), this->f_solution_.data());

        // Perform FFT on T, Y, U, I tensors to get T_hat, Y_hat, U_hat, I_hat
        this->block_compute_T_Y_U_I_hat(fft_c2c_4d_forward);

#pragma omp parallel for num_threads(nthreads)
        for (size_t idx_n1 = 0; idx_n1 < static_cast<size_t>(grid_size); ++idx_n1)
        {
            for (size_t idx_n2 = 0; idx_n2 < static_cast<size_t>(grid_size); ++idx_n2)
            {
                // Compute f_hat at current time step
                Q_hat_temp_4_(idx_n1, idx_n2) = compute_RHS(idx_n1, idx_n2, R2_RHS, C);

                this->f_hat_current_(idx_n1, idx_n2) =
                    this->f_hat_previous_(idx_n1, idx_n2) + dt / 6.0 * (Q_hat_temp_1_(idx_n1, idx_n2) + 2.0 * Q_hat_temp_2_(idx_n1, idx_n2) + 2.0 * Q_hat_temp_3_(idx_n1, idx_n2) + Q_hat_temp_4_(idx_n1, idx_n2));
            }
        }

        // Apply spectral stabilization if enabled
        this->block_apply_spectral_stabilization_f_hat();

        // Finish Stage 4 for RK4

        //////////////////////////////////////////////////////////////////////////

        // Update previous time step solution
        this->block_update_and_store_solution_for_the_next_time_step(fft_c2c_2d_backward, save_time_index, t_step);
    }
}

void Solver_2D_new::block_compute_T_Y_U_I_hat(FFT_Executor &fft_c2c_4d_forward)
{
    // Compute T, Y, U, I tensors
    if (this->spectral_config_.early_dealiasing.enabled)
    {
        this->apply_early_de_aliasing();
    }
    else
    {
        this->compute_T_Y_U_I_tensors();
    }

    // Perform FFT on T, Y, U, I tensors to get T_hat, Y_hat, U_hat, I_hat
    fft_c2c_4d_forward.execute_plan(this->T_.data(), this->T_.data());
    fft_c2c_4d_forward.execute_plan(this->Y_.data(), this->Y_.data());
    fft_c2c_4d_forward.execute_plan(this->U_.data(), this->U_.data());

    if (this->physical_model_->model_params_.equation_type == EquationType::quantum_Boltzmann)
    {
        fft_c2c_4d_forward.execute_plan(this->I_.data(), this->I_.data());
    }
}

void Solver_2D_new::block_create_fftw_plans(FFT_Executor &fft_c2c_4d_forward,
                                            FFT_Executor &fft_c2c_2d_backward, int grid_size, int nthreads_FFTW)
{
    // Create FFT plan C2C for reduction f_solution at each time step
    FFT_Config config_c2c_2d_backward = {
        .dim = 2,
        .shape = {grid_size, grid_size},
        .type = FFT_Type::C2C,
        .direction = FFT_Direction::Backward,
        .inplace = false,
        .normalize = false,
        .denormalize = false,
        .nthreads = nthreads_FFTW,
        .flags = FFT_Flag::Estimate}; // use Estimate to avoid overwriting f_hat data

    fft_c2c_2d_backward.create_plan(
        config_c2c_2d_backward,
        this->f_hat_current_.data(),
        this->f_solution_.data());

    string wisdom_file;
    if (grid_size == 64)
    {
        if (nthreads_FFTW == 16)
        {
            wisdom_file = "fftw_wisdom_4d_16threads.txt";
        }
        else if (nthreads_FFTW == 8)
        {
            wisdom_file = "fftw_wisdom_4d_8threads.txt";
        }
        else if (nthreads_FFTW == 4)
        {
            wisdom_file = "fftw_wisdom_4d_4threads.txt";
        }
        else if (nthreads_FFTW == 2)
        {
            wisdom_file = "fftw_wisdom_4d_2threads.txt";
        }
        else
        {
            wisdom_file = "fftw_wisdom_4d_1thread.txt";
        }
    }
    else if (grid_size == 128)
    {
        if (nthreads_FFTW == 16)
        {
            wisdom_file = "fftw_wisdom_4d_128_16threads.txt";
        }
        else if (nthreads_FFTW == 8)
        {
            wisdom_file = "fftw_wisdom_4d_128_8threads.txt";
        }
        else if (nthreads_FFTW == 4)
        {
            wisdom_file = "fftw_wisdom_4d_128_4threads.txt";
        }
        else if (nthreads_FFTW == 2)
        {
            wisdom_file = "fftw_wisdom_4d_128_2threads.txt";
        }
        else
        {
            wisdom_file = "fftw_wisdom_4d_128_1thread.txt";
        }
    }
    else
    {
        wisdom_file = "fftw_wisdom_4d_custom.txt";
    }

    string wisdom_file_input = constants::INPUT_PATH + wisdom_file;

    if (fftw_import_wisdom_from_filename(wisdom_file_input.c_str()) != 0)
    {
        cout << "Loaded FFTW wisdom from: " << wisdom_file_input << " at " + get_local_time() << endl;
    }
    else
    {
        cout << "No FFTW wisdom found. 4D PATIENT plan may take long on first run" << " at " + get_local_time() << endl;
    }
    FFT_Config config_c2c_4d_forward = {
        .dim = 4,
        .shape = {grid_size, grid_size, grid_size, grid_size},
        .type = FFT_Type::C2C,
        .direction = FFT_Direction::Forward,
        .inplace = true, // in-place to save memory
        .normalize = true,
        .denormalize = false,
        .nthreads = nthreads_FFTW,
        .flags = FFT_Flag::Patient}; // use Patient for better performance in the time-stepping loop

    fft_c2c_4d_forward.create_plan(
        config_c2c_4d_forward,
        this->T_.data(),
        this->T_.data());

    string wisdom_file_output = this->sim_params_.output_directory + "/" + wisdom_file;
    if (fftw_export_wisdom_to_filename(wisdom_file_output.c_str()) != 0)
    {
        cout << "Saved FFTW wisdom to: " << wisdom_file_output << " at " + get_local_time() << endl;
    }
    else
    {
        cout << "Warning: Could not save FFTW wisdom to: " << wisdom_file_output << " at " + get_local_time() << endl;
    }
}
