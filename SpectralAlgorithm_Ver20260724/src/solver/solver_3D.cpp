#include "solver_3D.h"
#include "constants.h"

Solver_3D::Solver_3D(const ModelParameters &model_params, SimulationParameters &sim_params, const SpectralStabilizationConfig &spectral_stabilization_config)
    : Solver_Base(model_params, sim_params, spectral_stabilization_config)
{
    // Constructor implementation (if any)
}

Solver_3D::~Solver_3D()
{
    this->k_grid_.Destroy();
    this->t_save_points_.Destroy();
    this->fft_idx_.clear();

    this->f_initial_.Destroy();
    this->f_solution_.Destroy();
    this->f_solution_time_.Destroy();

    this->f_hat_previous_.Destroy();
    this->f_hat_current_.Destroy();
    this->f_hat_time_.Destroy();

    this->W_tensor_.Destroy();
    this->T_Y_U_I_.Destroy();

    this->T_RHS_.Destroy();
    this->Y_RHS_.Destroy();
    this->U_RHS_.Destroy();
    this->I_RHS_.Destroy();
}

void Solver_3D::check_computational_data()
{
    const int N = this->sim_params_.N;

    // ✅ ADD: Warning for large N (no hard stop)
    const int WARN_N_3D = 16; // Practical warning threshold
    if (N > WARN_N_3D)
    {
        cout << "WARNING: N=" << N
             << " may be too large for practical runtime.\n"
             << "Reason: cost grows as O(N^6).\n"
             << "Consider reducing N or using float32 W_tensor." << endl;
    }

    size_t grid_size = static_cast<size_t>(2 * this->sim_params_.N);
    size_t N_time = static_cast<size_t>(this->sim_params_.total_number_of_times_saves);

    double estimated_memory_GB = 0.0;

    // Estimate memory usage for k_grid and fft_idx
    estimated_memory_GB += 1 * estimate_memory_usage_GB<double>({grid_size});
    // Estimate memory usage for fft_idx
    estimated_memory_GB += 1 * estimate_memory_usage_GB<int>({grid_size});
    // Estimate memory usage for t_save_points
    estimated_memory_GB += 1 * estimate_memory_usage_GB<double>({2, N_time});

    // Estimate memory usage for f_initial, f_solution, f_hat_previous, f_hat_current, T_RHS, Y_RHS, U_RHS, I_RHS
    estimated_memory_GB += 8 * estimate_memory_usage_GB<complex<double>>({grid_size, grid_size, grid_size});
    // Estimate memory usage for f_solution_time, f_hat_time
    estimated_memory_GB += 2 * estimate_memory_usage_GB<complex<double>>({N_time, grid_size, grid_size, grid_size});
    // Estimate memory usage for W_tensor
    estimated_memory_GB += 1 * estimate_memory_usage_GB<float>({grid_size, grid_size, grid_size, grid_size, grid_size, grid_size});
    // Estimate memory usage for T_Y_U_I_
    estimated_memory_GB += 1 * estimate_memory_usage_GB<complex<double>>({grid_size, grid_size, grid_size, grid_size, grid_size, grid_size});

    if (estimated_memory_GB > this->sim_params_.max_memory_GB)
    {
        throw runtime_error("Estimated memory usage " + to_string(estimated_memory_GB) + " GB exceeds the limit of " + to_string(this->sim_params_.max_memory_GB) + " GB. Please increase max_memory_GB or reduce grid size N.");
    }
    else
    {
        cout << "Estimated memory usage: " << estimated_memory_GB << " GB" << endl;
    }

    if (this->physical_model_->model_params_.dimension != 3)
    {
        throw runtime_error("Error: The dimension of the physical model does not match the solver dimension (3D).");
    }
}

void Solver_3D::create_objects()
{
    size_t grid_size = static_cast<size_t>(2 * this->sim_params_.N);
    size_t N_time = static_cast<size_t>(this->sim_params_.total_number_of_times_saves);

    this->k_grid_.resize(grid_size);
    this->t_save_points_.resize(2, N_time);
    this->fft_idx_.resize(grid_size);

    this->f_initial_.resize(grid_size, grid_size, grid_size);
    this->f_solution_.resize(grid_size, grid_size, grid_size);
    this->f_solution_time_.resize(N_time, grid_size, grid_size, grid_size);

    this->f_hat_previous_.resize(grid_size, grid_size, grid_size);
    this->f_hat_current_.resize(grid_size, grid_size, grid_size);
    this->f_hat_time_.resize(N_time, grid_size, grid_size, grid_size);

    this->W_tensor_.resize(grid_size, grid_size, grid_size, grid_size, grid_size, grid_size);
    this->T_Y_U_I_.resize(grid_size, grid_size, grid_size, grid_size, grid_size, grid_size);

    this->T_RHS_.resize(grid_size, grid_size, grid_size);
    this->Y_RHS_.resize(grid_size, grid_size, grid_size);
    this->U_RHS_.resize(grid_size, grid_size, grid_size);
    this->I_RHS_.resize(grid_size, grid_size, grid_size);
}

void Solver_3D::write_tensor_solution(string cout_filename, bool final)
{
    // Write solution to output files in HDF5 format
    if (this->sim_params_.save_hdf5_files)
    {
        this->f_solution_time_.WriteToHDF5(this->sim_params_.output_directory + "/f_solution_alltime.h5");
        this->f_hat_time_.WriteToHDF5(this->sim_params_.output_directory + "/f_hat_solution_alltime.h5");
    }

    // Write solution to output files in text format
    if (this->sim_params_.save_txt_files)
    {
        this->f_solution_time_.WriteToText(this->sim_params_.output_directory + "/f_solution_alltime.txt");
        this->f_hat_time_.WriteToText(this->sim_params_.output_directory + "/f_hat_solution_alltime.txt");
    }

    if (final)
    {
        this->f_solution_.WriteToHDF5(this->sim_params_.output_directory + "/" + cout_filename + ".h5");
        this->f_solution_.WriteToText(this->sim_params_.output_directory + "/" + cout_filename + ".txt");
    }
}

void Solver_3D::construct_f_initial_tensor()
{
    // Implementation of constructing f_initial_ tensor if needed
    vector<double> grid_vector = this->k_grid_.get_data_vector();

    // Call directly without wrapper - bind member function
    auto func_initial_bound = [this](const vector<double> &k)
    {
        return this->physical_model_->func_initial_condition(k);
    };

    this->f_initial_.fill_from_function({grid_vector, grid_vector, grid_vector}, func_initial_bound, this->sim_params_.num_threads);

    this->f_initial_.WriteToHDF5(this->sim_params_.output_directory + "/f_initial.h5");
    this->f_initial_.WriteToText(this->sim_params_.output_directory + "/f_initial.txt");
}

void Solver_3D::compute_f_initial_hat()
{
    // FFTW C2C for initial condition f_initial_
    const int Nx = 2 * this->sim_params_.N;

    FFT_Config config_c2c_3d = {
        .dim = 3,
        .shape = {Nx, Nx, Nx},
        .type = FFT_Type::C2C,
        .direction = FFT_Direction::Forward,
        .inplace = false,
        .normalize = true,
        .denormalize = false,
        .nthreads = this->nthreads_FFTW_,
        .flags = FFT_Flag::Estimate};

    FFT_Executor fft_c2c_3d;
    fft_c2c_3d.create_plan(
        config_c2c_3d,
        this->f_initial_.data(),
        this->f_hat_previous_.data());

    fft_c2c_3d.execute_plan(
        this->f_initial_.data(),
        this->f_hat_previous_.data());

    fft_c2c_3d.destroy_plan();

    this->f_hat_previous_.WriteToHDF5(this->sim_params_.output_directory + "/f_hat_initial.h5");
    this->f_hat_previous_.WriteToText(this->sim_params_.output_directory + "/f_hat_initial.txt");
}

void Solver_3D::compute_solution()
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

    // cout << 0 << "% completed." << endl;

    this->compute_f_initial_hat();
    this->f_hat_time_.insert_slice(0, 0, this->f_hat_previous_, nthreads);

    // cout << 1 << "% completed." << endl;

    this->construct_W_tensor();

    // cout << 0 << "% completed." << endl;

    // Create FFT plan 
    FFT_Executor fft_c2c_3d_backward;
    FFT_Executor fft_c2c_6d_forward;
    this->block_create_fftw_plans(fft_c2c_6d_forward, fft_c2c_3d_backward, grid_size, nthreads_FFTW);
    

    // Time-stepping loop implementation
    if (this->sim_params_.ode_solver == ODE_solver_type::Euler)
    {
        this->apply_ODE_solver_euler(fft_c2c_6d_forward,
                                     fft_c2c_3d_backward);
    }
    else if (this->sim_params_.ode_solver == ODE_solver_type::RK2)
    {
        cout << "RK2 solver is not implemented yet." << endl;
        exit(-1); // RK2 not implemented yet
    }
    else if (this->sim_params_.ode_solver == ODE_solver_type::RK3)
    {
        cout << "RK3 solver is not implemented yet." << endl;
        exit(-1); // RK3 not implemented yet
    }
    else if (this->sim_params_.ode_solver == ODE_solver_type::SSP_RK3)
    {
        cout << "SSP RK3 solver is not implemented yet." << endl;
        exit(-1); // SSP RK3 not implemented yet
    }
    else if (this->sim_params_.ode_solver == ODE_solver_type::RK4)
    {
        cout << "RK4 solver is not implemented yet." << endl;
        exit(-1); // RK4 not implemented yet
    }
    else
    {
        throw runtime_error("ODE solver not implemented yet.");
    }

    // Destroy FFT plans
    fft_c2c_3d_backward.destroy_plan();
    fft_c2c_6d_forward.destroy_plan();
}

void Solver_3D::construct_W_tensor()
{
    vector<double> grid_vector = this->k_grid_.get_data_vector();

    // Call directly without wrapper - bind member function
    auto func_W_bound = [this](const vector<double> &k)
    {
        return this->physical_model_->func_W(k);
    };

    this->W_tensor_.fill_from_function(
        {grid_vector, grid_vector, grid_vector, grid_vector, grid_vector, grid_vector}, func_W_bound, this->sim_params_.num_threads);
}

void Solver_3D::compute_T_tensors()
{
    const int N = this->sim_params_.N;
    const double cutoff_fraction = this->spectral_config_.early_dealiasing.cutoff_fraction;
    const int cutoff = static_cast<int>(std::floor(cutoff_fraction * N));
    const int cutoff_squared = cutoff * cutoff;

#pragma omp parallel for collapse(3) num_threads(this->sim_params_.num_threads) schedule(guided)
    for (int n1 = -N; n1 < N; ++n1)
    {
        for (int n2 = -N; n2 < N; ++n2)
        {
            for (int n3 = -N; n3 < N; ++n3)
            {
                size_t idx_n1 = this->fft_idx_[n1 + N];
                size_t idx_n2 = this->fft_idx_[n2 + N];
                size_t idx_n3 = this->fft_idx_[n3 + N];
                complex<double> f_k2 = this->f_solution_(idx_n1, idx_n2, idx_n3);

                for (int n4 = -N; n4 < N; ++n4)
                {
                    size_t idx_n4 = this->fft_idx_[n4 + N];
                    for (int n5 = -N; n5 < N; ++n5)
                    {
                        size_t idx_n5 = this->fft_idx_[n5 + N];
                        for (int n6 = -N; n6 < N; ++n6)
                        {
                            size_t idx_n6 = this->fft_idx_[n6 + N];

                            double W_val = 0.0;
                            complex<double> f_k3 = complex<double>(0.0, 0.0);

                            if (this->spectral_config_.early_dealiasing.enabled == true && this->spectral_config_.early_dealiasing.use_L2 == true)
                            {
                                if ((n1 * n1 + n2 * n2 + n3 * n3 + n4 * n4 + n5 * n5 + n6 * n6) > cutoff_squared)
                                {
                                    this->T_Y_U_I_(idx_n1, idx_n2, idx_n3, idx_n4, idx_n5, idx_n6) = complex<double>(0.0, 0.0);
                                    continue;
                                }
                            }
                            else if (this->spectral_config_.early_dealiasing.enabled == true && this->spectral_config_.early_dealiasing.use_L2 == false)
                            {
                                if (abs(n1) > cutoff || abs(n2) > cutoff || abs(n3) > cutoff || abs(n4) > cutoff || abs(n5) > cutoff || abs(n6) > cutoff)
                                {
                                    this->T_Y_U_I_(idx_n1, idx_n2, idx_n3, idx_n4, idx_n5, idx_n6) = complex<double>(0.0, 0.0);
                                    continue;
                                }
                            }

                            W_val = this->W_tensor_(idx_n1, idx_n2, idx_n3, idx_n4, idx_n5, idx_n6);
                            f_k3 = this->f_solution_(idx_n4, idx_n5, idx_n6);

                            this->T_Y_U_I_(idx_n1, idx_n2, idx_n3, idx_n4, idx_n5, idx_n6) = W_val * f_k2 * f_k3;
                        }
                    }
                }
            }
        }
    }
}

void Solver_3D::compute_Y_tensors()
{
    const int N = this->sim_params_.N;
    const double cutoff_fraction = this->spectral_config_.early_dealiasing.cutoff_fraction;
    const int cutoff = static_cast<int>(std::floor(cutoff_fraction * N));
    const int cutoff_squared = cutoff * cutoff;

#pragma omp parallel for collapse(3) num_threads(this->sim_params_.num_threads) schedule(guided)
    for (int n1 = -N; n1 < N; ++n1)
    {
        for (int n2 = -N; n2 < N; ++n2)
        {
            for (int n3 = -N; n3 < N; ++n3)
            {
                size_t idx_n1 = this->fft_idx_[n1 + N];
                size_t idx_n2 = this->fft_idx_[n2 + N];
                size_t idx_n3 = this->fft_idx_[n3 + N];
                complex<double> f_k2 = this->f_solution_(idx_n1, idx_n2, idx_n3);

                for (int n4 = -N; n4 < N; ++n4)
                {
                    size_t idx_n4 = this->fft_idx_[n4 + N];
                    for (int n5 = -N; n5 < N; ++n5)
                    {
                        size_t idx_n5 = this->fft_idx_[n5 + N];
                        for (int n6 = -N; n6 < N; ++n6)
                        {
                            size_t idx_n6 = this->fft_idx_[n6 + N];

                            double W_val = 0.0;
                            complex<double> f_sum = complex<double>(0.0, 0.0);

                            int sum1 = (n1 + n4);
                            int sum2 = (n2 + n5);
                            int sum3 = (n3 + n6);

                            size_t idx_sum1, idx_sum2, idx_sum3;
                            if (this->spectral_config_.early_dealiasing.enabled == true && this->spectral_config_.early_dealiasing.use_L2 == true)
                            {
                                // L2 de-aliasing: indices already within [-N, N), use fft_idx_
                                if ((n1 * n1 + n2 * n2 + n3 * n3 + n4 * n4 + n5 * n5 + n6 * n6) > cutoff_squared || (sum1 * sum1 + sum2 * sum2 + sum3 * sum3) > cutoff_squared)
                                {
                                    this->T_Y_U_I_(idx_n1, idx_n2, idx_n3, idx_n4, idx_n5, idx_n6) = complex<double>(0.0, 0.0);
                                    continue;
                                }
                                idx_sum1 = this->fft_idx_[sum1 + N];
                                idx_sum2 = this->fft_idx_[sum2 + N];
                                idx_sum3 = this->fft_idx_[sum3 + N];
                            }
                            else if (this->spectral_config_.early_dealiasing.enabled == true && this->spectral_config_.early_dealiasing.use_L2 == false)
                            {
                                // Rectangle de-aliasing: indices already within [-N, N), use fft_idx_
                                if (abs(n1) > cutoff || abs(n2) > cutoff || abs(n3) > cutoff ||
                                    abs(n4) > cutoff || abs(n5) > cutoff || abs(n6) > cutoff ||
                                    abs(sum1) > cutoff || abs(sum2) > cutoff || abs(sum3) > cutoff)
                                {
                                    this->T_Y_U_I_(idx_n1, idx_n2, idx_n3, idx_n4, idx_n5, idx_n6) = complex<double>(0.0, 0.0);
                                    continue;
                                }
                                idx_sum1 = this->fft_idx_[sum1 + N];
                                idx_sum2 = this->fft_idx_[sum2 + N];
                                idx_sum3 = this->fft_idx_[sum3 + N];
                            }
                            else
                            {
                                // No de-aliasing: sum1/2/3 may be outside [-N, N), use get_fft_index_from_math()
                                idx_sum1 = this->get_fft_index_from_math(sum1);
                                idx_sum2 = this->get_fft_index_from_math(sum2);
                                idx_sum3 = this->get_fft_index_from_math(sum3);
                            }

                            // Calculate Y: W * f_sum * f_k2
                            W_val = this->W_tensor_(idx_n1, idx_n2, idx_n3, idx_n4, idx_n5, idx_n6);
                            f_sum = this->f_solution_(idx_sum1, idx_sum2, idx_sum3);

                            this->T_Y_U_I_(idx_n1, idx_n2, idx_n3, idx_n4, idx_n5, idx_n6) = W_val * f_sum * f_k2;
                        }
                    }
                }
            }
        }
    }
}

void Solver_3D::compute_U_tensors()
{
    const int N = this->sim_params_.N;
    const double cutoff_fraction = this->spectral_config_.early_dealiasing.cutoff_fraction;
    const int cutoff = static_cast<int>(std::floor(cutoff_fraction * N));
    const int cutoff_squared = cutoff * cutoff;

#pragma omp parallel for collapse(3) num_threads(this->sim_params_.num_threads) schedule(guided)
    for (int n1 = -N; n1 < N; ++n1)
    {
        for (int n2 = -N; n2 < N; ++n2)
        {
            for (int n3 = -N; n3 < N; ++n3)
            {
                size_t idx_n1 = this->fft_idx_[n1 + N];
                size_t idx_n2 = this->fft_idx_[n2 + N];
                size_t idx_n3 = this->fft_idx_[n3 + N];
                for (int n4 = -N; n4 < N; ++n4)
                {
                    size_t idx_n4 = this->fft_idx_[n4 + N];
                    for (int n5 = -N; n5 < N; ++n5)
                    {
                        size_t idx_n5 = this->fft_idx_[n5 + N];
                        for (int n6 = -N; n6 < N; ++n6)
                        {
                            size_t idx_n6 = this->fft_idx_[n6 + N];

                            double W_val = 0.0;
                            complex<double> f_k3 = complex<double>(0.0, 0.0);
                            complex<double> f_sum = complex<double>(0.0, 0.0);

                            int sum1 = (n1 + n4);
                            int sum2 = (n2 + n5);
                            int sum3 = (n3 + n6);

                            size_t idx_sum1, idx_sum2, idx_sum3;
                            if (this->spectral_config_.early_dealiasing.enabled == true && this->spectral_config_.early_dealiasing.use_L2 == true)
                            {
                                // L2 de-aliasing: indices already within [-N, N), use fft_idx_
                                if ((n1 * n1 + n2 * n2 + n3 * n3 + n4 * n4 + n5 * n5 + n6 * n6) > (cutoff_squared) || (sum1 * sum1 + sum2 * sum2 + sum3 * sum3) > (cutoff_squared))
                                {
                                    this->T_Y_U_I_(idx_n1, idx_n2, idx_n3, idx_n4, idx_n5, idx_n6) = complex<double>(0.0, 0.0);
                                    continue;
                                }
                                idx_sum1 = this->fft_idx_[sum1 + N];
                                idx_sum2 = this->fft_idx_[sum2 + N];
                                idx_sum3 = this->fft_idx_[sum3 + N];
                            }
                            else if (this->spectral_config_.early_dealiasing.enabled == true && this->spectral_config_.early_dealiasing.use_L2 == false)
                            {
                                // Rectangle de-aliasing: indices already within [-N, N), use fft_idx_
                                if (abs(n1) > cutoff || abs(n2) > cutoff || abs(n3) > cutoff ||
                                    abs(n4) > cutoff || abs(n5) > cutoff || abs(n6) > cutoff ||
                                    abs(sum1) > cutoff || abs(sum2) > cutoff || abs(sum3) > cutoff)
                                {
                                    this->T_Y_U_I_(idx_n1, idx_n2, idx_n3, idx_n4, idx_n5, idx_n6) = complex<double>(0.0, 0.0);
                                    continue;
                                }
                                idx_sum1 = this->fft_idx_[sum1 + N];
                                idx_sum2 = this->fft_idx_[sum2 + N];
                                idx_sum3 = this->fft_idx_[sum3 + N];
                            }
                            else
                            {
                                // No de-aliasing: sum1/2/3 may be outside [-N, N), use get_fft_index_from_math()
                                idx_sum1 = this->get_fft_index_from_math(sum1);
                                idx_sum2 = this->get_fft_index_from_math(sum2);
                                idx_sum3 = this->get_fft_index_from_math(sum3);
                            }

                            // Calculate Y: W * f_sum * f_k3
                            W_val = this->W_tensor_(idx_n1, idx_n2, idx_n3, idx_n4, idx_n5, idx_n6);
                            f_k3 = this->f_solution_(idx_n4, idx_n5, idx_n6);
                            f_sum = this->f_solution_(idx_sum1, idx_sum2, idx_sum3);

                            this->T_Y_U_I_(idx_n1, idx_n2, idx_n3, idx_n4, idx_n5, idx_n6) = W_val * f_sum * f_k3;
                        }
                    }
                }
            }
        }
    }
}

void Solver_3D::compute_I_tensors()
{
    const int N = this->sim_params_.N;
    const double cutoff_fraction = this->spectral_config_.early_dealiasing.cutoff_fraction;
    const int cutoff = static_cast<int>(std::floor(cutoff_fraction * N));
    const int cutoff_squared = cutoff * cutoff;

#pragma omp parallel for collapse(3) num_threads(this->sim_params_.num_threads) schedule(guided)
    for (int n1 = -N; n1 < N; ++n1)
    {
        for (int n2 = -N; n2 < N; ++n2)
        {
            for (int n3 = -N; n3 < N; ++n3)
            {
                size_t idx_n1 = this->fft_idx_[n1 + N];
                size_t idx_n2 = this->fft_idx_[n2 + N];
                size_t idx_n3 = this->fft_idx_[n3 + N];
                for (int n4 = -N; n4 < N; ++n4)
                {
                    size_t idx_n4 = this->fft_idx_[n4 + N];
                    for (int n5 = -N; n5 < N; ++n5)
                    {
                        size_t idx_n5 = this->fft_idx_[n5 + N];
                        for (int n6 = -N; n6 < N; ++n6)
                        {
                            size_t idx_n6 = this->fft_idx_[n6 + N];

                            double W_val = 0.0;
                            complex<double> f_sum = complex<double>(0.0, 0.0);

                            int sum1 = (n1 + n4);
                            int sum2 = (n2 + n5);
                            int sum3 = (n3 + n6);

                            size_t idx_sum1, idx_sum2, idx_sum3;

                            if (this->spectral_config_.early_dealiasing.enabled == true && this->spectral_config_.early_dealiasing.use_L2 == true)
                            {
                                // L2 de-aliasing: indices already within [-N, N), use fft_idx_
                                if ((n1 * n1 + n2 * n2 + n3 * n3 + n4 * n4 + n5 * n5 + n6 * n6) > (cutoff_squared) || (sum1 * sum1 + sum2 * sum2 + sum3 * sum3) > (cutoff_squared))
                                {
                                    this->T_Y_U_I_(idx_n1, idx_n2, idx_n3, idx_n4, idx_n5, idx_n6) = complex<double>(0.0, 0.0);
                                    continue;
                                }
                                idx_sum1 = this->fft_idx_[sum1 + N];
                                idx_sum2 = this->fft_idx_[sum2 + N];
                                idx_sum3 = this->fft_idx_[sum3 + N];
                            }
                            else if (this->spectral_config_.early_dealiasing.enabled == true && this->spectral_config_.early_dealiasing.use_L2 == false)
                            {
                                // Rectangle de-aliasing: indices already within [-N, N), use fft_idx_
                                if (abs(n1) > cutoff || abs(n2) > cutoff || abs(n3) > cutoff ||
                                    abs(n4) > cutoff || abs(n5) > cutoff || abs(n6) > cutoff ||
                                    abs(sum1) > cutoff || abs(sum2) > cutoff || abs(sum3) > cutoff)
                                {
                                    this->T_Y_U_I_(idx_n1, idx_n2, idx_n3, idx_n4, idx_n5, idx_n6) = complex<double>(0.0, 0.0);
                                    continue;
                                }
                                idx_sum1 = this->fft_idx_[sum1 + N];
                                idx_sum2 = this->fft_idx_[sum2 + N];
                                idx_sum3 = this->fft_idx_[sum3 + N];
                            }
                            else
                            {
                                // No de-aliasing: sum1/2/3 may be outside [-N, N), use get_fft_index_from_math()
                                idx_sum1 = this->get_fft_index_from_math(sum1);
                                idx_sum2 = this->get_fft_index_from_math(sum2);
                                idx_sum3 = this->get_fft_index_from_math(sum3);
                            }

                            // Calculate Y: W * f_sum
                            W_val = this->W_tensor_(idx_n1, idx_n2, idx_n3, idx_n4, idx_n5, idx_n6);
                            f_sum = this->f_solution_(idx_sum1, idx_sum2, idx_sum3);

                            this->T_Y_U_I_(idx_n1, idx_n2, idx_n3, idx_n4, idx_n5, idx_n6) = W_val * f_sum;
                        }
                    }
                }
            }
        }
    }
}

void Solver_3D::block_create_fftw_plans(FFT_Executor &fft_c2c_6d_forward,
                                        FFT_Executor &fft_c2c_3d_backward, int grid_size, int nthreads_FFTW)
{
    // Create FFT plan C2C for reduction f_solution at each time step
    FFT_Config config_c2c_3d_backward = {
        .dim = 3,
        .shape = {grid_size, grid_size, grid_size},
        .type = FFT_Type::C2C,
        .direction = FFT_Direction::Backward,
        .inplace = false,
        .normalize = false,
        .denormalize = false,
        .nthreads = nthreads_FFTW,
        .flags = FFT_Flag::Estimate}; // use Estimate to avoid overwriting f_hat data

    fft_c2c_3d_backward.create_plan(
        config_c2c_3d_backward,
        this->f_hat_current_.data(),
        this->f_solution_.data());

    // cout << 1 << "% completed." << endl;

    // cout <<"Start time: " << get_local_time() << endl;
    
    string wisdom_file; 
    if (nthreads_FFTW == 16)
    {
        wisdom_file = "fftw_wisdom_6d_16threads.txt";
    }
    else if (nthreads_FFTW == 8)
    {
        wisdom_file = "fftw_wisdom_6d_8threads.txt";
    }
    else if (nthreads_FFTW == 4)
    {
        wisdom_file = "fftw_wisdom_6d_4threads.txt";
    }
    else if (nthreads_FFTW == 2)
    {
        wisdom_file = "fftw_wisdom_6d_2threads.txt";
    }
    else
    {
        wisdom_file = "fftw_wisdom_6d_1thread.txt";
    }

    string wisdom_file_input = constants::INPUT_PATH + wisdom_file;
    
    if (fftw_import_wisdom_from_filename(wisdom_file_input.c_str()) != 0)
    {
        cout << "Loaded FFTW wisdom from: " << wisdom_file_input << " at " + get_local_time() << endl;
    }
    else
    {
        cout << "No FFTW wisdom found. 6D MEASURE plan may take long on first run " << " at " + get_local_time() << endl;
    }

    FFT_Config config_c2c_6d_forward = {
        .dim = 6,
        .shape = {grid_size, grid_size, grid_size, grid_size, grid_size, grid_size},
        .type = FFT_Type::C2C,
        .direction = FFT_Direction::Forward,
        .inplace = true, // in-place to save memory
        .normalize = true,
        .denormalize = false,
        .nthreads = nthreads_FFTW,
        .flags = FFT_Flag::Measure}; // Measure is too expensive for 6D size (32^6), use Estimate for practical startup time

    fft_c2c_6d_forward.create_plan(
        config_c2c_6d_forward,
        this->T_Y_U_I_.data(),
        this->T_Y_U_I_.data());
    
    string wisdom_file_output = this->sim_params_.output_directory + "/" + wisdom_file;
    if (fftw_export_wisdom_to_filename(wisdom_file_output.c_str()) != 0)
    {
        cout << "Saved FFTW wisdom to: " << wisdom_file_output << " at " + get_local_time() << endl;
    }
    else
    {
        cout << "Warning: Could not save FFTW wisdom to: " << wisdom_file_output << " at " + get_local_time() << endl;
    }

    // cout <<"FFT plans created at " << get_local_time() << endl;
}

void Solver_3D::block_update_and_store_solution_for_the_next_time_step(FFT_Executor &fft_c2c_3d_backward, int &save_time_index, int t_step, const int nthreads)
{
    // Update previous time step solution
    copy_tensor(this->f_hat_previous_, this->f_hat_current_, nthreads);

    fft_c2c_3d_backward.execute_plan(this->f_hat_current_.data(), this->f_solution_.data());

    // Store solution at current time step
    if (t_step == this->t_save_points_(0, save_time_index) && save_time_index < int(this->t_save_points_.size() / 2))
    {
        this->f_solution_time_.insert_slice(0, save_time_index, this->f_solution_, nthreads);
        this->f_hat_time_.insert_slice(0, save_time_index, this->f_hat_current_, nthreads);
        save_time_index++;
    }

    this->print_infomation_each_timestep(t_step);
}

void Solver_3D::block_compute_T_Y_U_I_hat(FFT_Executor &fft_c2c_6d_forward, const int grid_size, const int nthreads)
{
    this->compute_T_tensors();
    fft_c2c_6d_forward.execute_plan(this->T_Y_U_I_.data(), this->T_Y_U_I_.data());

#pragma omp parallel for num_threads(nthreads)
    for (size_t idx_n1 = 0; idx_n1 < grid_size; ++idx_n1)
    {
        for (size_t idx_n2 = 0; idx_n2 < grid_size; ++idx_n2)
        {
            for (size_t idx_n3 = 0; idx_n3 < grid_size; ++idx_n3)
            {
                this->T_RHS_(idx_n1, idx_n2, idx_n3) = this->T_Y_U_I_(idx_n1, idx_n2, idx_n3, idx_n1, idx_n2, idx_n3) - this->T_Y_U_I_(idx_n1, idx_n2, idx_n3, 0, 0, 0) - this->T_Y_U_I_(0, 0, 0, idx_n1, idx_n2, idx_n3);
            }
        }
    }

    this->compute_Y_tensors();
    fft_c2c_6d_forward.execute_plan(this->T_Y_U_I_.data(), this->T_Y_U_I_.data());
#pragma omp parallel for num_threads(nthreads)
    for (size_t idx_n1 = 0; idx_n1 < grid_size; ++idx_n1)
    {
        for (size_t idx_n2 = 0; idx_n2 < grid_size; ++idx_n2)
        {
            for (size_t idx_n3 = 0; idx_n3 < grid_size; ++idx_n3)
            {
                this->Y_RHS_(idx_n1, idx_n2, idx_n3) = this->T_Y_U_I_(idx_n1, idx_n2, idx_n3, idx_n1, idx_n2, idx_n3) - this->T_Y_U_I_(idx_n1, idx_n2, idx_n3, 0, 0, 0) - this->T_Y_U_I_(0, 0, 0, idx_n1, idx_n2, idx_n3);
            }
        }
    }

    this->compute_U_tensors();
    fft_c2c_6d_forward.execute_plan(this->T_Y_U_I_.data(), this->T_Y_U_I_.data());
#pragma omp parallel for num_threads(nthreads)
    for (size_t idx_n1 = 0; idx_n1 < grid_size; ++idx_n1)
    {
        for (size_t idx_n2 = 0; idx_n2 < grid_size; ++idx_n2)
        {
            for (size_t idx_n3 = 0; idx_n3 < grid_size; ++idx_n3)
            {
                this->U_RHS_(idx_n1, idx_n2, idx_n3) = this->T_Y_U_I_(idx_n1, idx_n2, idx_n3, idx_n1, idx_n2, idx_n3) - this->T_Y_U_I_(idx_n1, idx_n2, idx_n3, 0, 0, 0) - this->T_Y_U_I_(0, 0, 0, idx_n1, idx_n2, idx_n3);
            }
        }
    }

    if (this->physical_model_->model_params_.equation_type == EquationType::quantum_Boltzmann)
    {
        this->compute_I_tensors();
        fft_c2c_6d_forward.execute_plan(this->T_Y_U_I_.data(), this->T_Y_U_I_.data());
#pragma omp parallel for num_threads(nthreads)
        for (size_t idx_n1 = 0; idx_n1 < grid_size; ++idx_n1)
        {
            for (size_t idx_n2 = 0; idx_n2 < grid_size; ++idx_n2)
            {
                for (size_t idx_n3 = 0; idx_n3 < grid_size; ++idx_n3)
                {
                    this->I_RHS_(idx_n1, idx_n2, idx_n3) = this->T_Y_U_I_(idx_n1, idx_n2, idx_n3, idx_n1, idx_n2, idx_n3) - this->T_Y_U_I_(idx_n1, idx_n2, idx_n3, 0, 0, 0) - this->T_Y_U_I_(0, 0, 0, idx_n1, idx_n2, idx_n3);
                }
            }
        }
    }
}

void Solver_3D::apply_zero_imaginary_part()
{
#pragma omp parallel for num_threads(this->sim_params_.num_threads)
    for (size_t i = 0; i < f_hat_current_.size(); ++i)
    {
        f_hat_current_[i].imag(0.0);
    }
}

void Solver_3D::apply_de_aliasing_two_thirds_f_hat()
{
    const int N = this->sim_params_.N;
    const double cutoff_fraction = this->spectral_config_.two_thirds_dealiasing.cutoff_fraction; // typically 2/3
    const int cutoff = static_cast<int>(std::floor(cutoff_fraction * N));
    const int cutoff_squared = cutoff * cutoff;

#pragma omp parallel for collapse(3) num_threads(this->sim_params_.num_threads) schedule(guided)
    for (int n1 = -N; n1 < N; ++n1)
    {
        for (int n2 = -N; n2 < N; ++n2)
        {
            for (int n3 = -N; n3 < N; ++n3)
            {
                if (this->spectral_config_.two_thirds_dealiasing.use_L2)
                {
                    if ((n1 * n1 + n2 * n2 + n3 * n3) > cutoff_squared)
                    {
                        size_t idx_n1 = this->fft_idx_[n1 + N];
                        size_t idx_n2 = this->fft_idx_[n2 + N];
                        size_t idx_n3 = this->fft_idx_[n3 + N];
                        this->f_hat_current_(idx_n1, idx_n2, idx_n3) = complex<double>(0.0, 0.0);
                    }
                }
                else
                {
                    if (std::abs(n1) > cutoff || std::abs(n2) > cutoff || std::abs(n3) > cutoff)
                    {
                        size_t idx_n1 = this->fft_idx_[n1 + N];
                        size_t idx_n2 = this->fft_idx_[n2 + N];
                        size_t idx_n3 = this->fft_idx_[n3 + N];
                        this->f_hat_current_(idx_n1, idx_n2, idx_n3) = complex<double>(0.0, 0.0);
                    }
                }
            }
        }
    }
}

void Solver_3D::apply_exponential_filter()
{
    const int N = this->sim_params_.N;
    const double kmax_coefficient = this->spectral_config_.exponential_filter.kmax_cofficient;
    const double kmax = kmax_coefficient * N; // maximum wavenumber magnitude in 3D grid

    const double alpha = this->spectral_config_.exponential_filter.strength; // ví dụ 4–10
    const int p = this->spectral_config_.exponential_filter.order;           // ví dụ 8

#pragma omp parallel for num_threads(this->sim_params_.num_threads)
    for (int n1 = -N; n1 < N; ++n1)
    {
        size_t idx_n1 = this->fft_idx_[n1 + N];
        for (int n2 = -N; n2 < N; ++n2)
        {
            size_t idx_n2 = this->fft_idx_[n2 + N];

            for (int n3 = -N; n3 < N; ++n3)
            {
                size_t idx_n3 = this->fft_idx_[n3 + N];

                double k_norm = std::sqrt(double(n1 * n1 + n2 * n2 + n3 * n3));

                double eta = k_norm / kmax;
                if (eta > 1.0)
                {
                    eta = 1.0; // Cap eta at 1.0 to avoid numerical issues
                }

                double sigma = std::exp(-alpha * std::pow(eta, p));

                this->f_hat_current_(idx_n1, idx_n2, idx_n3) *= sigma;
            }
        }
    }
}

void Solver_3D::apply_ODE_solver_euler(FFT_Executor &fft_c2c_6d_forward,
                                       FFT_Executor &fft_c2c_3d_backward)
{
    int save_time_index = 1; // start from 1 since 0 is already saved

    const int nthreads = this->sim_params_.num_threads;
    const int grid_size = 2 * this->sim_params_.N;
    const double dt = this->sim_params_.dt;
    const double C = this->physical_model_->C_;
    const double R = this->physical_model_->model_params_.R;
    const double R3_RHS = 8.0 * R * R * R;

    for (int t_step = 1; t_step < this->N_time_; ++t_step)
    {
        // Compute T_hat, Y_hat, U_hat, I_hat
        this->block_compute_T_Y_U_I_hat(fft_c2c_6d_forward, grid_size, nthreads);

#pragma omp parallel for num_threads(nthreads)
        for (size_t idx_n1 = 0; idx_n1 < static_cast<size_t>(grid_size); ++idx_n1)
        {
            for (size_t idx_n2 = 0; idx_n2 < static_cast<size_t>(grid_size); ++idx_n2)
            {
                for (size_t idx_n3 = 0; idx_n3 < static_cast<size_t>(grid_size); ++idx_n3)
                {
                    complex<double> rhs = this->compute_RHS(idx_n1, idx_n2, idx_n3, R3_RHS, C);

                    this->f_hat_current_(idx_n1, idx_n2, idx_n3) = this->f_hat_previous_(idx_n1, idx_n2, idx_n3) + dt * rhs;
                }
            }
        }

        // Apply spectral stabilization if enabled
        this->block_apply_spectral_stabilization_f_hat();

        // Update and store solution for the next time step
        this->block_update_and_store_solution_for_the_next_time_step(fft_c2c_3d_backward, save_time_index, t_step, nthreads);
    }
}