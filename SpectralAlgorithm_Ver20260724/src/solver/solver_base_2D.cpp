#include "solver_base_2D.h"

#include "fft_executor.h"
#include "utilities.h"
#include "tensor_free_function.h"
#include <complex>
#include <cmath>
using namespace std;

Solver_Base_2D::Solver_Base_2D(const ModelParameters &model_params, SimulationParameters &sim_params, const SpectralStabilizationConfig &spectral_stabilization_config)
    : Solver_Base(model_params, sim_params, spectral_stabilization_config)
{
    // Constructor implementation (if any)
}

Solver_Base_2D::~Solver_Base_2D()
{
    // Destructor implementation (if any)
}

void Solver_Base_2D::construct_f_initial_tensor()
{
    // Implementation of constructing f_initial_ tensor if needed
    vector<double> grid_vector = this->k_grid_.get_data_vector();

    // Call directly without wrapper - bind member function
    auto func_initial_bound = [this](const vector<double> &k)
    {
        return this->physical_model_->func_initial_condition(k);
    };

    this->f_initial_.fill_from_function({grid_vector, grid_vector}, func_initial_bound, this->sim_params_.num_threads);

    this->f_initial_.WriteToHDF5(this->sim_params_.output_directory + "/f_initial.h5");
    this->f_initial_.WriteToText(this->sim_params_.output_directory + "/f_initial.txt");
}

void Solver_Base_2D::compute_f_initial_hat()
{
    // FFTW R2C for initial condition f_initial_
    int Nx = 2 * this->sim_params_.N;
    int Ny = 2 * this->sim_params_.N;

    FFT_Config config_c2c_2d = {
        .dim = 2,
        .shape = {Nx, Ny},
        .type = FFT_Type::C2C,
        .direction = FFT_Direction::Forward,
        .inplace = false,
        .normalize = true,
        .denormalize = false,
        .nthreads = this->nthreads_FFTW_,
        .flags = FFT_Flag::Estimate};

    FFT_Executor fft_c2c_2d;
    fft_c2c_2d.create_plan(
        config_c2c_2d,
        this->f_initial_.data(),
        this->f_hat_previous_.data());

    fft_c2c_2d.execute_plan(
        config_c2c_2d,
        this->f_initial_.data(),
        this->f_hat_previous_.data());

    fft_c2c_2d.destroy_plan();

    this->f_hat_previous_.WriteToHDF5(this->sim_params_.output_directory + "/f_hat_initial.h5");
    this->f_hat_previous_.WriteToText(this->sim_params_.output_directory + "/f_hat_initial.txt");
}

void Solver_Base_2D::construct_W_tensor()
{
    vector<double> grid_vector = this->k_grid_.get_data_vector();

    // Call directly without wrapper - bind member function
    auto func_W_bound = [this](const vector<double> &k)
    {
        return this->physical_model_->func_W(k);
    };

    this->W_tensor_.fill_from_function(
        {grid_vector, grid_vector, grid_vector, grid_vector}, func_W_bound, this->sim_params_.num_threads);

    // if (this->sim_params_.save_hdf5_files)
    // {
    //     this->W_tensor_.WriteToHDF5(this->sim_params_.output_directory + "/W_tensor.h5");
    // }

    // if (this->sim_params_.save_txt_files)
    // {
    //     this->W_tensor_.WriteToText(this->sim_params_.output_directory + "/W_tensor.txt");
    // }
}

void Solver_Base_2D::write_tensor_solution(string cout_filename, bool final)
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

    //this->f_solution_.WriteToText(this->sim_params_.output_directory + "/" + cout_filename + ".txt");
}

void Solver_Base_2D::check_computational_data()
{
    if (this->physical_model_->model_params_.dimension != 2)
    {
        throw runtime_error("Error: The dimension of the physical model does not match the solver dimension (2D).");
    }
}

void Solver_Base_2D::apply_zero_imaginary_part()
{
#pragma omp parallel for num_threads(this->sim_params_.num_threads)
    for (size_t i = 0; i < f_hat_current_.size(); ++i)
    {
        f_hat_current_[i].imag(0.0);
    }
}

void Solver_Base_2D::apply_de_aliasing_two_thirds_f_hat()
{
    int N = this->sim_params_.N;
    double cutoff_fraction = this->spectral_config_.two_thirds_dealiasing.cutoff_fraction;
    int cutoff = static_cast<int>(std::floor(cutoff_fraction * N));

    if (this->spectral_config_.two_thirds_dealiasing.use_L2)
    {
#pragma omp parallel for collapse(2) num_threads(this->sim_params_.num_threads) schedule(guided)
        for (int n1 = -N; n1 < N; ++n1)
        {
            for (int n2 = -N; n2 < N; ++n2)
            {
                if ((n1 * n1 + n2 * n2) > cutoff * cutoff)
                {
                    size_t idx_n1 = this->fft_idx_[n1 + N];
                    size_t idx_n2 = this->fft_idx_[n2 + N];
                    this->f_hat_current_(idx_n1, idx_n2) = complex<double>(0.0, 0.0);
                }
            }
        }
    }
    else
    {
#pragma omp parallel for collapse(2) num_threads(this->sim_params_.num_threads)
        for (int n1 = -N; n1 < N; ++n1)
        {
            for (int n2 = -N; n2 < N; ++n2)
            {
                if (std::abs(n1) > cutoff || std::abs(n2) > cutoff)
                {
                    size_t idx_n1 = this->fft_idx_[n1 + N];
                    size_t idx_n2 = this->fft_idx_[n2 + N];
                    this->f_hat_current_(idx_n1, idx_n2) = complex<double>(0.0, 0.0);
                }
            }
        }
    }
}

void Solver_Base_2D::apply_exponential_filter()
{
    const int N = this->sim_params_.N;
    const double kmax_coefficient = this->spectral_config_.exponential_filter.kmax_cofficient;
    const double kmax = kmax_coefficient * N; // Maximum wavenumber in 2D is sqrt(2)*N

    const double alpha = this->spectral_config_.exponential_filter.strength; // ví dụ 4–10
    const int p = this->spectral_config_.exponential_filter.order;           // ví dụ 8

#pragma omp parallel for collapse(2) num_threads(this->sim_params_.num_threads)
    for (int n1 = -N; n1 < N; ++n1)
    {
        for (int n2 = -N; n2 < N; ++n2)
        {   
            size_t idx_n1 = this->fft_idx_[n1 + N];
            size_t idx_n2 = this->fft_idx_[n2 + N];

            double k_norm = std::sqrt(double(n1 * n1 + n2 * n2));

            double eta = k_norm / kmax;
            if (eta > 1.0)
            {
                eta = 1.0; // Cap eta at 1.0 to avoid numerical issues
            }
            
            double sigma = std::exp(-alpha * std::pow(eta, p));

            this->f_hat_current_(idx_n1, idx_n2) *= sigma;
        }
    }
}

void Solver_Base_2D::block_update_and_store_solution_for_the_next_time_step(FFT_Executor &fft_c2c_2d_backward, int &save_time_index, int t_step)
{
    int nthreads = this->sim_params_.num_threads;
    // Update previous time step solution
    copy_tensor(this->f_hat_previous_, this->f_hat_current_, nthreads);

    fft_c2c_2d_backward.execute_plan(this->f_hat_current_.data(), this->f_solution_.data());

    // Store solution at current time step
    if (t_step == this->t_save_points_(0, save_time_index) && save_time_index < int(this->t_save_points_.size() / 2))
    {
        this->f_solution_time_.insert_slice(0, save_time_index, this->f_solution_, nthreads);
        this->f_hat_time_.insert_slice(0, save_time_index, this->f_hat_current_, nthreads);
        save_time_index++;
    }

    this->print_infomation_each_timestep(t_step);
}