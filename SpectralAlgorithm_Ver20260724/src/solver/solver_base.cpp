#include "solver_base.h"
#include <filesystem>
#include <iostream>
#include <fstream>
#include <ctime>
#include <stdexcept>
#include "constants.h"
using namespace std;

Solver_Base::Solver_Base(const ModelParameters &model_params, SimulationParameters &sim_params, const SpectralStabilizationConfig &spectral_stabilization_config)
{   
    sim_params.validate();
    model_params.validate();
    spectral_stabilization_config.validate();

    this->sim_params_ = sim_params;
    this->spectral_config_ = spectral_stabilization_config;

    // this->N_ = sim_params.N;
    // this->dt_ = sim_params.dt;
    // this->final_time_ = sim_params.final_time;
    // this->output_directory_ = constants::OUTPUT_PATH + sim_params.output_directory;
    // this->num_threads_ = sim_params.num_threads;

    this->sim_params_.output_directory = constants::OUTPUT_PATH + sim_params.output_directory;

    this->N_time_ = static_cast<int>(this->sim_params_.final_time / this->sim_params_.dt) + 1;
    this->nthreads_FFTW_ = this->sim_params_.num_threads;

    if (this->sim_params_.num_threads > 1)
        this->is_parallel_ = true;
    else
        this->is_parallel_ = false;

    this->cpu_time_total_ = 0.0;
    this->real_time_total_ = 0.0;
    this->last_reported_percent_ = 0;
    this->is_manual_ = false;

    this->physical_model_ = new PhysicalModel(model_params);
}

Solver_Base::~Solver_Base()
{
    delete this->physical_model_;
}

void Solver_Base::solver_detailed()
{
    double start_time = get_cpu_time_hour();
    time_t start_now = time(0);

    cout << "1. Get all Input Parameters at " + get_local_time() << endl;

    cout << "2. Check the correctness of computational data" << endl;
    this->check_computational_data();

    cout << "3. Clean the output directory before computation!" << endl;
    this->clean_output_directory();

    cout << "4. Write the computation data!" << endl;
    this->write_computational_data();

    cout << "5. Create objects for solver before computations!" << endl;
    this->create_objects();

    cout << "6. Compute the solution:" << endl;
    cout << "The number of thread = " << this->sim_params_.num_threads << endl;
    cout << "The output directory is " << this->sim_params_.output_directory << endl;
    if(this->is_manual_)
        this->compute_solution_manual();
    else
    {
        this->compute_solution();
    }

    double finish_time = get_cpu_time_hour();
    this->cpu_time_total_ = finish_time - start_time;

    time_t end_now = time(0);
    this->real_time_total_ = (end_now - start_now) / 3600.0;

    cout << "7. Write the solution!" << endl;
    this->write_solution_final();

    cout << "8. Finish successfully at " + get_local_time() << endl;
    cout << "The CPU time is " << this->cpu_time_total_ << " hours!" << endl;
    cout << "The real time is " << this->real_time_total_ << " hours!" << endl;
    cout << endl;
    cout << "----------------------------------------------------------------" << endl;
    cout << endl;
}

void Solver_Base::clean_output_directory()
{
    namespace fs = std::filesystem;
    fs::path dir = this->sim_params_.output_directory;

    std::error_code ec;
    if (!fs::exists(dir, ec)) return;

    for (const auto& entry : fs::directory_iterator(dir, ec)) {
        if (ec) break;
        // remove all files (and symlinks), keep subdirectories
        if (entry.is_regular_file(ec) || entry.is_symlink(ec)) {
            fs::remove(entry.path(), ec);
        }
    }
}

void Solver_Base::print_infomation_each_timestep(int current_time_step)
{   
    // show the percent of completed computation and write solution at each 10%
    int percent_done = (100 * current_time_step) / this->N_time_;

    if (percent_done >= last_reported_percent_ + 10)
    {
        last_reported_percent_ = (percent_done / 10) * 10;

        cout << "The computation is completed " << last_reported_percent_ << "%!" << endl;
        cout.flush();  // Force flush to ensure output in nohup.out

        string cout_filename = "f_solution_at_" + to_string(last_reported_percent_) + "percent";

        this->write_tensor_solution(cout_filename);

        cout << "Writing the solution at " << last_reported_percent_ << "% of the number of time steps!" << endl;
        cout.flush();  // Force flush
    }
}

void Solver_Base::run_solver()
{
    this->solver_detailed();
}

void Solver_Base::run_solver_manual(double manual_step_size)
{
    this->is_manual_ = true;
    this->manual_step_size_ = manual_step_size;
    this->solver_detailed();
}

void Solver_Base::write_solution_final()
{
    this->write_tensor_solution("f_solution_final", true);

    string filename = this->sim_params_.output_directory + constants::COMPUTATIONAL_DATA_FILENAME;
    ofstream writefile(filename, ios::out | ios::app);

    // if the file can not be open then stop
    if (!writefile)
    {
        cerr << "Impossible to open the " << filename.c_str() << " file!" << endl;
        exit(EXIT_FAILURE);
    }

    // DEBUG: in đường dẫn file
    //cout << "DEBUG: Writing to: " << filename << endl;
    //cout << "DEBUG: output_directory_ = " << this->output_directory_ << endl;

    // Write computation information
    writefile << "\n#####################################################################" << endl;
    writefile << "# Computation information" << endl;
    writefile << "#####################################################################" << endl;
    writefile << "\nNumber of threads: " << this->sim_params_.num_threads << endl;
    writefile << "\nCPU time (hours): " << this->cpu_time_total_ << endl;
    writefile << "\nReal time (hours): " << this->real_time_total_ << endl;
    writefile << "\nThe simulation finish successfully at " + get_local_time() << endl;

    writefile.close();
}

void Solver_Base::write_computational_data()
{
    string filename = this->sim_params_.output_directory + constants::COMPUTATIONAL_DATA_FILENAME;

    //this->physical_model_->write_model_parameters(filename);

    this->physical_model_->model_params_.write_model_parameters(filename);
    this->sim_params_.write_simulation_parameters(filename);
    this->spectral_config_.write_spectral_stabilization_config(filename);

    // ofstream writefile(filename, ios::out | ios::app);

    // // if the file can not be open then stop
    // if (!writefile)
    // {
    //     cerr << "Impossible to open the " << filename.c_str() << " file!" << endl;
    //     exit(EXIT_FAILURE);
    // }

    // writefile << endl;
    // writefile << "\n#####################################################################" << endl;
    // writefile << "# Simulation parameters" << endl;
    // writefile << "#####################################################################" << endl;

    // writefile.precision(constants::PRECISION);

    // writefile << "\nThe Fourier series truncation parameter <N>:: " << this->sim_params_.N << endl;
    // writefile << "\nThe time step size <dt>:: " << this->sim_params_.dt << endl;
    // writefile << "\nThe final simulation time <final_time>:: " << this->sim_params_.final_time << endl;
    // writefile << "\nThe number of threads for parallel computation <num_threads>:: " << this->sim_params_.num_threads << endl;

    // writefile.close();
}

void Solver_Base::construct_k_grid()
{
    this->k_grid_.create_vector_FFTW(this->physical_model_->model_params_.R, this->sim_params_.N);

    this->k_grid_.WriteToHDF5(this->sim_params_.output_directory + "/k_grid.h5");
    this->k_grid_.WriteToText(this->sim_params_.output_directory + "/k_grid.txt");
}

void Solver_Base::compute_fftw_indices()
{
    int N = this->sim_params_.N;
    for (int k = - N; k < N; ++k)
    {
        fft_idx_[k + N] = (k >= 0 ? k : k + 2*N);
    }
}

void Solver_Base::make_saved_times_points()
{
    // Implementation of make_saved_times_points
    const int M = this->sim_params_.total_number_of_times_saves;
    const int N_time = this->N_time_;
    const double dt = this->sim_params_.dt;

    if (M == 1)
    {
        this->t_save_points_(0, 0) = N_time;
        this->t_save_points_(1, 0) = N_time * dt;
        return;
    }

    int last_step = -1;

    for (int m = 0; m < M; ++m)
    {
        int step = static_cast<int>(
            std::round(double(m) * double(N_time - 1) / double(M - 1))
        );

        // đảm bảo tăng đơn điệu
        if (step <= last_step)
            step = last_step + 1;

        // chặn trên
        if (step > N_time)
            step = N_time;
            
        this->t_save_points_(0, m) = step;
        this->t_save_points_(1, m) = step * dt;

        last_step = step;
    }

    this->t_save_points_.WriteToHDF5(this->sim_params_.output_directory + "/t_save_points.h5");
    this->t_save_points_.WriteToText(this->sim_params_.output_directory + "/t_save_points.txt");
}

void Solver_Base::block_apply_spectral_stabilization_f_hat()
{
    if (this->spectral_config_.zero_imaginary.enabled)
    {
        this->apply_zero_imaginary_part();
    }

    if (this->spectral_config_.two_thirds_dealiasing.enabled)
    {
        this->apply_de_aliasing_two_thirds_f_hat();
    }

    if (this->spectral_config_.exponential_filter.enabled)
    {
        this->apply_exponential_filter();
    }
}