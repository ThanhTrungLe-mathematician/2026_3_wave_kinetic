#ifndef SOLVER_BASE_H
#define SOLVER_BASE_H

#include "physicalModel.h"
#include "simulationParameters.h"
#include "spectralStabilizationConfig.h"
#include "utilities.h"

class Solver_Base
{
protected:
    // simulation parameters
    // int N_;
    // double dt_;
    // double final_time_;
    // string output_directory_;
    // int num_threads_;
    SimulationParameters sim_params_;

    int N_time_;
    int nthreads_FFTW_;
    bool is_parallel_;

    // spectral stabilization config
    SpectralStabilizationConfig spectral_config_;
    

    //local parameters
    double cpu_time_total_;
    double real_time_total_;
    int last_reported_percent_;

    bool is_manual_;
    double manual_step_size_;

    PhysicalModel* physical_model_;

    Tensor1DReal k_grid_;        ///< Wave-vector grid points in each dimension

    Tensor2DReal t_save_points_;  ///< Time points to save solution, shape (N_time, 2), col 0: index in time steps, col 1: time value
    
    /**
     * @brief FFTW index mapping for wave-vector grid, used for efficient FFT computations 
     * 
     * n ∈ [-N, N-1] mapped to fft_idx_[n + N] for index in [0, 2N-1]
     */
    vector<size_t> fft_idx_;  

    // Constructor
    Solver_Base(const ModelParameters &model_params, SimulationParameters &sim_params, const SpectralStabilizationConfig &spectral_stabilization_config);
    ~Solver_Base();
public:
    void solver_detailed();

    void run_solver();

    void run_solver_manual(double manual_step_size);

    void clean_output_directory();

    void write_solution_final();

    void print_infomation_each_timestep(int current_time_step);
    
    void write_computational_data();

    /**
     * @brief Get FFTW index from mathematical index
     *
     * @param k Mathematical index
     * @return Corresponding FFTW index
     */
    inline size_t get_fft_index_from_math(int k)
    {
        return get_fft_index_from_math_global(k, this->sim_params_.N);
    }

protected:
    virtual void compute_f_initial_hat() = 0;

    virtual void construct_f_initial_tensor() = 0;

    void construct_k_grid();

    void make_saved_times_points();

    void compute_fftw_indices();

    virtual void check_computational_data() = 0;

    virtual void create_objects() = 0;

    virtual void write_tensor_solution(string cout_filename, bool final = false) = 0;

    virtual void compute_solution() = 0;

    virtual void compute_solution_manual(){};

    void block_apply_spectral_stabilization_f_hat();

    virtual void apply_zero_imaginary_part() = 0;

    virtual void apply_exponential_filter() = 0;

    virtual void apply_de_aliasing_two_thirds_f_hat() = 0;
};

#endif // SOLVER_BASE_H