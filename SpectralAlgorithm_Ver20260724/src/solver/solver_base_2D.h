#ifndef SOLVER_BASE_2D_H
#define SOLVER_BASE_2D_H
#include "solver_base.h"
#include "fft_executor.h"

class Solver_Base_2D : public Solver_Base
{
public:
    Solver_Base_2D(const ModelParameters &model_params, SimulationParameters &sim_params, const SpectralStabilizationConfig &spectral_stabilization_config);
    ~Solver_Base_2D();

    // Getter methods for accessing solution data
    const Tensor3DComplex& get_f_solution_time() const { return f_solution_time_; }
    const Tensor2DComplex& get_final_solution() const { return f_solution_;}
    const Tensor2DReal& get_t_save_points() const { return t_save_points_; }

protected:
    Tensor2DComplex f_initial_;     ///< Initial condition tensor with shape (grid_size, grid_size)
    Tensor2DComplex f_solution_;      ///< Solution tensor with shape (grid_size, grid_size)
    Tensor3DComplex f_solution_time_;    ///< Solution tensor with shape (N_time, grid_size, grid_size)

    Tensor4DComplex W_tensor_;      ///< W tensor with shape (grid_size, grid_size, grid_size, grid_size)
    
    Tensor2DComplex f_hat_previous_; ///< Previous time step solution in Fourier space with shape (grid_size, grid_size)
    Tensor2DComplex f_hat_current_;  ///< Current time step solution in Fourier space with shape (grid_size, grid_size)
    Tensor3DComplex f_hat_time_;      ///< Solution in Fourier space over time with shape (N_time, grid_size, grid_size)

    
protected:
    void construct_W_tensor();

    void check_computational_data() override;

    void write_tensor_solution(string cout_filename, bool final = false) override;

    void compute_f_initial_hat() override;

    void construct_f_initial_tensor() override;

    void apply_zero_imaginary_part() override;

    void apply_exponential_filter() override;

    void apply_de_aliasing_two_thirds_f_hat() override;

    void block_update_and_store_solution_for_the_next_time_step(FFT_Executor &fft_c2c_2d_backward, int &save_time_index, int t_step);

    
};
#endif // SOLVER_BASE_2D_H