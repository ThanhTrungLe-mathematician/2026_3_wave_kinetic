#ifndef SOLVER_3D_H
#define SOLVER_3D_H

#include "solver_base.h"
#include "fft_executor.h"
#include "tensor_free_function.h"
#include <functional>
using namespace std;

class Solver_3D : public Solver_Base
{
public:
    Solver_3D(const ModelParameters &model_params, SimulationParameters &sim_params, const SpectralStabilizationConfig &spectral_stabilization_config);
    ~Solver_3D();

protected:
    Tensor3DComplex f_initial_;       ///< Initial condition tensor with shape (grid_size, grid_size, grid_size)
    Tensor3DComplex f_solution_;      ///< Solution tensor with shape (grid_size, grid_size, grid_size)
    Tensor4DComplex f_solution_time_; ///< Solution tensor with shape (N_time, grid_size, grid_size, grid_size)

    Tensor3DComplex f_hat_previous_; ///< Previous Fourier transform tensor with shape (grid_size, grid_size, grid_size)
    Tensor3DComplex f_hat_current_;  ///< Current Fourier transform tensor with shape (grid_size, grid_size, grid_size)
    Tensor4DComplex f_hat_time_;     ///< Fourier transform tensor over time with shape (N_time, grid_size, grid_size, grid_size)

    //Tensor6DReal W_tensor_;   ///< W tensor with shape (grid_size, grid_size, grid_size, grid_size, grid_size, grid_size)
    Tensor6DReal_Float32 W_tensor_;   ///< W tensor with shape (grid_size, grid_size, grid_size, grid_size, grid_size, grid_size)
    Tensor6DComplex T_Y_U_I_; ///< Combined T, Y, U, I tensor with shape (grid_size, grid_size, grid_size, grid_size, grid_size, grid_size)

    Tensor3DComplex T_RHS_; ///< T tensor for RHS computation with shape (grid_size, grid_size, grid_size)
    Tensor3DComplex Y_RHS_; ///< Y tensor for RHS computation with shape (grid_size, grid_size, grid_size)
    Tensor3DComplex U_RHS_; ///< U tensor for RHS computation with shape (grid_size, grid_size, grid_size)
    Tensor3DComplex I_RHS_; ///< I tensor for RHS computation with shape (grid_size, grid_size, grid_size)

    void check_computational_data() override;

    void create_objects() override;

    void write_tensor_solution(string cout_filename, bool final = false) override;

    void construct_f_initial_tensor() override;

    void compute_f_initial_hat() override;

    void compute_solution() override;

    void construct_W_tensor();

    void compute_T_tensors();

    void compute_Y_tensors();

    void compute_U_tensors();

    void compute_I_tensors();

    void apply_zero_imaginary_part() override;

    void apply_exponential_filter() override;

    void apply_de_aliasing_two_thirds_f_hat() override;

    void block_create_fftw_plans(FFT_Executor &fft_c2c_6d_forward,  
                                FFT_Executor &fft_c2c_3d_backward, int grid_size, int nthreads_FFTW);

    void block_update_and_store_solution_for_the_next_time_step(FFT_Executor &fft_c2c_3d_backward, int &save_time_index, int t_step, const int nthreads);

    void block_compute_T_Y_U_I_hat(FFT_Executor &fft_c2c_6d_forward, const int grid_size, const int nthreads);

    void apply_ODE_solver_euler(FFT_Executor &fft_c2c_6d_forward,
                                FFT_Executor &fft_c2c_3d_backward);

    
    inline complex<double> compute_RHS(size_t idx_n1, size_t idx_n2, size_t idx_n3, double R3_RHS, double C)
    {
        if (this->physical_model_->model_params_.equation_type == EquationType::quantum_Boltzmann)
        {
            // Multiply term-by-term to avoid catastrophic cancellation and preserve numerical accuracy
            // when subtracting small differences of similar magnitudes
            return (R3_RHS * T_RHS_(idx_n1, idx_n2, idx_n3)) 
                   - (R3_RHS * Y_RHS_(idx_n1, idx_n2, idx_n3))
                   - (R3_RHS * U_RHS_(idx_n1, idx_n2, idx_n3))
                   - (C * R3_RHS * I_RHS_(idx_n1, idx_n2, idx_n3));
        }
        else // three_wave_kinetic
        {
            return (R3_RHS * T_RHS_(idx_n1, idx_n2, idx_n3))
                   - (R3_RHS * Y_RHS_(idx_n1, idx_n2, idx_n3))
                   - (R3_RHS * U_RHS_(idx_n1, idx_n2, idx_n3));
        }
    }
};
#endif // SOLVER_3D_H