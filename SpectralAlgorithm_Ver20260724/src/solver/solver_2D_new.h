#ifndef SOLVER_2D_NEW_H
#define SOLVER_2D_NEW_H
#include "solver_base_2D.h"
#include "fft_executor.h"


class Solver_2D_new : public Solver_Base_2D
{
private:
    
    Tensor4DComplex T_;     ///< T tensor with shape (grid_size, grid_size, grid_size, grid_size)
    //Tensor4DComplex T_hat_;     ///< T_hat tensor with shape (grid_size, grid_size, grid_size, grid_size)
    
    Tensor4DComplex Y_;     ///< Y tensor with shape (grid_size, grid_size, grid_size, grid_size)
    //Tensor4DComplex Y_hat_;     ///< Y_hat tensor with shape (grid_size, grid_size, grid_size, grid_size)

    Tensor4DComplex U_;     ///< U tensor with shape (grid_size, grid_size, grid_size, grid_size)
    //Tensor4DComplex U_hat_;     ///< U_hat tensor with shape (grid_size, grid_size, grid_size, grid_size)

    Tensor4DComplex I_;     ///< I tensor with shape (grid_size, grid_size, grid_size, grid_size)
    //Tensor4DComplex I_hat_;     ///< I_hat tensor with shape (grid_size, grid_size, grid_size, grid_size)

    Tensor2DComplex Q_hat_temp_1_; ///< Temporary tensor for Runge-Kutta stages
    Tensor2DComplex Q_hat_temp_2_; ///< Temporary tensor for Runge-Kutta stages
    Tensor2DComplex Q_hat_temp_3_; ///< Temporary tensor for Runge-Kutta stages
    Tensor2DComplex Q_hat_temp_4_; ///< Temporary tensor for Runge-Kutta stages

public:
    Solver_2D_new(const ModelParameters &model_params, SimulationParameters &sim_params, const SpectralStabilizationConfig &spectral_stabilization_config);
    ~Solver_2D_new();

private:
    void create_objects() override;

    void compute_solution() override;

    void compute_T_Y_U_I_tensors();

    void apply_de_aliasing_two_thirds_T_Y_U_I_hat();

    void apply_early_de_aliasing();

    
    void block_compute_T_Y_U_I_hat(FFT_Executor &fft_c2c_4d_forward);

    void block_create_fftw_plans(FFT_Executor &fft_c2c_4d_forward,  
                                FFT_Executor &fft_c2c_2d_backward, int grid_size, int nthreads_FFTW);

    void apply_ODE_solver_euler(FFT_Executor &fft_c2c_4d_forward,  
                                       FFT_Executor &fft_c2c_2d_backward);

    void apply_ODE_solver_RK2(FFT_Executor &fft_c2c_4d_forward,  
                                      FFT_Executor &fft_c2c_2d_backward);

    void apply_ODE_solver_RK3(FFT_Executor &fft_c2c_4d_forward,  
                                           FFT_Executor &fft_c2c_2d_backward);

    void apply_ODE_solver_SSP_RK3(FFT_Executor &fft_c2c_4d_forward,  
                                            FFT_Executor &fft_c2c_2d_backward);

    void apply_ODE_solver_RK4(FFT_Executor &fft_c2c_4d_forward,  
                                            FFT_Executor &fft_c2c_2d_backward);

    inline complex<double> compute_RHS(size_t idx_n1, size_t idx_n2, double R2_RHS, double C)
    {
        complex<double> T_coefficient = T_(idx_n1, idx_n2, idx_n1, idx_n2) - T_(idx_n1, idx_n2, 0, 0) - T_(0, 0, idx_n1, idx_n2);
        complex<double> Y_coefficient = Y_(idx_n1, idx_n2, idx_n1, idx_n2) - Y_(idx_n1, idx_n2, 0, 0) - Y_(0, 0, idx_n1, idx_n2);
        complex<double> U_coefficient = U_(idx_n1, idx_n2, idx_n1, idx_n2) - U_(idx_n1, idx_n2, 0, 0) - U_(0, 0, idx_n1, idx_n2);

        if (C != 0.0)
        {
            complex<double> I_coefficient = I_(idx_n1, idx_n2, idx_n1, idx_n2) - I_(idx_n1, idx_n2, 0, 0) - I_(0, 0, idx_n1, idx_n2);
            return  (R2_RHS * T_coefficient - R2_RHS * Y_coefficient - R2_RHS * U_coefficient - R2_RHS * C * I_coefficient);
        }
        else
        {
            return  (R2_RHS * T_coefficient - R2_RHS * Y_coefficient - R2_RHS * U_coefficient);
        }
    }

};
#endif // SOLVER_2D_NEW_H