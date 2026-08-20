#ifndef SOLVER_2D_H
#define SOLVER_2D_H
#include "solver_base_2D.h"


class Solver_2D : public Solver_Base_2D
{
private:
    //Tensor4DComplex W_tensor_;      ///< W tensor with shape (grid_size, grid_size, grid_size, grid_size)
    Tensor4DComplex W_hat_tensor_; ///< Fourier transform of the W tensor with shape (grid_size, grid_size, grid_size, grid_size)
    //Tensor2DComplex f_initial_;     ///< Initial condition tensor with shape (grid_size, grid_size)
    //Tensor3DComplex f_solution_time_;    ///< Solution tensor with shape (N_time, grid_size, grid_size)
    //Tensor3DComplex f_hat_time_;      ///< Solution in Fourier space over time with shape (N_time, grid_size, grid_size)
    
    //Tensor2DComplex f_hat_previous_; ///< Previous time step solution in Fourier space with shape (grid_size, grid_size)
    //Tensor2DComplex f_hat_current_;  ///< Current time step solution in Fourier space with shape (grid_size, grid_size)
    //Tensor2DComplex f_solution_;      ///< Solution tensor with shape (grid_size, grid_size)

    //Tensor1DReal k_grid_;        ///< Wave-vector grid points in each dimension

public:
    Solver_2D(const ModelParameters &model_params, SimulationParameters &sim_params, const SpectralStabilizationConfig &spectral_stabilization_config);
    ~Solver_2D();

private:

    void create_objects() override;

    void compute_solution() override;

    void compute_W_hat();

    void compute_solution_manual() override;

    void compute_f_initial_hat_manual();

    void compute_W_hat_manual();

    void recontruct_solution_manual();

    inline complex<double> compute_H_np(int n1, int n2, int p1, int p2, double R);

    inline complex<double> compute_K_npm(int n1, int n2, int p1, int p2, int m1, int m2, double R);
};
#endif // SOLVER_2D_H