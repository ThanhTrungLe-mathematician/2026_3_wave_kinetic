#include "solver_3D.h"

void run_solver_3D_TC2(double parameter, string output_dir)
{
    ModelParameters model_params = {
        .dimension = 3,
        .R = 10.0,
        .M = 4.0,
        .rho = parameter,
        .equation_type = EquationType::three_wave_kinetic,
        .dispersion_relation = {
            .polynomial_form_1 = {
                .enabled = true,
                .alpha = 1.5
            }
        },
        .initial_condition = {
            .initial_condition_type = InitialConditionType::exponential,
            .amplitude_initial = 1.0,
            .power_initial = 2.0
        }    
    };

    SimulationParameters sim_params = {
        .N = 16,
        .dt = 0.0005,
        .final_time = 0.05,
        .output_directory = output_dir,
        .num_threads = 16,
        .total_number_of_times_saves = 501,
        .ode_solver = ODE_solver_type::Euler,
        .max_memory_GB = 30,
    };

    SpectralStabilizationConfig spectral_config = {
        .early_dealiasing = {
            .enabled = true,
            .use_L2 = true
        },
        .zero_imaginary = {
            .enabled = false
        },
        .two_thirds_dealiasing = {
            .enabled = false,
            .use_L2 = true
        },
        .exponential_filter = {
            .enabled = true,
            .strength = 7.5,
            .order = 8.0,
            .kmax_cofficient = sqrt(2.0)
        }
    };

    Solver_3D solver_0(model_params, sim_params, spectral_config);
    solver_0.run_solver();
}

void run_SA_3D_TC2()
{
    run_solver_3D_TC2(2.0, "output_1");
}