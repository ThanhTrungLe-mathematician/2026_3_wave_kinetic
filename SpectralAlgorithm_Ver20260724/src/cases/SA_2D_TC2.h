#include "solver_2D_new.h"
#include <chrono>
#include <thread>

void run_solver_2D_TC2(double parameters, string output_dir)
{
    ModelParameters model_params = {
        .dimension = 2,
        .R = 10.0,
        .M = 4.0,
        .rho = parameters,
        .equation_type = EquationType::three_wave_kinetic,
        .dispersion_relation = {
            .polynomial_form_1 = {
                .enabled = true,
                .alpha = 1.5}},
        .initial_condition = {
            .initial_condition_type = InitialConditionType::exponential, 
            .amplitude_initial = 1.0, 
            .power_initial = 1.0
        }
    };

    SimulationParameters sim_params = {
        .N = 32,
        .dt = 0.0005,
        .final_time = 0.05,
        .output_directory = output_dir,
        .num_threads = 4,
        .total_number_of_times_saves = 501,
        .ode_solver = ODE_solver_type::Euler};

    SpectralStabilizationConfig spectral_config = {
        .early_dealiasing = {
            .enabled = true,
            .use_L2 = true},
        .exponential_filter = {.enabled = true, .strength = 7.5, .order = 8.0}};

    Solver_2D_new solver_0(model_params, sim_params, spectral_config);
    solver_0.run_solver();
}

void run_SA_2D_TC2()
{
// #pragma omp parallel sections
//     {
// #pragma omp section
//         {
//             run_solver_2D_TC2(1.1, "output_0");

//             //run_solver_2D_TC2(7.5, "output_1");
//         }
// #pragma omp section
//         {
//             std::this_thread::sleep_for(std::chrono::seconds(10));
//             run_solver_2D_TC2(1.2, "output_1");

//             //run_solver_2D_TC2(8.5, "output_3");
//         }
// #pragma omp section
//         {
//             std::this_thread::sleep_for(std::chrono::seconds(20));
//             run_solver_2D_TC2(1.3, "output_2");

//             //run_solver_2D_TC2(9.5, "output_5");
//         }
//     }

    run_solver_2D_TC2(2.0, "output_0");
}