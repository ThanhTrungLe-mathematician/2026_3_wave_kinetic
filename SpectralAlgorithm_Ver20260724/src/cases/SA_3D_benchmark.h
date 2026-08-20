#include "solver_3D.h"
#include <chrono>
#include <ctime>
#include <memory>

using namespace std;

void benchmark_3D(int N);

void benchmark_3D(int N)
{
    ofstream benchmark_log("../data/benmark.txt", ios::app);
    if (!benchmark_log) {
        return;
    }

    benchmark_log << "========================================" << endl;
    benchmark_log << "           3D SOLVER BENCHMARK          " << endl;
    benchmark_log << "========================================" << endl;
    benchmark_log << "Grid size N = " << N << endl;
    benchmark_log << endl;

    ModelParameters model_params = {
        .dimension = 3,
        .R = 10.0,
        .M = 4.0,
        .rho = 1.0,
        .equation_type = EquationType::quantum_Boltzmann,
        .dispersion_relation = {
            .polynomial_form_1 = {
                .enabled = true,
                .alpha = 1.0
            }
        },
        .initial_condition = {
            .initial_condition_type = InitialConditionType::exponential,
            .amplitude_initial = 1.0,
            .power_initial = 2.0
        }
    };

    SimulationParameters sim_params = {
        .N = N,
        .dt = 0.0005,
        .final_time = 0.05,
        .output_directory = "output_1",
        .num_threads = 8,
        .total_number_of_times_saves = 101,
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
            .order = 8.0
        }
    };

    benchmark_log << "Running solver_3D..." << endl;
    double start_solver = get_cpu_time_sec();

    Solver_3D solver(model_params, sim_params, spectral_config);
    solver.run_solver();

    double end_solver = get_cpu_time_sec();
    double elapsed = end_solver - start_solver;

    int total_steps = static_cast<int>(sim_params.final_time / sim_params.dt) + 1;
    double avg_time_per_step = (total_steps > 1) ? (elapsed / (total_steps - 1)) : 0.0;

    benchmark_log << endl;
    benchmark_log << "========================================" << endl;
    benchmark_log << "         BENCHMARK SUMMARY              " << endl;
    benchmark_log << "========================================" << endl;
    benchmark_log << "Total CPU time solver_3D:   " << elapsed << " s" << endl;
    benchmark_log << "Total time steps:           " << (total_steps - 1) << endl;
    benchmark_log << "Avg time/step solver_3D:    " << avg_time_per_step << " s" << endl;
    benchmark_log << "========================================" << endl;
    benchmark_log << "Log file: ../data/benmark.txt" << endl;
    benchmark_log << endl;
}