#include "solver_2D.h"
#include "solver_2D_new.h"
#include "tensor_free_function.h"

#include <chrono>
#include <cmath>
#include <ctime>
#include <limits>
#include <memory>

using namespace std;

int benchmark_2D(int N);
int benchmark_2D_L2_error(int N, int num_threads = 1);
int benchmark_2D_cpu_time(int N);
int benchmark_2D_self_convergence(const vector<int>& N_list = {8, 16, 32, 64}, int num_threads = 1);
int benchmark_2D_self_convergence_time(const vector<int>& N_list = {8, 16, 32, 64}, int num_threads = 1);
int benchmark_2D_self_convergence_base(ModelParameters model_params, SimulationParameters sim_params, SpectralStabilizationConfig spectral_config, const vector<int>& N_list = {8, 16, 32, 64}, int num_threads = 1, bool scale_dt_with_N = false, int dt_reference_N = 32, double dt_reference_value = 0.0005);


int benchmark_2D(int N)
{
    ofstream benchmark_log("../data/benmark.txt", ios::app);
    if (!benchmark_log) {
        return -1;
    }

    benchmark_log << "========================================" << endl;
    benchmark_log << "         2D BENCHMARK WRAPPER          " << endl;
    benchmark_log << "========================================" << endl;
    benchmark_log << "Grid size N = " << N << endl;

    int status_l2 = benchmark_2D_L2_error(N);
    int status_cpu = benchmark_2D_cpu_time(N);

    benchmark_log << "benchmark_2D_L2_error status:  " << status_l2 << endl;
    benchmark_log << "benchmark_2D_cpu_time status:  " << status_cpu << endl;
    benchmark_log << "========================================" << endl;
    benchmark_log << endl;

    return (status_l2 != 0) ? status_l2 : status_cpu;
}

/**
 * @brief Benchmark L2 error between solver_2D and solver_2D_new
 *
 * Output file:
 * - ../data/L2_errors_comparison.txt
 */
int benchmark_2D_L2_error(int N, int num_threads)
{
    ofstream benchmark_log("../data/benmark.txt", ios::app);
    if (!benchmark_log) {
        return -1;
    }

    benchmark_log << "========================================" << endl;
    benchmark_log << "      2D BENCHMARK - L2 ERROR          " << endl;
    benchmark_log << "========================================" << endl;
    benchmark_log << "Grid size N = " << N << endl;
    benchmark_log << endl;

    ModelParameters model_params = {
        .dimension = 2,
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
        .dt = 1.0,
        .final_time = 2, // run for 2 time steps to see the error growth
        .output_directory = "output_0",
        .num_threads = num_threads,
        .total_number_of_times_saves = 3,
        .ode_solver = ODE_solver_type::Euler
    };

    SpectralStabilizationConfig spectral_config_no_dealiasing = {};
    int N_time = static_cast<int>(sim_params.total_number_of_times_saves);

    unique_ptr<Tensor3DComplex> solution_old;
    unique_ptr<Tensor3DComplex> solution_new;
    unique_ptr<Tensor2DReal> t_save_points;

    benchmark_log << "Running solver_2D for reference solution..." << endl;
    {
        Solver_2D solver_old(model_params, sim_params, spectral_config_no_dealiasing);
        solver_old.run_solver();

        // Copy out the result, then destroy solver_old to release its internal buffers.
        solution_old = make_unique<Tensor3DComplex>(solver_old.get_f_solution_time());
        t_save_points = make_unique<Tensor2DReal>(solver_old.get_t_save_points());
    }

    sim_params.output_directory = "output_1";
    benchmark_log << "Running solver_2D_new for comparison..." << endl;
    {
        Solver_2D_new solver_new(model_params, sim_params, spectral_config_no_dealiasing);
        solver_new.run_solver();

        // Copy out the result, then destroy solver_new to release its internal buffers.
        solution_new = make_unique<Tensor3DComplex>(solver_new.get_f_solution_time());
    }

    benchmark_log << "Solver memory released after copying solutions." << endl;

    auto shape_old = solution_old->shape();
    auto shape_new = solution_new->shape();
    if (shape_old[0] != shape_new[0] || shape_old[1] != shape_new[1] || shape_old[2] != shape_new[2]) {
        benchmark_log << "ERROR: Solution dimensions do not match!" << endl;
        benchmark_log << "solver_2D shape: [" << shape_old[0] << ", " << shape_old[1] << ", " << shape_old[2] << "]" << endl;
        benchmark_log << "solver_2D_new shape: [" << shape_new[0] << ", " << shape_new[1] << ", " << shape_new[2] << "]" << endl;
        benchmark_log << endl;
        return -1;
    }

    vector<double> L2_errors(N_time, 0.0);
    vector<double> times(N_time, 0.0);
    for (int t = 0; t < N_time; ++t) {
        TensorNDBase<complex<double>> slice_old = solution_old->slice(0, t, sim_params.num_threads);
        TensorNDBase<complex<double>> slice_new = solution_new->slice(0, t, sim_params.num_threads);

        L2_errors[t] = compute_L2_discrete_error(slice_new, slice_old, sim_params.num_threads);
        times[t] = t_save_points->value_at({1, static_cast<size_t>(t)});
    }

    string benchmark_dir = "../data";
    string l2_error_file = benchmark_dir + "/L2_errors_comparison.txt";
    ofstream out_l2(l2_error_file);
    if (!out_l2) {
        benchmark_log << "ERROR: Cannot open file " << l2_error_file << endl;
        benchmark_log << endl;
        return -1;
    }

    out_l2 << "# L2 norm of error between solver_2D and solver_2D_new at each time step" << endl;
    out_l2 << "# Column 1: Time step index" << endl;
    out_l2 << "# Column 2: Time value" << endl;
    out_l2 << "# Column 3: L2 norm of error" << endl;
    out_l2 << scientific << setprecision(16);
    for (int t = 0; t < N_time; ++t) {
        out_l2 << t << "\t" << times[t] << "\t" << L2_errors[t] << endl;
    }
    out_l2.close();

    double max_L2_error = *max_element(L2_errors.begin(), L2_errors.end());
    double avg_L2_error = 0.0;
    for (double err : L2_errors) {
        avg_L2_error += err;
    }
    avg_L2_error /= N_time;

    benchmark_log << scientific << setprecision(6);
    benchmark_log << "Max L2 error: " << max_L2_error << endl;
    benchmark_log << "Avg L2 error: " << avg_L2_error << endl;
    benchmark_log << "L2 errors written to: " << l2_error_file << endl;
    benchmark_log << "Log file: ../data/benmark.txt" << endl;
    benchmark_log << "========================================" << endl;
    benchmark_log << endl;

    return 0;
}

/**
 * @brief Benchmark CPU time for solver_2D and solver_2D_new
 *
 * Output files:
 * - ../data/cpu_time_solver_2D.txt
 * - ../data/cpu_time_solver_2D_new.txt
 */
int benchmark_2D_cpu_time(int N)
{
    ofstream benchmark_log("../data/benmark.txt", ios::app);
    if (!benchmark_log) {
        return -1;
    }

    benchmark_log << "========================================" << endl;
    benchmark_log << "      2D BENCHMARK - CPU TIME           " << endl;
    benchmark_log << "========================================" << endl;
    benchmark_log << "Grid size N = " << N << endl;
    benchmark_log << endl;

    ModelParameters model_params = {
        .dimension = 2,
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
        .output_directory = "output_0",
        .num_threads = 1,
        .total_number_of_times_saves = 101,
        .ode_solver = ODE_solver_type::Euler
    };

    SpectralStabilizationConfig spectral_config_no_dealiasing = {};
    int N_time = static_cast<int>(sim_params.total_number_of_times_saves);
    bool run_comparison = (N <= 32);

    if (!run_comparison) {
        benchmark_log << "NOTE: N > 32 detected. solver_2D would take too long." << endl;
        benchmark_log << "      Only running solver_2D_new for timing measurement." << endl;
        benchmark_log << endl;
    }

    double elapsed_solver_2D = 0.0;

    if (run_comparison) {
        benchmark_log << "Running solver_2D..." << endl;
        std::clock_t start_solver_2D = std::clock();

        Solver_2D solver_old(model_params, sim_params, spectral_config_no_dealiasing);
        solver_old.run_solver();

        std::clock_t end_solver_2D = std::clock();
        elapsed_solver_2D = static_cast<double>(end_solver_2D - start_solver_2D) / CLOCKS_PER_SEC;
        benchmark_log << "solver_2D completed in " << elapsed_solver_2D << " CPU seconds" << endl;
        benchmark_log << endl;
    }

    sim_params.output_directory = "output_1";

    benchmark_log << "Running solver_2D_new..." << endl;
    std::clock_t start_solver_2D_new = std::clock();

    Solver_2D_new solver_new(model_params, sim_params, spectral_config_no_dealiasing);
    solver_new.run_solver();

    std::clock_t end_solver_2D_new = std::clock();
    double elapsed_solver_2D_new = static_cast<double>(end_solver_2D_new - start_solver_2D_new) / CLOCKS_PER_SEC;
    benchmark_log << "solver_2D_new completed in " << elapsed_solver_2D_new << " CPU seconds" << endl;
    benchmark_log << endl;

    if (!run_comparison) {
        benchmark_log << "========================================" << endl;
        benchmark_log << "         BENCHMARK SUMMARY              " << endl;
        benchmark_log << "========================================" << endl;
        benchmark_log << "Grid size N:                  " << N << endl;
        benchmark_log << "Grid points:                  " << (2*N) << " x " << (2*N) << endl;
        benchmark_log << "Total CPU time solver_2D_new: " << elapsed_solver_2D_new << " s" << endl;
        benchmark_log << "Avg time/step solver_2D_new:  " << elapsed_solver_2D_new / (N_time - 1) << " s" << endl;
        benchmark_log << "========================================" << endl;
        benchmark_log << endl;
        benchmark_log << "NOTE: solver_2D was NOT run (N > 32 too expensive)." << endl;
        benchmark_log << "Log file: ../data/benmark.txt" << endl;
        benchmark_log << endl;
        return 0;
    }

    vector<double> times_old(N_time, 0.0);
    vector<double> times_new(N_time, 0.0);

    double avg_time_per_step_old = elapsed_solver_2D / (N_time - 1);
    double avg_time_per_step_new = elapsed_solver_2D_new / (N_time - 1);

    for (int t = 1; t < N_time; ++t) {
        times_old[t] = avg_time_per_step_old;
        times_new[t] = avg_time_per_step_new;
    }
    times_old[0] = 0.0;
    times_new[0] = 0.0;

    string benchmark_dir = "../data";

    string timing_file_old = benchmark_dir + "/cpu_time_solver_2D.txt";
    ofstream out_old(timing_file_old);
    if (!out_old) {
        benchmark_log << "ERROR: Cannot open file " << timing_file_old << endl;
        benchmark_log << endl;
        return -1;
    }
    
    out_old << "# CPU time per timestep for solver_2D" << endl;
    out_old << "# Column 1: Time step index" << endl;
    out_old << "# Column 2: CPU time (seconds)" << endl;
    out_old << "# Column 3: Cumulative CPU time (seconds)" << endl;
    out_old << scientific << setprecision(16);
    
    double cumulative_old = 0.0;
    for (int t = 0; t < N_time; ++t) {
        cumulative_old += times_old[t];
        out_old << t << "\t" << times_old[t] << "\t" << cumulative_old << endl;
    }
    out_old.close();
    benchmark_log << "solver_2D timing written to: " << timing_file_old << endl;
    
    string timing_file_new = benchmark_dir + "/cpu_time_solver_2D_new.txt";
    ofstream out_new(timing_file_new);
    if (!out_new) {
        benchmark_log << "ERROR: Cannot open file " << timing_file_new << endl;
        benchmark_log << endl;
        return -1;
    }
    
    out_new << "# CPU time per timestep for solver_2D_new" << endl;
    out_new << "# Column 1: Time step index" << endl;
    out_new << "# Column 2: CPU time (seconds)" << endl;
    out_new << "# Column 3: Cumulative CPU time (seconds)" << endl;
    out_new << scientific << setprecision(16);
    
    double cumulative_new = 0.0;
    for (int t = 0; t < N_time; ++t) {
        cumulative_new += times_new[t];
        out_new << t << "\t" << times_new[t] << "\t" << cumulative_new << endl;
    }
    out_new.close();
    benchmark_log << "solver_2D_new timing written to: " << timing_file_new << endl;
    benchmark_log << endl;

    benchmark_log << "========================================" << endl;
    benchmark_log << "         BENCHMARK SUMMARY              " << endl;
    benchmark_log << "========================================" << endl;
    benchmark_log << scientific << setprecision(6);
    benchmark_log << "Total CPU time solver_2D:     " << elapsed_solver_2D << " s" << endl;
    benchmark_log << "Total CPU time solver_2D_new: " << elapsed_solver_2D_new << " s" << endl;
    benchmark_log << "Speedup factor:               " << elapsed_solver_2D / elapsed_solver_2D_new << "x" << endl;
    benchmark_log << endl;
    benchmark_log << "Avg time/step solver_2D:      " << avg_time_per_step_old << " s" << endl;
    benchmark_log << "Avg time/step solver_2D_new:  " << avg_time_per_step_new << " s" << endl;
    benchmark_log << "========================================" << endl;
    benchmark_log << endl;
    benchmark_log << "All results saved to: " << benchmark_dir << "/" << endl;
    benchmark_log << "Log file: ../data/benmark.txt" << endl;
    benchmark_log << endl;
    
    return 0;
}

int benchmark_2D_self_convergence_base(ModelParameters model_params, SimulationParameters sim_params, SpectralStabilizationConfig spectral_config, const vector<int>& N_list , int num_threads, bool scale_dt_with_N, int dt_reference_N, double dt_reference_value)
{
    ofstream benchmark_log("../data/benmark.txt", ios::app);
    if (!benchmark_log) {
        return -1;
    }

    if (N_list.size() < 2) {
        benchmark_log << "ERROR: Need at least 2 grid sizes for self-convergence." << endl;
        benchmark_log << endl;
        return -1;
    }

    benchmark_log << "========================================" << endl;
    benchmark_log << "  2D BENCHMARK - SELF CONVERGENCE      " << endl;
    benchmark_log << "========================================" << endl;
    benchmark_log << "N list: ";
    for (size_t k = 0; k < N_list.size(); ++k) {
        benchmark_log << N_list[k] << (k + 1 < N_list.size() ? ", " : "");
    }
    benchmark_log << endl;

    if (scale_dt_with_N) {
        benchmark_log << "Time-step scaling enabled: dt(N) = dt_ref * N_ref / N" << endl;
        benchmark_log << "Reference values: N_ref = " << dt_reference_N
                      << ", dt_ref = " << dt_reference_value << endl;
    }

    for (size_t k = 0; k + 1 < N_list.size(); ++k) {
        if (N_list[k + 1] != 2 * N_list[k]) {
            benchmark_log << "ERROR: N list must be nested by factor 2. Found "
                          << N_list[k] << " and " << N_list[k + 1] << endl;
            benchmark_log << endl;
            return -1;
        }
    }

    vector<TensorNDBase<complex<double>>> final_solutions;
    final_solutions.reserve(N_list.size());

    for (size_t k = 0; k < N_list.size(); ++k) {
        const int N = N_list[k];
        sim_params.N = N;
        if (scale_dt_with_N) {
            sim_params.dt = dt_reference_value * static_cast<double>(dt_reference_N) / static_cast<double>(N);
        }
        sim_params.output_directory = "output_" + to_string(k);

        benchmark_log << "Running solver_2D_new for N = " << N << "..." << endl;
        if (scale_dt_with_N) {
            benchmark_log << "Using dt = " << sim_params.dt << " for N = " << N << endl;
        }
        Solver_2D_new solver(model_params, sim_params, spectral_config);
        solver.run_solver();

        Tensor2DComplex sol_time_final = solver.get_final_solution();
        final_solutions.push_back(sol_time_final);
    }

    vector<double> E_N(N_list.size() - 1, 0.0);
    vector<double> p_est(N_list.size() - 1, numeric_limits<double>::quiet_NaN());

    for (size_t k = 0; k + 1 < N_list.size(); ++k) {
        const TensorNDBase<complex<double>>& u_N = final_solutions[k];
        const TensorNDBase<complex<double>>& u_2N = final_solutions[k + 1];

        E_N[k] = compute_relative_L2_discrete_error_projection(u_N, u_2N, num_threads);
        //E_N[k] = compute_relative_L2_discrete_error(u_N, u_2N, num_threads);
    }

    for (size_t k = 0; k + 1 < E_N.size(); ++k) {
        if (E_N[k] > 0.0 && E_N[k + 1] > 0.0) {
            p_est[k] = log2(E_N[k] / E_N[k + 1]);
        }
    }

    const string out_file = "../data/self_convergence_2D.txt";
    ofstream out(out_file);
    if (!out) {
        benchmark_log << "ERROR: Cannot open file " << out_file << endl;
        benchmark_log << endl;
        return -1;
    }

    out << "# Self-convergence without exact solution (solver_2D_new)" << endl;
    out << "# Column 1: N" << endl;
    out << "# Column 2: 2N" << endl;
    out << "# Column 3: E_N = ||u_N - Pr(u_2N)||_2 / ||Pr(u_2N)||_2" << endl;
    out << "# Column 4: p_est = log2(E_N / E_2N)" << endl;
    out << scientific << setprecision(16);
    for (size_t k = 0; k + 1 < N_list.size(); ++k) {
        out << N_list[k] << "\t" << N_list[k + 1] << "\t" << E_N[k] << "\t" << p_est[k] << endl;
    }
    out.close();

    benchmark_log << "Self-convergence data written to: " << out_file << endl;
    benchmark_log << "========================================" << endl;
    benchmark_log << endl;

    return 0;
}

int benchmark_2D_self_convergence(const vector<int>& N_list, int num_threads)
{
    ModelParameters model_params = {
        .dimension = 2,
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
        .N = N_list.front(),
        .dt = 1,
        .final_time = 1, // run for 2 time steps to see the error growth
        .output_directory = "output_0",
        .num_threads = num_threads,
        .total_number_of_times_saves = 2,
        .ode_solver = ODE_solver_type::Euler
    };

    SpectralStabilizationConfig spectral_config_no_dealiasing = {};

    return benchmark_2D_self_convergence_base(model_params, sim_params, spectral_config_no_dealiasing, N_list, num_threads);
}

int benchmark_2D_self_convergence_time(const vector<int>& N_list, int num_threads)
{
    const int dt_reference_N = 64;
    const double dt_reference_value = 0.0005;

    ModelParameters model_params = {
        .dimension = 2,
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
        .N = N_list.front(),
        .dt = dt_reference_value,
        .final_time = 0.05, 
        .output_directory = "output_0",
        .num_threads = num_threads,
        .total_number_of_times_saves = 101,
        .ode_solver = ODE_solver_type::Euler
    };

    SpectralStabilizationConfig spectral_config = {
        .early_dealiasing = {
            .enabled = true,
            .use_L2 = true
        },
        .exponential_filter = {
            .enabled = true,
            .strength = 7.5,
            .order = 8.0
        }
    };

    return benchmark_2D_self_convergence_base(
        model_params,
        sim_params,
        spectral_config,
        N_list,
        num_threads,
        false, // scale_dt_with_N
        dt_reference_N,
        dt_reference_value
    );
}