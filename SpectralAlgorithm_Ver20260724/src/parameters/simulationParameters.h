#ifndef SIMULATION_PARAMETERS_H
#define SIMULATION_PARAMETERS_H

#include <string>
#include <stdexcept>
using namespace std;

enum class ODE_solver_type
{
    Euler,
    RK2,
    RK3,
    RK4,
    SSP_RK3
};

struct SimulationParameters
{
    int N;                                 ///< Fourier series truncation parameter
    double dt;                             ///< Time step size
    double final_time;                     ///< Final simulation time
    string output_directory;               ///< Directory to save output files
    int num_threads;                       ///< Number of threads for parallel computation
    int total_number_of_times_saves = 101; ///< Total number of time snapshots to save during the simulation
    ODE_solver_type ode_solver = ODE_solver_type::Euler;   ///< Type of ODE solver to use


    double max_memory_GB = 30;             ///< Maximum allowed memory usage in GB, default is 30 GB for my PC
    bool save_hdf5_files = true;           ///< Whether to save output files in HDF5 format
    bool save_txt_files = false;           ///< Whether to save output files in TXT format

    

    void validate()
    {
        if (dt <= 0)
        {
            throw invalid_argument("Time step size dt must be positive.");
        }
        if (final_time <= 0)
        {
            throw invalid_argument("Final simulation time must be positive.");
        }
        if (output_directory.empty())
        {
            throw invalid_argument("Output directory cannot be empty.");
        }
        if (num_threads < 1)
        {
            throw invalid_argument("Number of threads must be greater than or equal to 1.");
        }
        if (N <= 0)
        {
            throw invalid_argument("Fourier series truncation parameter N must be positive.");
        }
        if (max_memory_GB <= 0)
        {
            throw invalid_argument("Maximum memory usage must be positive.");
        }
        if (max_memory_GB > 60)
        {
            throw invalid_argument("Maximum memory usage must be less than 60 GB.");
        }

        if (total_number_of_times_saves < 101 || total_number_of_times_saves > int(final_time / dt) + 1)
        {
            total_number_of_times_saves = min(int(final_time / dt) + 1, 1001);
        }
    }

    void write_simulation_parameters(const string &filename) const
    {
        std::ofstream outfile(filename, std::ios::out | std::ios::app);
        if (!outfile.is_open())
        {
            throw std::runtime_error("Failed to open " + filename + " for writing.");
        }

        outfile.precision(constants::PRECISION);

        outfile << endl;
        outfile << "\n#####################################################################" << endl;
        outfile << "# Simulation parameters" << endl;
        outfile << "#####################################################################" << endl;

        outfile.precision(constants::PRECISION);

        outfile << "\nThe Fourier series truncation parameter N: " << N << endl;
        outfile << "\nThe time step size dt: " << dt << endl;
        outfile << "\nThe final simulation time final_time: " << final_time << endl;
        outfile << "\nThe output directory: " << output_directory << endl;
        outfile << "\nThe number of threads for parallel computation num_threads: " << num_threads << endl;
        outfile << "\nThe maximum allowed memory usage max_memory_GB: " << max_memory_GB << " GB" << endl;
        outfile << "\nThe total number of time snapshots to save total_number_of_times_saves: " << total_number_of_times_saves << endl;
        outfile << "\nThe ODE solver type ode_solver: ";
        switch (ode_solver)
        {
            case ODE_solver_type::Euler:
                outfile << "Euler" << endl;
                break;
            case ODE_solver_type::RK2:
                outfile << "Runge-Kutta 2 (RK2)" << endl;
                break;
            case ODE_solver_type::RK3:
                outfile << "Runge-Kutta 3 (RK3)" << endl;
                break;
            case ODE_solver_type::SSP_RK3:
                outfile << "Strong Stability Preserving Runge-Kutta 3 (SSP_RK3)" << endl;
                break;
            case ODE_solver_type::RK4:
                outfile << "Runge-Kutta 4 (RK4)" << endl;
                break;
        }

        outfile.close();
    }
};

#endif // SIMULATION_PARAMETERS_H