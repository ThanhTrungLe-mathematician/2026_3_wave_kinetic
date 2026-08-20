#ifndef MODEL_PARAMETERS_H
#define MODEL_PARAMETERS_H

#include <string>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <fstream>

using namespace std;

enum class EquationType
{
    three_wave_kinetic,
    quantum_Boltzmann
};

enum class InitialConditionType
{
    exponential,
    polynomial
};

struct InitialConditionConfig
{
    InitialConditionType initial_condition_type; ///< Type of initial condition
    double amplitude_initial;   ///< Amplitude for initial condition
    double power_initial;       ///< power for initial condition
};

struct PolynomialForm_1
{
    bool enabled = false;
    double alpha = 1;
};
struct PolynomialForm_2
{
    bool enabled = false;
    double alpha = 1;
    double beta = 1;
    double c_1 = 0.5;
    double c_2 = 0.5;
};

struct BogoliubovForm
{
    bool enabled = false;
};

struct DispersionRelation
{
    PolynomialForm_1 polynomial_form_1 = {};
    PolynomialForm_2 polynomial_form_2 = {};
    BogoliubovForm bogoliubov_form = {};
};

struct ModelParameters
{
    int dimension;              ///< Dimension of the problem
    double R;                   ///< Wave-vector domain cutoff
    double M;                   ///< Delta-function truncation parameter
    double rho;                 ///< kernel parameter
    EquationType equation_type; ///< Type of the kinetic equation
    DispersionRelation dispersion_relation; ///< Dispersion relation configuration
    InitialConditionConfig initial_condition; ///< Initial condition configuration
    

    void validate() const
    {
        // Validate basic parameters
        if (dimension <= 0)
        {
            throw invalid_argument("dimension must be positive.");
        }
        if (R <= 0)
        {
            throw invalid_argument("R must be positive.");
        }
        if (M <= 0)
        {
            throw invalid_argument("M must be positive.");
        }
        if (rho < 0)
        {
            throw invalid_argument("rho must be non-negative.");
        }
        
        // Validate equation type
        if (equation_type != EquationType::quantum_Boltzmann &&
            equation_type != EquationType::three_wave_kinetic)
        {
            throw invalid_argument("Invalid equation type.");
        }
        
        // Validate dispersion relation - exactly one form must be enabled
        int enabled_count = 0;
        if (dispersion_relation.polynomial_form_1.enabled) enabled_count++;
        if (dispersion_relation.polynomial_form_2.enabled) enabled_count++;
        if (dispersion_relation.bogoliubov_form.enabled) enabled_count++;
        
        if (enabled_count == 0)
        {
            throw invalid_argument("At least one dispersion relation form must be enabled.");
        }
        if (enabled_count > 1)
        {
            throw invalid_argument("Only one dispersion relation form can be enabled at a time.");
        }
        
        // Validate polynomial_form_1 parameters
        if (dispersion_relation.polynomial_form_1.enabled)
        {
            if (dispersion_relation.polynomial_form_1.alpha < 1)
            {
                throw invalid_argument("polynomial_form_1: alpha must be greater than or equal to 1.");
            }
        }
        
        // Validate polynomial_form_2 parameters
        if (dispersion_relation.polynomial_form_2.enabled)
        {
            if (dispersion_relation.polynomial_form_2.alpha < 1)
            {
                throw invalid_argument("polynomial_form_2: alpha must be greater than or equal to 1.");
            }
            if (dispersion_relation.polynomial_form_2.beta < 1)
            {
                throw invalid_argument("polynomial_form_2: beta must be greater than or equal to 1.");
            }
            if (dispersion_relation.polynomial_form_2.c_1 <= 0 )
            {
                throw invalid_argument("polynomial_form_2: c_1 must be positive.");
            }
            if (dispersion_relation.polynomial_form_2.c_2 <= 0 )
            {
                throw invalid_argument("polynomial_form_2: c_2 must be positive.");
            }
        }
        
        // Validate initial condition parameters
        if (initial_condition.initial_condition_type != InitialConditionType::exponential &&
            initial_condition.initial_condition_type != InitialConditionType::polynomial)
        {
            throw invalid_argument("Invalid initial condition type.");
        }
        if (initial_condition.amplitude_initial < 0)
        {
            throw invalid_argument("amplitude_initial must be non-negative.");
        }
        if (initial_condition.power_initial < 0)
        {
            throw invalid_argument("power_initial must be non-negative.");
        }
    }

    void write_model_parameters(const string &filename) const
    {
        std::ofstream outfile(filename, std::ios::out | std::ios::app);
        if (!outfile.is_open())
        {
            throw std::runtime_error("Failed to open " + filename + " for writing.");
        }

        outfile.precision(constants::PRECISION);

        // Header
        outfile << "#####################################################################" << std::endl;
        outfile << "# Physical Model Parameters" << std::endl;
        outfile << "#####################################################################" << std::endl;
        outfile << std::endl;

        // Basic parameters
        outfile << "Dimension <dimension>: " << dimension << std::endl;
        outfile << "Wave-vector domain cutoff <R>: " << R << std::endl;
        outfile << "Delta-function truncation parameter <M>: " << M << std::endl;
        outfile << "Kernel parameter <rho>: " << rho << std::endl;
        outfile << std::endl;

        // Equation type
        outfile << "Equation type <equation_type>: ";
        if (equation_type == EquationType::quantum_Boltzmann)
        {
            outfile << "quantum_Boltzmann (C = 1)" << std::endl;
        }
        else if (equation_type == EquationType::three_wave_kinetic)
        {
            outfile << "three_wave_kinetic (C = 0)" << std::endl;
        }
        outfile << std::endl;

        // Dispersion relation
        outfile << "Dispersion Relation Configuration:" << std::endl;
        if (dispersion_relation.polynomial_form_1.enabled)
        {
            outfile << "  Form: polynomial_form_1" << std::endl;
            outfile << "  alpha: " << dispersion_relation.polynomial_form_1.alpha << std::endl;
        }
        else if (dispersion_relation.polynomial_form_2.enabled)
        {
            outfile << "  Form: polynomial_form_2" << std::endl;
            outfile << "  alpha: " << dispersion_relation.polynomial_form_2.alpha << std::endl;
            outfile << "  beta: " << dispersion_relation.polynomial_form_2.beta << std::endl;
            outfile << "  c_1: " << dispersion_relation.polynomial_form_2.c_1 << std::endl;
            outfile << "  c_2: " << dispersion_relation.polynomial_form_2.c_2 << std::endl;
        }
        else if (dispersion_relation.bogoliubov_form.enabled)
        {
            outfile << "  Form: bogoliubov_form" << std::endl;
        }
        outfile << std::endl;

        // Initial condition
        outfile << "Initial Condition Configuration:" << std::endl;
        outfile << "  Type <initial_condition_type>: ";
        if (initial_condition.initial_condition_type == InitialConditionType::exponential)
        {
            outfile << "exponential (0)" << std::endl;
        }
        else if (initial_condition.initial_condition_type == InitialConditionType::polynomial)
        {
            outfile << "polynomial (1)" << std::endl;
        }
        outfile << "  Amplitude <amplitude_initial>: " << initial_condition.amplitude_initial << std::endl;
        outfile << "  Power <power_initial>: " << initial_condition.power_initial << std::endl;
        outfile << std::endl;

        outfile.close();
    }
};

#endif // MODEL_PARAMETERS_H