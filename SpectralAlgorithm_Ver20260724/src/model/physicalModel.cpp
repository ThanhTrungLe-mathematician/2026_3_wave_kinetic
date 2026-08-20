#include "physicalModel.h"
#include <fstream>
#include <stdexcept>
#include <cmath>

PhysicalModel::PhysicalModel(ModelParameters model_parameters)
{
    model_parameters.validate();
    // Initialize physical model parameters from ModelParameters struct
    // this->R_ = model_parameters.R;
    // this->M_ = model_parameters.M;
    
    // this->choice_omega_ = model_parameters.choice_omega;
    // this->alpha_ = model_parameters.alpha;
    // this->beta_ = model_parameters.beta;
    // this->rho_ = model_parameters.rho;
    // this->power_initial_ = model_parameters.power_initial;
    // this->dimension_ = model_parameters.dimension;

    this->model_params_ = model_parameters;
    this->C_ = (this->model_params_.equation_type == EquationType::quantum_Boltzmann) ? 1 : 0;
}

void PhysicalModel::write_model_parameters(const string &filename)
{
    model_params_.write_model_parameters(filename);
}

double PhysicalModel::func_initial_condition(const vector<double> &k)
{
    if (k.size() < static_cast<size_t>(this->model_params_.dimension))
        throw std::invalid_argument("Input k size is smaller than dimension");

    double sum_sq = 0.0;
    for (int i = 0; i < this->model_params_.dimension; ++i)
    {
        const double v = k[static_cast<size_t>(i)];
        sum_sq += v * v;
    }

    const double k_magnitude = std::sqrt(sum_sq);
    
    const double amplitude = this->model_params_.initial_condition.amplitude_initial;
    const double power = this->model_params_.initial_condition.power_initial;

    if (this->model_params_.initial_condition.initial_condition_type == InitialConditionType::exponential)
    {
        return amplitude * exp(-std::pow(k_magnitude, power));
    }
    else if (this->model_params_.initial_condition.initial_condition_type == InitialConditionType::polynomial)
    {
        return amplitude / (1.0 + std::pow(k_magnitude, power));
    }
    else
    {
        throw std::invalid_argument("Unsupported initial condition type");
    }
    
}

double PhysicalModel::func_dispersion_relation(const vector<double> &k)
{
    if (k.size() < static_cast<size_t>(this->model_params_.dimension))
        throw std::invalid_argument("Input k size is smaller than dimension");

    double sum_sq = 0.0;
    for (int i = 0; i < this->model_params_.dimension; ++i)
    {
        const double v = k[static_cast<size_t>(i)];
        sum_sq += v * v;
    }

    const double k_magnitude = std::sqrt(sum_sq);

    if(this->model_params_.dispersion_relation.polynomial_form_1.enabled)
    {
        const double alpha = this->model_params_.dispersion_relation.polynomial_form_1.alpha;
        return std::pow(k_magnitude, alpha);
    }
    else if(this->model_params_.dispersion_relation.polynomial_form_2.enabled)
    {
        const double alpha = this->model_params_.dispersion_relation.polynomial_form_2.alpha;
        const double beta = this->model_params_.dispersion_relation.polynomial_form_2.beta;
        const double c_1 = this->model_params_.dispersion_relation.polynomial_form_2.c_1;
        const double c_2 = this->model_params_.dispersion_relation.polynomial_form_2.c_2;
        return c_1 * std::pow(k_magnitude, alpha) + c_2 * std::pow(k_magnitude, beta);
    }
    else if(this->model_params_.dispersion_relation.bogoliubov_form.enabled)
    {
        // Bogoliubov form: omega(k) = sqrt(k^2 + k^4)
        return std::sqrt(k_magnitude * k_magnitude + k_magnitude * k_magnitude * k_magnitude * k_magnitude);
    }
    else
    {
        throw std::invalid_argument("No dispersion relation form is enabled");
    }
}

double PhysicalModel::func_kernel(const vector<double> &k)
{
    int dimension = this->model_params_.dimension;

    const std::size_t expected = static_cast<std::size_t>(3 * dimension);
    if (k.size() < expected)
        throw std::invalid_argument("Input k size is smaller than 3 * dimension");

    // Manually build k1, k2, k3 without helper lambdas for clarity
    vector<double> k1;
    vector<double> k2;
    vector<double> k3;
    k1.reserve(static_cast<std::size_t>(dimension));
    k2.reserve(static_cast<std::size_t>(dimension));
    k3.reserve(static_cast<std::size_t>(dimension));

    for (int i = 0; i < dimension; ++i)
    {
        k1.push_back(k[static_cast<std::size_t>(i)]);
        k2.push_back(k[static_cast<std::size_t>(dimension + i)]);
        k3.push_back(k[static_cast<std::size_t>(2 * dimension + i)]);
    }

    const double k1_dispersion = this->func_dispersion_relation(k1);
    const double k2_dispersion = this->func_dispersion_relation(k2);
    const double k3_dispersion = this->func_dispersion_relation(k3);

    return std::pow(k1_dispersion * k2_dispersion * k3_dispersion, this->model_params_.rho);
}

double PhysicalModel::func_W(const vector<double> &k)
{
    int dimension_ = this->model_params_.dimension;

    const std::size_t expected = static_cast<std::size_t>(2 * dimension_);
    if (k.size() < expected)
        throw std::invalid_argument("Input k size is smaller than 2 * dimension");

    vector<double> k2;
    vector<double> k3;
    k2.reserve(static_cast<std::size_t>(dimension_));
    k3.reserve(static_cast<std::size_t>(dimension_));

    for (int i = 0; i < dimension_; ++i)
    {
        k2.push_back(k[static_cast<std::size_t>(i)]);
        k3.push_back(k[static_cast<std::size_t>(dimension_ + i)]);
    }

    // k23 = k2 + k3 (component-wise)
    vector<double> k23;
    k23.reserve(static_cast<std::size_t>(dimension_));
    for (int i = 0; i < dimension_; ++i)
    {
        k23.push_back(k2[static_cast<std::size_t>(i)] + k3[static_cast<std::size_t>(i)]);
    }

    // Build input for func_kernel: [k23, k2, k3]
    vector<double> k_kernel;
    k_kernel.reserve(static_cast<std::size_t>(3 * dimension_));
    k_kernel.insert(k_kernel.end(), k23.begin(), k23.end());
    k_kernel.insert(k_kernel.end(), k2.begin(), k2.end());
    k_kernel.insert(k_kernel.end(), k3.begin(), k3.end());

    const double kernel_value = this->func_kernel(k_kernel);

    const double denominator = this->func_dispersion_relation(k23)
                              - this->func_dispersion_relation(k2)
                              - this->func_dispersion_relation(k3);

    if (std::abs(denominator) < 1e-8)
    {
        return (1.0 / constants::PI) * this->model_params_.M * kernel_value;
    }
    else
    {
        return (1.0 / constants::PI) * std::sin(this->model_params_.M * denominator) / denominator * kernel_value;
    }
}