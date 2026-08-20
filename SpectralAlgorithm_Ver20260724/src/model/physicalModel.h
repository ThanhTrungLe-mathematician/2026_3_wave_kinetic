#ifndef PHYSICAL_MODEL_H
#define PHYSICAL_MODEL_H

#include "tensorWrappers.h"
#include "modelParameters.h"
using namespace std;

class PhysicalModel
{
public:
    // double R_;                 ///< Wave-vector domain cutoff
    // double M_;              ///< Delta-function truncation parameter    
    //                  ///< quantum Boltzmann equation C = 1 or the 3-wave kinetic equation C = 0
    // int choice_omega_;       ///< Choice of dispersion relation
    // double alpha_;            ///< Dispersion relation parameter
    // double beta_;             ///< Dispersion relation parameter
    // double rho_;              ///< Kernel parameter
    // double power_initial_;   ///< Power of initial data as form e^{-|k|^power}
    // int dimension_;        ///< Dimension of the problem

    ModelParameters model_params_;
    int C_;

public:
    PhysicalModel(ModelParameters model_parameters);
    ~PhysicalModel() = default;

    /**
     * @brief Write Physical Model parameters to output file
     */
    void write_model_parameters(const string &filename);

    /**
     * @brief Initial condition function
     *
     * @param k Wave-vector coordinate
     * @return Initial condition value at k
     */
    double func_initial_condition(const vector<double> &k);

    /**
     * @brief Dispersion relation function
     *
     * @param k Wave-vector coordinate
     * @return Dispersion relation value at k
     */
    double func_dispersion_relation(const vector<double> &k);

    /**
     * @brief Kernel function
     *
     * @param k Wave-vector coordinate include k1, k2, k3
     * @return Kernel value at k1, k2, and k3
     */
    double func_kernel(const vector<double> &k);

    /**
     * @brief W function
     *
     * @param k Wave-vector coordinate include k2, k3
     * @return W value at k2 and k3
     */
    double func_W(const vector<double> &k);
};

#endif // PHYSICAL_MODEL_H