#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <string>
#include <numbers>
using namespace std;

namespace constants
{
    /* =====================================================
     *  Numerical constants
     * ===================================================== */

    inline constexpr double PI = std::numbers::pi; ///< Pi constant

    /**
     * @brief Numerical tolerance used in floating-point comparisons
     */
    inline constexpr double EPSILON = 1.0e-14;

    /**
     * @brief Output precision for floating-point values
     */
    inline constexpr int PRECISION = 15;

    /* =====================================================
     *  I/O paths
     * ===================================================== */

    /**
     * @brief Default input directory path
     */
    inline const string INPUT_PATH = "../data/input/";

    /**
     * @brief Default output directory path
     */
    inline const string OUTPUT_PATH = "../data/";

    /* =====================================================
     *  HDF5 constants
     * ===================================================== */

    /**
     * @brief Default dataset name for HDF5 files
     *
     * Used when each HDF5 file contains exactly one dataset.
     */
    inline const string DEFAULT_DATASET = "data";

    /**
     * @brief Dataset name for shape information in HDF5 files
     */
    inline const string SHAPE_DATASET = "shape";

    /**
     * @brief Dataset name for real and imaginary parts in HDF5 files
     */
    inline const string DATASET_REAL = "data_real";
    /**
     * @brief Dataset name for imaginary parts in HDF5 files
     */
    inline const string DATASET_IMAG = "data_imag";

    inline const string COMPUTATIONAL_DATA_FILENAME = "/computation_data.txt";
}

#endif // CONSTANTS_H
