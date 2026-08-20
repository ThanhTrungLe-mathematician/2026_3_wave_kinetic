#include "tensorNDBase.h"
#include <complex>

class TensorNDComplex : public TensorNDBase<std::complex<double>>
{
public:
    TensorNDComplex() = default;
    
    using TensorNDBase<std::complex<double>>::TensorNDBase;

    complex<double> get_max_by_modulus() const;
    complex<double> get_min_by_modulus() const;

    double get_max_real_part() const;
    double get_min_real_part() const;

    double get_max_imag_part() const;
    double get_min_imag_part() const;

    /**
     * @brief Write tensor to text file with shape information
     * @param filename Output file name
     */
    void WriteToText(const std::string &filename) const;

    /**
     * @brief Read tensor from text file with shape information
     * @param filename Input file name
     */
    void ReadFromText(const std::string &filename);

    /**
     * @brief Write tensor to HDF5 file (full control)
     * @param filename HDF5 file name
     * @param shape_dataset Dataset name for shape information
     * @param dataset_real Dataset name for real parts
     * @param dataset_imag Dataset name for imaginary parts
     */
    void WriteToHDF5(const std::string &filename, 
                     const std::string &shape_dataset,
                     const std::string &dataset_real,
                     const std::string &dataset_imag) const;

    /**
     * @brief Read tensor from HDF5 file (full control)
     * @param filename HDF5 file name
     * @param shape_dataset Dataset name for shape information
     * @param dataset_real Dataset name for real parts
     * @param dataset_imag Dataset name for imaginary parts
     */
    void ReadFromHDF5(const std::string &filename,
                      const std::string &shape_dataset,
                      const std::string &dataset_real,
                      const std::string &dataset_imag);

    /**
     * @brief Write tensor to HDF5 file with default dataset names
     * @param filename HDF5 file name
     */
    void WriteToHDF5(const std::string &filename) const;

    /**
     * @brief Read tensor from HDF5 file with default dataset names
     * @param filename HDF5 file name
     */
    void ReadFromHDF5(const std::string &filename);
};
