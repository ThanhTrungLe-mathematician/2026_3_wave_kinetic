#include "tensorNDBase.h"


class TensorNDReal : public TensorNDBase<double>
{
public:
    TensorNDReal() = default;
    
    using TensorNDBase<double>::TensorNDBase;

    double get_max_value() const;

    double get_min_value() const;

    /**
     * @brief Write tensor to text file with shape information
     * @param filename Output file name
     */
    void WriteToText(const string &filename) const;

    /**
     * @brief Read tensor from text file with shape information
     * @param filename Input file name
     */
    void ReadFromText(const string &filename);

    /**
     * @brief Write tensor to HDF5 file
     * @param filename HDF5 file name
     * @param shape_dataset Dataset name for shape information
     * @param dataset Dataset name for data
     */
    void WriteToHDF5(const string &filename, const string &shape_dataset, const string &dataset) const;

    /**
     * @brief Read tensor from HDF5 file
     * @param filename HDF5 file name
     * @param shape_dataset Dataset name for shape information
     * @param dataset Dataset name for data
     */
    void ReadFromHDF5(const string &filename, const string &shape_dataset, const string &dataset);

    /**
     * @brief Write tensor to HDF5 file with default dataset names
     * @param filename HDF5 file name
     */
    void WriteToHDF5(const string &filename) const;

    /**
     * @brief Read tensor from HDF5 file with default dataset names
     * @param filename HDF5 file name
     */
    void ReadFromHDF5(const string &filename);
};