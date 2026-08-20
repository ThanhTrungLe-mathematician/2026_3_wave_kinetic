#include "tensorNDReal.h"
#include <fstream>
#include <sstream>
#include <H5Cpp.h>

using namespace H5;

void TensorNDReal::WriteToText(const string &filename) const
{
    ofstream ofs(filename);
    if (!ofs)
        throw runtime_error("Cannot open file for writing: " + filename);

    // Write shape on first line
    const auto &shape = this->shape();
    for (size_t dim : shape)
        ofs << dim << " ";
    ofs << "\n";

    // Write all data on second line (space-separated)
    for (size_t i = 0; i < this->size(); ++i)
    {
        if (i > 0) ofs << " ";
        ofs << this->get(i);
    }
    ofs << "\n";

    ofs.close();
}

void TensorNDReal::ReadFromText(const string &filename)
{
    ifstream ifs(filename);
    if (!ifs)
        throw runtime_error("Cannot open file for reading: " + filename);

    // Read shape from first line
    vector<size_t> shape;
    size_t dim;
    string line;
    getline(ifs, line);
    istringstream iss(line);
    while (iss >> dim)
        shape.push_back(dim);

    this->resize(shape);

    // Read all data from second line
    getline(ifs, line);
    istringstream iss2(line);
    for (size_t i = 0; i < this->size(); ++i)
    {
        double value;
        iss2 >> value;
        this->set(i, value);
    }

    ifs.close();
}

void TensorNDReal::WriteToHDF5(const string &filename, const string &shape_dataset, const string &dataset) const
{
    H5File file(filename, H5F_ACC_TRUNC);

    // Write shape as dataset
    const auto &shape = this->shape();
    hsize_t shape_dims[1] = {shape.size()};
    DataSpace shape_space(1, shape_dims);
    DataSet shape_dset = file.createDataSet(
        shape_dataset, PredType::NATIVE_ULLONG, shape_space);
    shape_dset.write(shape.data(), PredType::NATIVE_ULLONG);

    // Write data as dataset
    hsize_t data_dims[1] = {this->size()};
    DataSpace data_space(1, data_dims);
    DataSet data_dset = file.createDataSet(
        dataset, PredType::NATIVE_DOUBLE, data_space);
    data_dset.write(this->data(), PredType::NATIVE_DOUBLE);

    file.close();
}

void TensorNDReal::WriteToHDF5(const string &filename) const
{
    WriteToHDF5(filename, constants::SHAPE_DATASET, constants::DEFAULT_DATASET);
}

void TensorNDReal::ReadFromHDF5(const string &filename, const string &shape_dataset, const string &dataset)
{
    H5File file(filename, H5F_ACC_RDONLY);

    // Read shape dataset first
    DataSet shape_dset = file.openDataSet(shape_dataset);
    DataSpace shape_space = shape_dset.getSpace();

    hsize_t shape_dims[1];
    shape_space.getSimpleExtentDims(shape_dims);

    vector<size_t> shape(shape_dims[0]);
    shape_dset.read(shape.data(), PredType::NATIVE_ULLONG);

    // Resize tensor with shape
    this->resize(shape);

    // Read data dataset
    DataSet data_dset = file.openDataSet(dataset);
    data_dset.read(this->data(), PredType::NATIVE_DOUBLE);

    file.close();
}

void TensorNDReal::ReadFromHDF5(const string &filename)
{
    ReadFromHDF5(filename, constants::SHAPE_DATASET, constants::DEFAULT_DATASET);
}

double TensorNDReal::get_max_value() const
{
    const double *data = this->data();
    const size_t n = this->size();

    if (n == 0)
        throw runtime_error("Cannot get max value of empty tensor");

    double max_val = data[0];
    for (size_t i = 1; i < n; ++i)
    {
        if (data[i] > max_val)
            max_val = data[i];
    }

    return max_val;
}

double TensorNDReal::get_min_value() const
{
    const double *data = this->data();
    const size_t n = this->size();

    if (n == 0)
        throw runtime_error("Cannot get min value of empty tensor");

    double min_val = data[0];
    for (size_t i = 1; i < n; ++i)
    {
        if (data[i] < min_val)
            min_val = data[i];
    }

    return min_val;
}