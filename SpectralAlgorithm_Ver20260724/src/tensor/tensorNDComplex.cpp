#include "tensorNDComplex.h"
#include <fstream>
#include <sstream>
#include <H5Cpp.h>
#include <limits>

using namespace H5;

double TensorNDComplex::get_max_real_part() const
{
    if (this->size() == 0)
        return 0.0;
    
    double max_real = -std::numeric_limits<double>::infinity();
    const std::complex<double>* data_ptr = this->data();
    
    for (size_t i = 0; i < this->size(); ++i)
    {
        double real_part = data_ptr[i].real();
        if (real_part > max_real)
            max_real = real_part;
    }
    
    return max_real;
}

double TensorNDComplex::get_min_real_part() const
{
    if (this->size() == 0)
        return 0.0;
    
    double min_real = std::numeric_limits<double>::infinity();
    const std::complex<double>* data_ptr = this->data();
    
    for (size_t i = 0; i < this->size(); ++i)
    {
        double real_part = data_ptr[i].real();
        if (real_part < min_real)
            min_real = real_part;
    }
    
    return min_real;
}

double TensorNDComplex::get_max_imag_part() const
{
    if (this->size() == 0)
        return 0.0;
    
    double max_imag = -std::numeric_limits<double>::infinity();
    const std::complex<double>* data_ptr = this->data();
    
    for (size_t i = 0; i < this->size(); ++i)
    {
        double imag_part = data_ptr[i].imag();
        if (imag_part > max_imag)
            max_imag = imag_part;
    }
    
    return max_imag;
}

double TensorNDComplex::get_min_imag_part() const
{
    if (this->size() == 0)
        return 0.0;
    
    double min_imag = std::numeric_limits<double>::infinity();
    const std::complex<double>* data_ptr = this->data();
    
    for (size_t i = 0; i < this->size(); ++i)
    {
        double imag_part = data_ptr[i].imag();
        if (imag_part < min_imag)
            min_imag = imag_part;
    }
    
    return min_imag;
}

void TensorNDComplex::WriteToText(const std::string &filename) const
{
    std::ofstream ofs(filename);
    if (!ofs)
        throw std::runtime_error("Cannot open file for writing: " + filename);

    // Write shape on first line
    const auto &shape = this->shape();
    for (size_t dim : shape)
        ofs << dim << " ";
    ofs << "\n";

    // Write all real parts on second line
    for (size_t i = 0; i < this->size(); ++i)
    {
        if (i) ofs << " ";
        ofs << this->get(i).real();
    }
    ofs << "\n";

    // Write all imag parts on third line
    for (size_t i = 0; i < this->size(); ++i)
    {
        if (i) ofs << " ";
        ofs << this->get(i).imag();
    }
    ofs << "\n";

    ofs.close();
}

void TensorNDComplex::ReadFromText(const std::string &filename)
{
    std::ifstream ifs(filename);
    if (!ifs)
        throw std::runtime_error("Cannot open file for reading: " + filename);

    // Read shape (first line)
    std::string line;
    std::getline(ifs, line);
    std::istringstream iss_shape(line);
    std::vector<size_t> shape;
    size_t dim;
    while (iss_shape >> dim) shape.push_back(dim);

    // Read real parts (second line)
    std::getline(ifs, line);
    std::istringstream iss_re(line);
    std::vector<double> re;
    double v;
    while (iss_re >> v) re.push_back(v);

    // Read imag parts (third line)
    std::getline(ifs, line);
    std::istringstream iss_im(line);
    std::vector<double> im;
    while (iss_im >> v) im.push_back(v);

    // Resize and validate
    this->resize(shape);
    if (re.size() != this->size() || im.size() != this->size())
        throw std::runtime_error("Data count mismatch when reading complex text file");

    // Assign values
    for (size_t i = 0; i < this->size(); ++i)
        this->set(i, std::complex<double>(re[i], im[i]));

    ifs.close();
}

void TensorNDComplex::WriteToHDF5(const std::string &filename,
                                   const std::string &shape_dataset,
                                   const std::string &dataset_real,
                                   const std::string &dataset_imag) const
{
    H5File file(filename, H5F_ACC_TRUNC);

    // Write shape as dataset
    const auto &shape = this->shape();
    hsize_t shape_dims[1] = {shape.size()};
    DataSpace shape_space(1, shape_dims);
    DataSet shape_dset = file.createDataSet(
        shape_dataset, PredType::NATIVE_ULLONG, shape_space);
    shape_dset.write(shape.data(), PredType::NATIVE_ULLONG);

    // Extract real and imaginary parts
    std::vector<double> real_parts(this->size());
    std::vector<double> imag_parts(this->size());
    for (size_t i = 0; i < this->size(); ++i)
    {
        std::complex<double> val = this->get(i);
        real_parts[i] = val.real();
        imag_parts[i] = val.imag();
    }

    // Write real parts
    hsize_t data_dims[1] = {this->size()};
    DataSpace data_space(1, data_dims);
    DataSet real_dset = file.createDataSet(
        dataset_real, PredType::NATIVE_DOUBLE, data_space);
    real_dset.write(real_parts.data(), PredType::NATIVE_DOUBLE);

    // Write imaginary parts
    DataSet imag_dset = file.createDataSet(
        dataset_imag, PredType::NATIVE_DOUBLE, data_space);
    imag_dset.write(imag_parts.data(), PredType::NATIVE_DOUBLE);

    file.close();
}

void TensorNDComplex::ReadFromHDF5(const std::string &filename,
                                    const std::string &shape_dataset,
                                    const std::string &dataset_real,
                                   const std::string &dataset_imag)
{
    H5File file(filename, H5F_ACC_RDONLY);

    // Read shape dataset first
    DataSet shape_dset = file.openDataSet(shape_dataset);
    DataSpace shape_space = shape_dset.getSpace();

    hsize_t shape_dims[1];
    shape_space.getSimpleExtentDims(shape_dims);

    std::vector<size_t> shape(shape_dims[0]);
    shape_dset.read(shape.data(), PredType::NATIVE_ULLONG);

    // Resize tensor with shape
    this->resize(shape);

    // Read real parts
    DataSet real_dset = file.openDataSet(dataset_real);
    std::vector<double> real_parts(this->size());
    real_dset.read(real_parts.data(), PredType::NATIVE_DOUBLE);

    // Read imaginary parts
    DataSet imag_dset = file.openDataSet(dataset_imag);
    std::vector<double> imag_parts(this->size());
    imag_dset.read(imag_parts.data(), PredType::NATIVE_DOUBLE);

    // Combine into complex values
    for (size_t i = 0; i < this->size(); ++i)
        this->set(i, std::complex<double>(real_parts[i], imag_parts[i]));

    file.close();
}

void TensorNDComplex::WriteToHDF5(const std::string &filename) const
{
    WriteToHDF5(filename, constants::SHAPE_DATASET, constants::DATASET_REAL, constants::DATASET_IMAG);
}

void TensorNDComplex::ReadFromHDF5(const std::string &filename)
{
    ReadFromHDF5(filename, constants::SHAPE_DATASET, constants::DATASET_REAL, constants::DATASET_IMAG);
}

std::complex<double> TensorNDComplex::get_max_by_modulus() const
{
    const std::complex<double> *data = this->data();
    const size_t n = this->size();

    if (n == 0)
        throw std::runtime_error("Cannot get max of empty tensor");

    std::complex<double> max_val = data[0];
    double max_modulus = std::abs(data[0]);

    for (size_t i = 1; i < n; ++i)
    {
        double modulus = std::abs(data[i]);
        if (modulus > max_modulus)
        {
            max_modulus = modulus;
            max_val = data[i];
        }
    }

    return max_val;
}

std::complex<double> TensorNDComplex::get_min_by_modulus() const
{
    const std::complex<double> *data = this->data();
    const size_t n = this->size();

    if (n == 0)
        throw std::runtime_error("Cannot get min of empty tensor");

    std::complex<double> min_val = data[0];
    double min_modulus = std::abs(data[0]);

    for (size_t i = 1; i < n; ++i)
    {
        double modulus = std::abs(data[i]);
        if (modulus < min_modulus)
        {
            min_modulus = modulus;
            min_val = data[i];
        }
    }

    return min_val;
}
