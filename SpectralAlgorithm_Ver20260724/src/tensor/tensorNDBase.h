#ifndef TENSOR_ND_BASE_H
#define TENSOR_ND_BASE_H

#include <iostream>
#include <vector>
#include <complex>
#include <stdexcept>
#include <cmath>
#include <cstddef> // for size_t
#include <algorithm>

#include "constants.h"

using namespace std;

/**
 * @brief N-dimensional tensor with contiguous memory storage
 *
 * This class is designed for high-performance scientific computing,
 * FFT-based algorithms, and large-scale numerical simulations.
 *
 * @tparam T Data type (double, complex<double>, ...)
 */
template <typename T>
class TensorNDBase
{
protected:
    /* ============================
     *  Internal data
     * ============================ */

    vector<size_t> shape_;  ///< Tensor dimensions
    vector<size_t> stride_; ///< Stride for flattened indexing
    vector<T> data_;        ///< Contiguous data storage
    size_t size_;           ///< Total number of elements

    /* ============================
     *  Internal helpers
     * ============================ */

    /**
     * @brief Compute stride array (row-major order)
     */
    void compute_stride()
    {
        stride_.resize(shape_.size());
        if (shape_.empty())
            return;

        stride_.back() = 1;
        for (int i = (int)shape_.size() - 2; i >= 0; --i)
            stride_[i] = stride_[i + 1] * shape_[i + 1];
    }

    /**
     * @brief Convert multi-index to flattened index
     * @param index Vector of indices
     * @return Flattened index
     */
    inline size_t flatten_index(const vector<size_t> &index) const
    {
        if (index.size() != shape_.size())
            throw invalid_argument("Dimension mismatch in flatten_index");

        size_t idx = 0;
        for (size_t i = 0; i < index.size(); ++i)
        {
            if (index[i] >= shape_[i])
                throw out_of_range("Index out of bounds");
            idx += index[i] * stride_[i];
        }
        return idx;
    }

    /**
     * @brief Convert multi-index to flattened index (no checks)
     * @param index Vector of indices
     * @return Flattened index
     */
    inline size_t flatten_index_unchecked(const vector<size_t> &index) const
    {
        size_t idx = 0;
        for (size_t i = 0; i < index.size(); ++i)
        {
            idx += index[i] * stride_[i];
        }
        return idx;
    }

public:
    /* ============================
     *  Constructors / Destructor
     * ============================ */

    /**
     * @brief Default constructor
     */
    TensorNDBase() : size_(0) {};

    /**
     * @brief Construct tensor with given shape
     * @param shape Vector of dimensions
     */
    explicit TensorNDBase(const vector<size_t> &shape)
    {
        this->resize(shape);
    }

    /**
     * @brief Destructor
     */
    ~TensorNDBase()
    {
        this->Destroy();
    }

    /**
     * @brief Explicitly release all allocated memory
     *
     * Use this function to free large tensors early
     * in long-running simulations.
     */
    void Destroy()
    {
        vector<T>().swap(data_);
        shape_.clear();
        stride_.clear();
        size_ = 0;
    }

    /* ============================
     *  Shape and size management
     * ============================ */

    /** Resize the tensor to the given shape
     * @param shape New shape vector
     */
    void resize(const vector<size_t> &shape)
    {
        shape_ = shape;
        compute_stride();

        size_ = 1;
        for (size_t dim : shape_)
            size_ *= dim;

        data_.resize(size_);
    }

    /**
     * @brief Get tensor shape
     * @return Shape vector
     */
    const vector<size_t> &shape() const
    {
        return shape_;
    }

    /**
     * @brief Get total number of elements
     * @return Total size
     */
    size_t size() const
    {
        return size_;
    }

    /**
     * @brief Get stride array
     * @return Stride vector
     */
    const vector<size_t>& stride() const
    {
        return stride_;
    }

    /**
     * @brief Check if this tensor has the same shape as another tensor
     * @param other Other tensor to compare
     * @return True if shapes are the same, false otherwise
     */
    bool same_shape(const TensorNDBase<T> &other) const
    {
        if (shape_.size() != other.shape_.size())
            return false;

        for (size_t i = 0; i < shape_.size(); ++i)
            if (shape_[i] != other.shape_[i])
                return false;

        return true;
    }

    /* ============================
     *  Raw data access
     * ============================ */

    /**
     * @brief Get raw pointer to contiguous data
     */
    T *data()
    {
        return data_.data();
    }

    /**
     * @brief Get raw pointer to contiguous data (read-only)
     */
    const T *data() const
    {
        return data_.data();
    }

    /**
     * @brief Get data as std::vector
     */
    vector<T> get_data_vector() const
    {
        return data_;
    }

    /* ============================
     *  Semantic access (FAST)
     * ============================ */

    /**
     * @brief Get element value using flattened index
     * @param idx Flattened index
     * @return Element value
     */
    T get(size_t idx) const
    {
        return data_[idx];
    }

    /**
     * @brief Get element value using multi-index
     * @param index Vector of indices
     * @return Element value
     */
    T get(const vector<size_t> &index) const
    {
        return data_[flatten_index(index)];
    }

    /**
     * @brief Set element value using flattened index
     * @param idx Flattened index
     * @param value Value to assign
     */
    void set(size_t idx, const T &value)
    {
        data_[idx] = value;
    }

    /**
     * @brief Set element value using multi-index
     * @param index Vector of indices
     * @param value Value to assign
     */
    void set(const vector<size_t> &index, const T &value)
    {
        data_[flatten_index(index)] = value;
    }

    /* ============================
     *  High-performance access
     * ============================ */

    /**
     * @brief Overloaded operator[] for linear indexing of the total size (read/write)
     * @param idx linear index
     * @return reference to the element at idx
     *
     * WARNING:
     *  - No bounds checking
     *  - Use only when index correctness is guaranteed
     */
    inline T &operator[](size_t idx)
    {
        return data_[idx];
    }

    /**
     * @brief Fast unchecked access using multi-index (read/write)
     *
     * WARNING:
     *  - No bounds checking
     *  - Use only when index correctness is guaranteed
     */
    inline T &value_at(const vector<size_t> &index)
    {
        return data_[flatten_index_unchecked(index)];
    }

    /**
     * @brief Fast unchecked access using multi-index (read-only)
     */
    inline const T &value_at(const vector<size_t> &index) const
    {
        return data_[flatten_index_unchecked(index)];
    }

    /* ============================
     *  Initialization helpers
     * ============================ */

    /**
     * @brief Fill tensor with a constant value
     * @param value Value to assign
     */
    void fill(const T &value)
    {
        std::fill(data_.begin(), data_.end(), value);
    }

    /**
     * @brief Fill tensor values from a scalar function defined on R^d
     *        using multiple threads
     *
     * @tparam Func Callable type: T(const vector<double>&)
     * @param grids Physical grids for each dimension
     * @param func  Function defined in physical space
     * @param number_of_threads Number of threads for parallel filling, default is 1 (no parallelism)
     */
    template <typename Func>
    void fill_from_function(
        const vector<vector<double>> &grids,
        Func func,
        int number_of_threads = 1)
    {
        const size_t ndim = shape_.size();

        if (grids.size() != ndim)
            throw runtime_error("Grid dimension mismatch");

        for (size_t d = 0; d < ndim; ++d)
        {
            if (grids[d].size() != shape_[d])
                throw runtime_error("Grid size mismatch at dimension " +
                                    std::to_string(d));
        }

        #pragma omp parallel for num_threads(number_of_threads)
        for (size_t linear = 0; linear < data_.size(); ++linear)
        {
            vector<size_t> index(ndim);
            vector<double> coords(ndim);
            // linear index -> multi-index (reverse of flatten_index)
            size_t tmp = linear;
            for (int d = int(ndim) - 1; d >= 0; --d)
            {
                index[d] = tmp % shape_[d];
                tmp /= shape_[d];
            }

            // multi-index -> physical coordinates
            for (size_t d = 0; d < ndim; ++d)
                coords[d] = grids[d][index[d]];

            // evaluate function
            data_[linear] = func(coords);
        }
    }

    /**
     * @brief Set all tensor elements to zero
     */
    void zero()
    {
        std::fill(data_.begin(), data_.end(), T(0));
    }

    /* ============================
     *  I/O: Screen
     * ============================ */

    /**
     * @brief Print tensor values to standard output
     *
     * For ND tensors, values are printed in flattened order.
     */
    void print() const
    {
        for (size_t i = 0; i < this->shape().size(); i++)
        {
            cout << this->shape()[i] << "\t";
        }

        cout << "\n";

        for (size_t i = 0; i < size_; ++i)
            cout << data_[i] << "\t";

        cout << "\n";
    }

    /**
     * @brief Extract a slice along a specified dimension at a given index
     *        using multiple threads
     * @param dim   Dimension to slice
     * @param idx Index along the dimension
     * @param number_of_threads Number of threads for parallel extraction, default is 1 (no parallelism)
     * @return      (N-1)-D tensor slice
     */
    TensorNDBase<T> slice(size_t dim, size_t idx, int number_of_threads = 1) const
    {
        if (dim >= shape_.size())
            throw runtime_error("Invalid dimension");

        if (idx >= shape_[dim])
            throw runtime_error("Index out of bounds");

        // new shape
        vector<size_t> new_shape;
        for (size_t d = 0; d < shape_.size(); ++d)
            if (d != dim)
                new_shape.push_back(shape_[d]);

        TensorNDBase<T> result(new_shape);

#pragma omp parallel for num_threads(number_of_threads)
        for (size_t linear = 0; linear < result.size(); ++linear)
        {
            // Each thread needs its own local copies to avoid race conditions
            vector<size_t> index(shape_.size());
            vector<size_t> sub_index(new_shape.size());

            // unflatten in result
            size_t tmp = linear;
            for (int d = int(new_shape.size()) - 1; d >= 0; --d)
            {
                sub_index[d] = tmp % new_shape[d];
                tmp /= new_shape[d];
            }

            // build full index
            size_t s = 0;
            for (size_t d = 0; d < shape_.size(); ++d)
            {
                if (d == dim)
                    index[d] = idx;
                else
                    index[d] = sub_index[s++];
            }

            result.data_[linear] = value_at(index);
        }

        return result;
    }

    /**
     * @brief Insert a (d-1)D tensor into this dD tensor
     *        along dimension dim at position index
     * @param dim   Dimension to fix
     * @param index Index along dim
     * @param sub   Tensor of dimension (d-1)
     * @param number_of_threads Number of threads for parallel insertion, default is 1 (no parallelism)
     */
    void insert_slice(size_t dim, size_t index, const TensorNDBase<T> &sub, int number_of_threads = 1)
    {
        const size_t ndim = shape_.size();

        if (dim >= ndim)
            throw runtime_error("Invalid dimension");

        if (index >= shape_[dim])
            throw runtime_error("Index out of bounds");

        if (sub.shape_.size() != ndim - 1)
            throw runtime_error("Sub-tensor has wrong dimension");

        // check shape compatibility
        for (size_t d = 0, s = 0; d < ndim; ++d)
        {
            if (d == dim)
                continue;
            if (shape_[d] != sub.shape_[s++])
                throw runtime_error("Shape mismatch in insert_slice");
        }

#pragma omp parallel for num_threads(number_of_threads)
        for (size_t linear = 0; linear < sub.data_.size(); ++linear)
        {
            // Each thread needs its own local copies to avoid race conditions
            vector<size_t> full_index(ndim);
            vector<size_t> sub_index(ndim - 1);

            // unflatten sub-tensor index
            size_t tmp = linear;
            for (int d = int(sub.shape_.size()) - 1; d >= 0; --d)
            {
                sub_index[d] = tmp % sub.shape_[d];
                tmp /= sub.shape_[d];
            }

            // build full index
            size_t s = 0;
            for (size_t d = 0; d < ndim; ++d)
            {
                if (d == dim)
                    full_index[d] = index;
                else
                    full_index[d] = sub_index[s++];
            }

            value_at(full_index) = sub.data_[linear];
        }
    }

    /* ============================
     *  Norms and statistics
     * ============================ */

    /**
     * @brief Compute discrete L2 norm: sqrt(1/n * sum(|x|^2))
     * @return discrete L2 norm value
     */
    double get_L2_discrete_norm() const
    {
        double sum_sq = 0.0;
        const T *data = this->data();
        const size_t n = this->size();

        for (size_t i = 0; i < n; ++i)
        {
            T val = data[i];
            // For complex numbers: |z|^2 = real^2 + imag^2
            // For real numbers: |z|^2 = z^2
            double abs_val = std::abs(val);
            sum_sq += abs_val * abs_val;
        }

        return std::sqrt(sum_sq / static_cast<double>(n));
    }

    /**
     * @brief Compute discrete L1 norm: 1/n * sum(|x|)
     * @return discrete L1 norm value
     */
    double get_L1_discrete_norm() const
    {
        double sum_abs = 0.0;
        const T *data = this->data();
        const size_t n = this->size();

        for (size_t i = 0; i < n; ++i)
        {
            T val = data[i];
            double abs_val = std::abs(val);
            sum_abs += abs_val;
        }

        return sum_abs / static_cast<double>(n);
    }

    /**
     * @brief Compute infinity norm: max(|x|)
     * @return Infinity norm value
     */
    double get_inf_norm() const
    {
        double inf_norm = 0.0;
        const T *data = this->data();
        const size_t n = this->size();

        for (size_t i = 0; i < n; ++i)
        {
            double abs_val = std::abs(data[i]);
            if (abs_val > inf_norm)
                inf_norm = abs_val;
        }

        return inf_norm;
    }

    /**
     * @brief Calculate memory usage in bytes
     * @return Total memory usage in bytes (data + shape + stride)
     */
    size_t get_memory_usage_bytes() const
    {
        size_t total = 0;

        // vector objects themselves
        total += sizeof(data_);
        total += sizeof(shape_);
        total += sizeof(stride_);

        // buffers
        total += sizeof(double) * data_.capacity();
        total += sizeof(size_t) * shape_.capacity();
        total += sizeof(size_t) * stride_.capacity();

        // scalar members
        total += sizeof(size_);

        return total;
    }

    /**
     * @brief Calculate memory usage in MB
     * @return Total memory usage in megabytes
     */
    double get_memory_usage_MB() const
    {
        return static_cast<double>(get_memory_usage_bytes()) / (1024.0 * 1024.0);
    }
};

#endif // TENSOR_ND_BASE_H
