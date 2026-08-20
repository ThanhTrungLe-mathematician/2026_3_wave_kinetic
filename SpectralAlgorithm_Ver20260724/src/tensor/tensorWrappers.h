#ifndef TENSOR_WRAPPERS_H
#define TENSOR_WRAPPERS_H

#include "tensorNDReal.h"
#include "tensorNDComplex.h"

using std::size_t;

/* =========================================================
 *  REAL TENSORS
 * ========================================================= */

// -------- 1D --------
class Tensor1DReal : public TensorNDReal
{
public:
    Tensor1DReal() = default;
    explicit Tensor1DReal(size_t n0)
        : TensorNDReal({n0}) {}

    /**
     * @brief Resize the tensor to the given shape
     * @param n0 New size for the 1D tensor
     */
    void resize(size_t n0)
    {
        TensorNDReal::resize({n0});
    }

    /**
     * @brief Overloaded operator() for indexing in 1D tensor (read/write)
     * @param idx 1D index
     * @return reference to the element at idx
     *
     * WARNING:
     *  - No bounds checking
     *  - Use only when index correctness is guaranteed
     */
    inline double &operator()(size_t idx)
    {
        return data_[idx];
    }

    /**
     * @brief Create a 1D vector with specified start, end, and step
     * @param start_point Starting value
     * @param end_point Ending value
     * @param step Step size
     */
    void create_vector(double start_point, double end_point, double step)
    {
        size_t N = static_cast<size_t>((end_point - start_point) / step) + 1;
        this->resize(N);
        for (size_t i = 0; i < N; ++i)
            this->set(i, start_point + i * step);
    }

    /**
     * @brief Create a 1D vector with specified start, end, and number of points (FFTW standard)
     * @param start_point Starting value
     * @param end_point Ending value
     * @param number_of_points Number of points (must be >= 2)
     *
     * Creates a uniform grid from start_point to end_point with exactly number_of_points points.
     * The step size is (end_point - start_point) / (number_of_points - 1) to properly include the endpoint.
     */
    void create_vector(double start_point, double end_point, int number_of_points)
    {
        if (number_of_points < 2)
            throw std::runtime_error("Number of points must be at least 2 for FFTW standard");

        size_t N = static_cast<size_t>(number_of_points);
        this->resize(N);

        if (N == 1)
        {
            this->set(0, start_point);
            return;
        }

        double step = (end_point - start_point) / (N - 1);
        for (size_t i = 0; i < N; ++i)
            this->set(i, start_point + i * step);
    }

    /**
     * @brief Create physical grid k in FFTW ordering.
     *
     * This function constructs the physical grid for k ∈ [-R, R]
     * but stored in FFTW-compatible order:
     *
     *   [ 0 → R )  followed by  ( -R → 0 )
     *
     * That is:
     *   k[0]     = 0
     *   k[1]     = h
     *   ...
     *   k[N-1]   = R - h
     *   k[N]     = -R
     *   k[N+1]   = -R + h
     *   ...
     *   k[2N-1]  = -h
     *
     * This ordering removes the need for phase correction (-1)^n
     * when using FFTW, since the grid origin is aligned with FFTW's convention.
     *
     * @param R  Half-length of the physical domain [-R, R]
     * @param N  Half number of grid points (total points = 2N)
     * @return   std::vector<double> of size 2N storing k-grid in FFTW order
     */
    void create_vector_FFTW(double R, int N)
    {
        const int M = 2 * N;
        const double h = R / N;

        if (N < 2)
            throw std::runtime_error("Number of points must be at least 2 for FFTW standard");

        size_t size_2N = static_cast<size_t>(M);
        this->resize(size_2N);

        for (int ell = 0; ell < M; ++ell)
        {
            if (ell < N)
            {
                // [0 → R)
                this->set(ell, ell * h);
            }
            else
            {
                // (-R → 0)
                this->set(ell, (ell - M) * h);
            }
        }
    }
};

// -------- 2D --------
class Tensor2DReal : public TensorNDReal
{
public:
    Tensor2DReal() = default;
    Tensor2DReal(size_t n0, size_t n1)
        : TensorNDReal({n0, n1}) {}

        /**
         * @brief Resize the 2D tensor to the given shape
         * @param n0 New size for the first dimension
         * @param n1 New size for the second dimension
         */
    void resize(size_t n0, size_t n1)
    {
        TensorNDReal::resize({n0, n1});
    }

    /**
     * @brief Access element at (idx, jdx) in the 2D tensor (read/write)
     * 
     * @param idx  index along the first dimension
     * @param jdx  index along the second dimension
     * @return double& Reference to the element at (idx, jdx)
     */
    inline double &operator()(size_t idx, size_t jdx)
    {
        size_t linear_index = idx*stride_[0] + jdx*stride_[1];
        return data_[linear_index];
    }

};

// -------- 3D --------
class Tensor3DReal : public TensorNDReal
{
public:
    Tensor3DReal() = default;
    Tensor3DReal(size_t n0, size_t n1, size_t n2)
        : TensorNDReal({n0, n1, n2}) {}

    void resize(size_t n0, size_t n1, size_t n2)
    {
        TensorNDReal::resize({n0, n1, n2});
    }

    /**
     * @brief Access element at (idx, jdx, kdx) in the 3D tensor (read/write)
     * 
     * @param idx  index along the first dimension
     * @param jdx  index along the second dimension
     * @param kdx  index along the third dimension
     * @return double& Reference to the element at (idx, jdx, kdx)
     */
    inline double &operator()(size_t idx, size_t jdx, size_t kdx)
    {
        size_t linear_index = idx*stride_[0] + jdx*stride_[1] + kdx*stride_[2];
        return data_[linear_index];
    }
};

// -------- 4D --------
class Tensor4DReal : public TensorNDReal
{
public:
    Tensor4DReal() = default;
    Tensor4DReal(size_t n0, size_t n1, size_t n2, size_t n3)
        : TensorNDReal({n0, n1, n2, n3}) {}

    void resize(size_t n0, size_t n1, size_t n2, size_t n3)
    {
        TensorNDReal::resize({n0, n1, n2, n3});
    }

    /**
     * @brief Access element at (idx, jdx, kdx, ldx) in the 4D tensor (read/write)
     * 
     * @param idx  index along the first dimension
     * @param jdx  index along the second dimension
     * @param kdx  index along the third dimension
     * @param ldx  index along the fourth dimension
     * @return double& Reference to the element at (idx, jdx, kdx, ldx)
     */
    inline double &operator()(size_t idx, size_t jdx, size_t kdx, size_t ldx)
    {
        size_t linear_index = idx*stride_[0] + jdx*stride_[1] + kdx*stride_[2] + ldx*stride_[3];
        return data_[linear_index];
    }
};

// -------- 5D --------
class Tensor5DReal : public TensorNDReal
{
public:
    Tensor5DReal() = default;
    Tensor5DReal(size_t n0, size_t n1, size_t n2,
                 size_t n3, size_t n4)
        : TensorNDReal({n0, n1, n2, n3, n4}) {}

    void resize(size_t n0, size_t n1, size_t n2,
                size_t n3, size_t n4)
    {
        TensorNDReal::resize({n0, n1, n2, n3, n4});
    }

    /**
     * @brief Access element in the 5D tensor (read/write)
     */
    inline double &operator()(size_t i0, size_t i1, size_t i2, size_t i3, size_t i4)
    {
        size_t linear_index = i0*stride_[0] + i1*stride_[1] + i2*stride_[2] + i3*stride_[3] + i4*stride_[4];
        return data_[linear_index];
    }
};

// -------- 6D --------
class Tensor6DReal : public TensorNDReal
{
public:
    Tensor6DReal() = default;
    Tensor6DReal(size_t n0, size_t n1, size_t n2,
                 size_t n3, size_t n4, size_t n5)
        : TensorNDReal({n0, n1, n2, n3, n4, n5}) {}

    void resize(size_t n0, size_t n1, size_t n2,
                size_t n3, size_t n4, size_t n5)
    {
        TensorNDReal::resize({n0, n1, n2, n3, n4, n5});
    }

    /**
     * @brief Access element in the 6D tensor (read/write)
     */
    inline double &operator()(size_t i0, size_t i1, size_t i2, size_t i3, size_t i4, size_t i5)
    {
        size_t linear_index = i0*stride_[0] + i1*stride_[1] + i2*stride_[2] + i3*stride_[3] + i4*stride_[4] + i5*stride_[5];
        return data_[linear_index];
    }
};

// -------- 7D --------
class Tensor7DReal : public TensorNDReal
{
public:
    Tensor7DReal() = default;
    Tensor7DReal(size_t n0, size_t n1, size_t n2,
                 size_t n3, size_t n4, size_t n5, size_t n6)
        : TensorNDReal({n0, n1, n2, n3, n4, n5, n6}) {}

    void resize(size_t n0, size_t n1, size_t n2,
                size_t n3, size_t n4, size_t n5, size_t n6)
    {
        TensorNDReal::resize({n0, n1, n2, n3, n4, n5, n6});
    }

    /**
     * @brief Access element in the 7D tensor (read/write)
     */
    inline double &operator()(size_t i0, size_t i1, size_t i2, size_t i3, size_t i4, size_t i5, size_t i6)
    {
        size_t linear_index = i0*stride_[0] + i1*stride_[1] + i2*stride_[2] + i3*stride_[3] + i4*stride_[4] + i5*stride_[5] + i6*stride_[6];
        return data_[linear_index];
    }
};

// -------- 8D --------
class Tensor8DReal : public TensorNDReal
{
public:
    Tensor8DReal() = default;
    Tensor8DReal(size_t n0, size_t n1, size_t n2, size_t n3,
                 size_t n4, size_t n5, size_t n6, size_t n7)
        : TensorNDReal({n0, n1, n2, n3, n4, n5, n6, n7}) {}

    void resize(size_t n0, size_t n1, size_t n2, size_t n3,
                size_t n4, size_t n5, size_t n6, size_t n7)
    {
        TensorNDReal::resize({n0, n1, n2, n3, n4, n5, n6, n7});
    }

    /**
     * @brief Access element in the 8D tensor (read/write)
     */
    inline double &operator()(size_t i0, size_t i1, size_t i2, size_t i3, size_t i4, size_t i5, size_t i6, size_t i7)
    {
        size_t linear_index = i0*stride_[0] + i1*stride_[1] + i2*stride_[2] + i3*stride_[3] + i4*stride_[4] + i5*stride_[5] + i6*stride_[6] + i7*stride_[7];
        return data_[linear_index];
    }
};

// -------- 9D --------
class Tensor9DReal : public TensorNDReal
{
public:
    Tensor9DReal() = default;
    Tensor9DReal(size_t n0, size_t n1, size_t n2, size_t n3,
                 size_t n4, size_t n5, size_t n6, size_t n7, size_t n8)
        : TensorNDReal({n0, n1, n2, n3, n4, n5, n6, n7, n8}) {}

    void resize(size_t n0, size_t n1, size_t n2, size_t n3,
                size_t n4, size_t n5, size_t n6, size_t n7, size_t n8)
    {
        TensorNDReal::resize({n0, n1, n2, n3, n4, n5, n6, n7, n8});
    }

    /**
     * @brief Access element in the 9D tensor (read/write)
     */
    inline double &operator()(size_t i0, size_t i1, size_t i2, size_t i3, size_t i4, size_t i5, size_t i6, size_t i7, size_t i8)
    {
        size_t linear_index = i0*stride_[0] + i1*stride_[1] + i2*stride_[2] + i3*stride_[3] + i4*stride_[4] + i5*stride_[5] + i6*stride_[6] + i7*stride_[7] + i8;
        return data_[linear_index];
    }
};

/* =========================================================
 *  REAL TENSORS (FLOAT32)
 * ========================================================= */

// -------- 1D --------
class Tensor1DReal_Float32 : public TensorNDBase<float>
{
public:
    Tensor1DReal_Float32() = default;
    explicit Tensor1DReal_Float32(size_t n0)
        : TensorNDBase<float>({n0}) {}

    void resize(size_t n0)
    {
        TensorNDBase<float>::resize({n0});
    }

    inline float &operator()(size_t idx)
    {
        return data_[idx];
    }
};

// -------- 2D --------
class Tensor2DReal_Float32 : public TensorNDBase<float>
{
public:
    Tensor2DReal_Float32() = default;
    Tensor2DReal_Float32(size_t n0, size_t n1)
        : TensorNDBase<float>({n0, n1}) {}

    void resize(size_t n0, size_t n1)
    {
        TensorNDBase<float>::resize({n0, n1});
    }

    inline float &operator()(size_t idx, size_t jdx)
    {
        size_t linear_index = idx * stride_[0] + jdx * stride_[1];
        return data_[linear_index];
    }
};

// -------- 3D --------
class Tensor3DReal_Float32 : public TensorNDBase<float>
{
public:
    Tensor3DReal_Float32() = default;
    Tensor3DReal_Float32(size_t n0, size_t n1, size_t n2)
        : TensorNDBase<float>({n0, n1, n2}) {}

    void resize(size_t n0, size_t n1, size_t n2)
    {
        TensorNDBase<float>::resize({n0, n1, n2});
    }

    inline float &operator()(size_t idx, size_t jdx, size_t kdx)
    {
        size_t linear_index = idx * stride_[0] + jdx * stride_[1] + kdx * stride_[2];
        return data_[linear_index];
    }
};

// -------- 4D --------
class Tensor4DReal_Float32 : public TensorNDBase<float>
{
public:
    Tensor4DReal_Float32() = default;
    Tensor4DReal_Float32(size_t n0, size_t n1, size_t n2, size_t n3)
        : TensorNDBase<float>({n0, n1, n2, n3}) {}

    void resize(size_t n0, size_t n1, size_t n2, size_t n3)
    {
        TensorNDBase<float>::resize({n0, n1, n2, n3});
    }

    inline float &operator()(size_t idx, size_t jdx, size_t kdx, size_t ldx)
    {
        size_t linear_index = idx * stride_[0] + jdx * stride_[1] + kdx * stride_[2] + ldx * stride_[3];
        return data_[linear_index];
    }
};

// -------- 5D --------
class Tensor5DReal_Float32 : public TensorNDBase<float>
{
public:
    Tensor5DReal_Float32() = default;
    Tensor5DReal_Float32(size_t n0, size_t n1, size_t n2, size_t n3, size_t n4)
        : TensorNDBase<float>({n0, n1, n2, n3, n4}) {}

    void resize(size_t n0, size_t n1, size_t n2, size_t n3, size_t n4)
    {
        TensorNDBase<float>::resize({n0, n1, n2, n3, n4});
    }

    inline float &operator()(size_t i0, size_t i1, size_t i2, size_t i3, size_t i4)
    {
        size_t linear_index = i0 * stride_[0] + i1 * stride_[1] + i2 * stride_[2] + i3 * stride_[3] + i4 * stride_[4];
        return data_[linear_index];
    }
};

// -------- 6D --------
class Tensor6DReal_Float32 : public TensorNDBase<float>
{
public:
    Tensor6DReal_Float32() = default;
    Tensor6DReal_Float32(size_t n0, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5)
        : TensorNDBase<float>({n0, n1, n2, n3, n4, n5}) {}

    void resize(size_t n0, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5)
    {
        TensorNDBase<float>::resize({n0, n1, n2, n3, n4, n5});
    }

    inline float &operator()(size_t i0, size_t i1, size_t i2, size_t i3, size_t i4, size_t i5)
    {
        size_t linear_index = i0 * stride_[0] + i1 * stride_[1] + i2 * stride_[2] + i3 * stride_[3] + i4 * stride_[4] + i5 * stride_[5];
        return data_[linear_index];
    }
};

// -------- 7D --------
class Tensor7DReal_Float32 : public TensorNDBase<float>
{
public:
    Tensor7DReal_Float32() = default;
    Tensor7DReal_Float32(size_t n0, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6)
        : TensorNDBase<float>({n0, n1, n2, n3, n4, n5, n6}) {}

    void resize(size_t n0, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6)
    {
        TensorNDBase<float>::resize({n0, n1, n2, n3, n4, n5, n6});
    }

    inline float &operator()(size_t i0, size_t i1, size_t i2, size_t i3, size_t i4, size_t i5, size_t i6)
    {
        size_t linear_index = i0 * stride_[0] + i1 * stride_[1] + i2 * stride_[2] + i3 * stride_[3] + i4 * stride_[4] + i5 * stride_[5] + i6 * stride_[6];
        return data_[linear_index];
    }
};

// -------- 8D --------
class Tensor8DReal_Float32 : public TensorNDBase<float>
{
public:
    Tensor8DReal_Float32() = default;
    Tensor8DReal_Float32(size_t n0, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7)
        : TensorNDBase<float>({n0, n1, n2, n3, n4, n5, n6, n7}) {}

    void resize(size_t n0, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7)
    {
        TensorNDBase<float>::resize({n0, n1, n2, n3, n4, n5, n6, n7});
    }

    inline float &operator()(size_t i0, size_t i1, size_t i2, size_t i3, size_t i4, size_t i5, size_t i6, size_t i7)
    {
        size_t linear_index = i0 * stride_[0] + i1 * stride_[1] + i2 * stride_[2] + i3 * stride_[3] + i4 * stride_[4] + i5 * stride_[5] + i6 * stride_[6] + i7 * stride_[7];
        return data_[linear_index];
    }
};

// -------- 9D --------
class Tensor9DReal_Float32 : public TensorNDBase<float>
{
public:
    Tensor9DReal_Float32() = default;
    Tensor9DReal_Float32(size_t n0, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, size_t n8)
        : TensorNDBase<float>({n0, n1, n2, n3, n4, n5, n6, n7, n8}) {}

    void resize(size_t n0, size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t n7, size_t n8)
    {
        TensorNDBase<float>::resize({n0, n1, n2, n3, n4, n5, n6, n7, n8});
    }

    inline float &operator()(size_t i0, size_t i1, size_t i2, size_t i3, size_t i4, size_t i5, size_t i6, size_t i7, size_t i8)
    {
        size_t linear_index = i0 * stride_[0] + i1 * stride_[1] + i2 * stride_[2] + i3 * stride_[3] + i4 * stride_[4] + i5 * stride_[5] + i6 * stride_[6] + i7 * stride_[7] + i8;
        return data_[linear_index];
    }
};

/* =========================================================
 *  COMPLEX TENSORS (use TensorComplexBase)
 * ========================================================= */

// -------- 1D --------
class Tensor1DComplex : public TensorNDComplex
{
public:
    Tensor1DComplex() = default;
    explicit Tensor1DComplex(size_t n0)
        : TensorNDComplex({n0}) {}

    void resize(size_t n0)
    {
        TensorNDComplex::resize({n0});
    }

    /**
     * @brief Access element at idx in the 1D tensor (read/write)
     * @param idx index along the first dimension
     * @return std::complex<double>& Reference to the element at idx
     */
    inline std::complex<double> &operator()(size_t idx)
    {
        return data_[idx];
    }
};

// -------- 2D --------
class Tensor2DComplex : public TensorNDComplex
{
public:
    Tensor2DComplex() = default;
    Tensor2DComplex(size_t n0, size_t n1)
        : TensorNDComplex({n0, n1}) {}
    void resize(size_t n0, size_t n1)
    {
        TensorNDComplex::resize({n0, n1});
    }

    /**
     * @brief Access element at (idx, jdx) in the 2D tensor (read/write)
     * 
     * @param idx  index along the first dimension
     * @param jdx  index along the second dimension
     * @return std::complex<double>& Reference to the element at (idx, jdx)
     */
    inline std::complex<double> &operator()(size_t idx, size_t jdx)
    {
        size_t linear_index = idx*stride_[0] + jdx*stride_[1];
        return data_[linear_index];
    }
};

// -------- 3D --------
class Tensor3DComplex : public TensorNDComplex
{
public:
    Tensor3DComplex() = default;
    Tensor3DComplex(size_t n0, size_t n1, size_t n2)
        : TensorNDComplex({n0, n1, n2}) {}
    void resize(size_t n0, size_t n1, size_t n2)
    {
        TensorNDComplex::resize({n0, n1, n2});
    }

    /**
     * @brief Access element at (idx, jdx, kdx) in the 3D tensor (read/write)
     */
    inline std::complex<double> &operator()(size_t idx, size_t jdx, size_t kdx)
    {
        size_t linear_index = idx*stride_[0] + jdx*stride_[1] + kdx*stride_[2];
        return data_[linear_index];
    }
};

// -------- 4D --------
class Tensor4DComplex : public TensorNDComplex
{
public:
    Tensor4DComplex() = default;
    Tensor4DComplex(size_t n0, size_t n1, size_t n2, size_t n3)
        : TensorNDComplex({n0, n1, n2, n3}) {}
    void resize(size_t n0, size_t n1, size_t n2, size_t n3)
    {
        TensorNDComplex::resize({n0, n1, n2, n3});
    }

    /**
     * @brief Access element in the 4D tensor (read/write)
     */
    inline std::complex<double> &operator()(size_t i0, size_t i1, size_t i2, size_t i3)
    {
        size_t linear_index = i0*stride_[0] + i1*stride_[1] + i2*stride_[2] + i3*stride_[3];
        return data_[linear_index];
    }
};

// -------- 5D --------
class Tensor5DComplex : public TensorNDComplex
{
public:
    Tensor5DComplex() = default;
    Tensor5DComplex(size_t n0, size_t n1, size_t n2,
                    size_t n3, size_t n4)
        : TensorNDComplex({n0, n1, n2, n3, n4}) {}
    void resize(size_t n0, size_t n1, size_t n2,
                size_t n3, size_t n4)
    {
        TensorNDComplex::resize({n0, n1, n2, n3, n4});
    }

    /**
     * @brief Access element in the 5D tensor (read/write)
     */
    inline std::complex<double> &operator()(size_t i0, size_t i1, size_t i2, size_t i3, size_t i4)
    {
        size_t linear_index = i0*stride_[0] + i1*stride_[1] + i2*stride_[2] + i3*stride_[3] + i4*stride_[4];
        return data_[linear_index];
    }
};

// -------- 6D --------
class Tensor6DComplex : public TensorNDComplex
{
public:
    Tensor6DComplex() = default;
    Tensor6DComplex(size_t n0, size_t n1, size_t n2,
                    size_t n3, size_t n4, size_t n5)
        : TensorNDComplex({n0, n1, n2, n3, n4, n5}) {}
    void resize(size_t n0, size_t n1, size_t n2,
                size_t n3, size_t n4, size_t n5)
    {
        TensorNDComplex::resize({n0, n1, n2, n3, n4, n5});
    }

    /**
     * @brief Access element in the 6D tensor (read/write)
     */
    inline std::complex<double> &operator()(size_t i0, size_t i1, size_t i2, size_t i3, size_t i4, size_t i5)
    {
        size_t linear_index = i0*stride_[0] + i1*stride_[1] + i2*stride_[2] + i3*stride_[3] + i4*stride_[4] + i5*stride_[5];
        return data_[linear_index];
    }
};

// -------- 7D --------
class Tensor7DComplex : public TensorNDComplex
{
public:
    Tensor7DComplex() = default;
    Tensor7DComplex(size_t n0, size_t n1, size_t n2,
                    size_t n3, size_t n4, size_t n5, size_t n6)
        : TensorNDComplex({n0, n1, n2, n3, n4, n5, n6}) {}
    void resize(size_t n0, size_t n1, size_t n2,
                size_t n3, size_t n4, size_t n5, size_t n6)
    {
        TensorNDComplex::resize({n0, n1, n2, n3, n4, n5, n6});
    }

    /**
     * @brief Access element in the 7D tensor (read/write)
     */
    inline std::complex<double> &operator()(size_t i0, size_t i1, size_t i2, size_t i3, size_t i4, size_t i5, size_t i6)
    {
        size_t linear_index = i0*stride_[0] + i1*stride_[1] + i2*stride_[2] + i3*stride_[3] + i4*stride_[4] + i5*stride_[5] + i6*stride_[6];
        return data_[linear_index];
    }
};

// -------- 8D --------
class Tensor8DComplex : public TensorNDComplex
{
public:
    Tensor8DComplex() = default;
    Tensor8DComplex(size_t n0, size_t n1, size_t n2, size_t n3,
                    size_t n4, size_t n5, size_t n6, size_t n7)
        : TensorNDComplex({n0, n1, n2, n3, n4, n5, n6, n7}) {}
    void resize(size_t n0, size_t n1, size_t n2, size_t n3,
                size_t n4, size_t n5, size_t n6, size_t n7)
    {
        TensorNDComplex::resize({n0, n1, n2, n3, n4, n5, n6, n7});
    }

    /**
     * @brief Access element in the 8D tensor (read/write)
     */
    inline std::complex<double> &operator()(size_t i0, size_t i1, size_t i2, size_t i3, size_t i4, size_t i5, size_t i6, size_t i7)
    {
        size_t linear_index = i0*stride_[0] + i1*stride_[1] + i2*stride_[2] + i3*stride_[3] + i4*stride_[4] + i5*stride_[5] + i6*stride_[6] + i7*stride_[7];
        return data_[linear_index];
    }
};

// -------- 9D --------
class Tensor9DComplex : public TensorNDComplex
{
public:
    Tensor9DComplex() = default;
    Tensor9DComplex(size_t n0, size_t n1, size_t n2, size_t n3,
                    size_t n4, size_t n5, size_t n6, size_t n7, size_t n8)
        : TensorNDComplex({n0, n1, n2, n3, n4, n5, n6, n7, n8}) {}
    void resize(size_t n0, size_t n1, size_t n2, size_t n3,
                size_t n4, size_t n5, size_t n6, size_t n7, size_t n8)
    {
        TensorNDComplex::resize({n0, n1, n2, n3, n4, n5, n6, n7, n8});
    }

    /**
     * @brief Access element in the 9D tensor (read/write)
     */
    inline std::complex<double> &operator()(size_t i0, size_t i1, size_t i2, size_t i3, size_t i4, size_t i5, size_t i6, size_t i7, size_t i8)
    {
        size_t linear_index = i0*stride_[0] + i1*stride_[1] + i2*stride_[2] + i3*stride_[3] + i4*stride_[4] + i5*stride_[5] + i6*stride_[6] + i7*stride_[7] + i8;
        return data_[linear_index];
    }
};

#endif // TENSOR_WRAPPERS_H
