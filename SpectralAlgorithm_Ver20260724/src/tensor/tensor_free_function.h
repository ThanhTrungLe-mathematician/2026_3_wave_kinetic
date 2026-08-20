#ifndef FREE_FUNCTION_TENSOR_H
#define FREE_FUNCTION_TENSOR_H

#include "tensorNDBase.h"
#include "fft_executor.h"

#include <limits>
#include <type_traits>

/**
 * @brief Sum two tensors: A = B + C using multiple threads
 *
 * @tparam T Data type
 * @param A Result tensor
 * @param B First input tensor
 * @param C Second input tensor
 * @param number_of_threads Number of threads to use
 */
template <typename T>
void sum_tensors(TensorNDBase<T> &A, const TensorNDBase<T> &B, const TensorNDBase<T> &C, int number_of_threads = 1)
{
    if (!B.same_shape(C))
        throw std::invalid_argument("Tensor shape mismatch");

    A.resize(B.shape());

    const T *b = B.data();
    const T *c = C.data();
    T *a = A.data();

    const size_t n = B.size();
#pragma omp parallel for num_threads(number_of_threads)
    for (size_t i = 0; i < n; ++i)
        a[i] = b[i] + c[i];
}

/**
 * @brief Subtract two tensors: A = B - C using multiple threads
 *
 * @tparam T Data type
 * @param A Result tensor
 * @param B Minuend tensor
 * @param C Subtrahend tensor
 * @param number_of_threads Number of threads to use
 */
template <typename T>
void subtract_tensors(TensorNDBase<T> &A, const TensorNDBase<T> &B, const TensorNDBase<T> &C, int number_of_threads = 1)
{
    if (!B.same_shape(C))
        throw std::invalid_argument("Tensor shape mismatch");

    A.resize(B.shape());

    const T *b = B.data();
    const T *c = C.data();
    T *a = A.data();

    const size_t n = B.size();
#pragma omp parallel for num_threads(number_of_threads)
    for (size_t i = 0; i < n; ++i)
        a[i] = b[i] - c[i];
}

/**
 * @brief Copy tensor data from B to A using multiple threads
 *
 * @tparam T Data type
 * @param A Destination tensor
 * @param B Source tensor
 * @param number_of_threads Number of threads to use
 */
template <typename T>
void copy_tensor(TensorNDBase<T> &A, const TensorNDBase<T> &B, int number_of_threads = 1)
{
    if (!A.same_shape(B))
        throw std::invalid_argument("Tensor shape mismatch");

    const T *b = B.data();
    T *a = A.data();

    const size_t n = B.size();
#pragma omp parallel for num_threads(number_of_threads)
    for (size_t i = 0; i < n; ++i)
        a[i] = b[i];
}

/**
 * @brief Estimate the memory usage of a tensor in bytes
 *
 * @tparam T Data type of the tensor elements
 * @param shape The shape of the tensor
 * @return size_t The estimated memory usage in bytes
 */
template <typename T>
static size_t estimate_memory_bytes(const std::vector<size_t> &shape)
{
    size_t n_elem = 1;
    for (size_t d : shape)
        n_elem *= d;

    size_t total = 0;

    // data buffer
    total += n_elem * sizeof(T);

    // shape + stride arrays (assume exact size, no capacity waste)
    total += 3 * shape.size() * sizeof(size_t); // shape, stride, temporary vectors

    // scalar members
    total += sizeof(size_t); // size_

    return total;
}

/**
 * @brief Estimate the memory usage of a tensor in megabytes
 *
 * @tparam T Data type of the tensor elements
 * @param shape The shape of the tensor
 * @return double The estimated memory usage in megabytes
 */
template <typename T>
double estimate_memory_usage_MB(const std::vector<size_t> &shape)
{
    return static_cast<double>(estimate_memory_bytes<T>(shape)) / (1024.0 * 1024.0);
}

/**
 * @brief Estimate the memory usage of a tensor in gigabytes
 *
 * @tparam T Data type of the tensor elements
 * @param shape The shape of the tensor
 * @return double The estimated memory usage in gigabytes
 */
template <typename T>
double estimate_memory_usage_GB(const std::vector<size_t> &shape)
{
    return static_cast<double>(estimate_memory_bytes<T>(shape)) / (1024.0 * 1024.0 * 1024.0);
}

/**
 * @brief Build difference tensor on numerical grid from a (possibly finer) reference tensor
 *
 * @tparam T Data type
 * @param numerical_sol Numerical solution tensor (coarser or same grid)
 * @param reference_sol Reference solution tensor (finer or same grid)
 * @param number_of_threads Number of threads to use for computation, default is 1 (no parallelism)
 * @return TensorNDBase<T> Difference tensor sampled on numerical grid
 */
template <typename T>
TensorNDBase<T> build_difference_tensor_on_numerical_grid(
    const TensorNDBase<T> &numerical_sol,
    const TensorNDBase<T> &reference_sol,
    int number_of_threads = 1,
    TensorNDBase<T> *ref_sampled_out = nullptr)
{
    const std::vector<size_t> &num_shape = numerical_sol.shape();
    const std::vector<size_t> &ref_shape = reference_sol.shape();

    if (num_shape.size() != ref_shape.size())
        throw std::invalid_argument("Tensor dimension mismatch");

    const size_t ndim = num_shape.size();

    TensorNDBase<T> diff;
    diff.resize(num_shape);

    if (numerical_sol.same_shape(reference_sol))
    {
        if (ref_sampled_out)
        {
            ref_sampled_out->resize(reference_sol.shape());
            copy_tensor(*ref_sampled_out, reference_sol, number_of_threads);
        }
        subtract_tensors(diff, numerical_sol, reference_sol, number_of_threads);
        return diff;
    }

    std::vector<size_t> scale(ndim);

    for (size_t d = 0; d < ndim; ++d)
    {
        if (num_shape[d] < 2 || ref_shape[d] < 2)
            throw std::invalid_argument("Each tensor dimension must be >= 2");

        if (ref_shape[d] < num_shape[d])
            throw std::invalid_argument("Reference shape must be finer or equal to numerical shape");

        // Periodic nested grid: reference size must be an integer multiple of numerical size.
        if (ref_shape[d] % num_shape[d] != 0)
            throw std::invalid_argument("Reference shape incompatible with numerical shape");

        scale[d] = ref_shape[d] / num_shape[d];
    }

    const std::vector<size_t> &num_stride = numerical_sol.stride();
    const std::vector<size_t> &ref_stride = reference_sol.stride();

    const T *num_data = numerical_sol.data();
    const T *ref_data = reference_sol.data();
    T *diff_data = diff.data();

    TensorNDBase<T> ref_sampled;
    T *ref_sampled_data = nullptr;
    if (ref_sampled_out)
    {
        ref_sampled.resize(num_shape);
        ref_sampled_data = ref_sampled.data();
    }

    const size_t N = numerical_sol.size();

#pragma omp parallel for num_threads(number_of_threads)
    for (size_t linear = 0; linear < N; ++linear)
    {
        size_t tmp = linear;
        size_t ref_linear = 0;

        for (size_t d = 0; d < ndim; ++d)
        {
            size_t idx = tmp / num_stride[d];
            tmp %= num_stride[d];

            size_t ref_idx = idx * scale[d];
            ref_linear += ref_idx * ref_stride[d];
        }

        diff_data[linear] = num_data[linear] - ref_data[ref_linear];
        if (ref_sampled_data)
            ref_sampled_data[linear] = ref_data[ref_linear];
    }

    if (ref_sampled_out)
    {
        ref_sampled_out->resize(num_shape);
        copy_tensor(*ref_sampled_out, ref_sampled, number_of_threads);
    }

    return diff;
}

/**
 * @brief Build difference tensor on numerical grid by spectral projection
 *
 * Algorithm:
 * 1) If shapes are equal: diff = numerical_sol - reference_sol directly.
 * 2) Otherwise: FFT(reference_sol), keep only low-frequency modes selected by
 *    an L2 cutoff compatible with numerical grid size, then IFFT back on
 *    numerical grid and subtract from numerical solution.
 *
 * @param numerical_sol Numerical solution on coarse grid
 * @param reference_sol Reference solution on fine grid (same ndim, each dim >= numerical)
 * @param number_of_threads Number of FFT / OpenMP threads
 * @param ref_projected_out Optional output pointer to store projected reference tensor on numerical grid
 * @return TensorNDBase<std::complex<double>> Difference tensor on numerical grid
 */
template <typename T>
TensorNDBase<T> build_difference_tensor_on_numerical_grid_projection(
    const TensorNDBase<T> &numerical_sol,
    const TensorNDBase<T> &reference_sol,
    int number_of_threads = 1,
    TensorNDBase<T> *ref_projected_out = nullptr)
{
    static_assert(std::is_same<T, std::complex<double>>::value,
                  "build_difference_tensor_on_numerical_grid_projection currently supports T = std::complex<double> only");

    const std::vector<size_t> &num_shape = numerical_sol.shape();
    const std::vector<size_t> &ref_shape = reference_sol.shape();

    if (num_shape.size() != ref_shape.size())
        throw std::invalid_argument("Tensor dimension mismatch");

    const size_t ndim = num_shape.size();

    TensorNDBase<T> diff;
    diff.resize(num_shape);

    if (numerical_sol.same_shape(reference_sol))
    {
        if (ref_projected_out)
        {
            ref_projected_out->resize(reference_sol.shape());
            copy_tensor(*ref_projected_out, reference_sol, number_of_threads);
        }
        subtract_tensors(diff, numerical_sol, reference_sol, number_of_threads);
        return diff;
    }

    std::vector<int> num_shape_i(ndim), ref_shape_i(ndim);
    for (size_t d = 0; d < ndim; ++d)
    {
        if (num_shape[d] < 2 || ref_shape[d] < 2)
            throw std::invalid_argument("Each tensor dimension must be >= 2");
        if (ref_shape[d] < num_shape[d])
            throw std::invalid_argument("Reference shape must be finer or equal to numerical shape");
        if (num_shape[d] > static_cast<size_t>(std::numeric_limits<int>::max()) ||
            ref_shape[d] > static_cast<size_t>(std::numeric_limits<int>::max()))
            throw std::invalid_argument("Tensor dimension too large for FFTW int shape");

        num_shape_i[d] = static_cast<int>(num_shape[d]);
        ref_shape_i[d] = static_cast<int>(ref_shape[d]);
    }

    TensorNDBase<T> ref_hat(ref_shape);
    TensorNDBase<T> proj_hat(num_shape);
    proj_hat.fill(T(0.0, 0.0));

    FFT_Executor fft_forward_ref;
    FFT_Config cfg_forward_ref = {
        .dim = static_cast<int>(ndim),
        .shape = ref_shape_i,
        .type = FFT_Type::C2C,
        .direction = FFT_Direction::Forward,
        .inplace = false,
        .normalize = true,
        .denormalize = false,
        .nthreads = number_of_threads,
        .flags = FFT_Flag::Estimate};

    T *ref_input_ptr = const_cast<T *>(reference_sol.data());
    fft_forward_ref.create_plan(cfg_forward_ref, ref_input_ptr, ref_hat.data());
    fft_forward_ref.execute_plan(ref_input_ptr, ref_hat.data());

    const std::vector<size_t> &num_stride = numerical_sol.stride();
    const std::vector<size_t> &ref_stride = reference_sol.stride();

    const T *ref_hat_data = ref_hat.data();
    T *proj_hat_data = proj_hat.data();

    const size_t N_num = numerical_sol.size();

#pragma omp parallel for num_threads(number_of_threads)
    for (size_t linear = 0; linear < N_num; ++linear)
    {
        size_t tmp = linear;
        size_t ref_linear = 0;

        for (size_t d = 0; d < ndim; ++d)
        {
            const size_t idx_num = tmp / num_stride[d];
            tmp %= num_stride[d];

            const double Nn = static_cast<double>(num_shape[d]);
            const double Nr = static_cast<double>(ref_shape[d]);
            const size_t cutoff = (num_shape[d] + 1) / 2;
            const double kd = (idx_num < cutoff)
                                  ? static_cast<double>(idx_num)
                                  : static_cast<double>(idx_num) - Nn;

            const double idx_ref_d = (kd >= 0.0) ? kd : (kd + Nr);
            const size_t idx_ref = static_cast<size_t>(idx_ref_d);
            ref_linear += idx_ref * ref_stride[d];
        }

        proj_hat_data[linear] = ref_hat_data[ref_linear];
    }

    TensorNDBase<T> ref_projected(num_shape);

    FFT_Executor fft_backward_num;
    FFT_Config cfg_backward_num = {
        .dim = static_cast<int>(ndim),
        .shape = num_shape_i,
        .type = FFT_Type::C2C,
        .direction = FFT_Direction::Backward,
        .inplace = false,
        .normalize = false,
        .denormalize = false,
        .nthreads = number_of_threads,
        .flags = FFT_Flag::Estimate};

    fft_backward_num.create_plan(cfg_backward_num, proj_hat.data(), ref_projected.data());
    fft_backward_num.execute_plan(proj_hat.data(), ref_projected.data());

    if (ref_projected_out)
    {
        ref_projected_out->resize(ref_projected.shape());
        copy_tensor(*ref_projected_out, ref_projected, number_of_threads);
    }

    subtract_tensors(diff, numerical_sol, ref_projected, number_of_threads);
    return diff;
}

/**
 * @brief Compute L2 discrete error using spectral projection of reference solution
 *
 * @param numerical_sol Numerical solution tensor (coarser grid)
 * @param reference_sol Reference solution tensor (finer or same grid)
 * @param number_of_threads Number of threads for FFT and tensor operations
 * @return double L2 discrete error on numerical grid
 */
template <typename T>
double compute_L2_discrete_error_projection(
    const TensorNDBase<T> &numerical_sol,
    const TensorNDBase<T> &reference_sol,
    int number_of_threads = 1)
{
    static_assert(std::is_same<T, std::complex<double>>::value,
                  "compute_L2_discrete_error_projection currently supports T = std::complex<double> only");

    TensorNDBase<T> diff =
        build_difference_tensor_on_numerical_grid_projection(numerical_sol, reference_sol, number_of_threads);
    return diff.get_L2_discrete_norm();
}

/**
 * @brief Compute L1 discrete error using spectral projection of reference solution
 *
 * @param numerical_sol Numerical solution tensor (coarser grid)
 * @param reference_sol Reference solution tensor (finer or same grid)
 * @param number_of_threads Number of threads for FFT and tensor operations
 * @return double L1 discrete error on numerical grid
 */
template <typename T>
double compute_L1_discrete_error_projection(
    const TensorNDBase<T> &numerical_sol,
    const TensorNDBase<T> &reference_sol,
    int number_of_threads = 1)
{
    static_assert(std::is_same<T, std::complex<double>>::value,
                  "compute_L1_discrete_error_projection currently supports T = std::complex<double> only");

    TensorNDBase<T> diff =
        build_difference_tensor_on_numerical_grid_projection(numerical_sol, reference_sol, number_of_threads);
    return diff.get_L1_discrete_norm();
}

/**
 * @brief Compute infinity-norm error using spectral projection of reference solution
 *
 * @param numerical_sol Numerical solution tensor (coarser grid)
 * @param reference_sol Reference solution tensor (finer or same grid)
 * @param number_of_threads Number of threads for FFT and tensor operations
 * @return double Infinity-norm error on numerical grid
 */
template <typename T>
double compute_inf_discrete_error_projection(
    const TensorNDBase<T> &numerical_sol,
    const TensorNDBase<T> &reference_sol,
    int number_of_threads = 1)
{
    static_assert(std::is_same<T, std::complex<double>>::value,
                  "compute_inf_discrete_error_projection currently supports T = std::complex<double> only");

    TensorNDBase<T> diff =
        build_difference_tensor_on_numerical_grid_projection(numerical_sol, reference_sol, number_of_threads);
    return diff.get_inf_norm();
}

/**
 * @brief Compute relative L2 discrete error using spectral projection of reference solution
 *
 * relative_L2 = ||numerical - projected(reference)||_2 / ||projected(reference)||_2
 */
template <typename T>
double compute_relative_L2_discrete_error_projection(
    const TensorNDBase<T> &numerical_sol,
    const TensorNDBase<T> &reference_sol,
    int number_of_threads = 1)
{
    static_assert(std::is_same<T, std::complex<double>>::value,
                  "compute_relative_L2_discrete_error_projection currently supports T = std::complex<double> only");

    TensorNDBase<T> ref_projected;
    TensorNDBase<T> diff =
        build_difference_tensor_on_numerical_grid_projection(numerical_sol, reference_sol, number_of_threads, &ref_projected);

    const double denom = ref_projected.get_L2_discrete_norm();
    if (denom == 0.0)
        throw std::runtime_error("Relative L2 error undefined: projected reference norm is zero");

    return diff.get_L2_discrete_norm() / denom;
}

/**
 * @brief Compute relative L1 discrete error using spectral projection of reference solution
 *
 * relative_L1 = ||numerical - projected(reference)||_1 / ||projected(reference)||_1
 */
template <typename T>
double compute_relative_L1_discrete_error_projection(
    const TensorNDBase<T> &numerical_sol,
    const TensorNDBase<T> &reference_sol,
    int number_of_threads = 1)
{
    static_assert(std::is_same<T, std::complex<double>>::value,
                  "compute_relative_L1_discrete_error_projection currently supports T = std::complex<double> only");

    TensorNDBase<T> ref_projected;
    TensorNDBase<T> diff =
        build_difference_tensor_on_numerical_grid_projection(numerical_sol, reference_sol, number_of_threads, &ref_projected);

    const double denom = ref_projected.get_L1_discrete_norm();
    if (denom == 0.0)
        throw std::runtime_error("Relative L1 error undefined: projected reference norm is zero");

    return diff.get_L1_discrete_norm() / denom;
}

/**
 * @brief Compute relative infinity-norm error using spectral projection of reference solution
 *
 * relative_inf = ||numerical - projected(reference)||_inf / ||projected(reference)||_inf
 */
template <typename T>
double compute_relative_inf_discrete_error_projection(
    const TensorNDBase<T> &numerical_sol,
    const TensorNDBase<T> &reference_sol,
    int number_of_threads = 1)
{
    static_assert(std::is_same<T, std::complex<double>>::value,
                  "compute_relative_inf_discrete_error_projection currently supports T = std::complex<double> only");

    TensorNDBase<T> ref_projected;
    TensorNDBase<T> diff =
        build_difference_tensor_on_numerical_grid_projection(numerical_sol, reference_sol, number_of_threads, &ref_projected);

    const double denom = ref_projected.get_inf_norm();
    if (denom == 0.0)
        throw std::runtime_error("Relative infinity error undefined: projected reference norm is zero");

    return diff.get_inf_norm() / denom;
}

/**
 * @brief Compute relative L2 discrete error on numerical grid
 *
 * relative_L2 = ||numerical - sampled(reference)||_2 / ||sampled(reference)||_2
 */
template <typename T>
double compute_relative_L2_discrete_error(
    const TensorNDBase<T> &numerical_sol,
    const TensorNDBase<T> &reference_sol,
    int number_of_threads = 1)
{
    TensorNDBase<T> ref_sampled;
    TensorNDBase<T> diff =
        build_difference_tensor_on_numerical_grid(numerical_sol, reference_sol, number_of_threads, &ref_sampled);

    const double denom = ref_sampled.get_L2_discrete_norm();
    if (denom == 0.0)
        throw std::runtime_error("Relative L2 error undefined: sampled reference norm is zero");

    return diff.get_L2_discrete_norm() / denom;
}

/**
 * @brief Compute relative L1 discrete error on numerical grid
 *
 * relative_L1 = ||numerical - sampled(reference)||_1 / ||sampled(reference)||_1
 */
template <typename T>
double compute_relative_L1_discrete_error(
    const TensorNDBase<T> &numerical_sol,
    const TensorNDBase<T> &reference_sol,
    int number_of_threads = 1)
{
    TensorNDBase<T> ref_sampled;
    TensorNDBase<T> diff =
        build_difference_tensor_on_numerical_grid(numerical_sol, reference_sol, number_of_threads, &ref_sampled);

    const double denom = ref_sampled.get_L1_discrete_norm();
    if (denom == 0.0)
        throw std::runtime_error("Relative L1 error undefined: sampled reference norm is zero");

    return diff.get_L1_discrete_norm() / denom;
}

/**
 * @brief Compute relative infinity-norm error on numerical grid
 *
 * relative_inf = ||numerical - sampled(reference)||_inf / ||sampled(reference)||_inf
 */
template <typename T>
double compute_relative_inf_discrete_error(
    const TensorNDBase<T> &numerical_sol,
    const TensorNDBase<T> &reference_sol,
    int number_of_threads = 1)
{
    TensorNDBase<T> ref_sampled;
    TensorNDBase<T> diff =
        build_difference_tensor_on_numerical_grid(numerical_sol, reference_sol, number_of_threads, &ref_sampled);

    const double denom = ref_sampled.get_inf_norm();
    if (denom == 0.0)
        throw std::runtime_error("Relative infinity error undefined: sampled reference norm is zero");

    return diff.get_inf_norm() / denom;
}

/**
 * @brief Compute the L2 discrete error between numerical solution and reference solution
 *
 * @tparam T Data type
 * @param numerical_sol Numerical solution tensor
 * @param reference_sol Reference solution tensor
 * @param number_of_threads Number of threads to use for computation, default is 1 (no parallelism)
 * @return double The L2 discrete error
 */
template <typename T>
double compute_L2_discrete_error(const TensorNDBase<T> &numerical_sol, const TensorNDBase<T> &reference_sol, int number_of_threads = 1)
{
    TensorNDBase<T> diff = build_difference_tensor_on_numerical_grid(numerical_sol, reference_sol, number_of_threads);
    return diff.get_L2_discrete_norm();
}

/**
 * @brief Compute the L1 discrete error between numerical solution and reference solution
 *
 * @tparam T Data type
 * @param numerical_sol Numerical solution tensor
 * @param reference_sol Reference solution tensor
 * @param number_of_threads Number of threads to use for computation, default is 1 (no parallelism)
 * @return double The L1 discrete error
 */
template <typename T>
double compute_L1_discrete_error(const TensorNDBase<T> &numerical_sol, const TensorNDBase<T> &reference_sol, int number_of_threads = 1)
{
    TensorNDBase<T> diff = build_difference_tensor_on_numerical_grid(numerical_sol, reference_sol, number_of_threads);
    return diff.get_L1_discrete_norm();
}

/**
 * @brief Compute the infinity discrete error between numerical solution and reference solution
 *
 * @tparam T Data type
 * @param numerical_sol Numerical solution tensor
 * @param reference_sol Reference solution tensor
 * @param number_of_threads Number of threads to use for computation, default is 1 (no parallelism)
 * @return double The infinity discrete error
 */
template <typename T>
double compute_inf_discrete_error(const TensorNDBase<T> &numerical_sol, const TensorNDBase<T> &reference_sol, int number_of_threads = 1)
{
    TensorNDBase<T> diff = build_difference_tensor_on_numerical_grid(numerical_sol, reference_sol, number_of_threads);
    return diff.get_inf_norm();
}

#endif // FREE_FUNCTION_TENSOR_H