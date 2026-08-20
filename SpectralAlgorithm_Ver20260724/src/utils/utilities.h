//
//  utilities.h
//
//
//  This file contains all the includes , types, and all global constants and functions
//



#ifndef _utilities_h
#define _utilities_h

#include <string>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <stdexcept>

using namespace std;

/**
 * @brief Get the cpu time sec object
 * 
 * @return double 
 */
double get_cpu_time_sec();

/**
 * @brief Get the CPU time in hours
 * 
 * @return double CPU time in hours
 */
double get_cpu_time_hour();

/**
 * @brief Get the current local time as a string
 * 
 * @return string Local time string
 */
string get_local_time();

/**
 * @brief Compute the modulo of two double values
 * 
 * @param a Dividend
 * @param b Divisor
 * @return double Result of a mod b
 */
double modulo_of_double(double a, double b);

/**
 * @brief Generate a random double in the range [min, max)
 * 
 * @param min Minimum value (inclusive)
 * @param max Maximum value (exclusive)
 * @return double Random double in the specified range
 */
double rand_min_max(double min, double max);

/**
 * @brief Get the fft index from math object
 * 
 * @param k math index
 * @param N Fourier series truncation parameter
 * @return size_t 
 */
inline size_t get_fft_index_from_math_global(int k, int N)
{
    const int size = 2 * N;

    // FAST PATH: k trong [-N, N-1]
    if (k >= -N && k < N) {
        return static_cast<size_t>(k >= 0 ? k : k + size);
    }

    // SLOW PATH: k vượt miền (cả dương lẫn âm)
    int r = k % size;
    if (r < 0) r += size;

    return static_cast<size_t>(r);
}

#endif
