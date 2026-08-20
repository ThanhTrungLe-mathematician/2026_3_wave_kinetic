//
//	This file contains all global functions
//

#include "utilities.h"

double rand_min_max(double min, double max)
{
    // srand((unsigned)time(NULL));
    double rand_01 = (double)rand() / (RAND_MAX);
    return min + rand_01 * (max - min);
}

double modulo_of_double(double a, double b)
{
    return a - b * int(a / b);
}

double get_cpu_time_sec()
{
    return (double)clock() / (double)CLOCKS_PER_SEC;
}

double get_cpu_time_hour()
{
    return (double)clock() / (double)CLOCKS_PER_SEC / 3600.0;
}

string get_local_time()
{
    time_t now = time(0);
    string local_time = ctime(&now);
    if (!local_time.empty() && local_time.back() == '\n')
    {
        local_time.pop_back();
    }
    return local_time;
}
