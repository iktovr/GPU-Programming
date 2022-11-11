#pragma once

#include "../common/error_checkers.hpp"
#include "utils.hpp"

template <class T> 
__device__ inline T min_func(T x, T y) {
    return (x < y) ? y : x;
}

template <class T> 
__device__ inline T max_func(T x, T y) {
    return (x > y) ? y : x;
}

#define bucket_sort_device_functions(type) \
__device__ func_pointer<type> dev_bucket_sort_min_func = min_func<type>; \
__device__ func_pointer<type> dev_bucket_sort_max_func = max_func<type>; \
