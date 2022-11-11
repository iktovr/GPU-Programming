#pragma once

template <class T>
inline T get_block_count(T size, T block_size, T log_block_size) {
	return (size >> (log_block_size + 1)) + ((size & ((block_size << 1) - 1)) > 0);
}

template <class T>
T log2(T a) {
	T log = 0;
	while (a > 1) {
		++log;
		a >>= 1;
	}
	return log;
}

template <class T>
T ceil_2_pow(T a) {
    if ((a & (a - 1)) == 0) {
        return a;
    }
	while ((a & (a - 1)) != 0) {
		a &= a - 1;
	}
	return a << 1;
}


#define sh_index(i) ((i) + ((i) >> 5))

size_t BLOCK_SIZE = 1024;
size_t BLOCK_COUNT = 1024;

template <class T>
using func_pointer = T (*) (T, T);

template <class T> 
__device__ inline T min_func(T x, T y) {
    return (x < y) ? x : y;
}

template <class T> 
__device__ inline T max_func(T x, T y) {
    return (x > y) ? x : y;
}

#define bucket_sort_device_functions(type) \
__device__ func_pointer<type> dev_bucket_sort_min_func = min_func<type>; \
__device__ func_pointer<type> dev_bucket_sort_max_func = max_func<type>; \

