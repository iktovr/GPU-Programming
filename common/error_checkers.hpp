#pragma once
#include <iostream>

#ifdef __CUDACC__

#define cudaCheck(val) __cudaCheck((val), __FILE__, __LINE__)

inline void __cudaCheck(cudaError_t err, const char *const file, const int line) {
	if (err != cudaSuccess) {
		std::cerr << "ERROR: CUDA error " << static_cast<unsigned int>(err) << " (" << 
		             cudaGetErrorString(err) << ") at " << file << ":" << line << '\n';
		exit(0);
	}
}

#define cudaCheckLastError() __cudaCheck(cudaGetLastError(), __FILE__, __LINE__)

#endif

#define check(value, err_value, msg) \
if ((value) == (err_value)) { \
	std::cerr << "ERROR: " << (msg) << " at " << __FILE__ << ":" << __LINE__ << '\n'; \
	exit(0); \
}
