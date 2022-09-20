#pragma once
#include <iostream>

#define checkCudaError(val) __checkCudaError((val), __FILE__, __LINE__)

inline void __checkCudaError(cudaError_t err, const char *const file, const int line) {
	if (err != cudaSuccess) {
		std::cerr << "ERROR: CUDA error " << static_cast<unsigned int>(err) << " (" << 
		             cudaGetErrorString(err) << ") at " << file << ":" << line << '\n';
		exit(0);
	}
}

#define checkLastCudaError() __checkCudaError(cudaGetLastError(), __FILE__, __LINE__)

#define checkError(value, msg, err_value) __checkError(value, err_value, msg, __FILE__, __LINE__)

template <class T>
inline void __checkError(const T value, const T err_value, const char *msg, const char *const file, const int line) {
	if (value == err_value) { 
		std::cerr << "ERROR: " << msg << " at " << file << ":" << line << '\n';
		exit(0); 
	}
}
