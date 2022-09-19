// Заимствовано из официальных примеров
// https://github.com/NVIDIA/cuda-samples/blob/master/Common/helper_cuda.h

#pragma once
#include <iostream>

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

template <typename T>
void check(T result, char const *const func, const char *const file, int const line) {
	if (result) {
		std::cerr << "CUDA error at " << file << ":" << line << " code=" << 
                     static_cast<unsigned int>(result) << " \"" << func << "\"\n";
		exit(0);
	}
}

#define getLastCudaError(msg) __getLastCudaError(msg, __FILE__, __LINE__)

inline void __getLastCudaError(const char *errorMessage, const char *file, const int line) {
	cudaError_t err = cudaGetLastError();

	if (err != cudaSuccess) {
        std::cerr << file << "(" << line << ") : getLastCudaError() CUDA error : " << errorMessage <<
                     " : (" << static_cast<int>(err) << ") " << cudaGetErrorString(err) << '\n';
		exit(0);
	}
}
