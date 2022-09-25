#include "error_checkers.hpp"

#define cudaStartTimer() \
	cudaEvent_t __start, __end; \
	cudaCheck(cudaEventCreate(&__start)); \
	cudaCheck(cudaEventCreate(&__end)); \
	cudaCheck(cudaEventRecord(&__start)); \

#define cudaEndTimer(time) \
	cudaCheck(cudaEventRecord(__stop)); \
	cudaCheck(cudaEventSynchronize(__stop)); \
	cudaCheck(cudaEventElapsedTime(&time, __start, __stop)); \
	cudaCheck(cudaEventDestroy(__start)); \
	cudaCheck(cudaEventDestroy(__stop)); \

