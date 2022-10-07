#pragma once

#include "error_checkers.hpp"

#define cudaStartTimer() \
	cudaEvent_t __start, __end; \
	cudaCheck(cudaEventCreate(&__start)); \
	cudaCheck(cudaEventCreate(&__end)); \
	cudaCheck(cudaEventRecord(__start)); \

#define cudaEndTimer(time) \
	cudaCheck(cudaEventRecord(__end)); \
	cudaCheck(cudaEventSynchronize(__end)); \
	cudaCheck(cudaEventElapsedTime(&time, __start, __end)); \
	cudaCheck(cudaEventDestroy(__start)); \
	cudaCheck(cudaEventDestroy(__end)); \

