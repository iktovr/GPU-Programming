#pragma once

#include <vector>

#include "../common/error_checkers.hpp"
#include "utils.hpp"

template <class T>
__global__ void reduce(T* idata, int size, T* odata, func_pointer<T> func) {
	extern __shared__ T sdata[];

	int tid = threadIdx.x;
	int id = blockDim.x * 2 * blockIdx.x + threadIdx.x;
	int offset = blockDim.x * 2 * gridDim.x;
	int blockId = blockIdx.x;

	while (id < size) {
		sdata[tid] = func(idata[id], idata[id + blockDim.x]);
		__syncthreads();
		for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
			if (tid < s) {
				sdata[tid] = func(sdata[tid], sdata[tid + s]);
			}
			__syncthreads();
		}

		if (tid == 0) {
			odata[blockId] = sdata[0];
		}
		id += offset;
		blockId += gridDim.x;
	}
}

template <class T>
T reduce(T *dev_data, size_t size, func_pointer<T> func, T identity) {
	T res;
	if (size == 1) {
		cudaCheck(cudaMemcpy(&res, dev_data, sizeof(T), cudaMemcpyDeviceToHost));
		return res;
	}

	size_t pad_data_size = ceil_2_pow(size);
	std::vector<T> fill(pad_data_size - size, identity);
	size_t log_block_size = log2(BLOCK_SIZE);
	size_t res_size = get_block_count(pad_data_size, BLOCK_SIZE, log_block_size);

	T *pad_dev_data, *dev_res;
	cudaCheck(cudaMalloc(&pad_dev_data, sizeof(T) * pad_data_size));
	cudaCheck(cudaMemcpy(pad_dev_data, dev_data, sizeof(T) * size, cudaMemcpyDeviceToDevice));
	cudaCheck(cudaMemcpy(pad_dev_data + size, fill.data(), sizeof(T) * fill.size(), cudaMemcpyHostToDevice));
	cudaCheck(cudaMalloc(&dev_res, sizeof(T) * res_size));

	while (res_size > 1) {
		reduce<<<min(res_size, BLOCK_COUNT), BLOCK_SIZE, sizeof(T) * BLOCK_SIZE>>>(pad_dev_data, pad_data_size, dev_res, func);
		cudaCheck(cudaDeviceSynchronize());
		cudaCheckLastError();
		std::swap(pad_dev_data, dev_res);
		pad_data_size = res_size;
		res_size = get_block_count(pad_data_size, BLOCK_SIZE, log_block_size);
	}

	reduce<<<1, (pad_data_size >> 1), sizeof(T) * (pad_data_size >> 1)>>>(pad_dev_data, pad_data_size, dev_res, func);
	cudaCheck(cudaDeviceSynchronize());
	cudaCheckLastError();

	cudaCheck(cudaMemcpy(&res, dev_res, sizeof(T), cudaMemcpyDeviceToHost));
	cudaCheck(cudaFree(pad_dev_data));
	cudaCheck(cudaFree(dev_res));
	return res;
}