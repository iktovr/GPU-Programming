#pragma once

#include <vector>

#include "../common/error_checkers.hpp"
#include "utils.hpp"

template <class T>
__device__ void bitonic_merge(int i, T *data, int m, int start_b) {
    int k;
    T tmp;
    for (int b = start_b; b >= 2; b >>= 1) {
        if ((i & (b - 1)) < (b >> 1)) {
            k = i + (b >> 1);
            if (((i & m) && (data[i] < data[k])) || 
                (!(i & m) && (data[i] > data[k]))) {
                tmp = data[i];
                data[i] = data[k];
                data[k] = tmp;
            }
        }
        __syncthreads();
    }
}

template <class T>
__global__ void bitonic_sort_shared_memory(T *data, int size) {
    extern __shared__ T sdata[];

    int tid = threadIdx.x;
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int offset = blockDim.x * gridDim.x;
    int max_m = (size < BLOCK_SIZE) ? size : BLOCK_SIZE;

    while (id < size) {
        sdata[tid] = data[id];
		__syncthreads();

        for (int m = 2; m <= max_m; m <<= 1) {
            bitonic_merge(tid, sdata, m, m);
        }

        data[id] = sdata[tid];
		
        id += offset;
    }
}

template <class T>
__global__ void bitonic_sort_shared_memory(T *data, int size, int m) {
    extern __shared__ T sdata[];

    int tid = threadIdx.x;
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int offset = blockDim.x * gridDim.x;

	while (id < size) {
		sdata[tid] = data[id];
		__syncthreads();

        bitonic_merge(tid, sdata, m, BLOCK_SIZE);

        data[id] = sdata[tid];
		
        id += offset;
	}
}

template <class T>
__global__ void bitonic_sort_global_memory(T *data, int size, int m, int b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;
    int k;
    T tmp;

    while (i < size) {
        if ((i & (b - 1)) < (b >> 1)) {
            k = i + (b >> 1);
            if (((i & m) && (data[i] < data[k])) || 
                (!(i & m) && (data[i] > data[k]))) {
                tmp = data[i];
                data[i] = data[k];
                data[k] = tmp;
            }
        }

        i += offset;
    }
}

template <class T>
void bitonic_sort(T *dev_data, size_t size, T fill_value) {
    size_t pad_data_size = ceil_2_pow(size);
    std::vector<T> fill(pad_data_size - size, fill_value);
    T *pad_dev_data;
    cudaCheck(cudaMalloc(&pad_dev_data, sizeof(T) * pad_data_size));
    cudaCheck(cudaMemcpy(pad_dev_data, dev_data, sizeof(T) * size, cudaMemcpyDeviceToDevice));
    cudaCheck(cudaMemcpy(pad_dev_data + size, fill.data(), sizeof(T) * fill.size(), cudaMemcpyHostToDevice));

    bitonic_sort_shared_memory<<<BLOCK_COUNT, BLOCK_SIZE, sizeof(T) * BLOCK_SIZE>>>(pad_dev_data, pad_data_size);
    cudaCheck(cudaDeviceSynchronize());
    cudaCheckLastError();
    for (size_t m = BLOCK_SIZE << 1; m <= pad_data_size; m <<= 1) {
        for (size_t b = m; b > BLOCK_SIZE; b >>= 1) {
            bitonic_sort_global_memory<<<BLOCK_COUNT, BLOCK_SIZE>>>(pad_dev_data, pad_data_size, m, b);
            cudaCheck(cudaDeviceSynchronize());
            cudaCheckLastError();
        }
        bitonic_sort_shared_memory<<<BLOCK_COUNT, BLOCK_SIZE, sizeof(T) * BLOCK_SIZE>>>(pad_dev_data, pad_data_size, m);
        cudaCheck(cudaDeviceSynchronize());
        cudaCheckLastError();
    }

    cudaCheck(cudaMemcpy(dev_data, pad_dev_data, sizeof(T) * size, cudaMemcpyDeviceToDevice));
    cudaCheck(cudaFree(pad_dev_data));
}