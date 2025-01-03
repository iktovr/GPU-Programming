#pragma once

#include <vector>

#include "../common/error_checkers.hpp"
#include "utils.hpp"

template <class T>
__device__ void bitonic_merge(int i, int id, T *data, int m, int start_b) {
    int k;
    T tmp;
    for (int b = start_b; b >= 2; b >>= 1) {
        if ((i & (b - 1)) < (b >> 1)) {
            k = i + (b >> 1);
            if (((id & m) && (data[i] < data[k])) || 
                (!(id & m) && (data[i] > data[k]))) {
                tmp = data[i];
                data[i] = data[k];
                data[k] = tmp;
            }
        }
        __syncthreads();
    }
}

template <class T>
__global__ void bitonic_sort_shared_memory(T *data, int size, int real_size, T fill_value) {
    extern __shared__ uint8_t shared_memory[];
    T* sdata = (T*)shared_memory;

    int tid = threadIdx.x;
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int offset = blockDim.x * gridDim.x;
    int max_m = (size < blockDim.x) ? size : blockDim.x;

    sdata[tid] = fill_value;

    while (id < size) {
        if (id < real_size) {
            sdata[tid] = data[id];
        }
		__syncthreads();

        for (int m = 2; m <= max_m; m <<= 1) {
            bitonic_merge(tid, id, sdata, m, m);
        }

        if (id < real_size) {
            data[id] = sdata[tid];
        }
		
        id += offset;
    }
}

template <class T>
__global__ void bitonic_sort_merge_shared_memory(T *data, int size, int m) {
    extern __shared__ uint8_t shared_memory[];
    T* sdata = (T*)shared_memory;

    int tid = threadIdx.x;
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int offset = blockDim.x * gridDim.x;

	while (id < size) {
		sdata[tid] = data[id];
		__syncthreads();

        bitonic_merge(tid, id, sdata, m, blockDim.x);

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
    if (size == 0) {
        return;
    }

    size_t pad_data_size = ceil_2_pow(size);
    size_t log_block_size = log2(BLOCK_SIZE);
    size_t block_count = get_block_count<false>(pad_data_size, BLOCK_SIZE, log_block_size);
    T *pad_dev_data = dev_data;
    
    if (block_count > 1) {
        std::vector<T> fill(pad_data_size - size, fill_value);
        cudaCheck(cudaMalloc(&pad_dev_data, sizeof(T) * pad_data_size));
        cudaCheck(cudaMemcpy(pad_dev_data, dev_data, sizeof(T) * size, cudaMemcpyDeviceToDevice));
        cudaCheck(cudaMemcpy(pad_dev_data + size, fill.data(), sizeof(T) * fill.size(), cudaMemcpyHostToDevice));
        bitonic_sort_shared_memory<<<min(block_count, BLOCK_COUNT), BLOCK_SIZE, sizeof(T) * BLOCK_SIZE>>>(pad_dev_data, pad_data_size, pad_data_size, fill_value);
    } else {
        bitonic_sort_shared_memory<<<min(block_count, BLOCK_COUNT), BLOCK_SIZE, sizeof(T) * BLOCK_SIZE>>>(pad_dev_data, pad_data_size, size, fill_value);
    }
    cudaCheck(cudaDeviceSynchronize());
    cudaCheckLastError();

    for (size_t m = BLOCK_SIZE << 1; m <= pad_data_size; m <<= 1) {
        for (size_t b = m; b > BLOCK_SIZE; b >>= 1) {
            bitonic_sort_global_memory<<<min(block_count, BLOCK_COUNT), BLOCK_SIZE>>>(pad_dev_data, pad_data_size, m, b);
            cudaCheck(cudaDeviceSynchronize());
            cudaCheckLastError();
        }
        bitonic_sort_merge_shared_memory<<<min(block_count, BLOCK_COUNT), BLOCK_SIZE, sizeof(T) * BLOCK_SIZE>>>(pad_dev_data, pad_data_size, m);
        cudaCheck(cudaDeviceSynchronize());
        cudaCheckLastError();
    }

    if (block_count > 1) {
        cudaCheck(cudaMemcpy(dev_data, pad_dev_data, sizeof(T) * size, cudaMemcpyDeviceToDevice));
        cudaCheck(cudaFree(pad_dev_data));
    }
}