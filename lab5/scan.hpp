#pragma once

#include <vector>

#include "../common/error_checkers.hpp"
#include "utils.hpp"

template <class T>
__global__ void scan(T* data, func_pointer<T> func) {
    extern __shared__ T sdata[];

    int tid = threadIdx.x;
    int block_size = blockDim.x << 1;

    sdata[sh_index(tid)] = data[tid];
    sdata[sh_index(tid + blockDim.x)] = data[tid + blockDim.x];
    __syncthreads();

    for (int s = 1; s < block_size; s <<= 1) {
        int i = (s << 1) * (tid + 1) - 1;
        if (i < block_size) {
            sdata[sh_index(i)] = func(sdata[sh_index(i)], sdata[sh_index(i - s)]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        sdata[sh_index(block_size - 1)] = 0;
    }
    __syncthreads();

    T tmp;
    for (int s = blockDim.x; s > 0; s >>= 1) {
        int i = (s << 1) * (tid + 1) - 1;
        if (i < block_size) {
            tmp = sdata[sh_index(i)];
            sdata[sh_index(i)] = func(sdata[sh_index(i)], sdata[sh_index(i - s)]);
            sdata[sh_index(i - s)] = tmp;
        }
        __syncthreads();
    }

    data[tid] = sdata[sh_index(tid)];
    data[tid + blockDim.x] = sdata[sh_index(tid + blockDim.x)];
}

template <class T>
__global__ void per_block_sum(T *data, T *blocks, int size) {
    int id = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    int bid = blockIdx.x;
    int offset = gridDim.x * blockDim.x * 2;

    while (id < size) {
        data[id] += blocks[bid];
        data[id + blockDim.x] += blocks[bid];

        id += offset;
        bid += gridDim.x;
    }
}

template <class T>
__global__ void scan(T *data, int size, T *block_sum, func_pointer<T> func) {
    extern __shared__ T sdata[];

    int tid = threadIdx.x;
    int block_size = blockDim.x << 1;
    int id = block_size * blockIdx.x + threadIdx.x;
    int offset = block_size * gridDim.x;
	int blockId = blockIdx.x;

    while (id < size) {
        sdata[sh_index(tid)] = data[id];
        sdata[sh_index(tid + blockDim.x)] = data[id + blockDim.x];
        __syncthreads();

        for (int s = 1; s < block_size; s <<= 1) {
            int i = (s << 1) * (tid + 1) - 1;
            if (i < block_size) {
                sdata[sh_index(i)] = func(sdata[sh_index(i)], sdata[sh_index(i - s)]);
            }
            __syncthreads();
        }

        if (tid == 0) {
            block_sum[blockId] = sdata[sh_index(block_size - 1)];
            sdata[sh_index(block_size - 1)] = 0;
        }
        __syncthreads();

        T tmp;
        for (int s = blockDim.x; s > 0; s >>= 1) {
            int i = (s << 1) * (tid + 1) - 1;
            if (i < block_size) {
                tmp = sdata[sh_index(i)];
                sdata[sh_index(i)] = func(sdata[sh_index(i)], sdata[sh_index(i - s)]);
                sdata[sh_index(i - s)] = tmp;
            }
            __syncthreads();
        }

        data[id] = sdata[sh_index(tid)];
        data[id + blockDim.x] = sdata[sh_index(tid + blockDim.x)];

        id += offset;
        blockId += gridDim.x;
    }
}

template <class T>
void full_scan(T *data, size_t size, func_pointer<T> func) {
    size_t log_block_size = log2(BLOCK_SIZE);
    size_t block_count = get_block_count(size, BLOCK_SIZE, log_block_size);

    if (block_count == 1) {
        scan<<<1, size >> 1, sizeof(T) * sh_index(size)>>>(data, func);
        cudaCheck(cudaDeviceSynchronize());
        cudaCheckLastError();
        return;
    }

    size_t block_sum_size = get_block_count(block_count, BLOCK_SIZE, log_block_size) * 2 * BLOCK_SIZE;
    T *block_sum, *block_sum_scan;
    cudaCheck(cudaMalloc(&block_sum, sizeof(T) * block_sum_size));
    cudaCheck(cudaMalloc(&block_sum_scan, sizeof(T) * block_sum_size));

    scan<<<min(block_count, BLOCK_COUNT), BLOCK_SIZE, sizeof(T) * sh_index(BLOCK_SIZE * 2)>>>(data, size, block_sum, func);
    cudaCheck(cudaDeviceSynchronize());
    cudaCheckLastError();
    full_scan(block_sum, block_sum_scan, block_sum_size);
    per_block_sum<<<min(block_count, BLOCK_COUNT), BLOCK_SIZE>>>(data, block_sum_scan, size);
    cudaCheck(cudaDeviceSynchronize());
    cudaCheckLastError();

    cudaCheck(cudaFree(block_sum));
    cudaCheck(cudaFree(block_sum_scan));
}

template <class T>
void scan(T *dev_data, size_t size, func_pointer<T> func) {
    size_t log_block_size = log2(BLOCK_SIZE);
    size_t block_count = get_block_count(size, BLOCK_SIZE, log_block_size);
    size_t pad_data_size = block_count * 2 * BLOCK_SIZE;

    T *pad_dev_data;
    cudaCheck(cudaMalloc(&pad_dev_data, sizeof(T) * pad_data_size));
    cudaCheck(cudaMemcpy(pad_dev_data, dev_data, sizeof(T) * size, cudaMemcpyDeiceToDevice));
    cudaCheck(cudaMemset(pad_dev_data + size, 0, sizeof(T) * (pad_data_size - size)));

    full_scan(pad_dev_data, pad_data_size, func);

    cudaCheck(cudaMemcpy(dev_data, pad_dev_data, sizeof(T) * size, cudaMemcpyDeviceToDevice));
    cudaCheck(cudaFree(pad_dev_data));    
}