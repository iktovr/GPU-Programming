#pragma once

#include <vector>
#include <cmath>

#include "../common/error_checkers.hpp"
#include "utils.hpp"
#include "reduce.hpp"
#include "scan.hpp"
#include "hist.hpp"
#include "bitonic_sort.hpp"

template <class T> 
__device__ inline T add_func(T x, T y) {
    return x + y;
}

const int SPLIT_SIZE = 32;
const float SPLIT_COEF = 1. - 1e-6;
const float EPS = 1e-8;
using index_t = uint32_t;

__device__ func_pointer<index_t> dev_bucket_sort_add_func = add_func<index_t>;

template <class T>
__global__ void split(T *data, uint32_t *splits, int size, int n_splits, T min, T max) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;

    while (id < size) {
        splits[id] = SPLIT_COEF * (data[id] - min) / (max - min) * n_splits;

        id += offset;
    }
}

template<class T>
__global__ void group(T *data, T *res, int size, index_t *splits, index_t *hist) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;

    while (id < size) {
        res[atomicAdd(hist + splits[id], 1)] = data[id];

        id += offset;
    }
}

template <class T>
void bucket_sort(T *data, size_t size, T min_value, T max_value) {
    func_pointer<T> h_min_func, h_max_func;
    cudaCheck(cudaMemcpyFromSymbol(&h_min_func, dev_bucket_sort_min_func, sizeof(func_pointer<T>)));
    cudaCheck(cudaMemcpyFromSymbol(&h_max_func, dev_bucket_sort_max_func, sizeof(func_pointer<T>)));
    func_pointer<index_t> h_add_func;
    cudaCheck(cudaMemcpyFromSymbol(&h_add_func, dev_bucket_sort_add_func, sizeof(func_pointer<index_t>)));

    T min = reduce(data, size, h_min_func, max_value);
    T max = reduce(data, size, h_max_func, min_value);
    if (std::abs(min - max) < EPS) {
        return;
    }

    size_t n_splits = (size - 1) / SPLIT_SIZE + 1;
    index_t *splits;
    cudaCheck(cudaMalloc(&splits, sizeof(index_t) * size));
    split<<<BLOCK_COUNT, BLOCK_SIZE>>>(data, splits, size, n_splits, min, max);
    cudaCheck(cudaDeviceSynchronize());
    cudaCheckLastError();

    index_t *hist = histogram<index_t, index_t>(splits, size, n_splits);
    scan(hist, n_splits, h_add_func);

    std::vector<index_t> groups(n_splits + 1);
    cudaCheck(cudaMemcpy(groups.data(), hist, sizeof(index_t) * n_splits, cudaMemcpyDeviceToHost));
    groups[n_splits] = size;

    T *group_data;
    cudaCheck(cudaMalloc(&group_data, sizeof(T) * size));
    group<<<BLOCK_COUNT, BLOCK_SIZE>>>(data, group_data, size, splits, hist);
    cudaCheck(cudaDeviceSynchronize());
    cudaCheckLastError();

    size_t len, start;
    for (size_t i = 0; i < n_splits; ++i) {
        start = i;
        len = 0;
        while (i < n_splits && len + (groups[i + 1] - groups[i]) < BLOCK_SIZE) {
            len += groups[i + 1] - groups[i];
            ++i;
        }

        if (start == i) {
            bucket_sort(group_data + groups[start], groups[i + 1] - groups[i], min_value, max_value);
        } else {
            --i;
            bitonic_sort(group_data + groups[start], len, max_value);
        }
    }

    cudaCheck(cudaMemcpy(data, group_data, sizeof(T) * size, cudaMemcpyDeviceToDevice));

    cudaCheck(cudaFree(splits));
    cudaCheck(cudaFree(hist));
    cudaCheck(cudaFree(group_data));
}
