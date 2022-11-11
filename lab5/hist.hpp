#pragma once

#include "../common/error_checkers.hpp"
#include "utils.hpp"

template <class T, class U>
__global__ void histogram(T *data, int size, U *hist) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;

    while (id < size) {
        atomicAdd(hist + data[id], 1);

        id += offset;
    }
}

template <class T, class U>
U* histogram(T *dev_data, size_t size, size_t hist_size) {
    U *dev_hist;
    cudaCheck(cudaMalloc(&dev_hist, sizeof(U) * hist_size));
    cudaCheck(cudaMemset(dev_hist, static_cast<U>(0), sizeof(U) * hist_size));

    histogram<<<BLOCK_COUNT, BLOCK_SIZE>>>(dev_data, size, dev_hist);
    cudaCheck(cudaDeviceSynchronize());
    cudaCheckLastError();

    return dev_hist;
}