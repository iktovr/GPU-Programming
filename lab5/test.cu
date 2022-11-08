#include <iostream>

#include "../common/error_checkers.hpp"

__global__ void kernel(int* a, int* b, int* c, int (*p)(int*, int*)) {
    (*c) = p(a, b);
}

__device__ int Min(int* a, int* b) {
    return (*a < *b) ? *a : *b;
}

int main() {
    int a, b, c;
    std::cin >> a >> b;

    int *dev_a, *dev_b, *dev_c;
    cudaCheck(cudaMalloc(&dev_a, sizeof(int)));
    cudaCheck(cudaMemcpy(dev_a, &a, sizeof(int), cudaMemcpyHostToDevice));
    cudaCheck(cudaMalloc(&dev_b, sizeof(int)));
    cudaCheck(cudaMemcpy(dev_b, &b, sizeof(int), cudaMemcpyHostToDevice));
    cudaCheck(cudaMalloc(&dev_c, sizeof(int)));
    cudaCheck(cudaMemcpy(dev_c, &c, sizeof(int), cudaMemcpyHostToDevice));

    kernel<<<1, 32>>>(dev_a, dev_b, dev_c, &Min);
    cudaCheck(cudaDeviceSynchronize());
    cudaCheckLastError();

    cudaCheck(cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost));
    std::cout << c << '\n';
    cudaCheck(cudaFree(dev_a));
    cudaCheck(cudaFree(dev_b));
    cudaCheck(cudaFree(dev_c));
}
