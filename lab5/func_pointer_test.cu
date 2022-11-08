#include <iostream>

#include "../common/error_checkers.hpp"

template <class T>
using func_t = T (*) (T, T);

template <class T> 
__device__ inline T add_func (T x, T y)
{
    return x + y;
}

template <class T> 
__device__ inline T mul_func (T x, T y)
{
    return x * y;
}

// template <class T> 
__device__ func_t<int> p_add_func = add_func<int>;
// template <class T> 
// __device__ func_t<int> p_mul_func = mul_func<int>;

template <class T> 
__global__ void kernel(T* a, T* b, T* c, func_t<T> p) {
    (*c) = p(*a, *b);
}

int main() {
    int a, b, c;
    std::cin >> a >> b;

    func_t<int> h_add_func;
    int *dev_a, *dev_b, *dev_c;
    cudaCheck(cudaMalloc(&dev_a, sizeof(int)));
    cudaCheck(cudaMemcpy(dev_a, &a, sizeof(int), cudaMemcpyHostToDevice));
    cudaCheck(cudaMalloc(&dev_b, sizeof(int)));
    cudaCheck(cudaMemcpy(dev_b, &b, sizeof(int), cudaMemcpyHostToDevice));
    cudaCheck(cudaMalloc(&dev_c, sizeof(int)));
    cudaCheck(cudaMemcpy(dev_c, &c, sizeof(int), cudaMemcpyHostToDevice));

    cudaCheck(cudaMemcpyFromSymbol(&h_add_func, p_add_func, sizeof(func_t<int>)));

    kernel<int><<<1, 32>>>(dev_a, dev_b, dev_c, h_add_func);
    cudaCheck(cudaDeviceSynchronize());
    cudaCheckLastError();

    cudaCheck(cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost));
    std::cout << c << '\n';
    cudaCheck(cudaFree(dev_a));
    cudaCheck(cudaFree(dev_b));
    cudaCheck(cudaFree(dev_c));
}
