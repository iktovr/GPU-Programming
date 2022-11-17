#include <vector>
#include <iostream>

#include "../../common/error_checkers.hpp"
#include "../utils.hpp"
#include "../scan.hpp"

template <class T> 
__device__ inline T add_func(T x, T y) {
    return x + y;
}

__device__ func_pointer<long long> dev_add_func = add_func<long long>;

template <class T>
std::vector<T> scan(const std::vector<T>& data) {
    T *dev_data;
    cudaCheck(cudaMalloc(&dev_data, sizeof(T) * data.size()));
    cudaCheck(cudaMemcpy(dev_data, data.data(), sizeof(T) * data.size(), cudaMemcpyHostToDevice));

    func_pointer<T> h_add_func;
    cudaCheck(cudaMemcpyFromSymbol(&h_add_func, dev_add_func, sizeof(func_pointer<T>)));

    scan(dev_data, data.size(), h_add_func);

    std::vector<T> res(data.size());
    cudaCheck(cudaMemcpy(res.data(), dev_data, sizeof(T) * res.size(), cudaMemcpyDeviceToHost));
    cudaCheck(cudaFree(dev_data));
    return res;    
}

int main() {
	int n;
	std::cin >> n;
	std::vector<long long> data(n);
	for (int i = 0; i < n; ++i) {
		std::cin >> data[i];
	}

    std::vector<long long> pr_data = scan(data);
    for (int i = 0; i < n; ++i) {
		std::cout << pr_data[i] << ' ';
	}
    std::cout << '\n';
}