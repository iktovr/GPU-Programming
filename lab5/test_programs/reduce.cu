#include <vector>
#include <iostream>
#include <limits>

#include "../../common/error_checkers.hpp"
#include "../utils.hpp"
#include "../reduce.hpp"

__device__ func_pointer<long long> dev_max_func = max_func<long long>;
__device__ func_pointer<long long> dev_min_func = min_func<long long>;

template <class T>
T reduce_max(const std::vector<T>& data) {
	T *dev_data;
	cudaCheck(cudaMalloc(&dev_data, sizeof(T) * data.size()));
	cudaCheck(cudaMemcpy(dev_data, data.data(), sizeof(T) * data.size(), cudaMemcpyHostToDevice));

	func_pointer<T> h_max_func;
	cudaCheck(cudaMemcpyFromSymbol(&h_max_func, dev_max_func, sizeof(func_pointer<T>)));

	T res = reduce(dev_data, data.size(), h_max_func, std::numeric_limits<T>::lowest());
	cudaCheck(cudaFree(dev_data));
	return res;
}

template <class T>
T reduce_min(const std::vector<T>& data) {
	T *dev_data;
	cudaCheck(cudaMalloc(&dev_data, sizeof(T) * data.size()));
	cudaCheck(cudaMemcpy(dev_data, data.data(), sizeof(T) * data.size(), cudaMemcpyHostToDevice));

	func_pointer<T> h_min_func;
	cudaCheck(cudaMemcpyFromSymbol(&h_min_func, dev_min_func, sizeof(func_pointer<T>)));

	T res = reduce(dev_data, data.size(), h_min_func, std::numeric_limits<T>::max());
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

	std::cout << reduce_min(data) << ' ' << reduce_max(data) << '\n';
}