#include <vector>
#include <iostream>

#include "../../common/error_checkers.hpp"
#include "../utils.hpp"
#include "../reduce.hpp"

template <class T> 
__device__ inline T add_func(T x, T y) {
    return x + y;
}

__device__ func_pointer<int> dev_add_func = add_func<int>;

int reduce(const std::vector<int>& data) {
	int *dev_data;
	cudaCheck(cudaMalloc(&dev_data, sizeof(int) * data.size()));
	cudaCheck(cudaMemcpy(dev_data, data.data(), sizeof(int) * data.size(), cudaMemcpyHostToDevice));

	func_pointer<int> h_add_func;
	cudaCheck(cudaMemcpyFromSymbol(&h_add_func, dev_add_func, sizeof(func_pointer<int)));

	int res = reduce(dev_data, data.size(), h_add_func, 0);
	cudaCheck(cudaFree(dev_data));
	return res;
}

int main() {
	int n;
	std::cin >> n;
	std::vector<int> data(n);
	for (int i = 0; i < n; ++i) {
		std::cin >> data[i];
	}

	std::cout << reduce(data) << '\n';
}