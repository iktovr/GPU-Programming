#include <vector>
#include <iostream>

#include "../../common/error_checkers.hpp"
#include "../utils.hpp"
#include "../bitonic_sort.hpp"

std::vector<int> bitonic_sort(std::vector<int> data) {
    int *dev_data;
    cudaCheck(cudaMalloc(&dev_data, sizeof(int) * data.size()));
    cudaCheck(cudaMemcpy(dev_data, data.data(), sizeof(int) * data.size(), cudaMemcpyHostToDevice));

    bitonic_sort(dev_data, data.size(), 0x7FFFFFFF);

    cudaCheck(cudaMemcpy(data.data(), dev_data, sizeof(int) * data.size(), cudaMemcpyDeviceToHost));
    cudaCheck(cudaFree(dev_data));
    return data;
}

int main() {
	int n;
	std::cin >> n;
	std::vector<int> data(n);
	for (int i = 0; i < n; ++i) {
		std::cin >> data[i];
	}

    std::vector<int> res = bitonic_sort(data);
    for (int i = 0; i < n; ++i) {
		std::cout << res[i] << ' ';
	}
    std::cout << '\n';
}