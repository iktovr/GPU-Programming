#include <vector>
#include <iostream>
// #include <algorithm>

#include "../../common/error_checkers.hpp"
#include "../hist.hpp"

template <class T>
T max(std::vector<T> v) {
    T m = v[0];
    for (T a: v) {
        if (a > m) {
            m = a;
        }
    }
    return m;
}

std::vector<int> histogram(const std::vector<int>& data) {
    int *dev_data, *dev_hist;
    cudaCheck(cudaMalloc(&dev_data, sizeof(int) * data.size()));
    cudaCheck(cudaMemcpy(dev_data, data.data(), sizeof(int) * data.size(), cudaMemcpyHostToDevice));

    int hist_size = max(data) + 1;
    dev_hist = histogram<int, int>(dev_data, data.size(), hist_size);

    std::vector<int> hist(hist_size);
    cudaCheck(cudaMemcpy(hist.data(), dev_hist, sizeof(int) * hist_size, cudaMemcpyDeviceToHost));
    cudaCheck(cudaFree(dev_data));
    cudaCheck(cudaFree(dev_hist));
    return hist;
}

int main() {
	int n;
	std::cin >> n;
	std::vector<int> data(n);
	for (int i = 0; i < n; ++i) {
		std::cin >> data[i];
	}

    std::vector<int> res = histogram(data);
    for (size_t i = 0; i < res.size(); ++i) {
		std::cout << res[i] << ' ';
	}
    std::cout << '\n';
}