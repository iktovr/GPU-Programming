#include <vector>
#include <iostream>
#include <limits> 

#include "../../common/error_checkers.hpp"
#include "../utils.hpp"
#include "../bitonic_sort.hpp"

// #define TIME
#ifdef TIME
#include "../../common/cuda_timer.hpp"
#endif

template <class T>
void bitonic_sort(std::vector<T>& data) {
    T *dev_data;
    cudaCheck(cudaMalloc(&dev_data, sizeof(T) * data.size()));
    cudaCheck(cudaMemcpy(dev_data, data.data(), sizeof(T) * data.size(), cudaMemcpyHostToDevice));

#ifdef TIME
    cudaStartTimer();
#endif

    bitonic_sort(dev_data, data.size(), std::numeric_limits<T>::max());

#ifdef TIME
    float t;
    cudaEndTimer(t);
    std::cout << t;
#endif

    cudaCheck(cudaMemcpy(data.data(), dev_data, sizeof(T) * data.size(), cudaMemcpyDeviceToHost));
    cudaCheck(cudaFree(dev_data));
}

int main() {
	int n;
	std::cin >> n;
	std::vector<long long> data(n);
	for (int i = 0; i < n; ++i) {
		std::cin >> data[i];
	}

    bitonic_sort(data);

    #ifndef TIME
    for (int i = 0; i < n; ++i) {
		std::cout << data[i] << ' ';
	}
    std::cout << '\n';
    #endif
    return 0;
}