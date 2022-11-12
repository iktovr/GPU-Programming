#include <vector>
#include <limits>

#define CHECKER
#ifdef CHECKER
#include <cstdio>
#else
#include <iostream>
#endif

// #define TIME
#ifdef TIME
#include "../common/cuda_timer.hpp"
#endif

#include "../common/error_checkers.hpp"
#include "utils.hpp"
bucket_sort_device_functions(float);
#include "bucket_sort.hpp"

template <class T>
void bucket_sort(std::vector<T>& data) {
    T *dev_data;
    cudaCheck(cudaMalloc(&dev_data, sizeof(T) * data.size()));
    cudaCheck(cudaMemcpy(dev_data, data.data(), sizeof(T) * data.size(), cudaMemcpyHostToDevice));

#ifdef TIME
    cudaStartTimer();
#endif

    bucket_sort(dev_data, data.size(), std::numeric_limits<T>::lowest(), std::numeric_limits<T>::max());

#ifdef TIME
    float t;
    cudaEndTimer(t);
    std::cout << t;
#endif

    cudaCheck(cudaMemcpy(data.data(), dev_data, sizeof(T) * data.size(), cudaMemcpyDeviceToHost));
    cudaCheck(cudaFree(dev_data));
}

int main() {
	uint32_t n;
#ifdef CHECKER
    fread(&n, sizeof(n), 1, stdin);
#else
	std::cin >> n;
#endif
    if (n == 0) {
        return 0;
    }
	std::vector<float> data(n);

#ifdef CHECKER
    std::fread(data.data(), sizeof(float), n, stdin);
#else
	for (int i = 0; i < n; ++i) {
		std::cin >> data[i];
	}
#endif

    bucket_sort(data);

#ifndef TIME
#ifdef CHECKER
    std::fwrite(data.data(), sizeof(float), n, stdout);
#else
    for (int i = 0; i < n; ++i) {
		std::cout << data[i] << ' ';
	}
    std::cout << '\n';
#endif
#endif
    return 0;
}