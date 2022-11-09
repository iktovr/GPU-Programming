#include <vector>
#include <iostream>

#include "../common/error_checkers.hpp"

__global__ void hist(int *data, int size, int *hist) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;

    while (id < size) {
        atomicAdd(hist + data[id], 1);

        id += offset;
    }
}

std::vector<int> hist(const std::vector<int>& data, int hist_size) {
    int *dev_data, *dev_hist;
    cudaCheck(cudaMalloc(&dev_data, sizeof(int) * data.size()));
    cudaCheck(cudaMalloc(&dev_hist, sizeof(int) * hist_size));
    cudaCheck(cudaMemcpy(dev_data, data.data(), sizeof(int) * data.size(), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(dev_hist, 0, sizeof(int) * hist_size));

    hist<<<1024, 1024>>>(dev_data, data.size(), dev_hist);
    cudaCheck(cudaDeviceSynchronize());
    cudaCheckLastError();

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

    std::vector<int> res = hist(data, 32);
    for (size_t i = 0; i < res.size(); ++i) {
		std::cout << res[i] << ' ';
	}
    std::cout << '\n';
}