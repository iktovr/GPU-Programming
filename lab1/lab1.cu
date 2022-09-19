#include <iostream>
#include <vector>
#include <iomanip>

#include "../common/cuda_check.hpp"

__global__ void reverse(double *vec, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int offset = blockDim.x * gridDim.x;
	double p;
	while (idx <= n / 2 - 1) {
		p = vec[idx];
		vec[idx] = vec[n - 1 - idx];
		vec[n - 1 - idx] = p;

		idx += offset;
	}
}

int main() {
	std::ios::sync_with_stdio(false);
	int n;
	std::cin >> n;
	std::vector<double> vec(n);
	vec.shrink_to_fit();
	for (int i = 0; i < n; ++i) {
		std::cin >> vec[i];
	}

	double *dev_vec;
	checkCudaErrors(cudaMalloc(&dev_vec, sizeof(double) * n));
	checkCudaErrors(cudaMemcpy(dev_vec, vec.data(), sizeof(double) * n, cudaMemcpyDeviceToDevice));
	reverse<<<128, 128>>>(dev_vec, n);
	checkCudaErrors(cudaMemcpy(vec.data(), dev_vec, sizeof(double) * n, cudaMemcpyDeviceToHost));

	std::cout << std::setprecision(10) << std::fixed;
	for (int i = 0; i < n; ++i) {
		std::cout << vec[i] << ' ';
	}
	std::cout << '\n';

	checkCudaErrors(cudaFree(dev_vec));
}