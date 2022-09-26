#include <iostream>
#include <vector>
#include <iomanip>
#include <cstdlib>

#include "../../common/error_checkers.hpp"
#include "../../common/cuda_timer.hpp"

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

int main(int argc, char* argv[]) {
	char *end;
	int blocks = std::strtol(argv[1], &end, 10);
	int threads = std::strtol(argv[2], &end, 10);

	std::ios::sync_with_stdio(false);
	int n;
	std::cin >> n;
	std::vector<double> vec(n);
	vec.shrink_to_fit();
	for (int i = 0; i < n; ++i) {
		std::cin >> vec[i];
	}

	double *dev_vec;
	cudaCheck(cudaMalloc(&dev_vec, sizeof(double) * n));
	cudaCheck(cudaMemcpy(dev_vec, vec.data(), sizeof(double) * n, cudaMemcpyHostToDevice));

	cudaStartTimer();

	reverse<<<blocks, threads>>>(dev_vec, n);
	// cudaCheck(cudaDeviceSynchronize());
	cudaCheckLastError();

	float t;
	cudaEndTimer(t);

	std::cout << t;

	cudaCheck(cudaFree(dev_vec));
	return 0;
}