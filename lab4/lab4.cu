#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

#include <thrust/extrema.h>
#include <thrust/device_vector.h>

#include "../common/error_checkers.hpp"

#ifdef TIME
#include "../common/cuda_timer.hpp"
#endif

using std::abs;

struct comparator {
	__host__ __device__ bool operator()(double a, double b) {
		return abs(a) < abs(b);
	}
};

__device__ void swap(double& a, double& b) {
	double tmp = a;
	a = b;
	b = tmp;
}

__global__ void swap_rows(double *matrix, int i, int j, int n) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int offset = blockDim.x * gridDim.x;

	for (int k = idx; k < n + 1; k += offset) {
		swap(matrix[i + k * n], matrix[j + k * n]);
	}
}

__global__ void gaussian_solver_step(double *matrix, int row, int n) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;

	for (int j = row + 1 + idx; j < n + 1; j += offsetx) {
		for (int i = row + 1 + idy; i < n; i += offsety) {
			matrix[i + j * n] -= matrix[row + j * n] / matrix[row + row * n] * matrix[i + row * n];
		}
	}
}

int main(int argc, char* argv[]) {
	std::ios::sync_with_stdio(false);
#ifdef TIME
	check(argc < 5, true, "Expected 4 arguments");
	char *end;
	dim3 grid_dim(std::strtol(argv[1], &end, 10), std::strtol(argv[2], &end, 10)), 
	     block_dim(std::strtol(argv[3], &end, 10), std::strtol(argv[4], &end, 10));
#else
	(void)argc; (void)argv;
	dim3 grid_dim(16, 16), 
	     block_dim(32, 32);
#endif

	int n;
	std::cin >> n;
	std::vector<double> matrix(n * (n + 1));
	matrix.shrink_to_fit();
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			std::cin >> matrix[i + j * n];
		}
	}

	for (int i = 0; i < n; ++i) {
		std::cin >> matrix[i + n * n];
	}

	double *dev_matrix;
	cudaCheck(cudaMalloc(&dev_matrix, sizeof(double) * matrix.size()));
	cudaCheck(cudaMemcpy(dev_matrix, matrix.data(), sizeof(double) * matrix.size(), cudaMemcpyHostToDevice));
	thrust::device_ptr<double> p_matrix = thrust::device_pointer_cast(dev_matrix);

	comparator comp;
	int leading_row;

#ifdef TIME
	cudaStartTimer();
#endif

	for (int row = 0; row < n; ++row) {
		leading_row = static_cast<int>(thrust::max_element(p_matrix + row * n + row, p_matrix + (row + 1) * n, comp) - (p_matrix + row * n));
		if (leading_row != row) {
			swap_rows<<<grid_dim.x, block_dim.x>>>(dev_matrix, leading_row, row, n);
			cudaCheck(cudaDeviceSynchronize());
			cudaCheckLastError();
		}
		gaussian_solver_step<<<grid_dim, block_dim>>>(dev_matrix, row, n);
		cudaCheck(cudaDeviceSynchronize());
		cudaCheckLastError();
	}

#ifdef TIME
	float t;
	cudaEndTimer(t);
	std::cout << t;
#else

	cudaCheck(cudaMemcpy(matrix.data(), dev_matrix, sizeof(double) * matrix.size(), cudaMemcpyDeviceToHost));
	std::vector<double> x(n);

	for (int i = n-1; i >= 0; --i) {
		x[i] = matrix[i + n * n];
		for (int j = i+1; j < n; ++j) {
			x[i] -= x[j] * matrix[i + j * n];
		}
		x[i] /= matrix[i + i * n];
	}

	std::cout << std::setprecision(10) << std::fixed;
	for (int i = 0; i < n; ++i) {
		std::cout << x[i] << ' ';
	}
	std::cout << '\n';
#endif

	cudaCheck(cudaFree(dev_matrix));
	return 0;
}
