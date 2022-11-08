#include <vector>
#include <iostream>

#include "../common/error_checkers.hpp"

__global__ void reduce(int* idata, int n, int* odata) {
	int tid = threadIdx.x;
	int id = blockDim.x * blockIdx.x * 2 + threadIdx.x;

	extern __shared__ int sdata[];

	sdata[tid] = idata[id] + idata[id + blockDim.x];
	__syncthreads();
	for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}

	if (tid == 0) {
		odata[blockIdx.x] = sdata[0];
	}
}

const size_t BLOCK_SIZE = 1024;

template <class T>
inline T get_block_count(T size, T block_size, T log_block_size) {
	return (size >> (log_block_size + 1)) + ((size & ((block_size << 1) - 1)) > 0);
}

template <class T>
T ceil_2_pow(T a) {
	while ((a & (a - 1)) != 0) {
		a &= a - 1;
	}
	return a << 1;
}

template <class T>
T log2(T a) {
	T log = 0;
	while (a > 1) {
		++log;
		a >>= 1;
	}
	return log;
}

int reduce(const std::vector<int>& data) {
	size_t data_size = ceil_2_pow(data.size());
	std::vector<int> fill(data_size - data.size(), 0);
	size_t log_block_size = log2(BLOCK_SIZE);
	size_t res_size = get_block_count(data_size, BLOCK_SIZE, log_block_size);

	int *dev_data, *dev_res;
	cudaCheck(cudaMalloc(&dev_data, sizeof(int) * data_size));
	cudaCheck(cudaMemcpy(dev_data, data.data(), sizeof(int) * data.size(), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(dev_data + data.size(), fill.data(), sizeof(int) * fill.size(), cudaMemcpyHostToDevice));
	cudaCheck(cudaMalloc(&dev_res, sizeof(int) * res_size));

	while (res_size > 1) {
		// std::cout << data_size << ' ' << res_size << '\n';
		reduce<<<res_size, BLOCK_SIZE, sizeof(int) * BLOCK_SIZE>>>(dev_data, data_size, dev_res);
		cudaCheck(cudaDeviceSynchronize());
		cudaCheckLastError();
		std::swap(dev_data, dev_res);
		data_size = res_size;
		res_size = get_block_count(data_size, BLOCK_SIZE, log_block_size);
	}

	// std::cout << data_size << ' ' << res_size << '\n';
	reduce<<<1, (data_size >> 1), sizeof(int) * (data_size >> 1)>>>(dev_data, data_size, dev_res);
	cudaCheck(cudaDeviceSynchronize());
	cudaCheckLastError();

	int res;
	cudaCheck(cudaMemcpy(&res, dev_res, sizeof(int), cudaMemcpyDeviceToHost));
	cudaCheck(cudaFree(dev_data));
	cudaCheck(cudaFree(dev_res));
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
	cudaCheck(cudaDeviceSynchronize());
	cudaCheckLastError();
}