#include <vector>
#include <iostream>

#include "../common/error_checkers.hpp"

___global__ void reduce(int* idata, int n, int* odata) {
	int tid = threadIdx.x;
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int offset = blockDim.x * gridDim.x;

	extern __shared__ int sdata[];

	sdata[tid] = idata[id];
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

const int MAX_BLOCK_SIZE = 1024;

inline int optimal_block_size(int size) {
	if (size > MAX_BLOCK_SIZE) {
		return MAX_BLOCK_SIZE;
	} else {
		while (size & (size - 1) != 0) {
			size &= size - 1;
		}
		return max(size, 32);
	}
}

inline int get_block_count(int size, int block_size) {
	return (size / block_size + ((size & block_size) > 0)) * block_size;
}

__host__ int reduce(const std::vector<int>& data) {
	int block_size = optimal_block_size(size);
	int block_count = get_block_count(data.size(), block_size);
	int data_size = block_count * block_size, res_size = block_count;
	std::vector<int> fill(data_size - data.size(), 0);

	int *dev_data, *dev_res;
	cudaCheck(cudaMalloc(&dev_data, sizeof(int) * data_size));
	cudaCheck(cudaMemcpy(dev_data, data.data(), sizeof(int) * data.size(), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(dev_data + data.size(), fill.data(), sizeof(int) * fill.size(), cudaMemcpyHostToDevice));
	cudaCheck(cudaMalloc(&dev_res, sizeof(int) * res_size));
	reduce<<<block_count, block_size, sizeof(int) * block_size>>>(dev_data, dev_res);
	
	while (res_size > 1) {
		cudaCheck(cudaMemcpy(dev_data, dev_res, sizeof(int) * block_count, cudaMemcpyDeviceToDevice));
		block_size = optimal_block_size(block_count);
		block_count = get_block_count(block_count, block_size);
		data_size = block_count * block_size;

		fill.assign(data_size - res_size, 0);
		cudaCheck(cudaMemcpy(dev_data + res_size, fill.data(), sizeof(int) * fill.size(), cudaMemcpyHostToDevice));
		res_size = block_count;

		reduce<<<block_count, block_size, sizeof(int) * block_size>>>(dev_data, dev_res);
	}

	int res;
	cudaCheck(cudaMemcpy(&res, dev_res, sizeof(int), cudaMemcpyDeviceToHost));
	return res;
}

int main() {
	int n;
	std::cin >> n;
	std::vector<int> data(n);
	for (int i = 0; i < n; ++i) {
		std::cin >> data[i];
	}

	cout << reduce(data);
}