#include <vector>
#include <iostream>

#include "../common/error_checkers.hpp"

const size_t BLOCK_SIZE = 1024;
const size_t BLOCK_COUNT = 1024;

__device__ void bitonic_merge(int i, int *data, int m) {
    int tmp, k;
    for (int b = m; b >= 2; b >>= 1) {
        if ((i & (b - 1)) < (b >> 1)) {
            k = i + (b >> 1);
            if (((i & m) && (data[i] < data[k])) || 
                (!(i & m) && (data[i] > data[k]))) {
                tmp = data[i];
                data[i] = data[k];
                data[k] = tmp;
            }
        }
        __syncthreads();
    }
}

__global__ void bitonic_sort_shared_memory(int *data, int size, int m) {
    extern __shared__ int sdata[];

    int tid = threadIdx.x;
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int offset = blockDim.x * gridDim.x;
	int blockId = blockIdx.x;

	while (id < size) {
		sdata[tid] = data[id];
		__syncthreads();

        bitonic_merge(tid, sdata, m);

        data[id] = sdata[tid];
		
        id += offset;
		blockId += gridDim.x;
	}
}

__global__ void bitonic_sort_global_memory(int *data, int size, int m, int b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;
    int tmp, k;

    while (i < size) {
        if ((i & (b - 1)) < (b >> 1)) {
            k = i + (b >> 1);
            if (((i & m) && (data[i] < data[k])) || 
                (!(i & m) && (data[i] > data[k]))) {
                tmp = data[i];
                data[i] = data[k];
                data[k] = tmp;
            }
        }

        i += offset;
    }
}

std::vector<int> bitonic_sort(std::vector<int> data) {
    int *dev_data;
    cudaCheck(cudaMalloc(&dev_data, sizeof(int) * data.size()));
    cudaCheck(cudaMemcpy(dev_data, data.data(), sizeof(int) * data.size(), cudaMemcpyHostToDevice));

    for (size_t m = 2; m <= data.size(); m <<= 1) {
		// for (size_t b = m; b >= 2; b >>= 1) {
        //     bitonic_sort_global_memory<<<BLOCK_COUNT, BLOCK_SIZE>>>(dev_data, data.size(), m, b);
        // }
        bitonic_sort_shared_memory<<<1, data.size(), sizeof(int) * data.size()>>>(dev_data, data.size(), m);
        cudaCheck(cudaDeviceSynchronize());
        cudaCheckLastError();
    }

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