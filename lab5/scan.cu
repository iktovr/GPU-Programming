#include <vector>
#include <iostream>

#include "../common/error_checkers.hpp"

template <class T>
inline T get_block_count(T size, T block_size, T log_block_size) {
	return (size >> (log_block_size + 1)) + ((size & ((block_size << 1) - 1)) > 0);
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

const size_t BLOCK_SIZE = 1024;
const size_t BLOCK_COUNT = 1024;

#define sh_index(i) ((i) + ((i) >> 5))

__global__ void scan(int* idata, int* odata) {
    extern __shared__ int sdata[];

    int tid = threadIdx.x;
    int block_size = blockDim.x << 1;

    sdata[sh_index(tid)] = idata[tid];
    sdata[sh_index(tid + blockDim.x)] = idata[tid + blockDim.x];
    __syncthreads();

    for (int s = 1; s < block_size; s <<= 1) {
        int i = (s << 1) * (tid + 1) - 1;
        if (i < block_size) {
            sdata[sh_index(i)] += sdata[sh_index(i - s)];
        }
        __syncthreads();
    }

    if (tid == 0) {
        sdata[sh_index(block_size - 1)] = 0;
    }
    __syncthreads();

    int tmp;
    for (int s = blockDim.x; s > 0; s >>= 1) {
        int i = (s << 1) * (tid + 1) - 1;
        if (i < block_size) {
            tmp = sdata[sh_index(i)];
            sdata[sh_index(i)] += sdata[sh_index(i - s)];
            sdata[sh_index(i - s)] = tmp;
        }
        __syncthreads();
    }

    odata[tid] = sdata[sh_index(tid)];
    odata[tid + blockDim.x] = sdata[sh_index(tid + blockDim.x)];
}

__global__ void per_block_sum(int *data, int *blocks, int size) {
    int id = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    int bid = blockIdx.x;
    int offset = gridDim.x * blockDim.x * 2;

    while (id < size) {
        data[id] += blocks[bid];
        data[id + blockDim.x] += blocks[bid];

        id += offset;
        bid += gridDim.x;
    }
}

__global__ void scan(int *idata, int *odata, int size, int *block_sum) {
    extern __shared__ int sdata[];

    int tid = threadIdx.x;
    int block_size = blockDim.x << 1;
    int id = block_size * blockIdx.x + threadIdx.x;
    int offset = block_size * gridDim.x;
	int blockId = blockIdx.x;

    while (id < size) {
        sdata[sh_index(tid)] = idata[id];
        sdata[sh_index(tid + blockDim.x)] = idata[id + blockDim.x];
        __syncthreads();

        for (int s = 1; s < block_size; s <<= 1) {
            int i = (s << 1) * (tid + 1) - 1;
            if (i < block_size) {
                sdata[sh_index(i)] += sdata[sh_index(i - s)];
            }
            __syncthreads();
        }

        if (tid == 0) {
            block_sum[blockId] = sdata[sh_index(block_size - 1)];
            sdata[sh_index(block_size - 1)] = 0;
        }
        __syncthreads();

        int tmp;
        for (int s = blockDim.x; s > 0; s >>= 1) {
            int i = (s << 1) * (tid + 1) - 1;
            if (i < block_size) {
                tmp = sdata[sh_index(i)];
                sdata[sh_index(i)] += sdata[sh_index(i - s)];
                sdata[sh_index(i - s)] = tmp;
            }
            __syncthreads();
        }

        odata[id] = sdata[sh_index(tid)];
        odata[id + blockDim.x] = sdata[sh_index(tid + blockDim.x)];

        id += offset;
        blockId += gridDim.x;
    }
}

void scan(int *idata, int *odata, size_t size) {
    size_t log_block_size = log2(BLOCK_SIZE);
    size_t block_count = get_block_count(size, BLOCK_SIZE, log_block_size);

    if (block_count == 1) {
        scan<<<1, size >> 1, sizeof(int) * sh_index(size)>>>(idata, odata);
        cudaCheck(cudaDeviceSynchronize());
        cudaCheckLastError();
        return;
    }

    size_t block_sum_size = get_block_count(block_count, BLOCK_SIZE, log_block_size) * 2 * BLOCK_SIZE;
    int *block_sum, *block_sum_scan;
    cudaCheck(cudaMalloc(&block_sum, sizeof(int) * block_sum_size));
    cudaCheck(cudaMalloc(&block_sum_scan, sizeof(int) * block_sum_size));

    // std::cout << size << ' ' << block_count << ' ' << block_sum_size << '\n';

    scan<<<min(block_count, BLOCK_COUNT), BLOCK_SIZE, sizeof(int) * sh_index(BLOCK_SIZE * 2)>>>(idata, odata, size, block_sum);
    cudaCheck(cudaDeviceSynchronize());
    cudaCheckLastError();
    scan(block_sum, block_sum_scan, block_sum_size);
    per_block_sum<<<min(block_count, BLOCK_COUNT), BLOCK_SIZE>>>(odata, block_sum_scan, size);
    cudaCheck(cudaDeviceSynchronize());
    cudaCheckLastError();

    cudaCheck(cudaFree(block_sum));
    cudaCheck(cudaFree(block_sum_scan));
}

std::vector<int> scan(const std::vector<int>& data) {
    size_t log_block_size = log2(BLOCK_SIZE);
    size_t block_count = get_block_count(data.size(), BLOCK_SIZE, log_block_size);
    size_t data_size = block_count * 2 * BLOCK_SIZE;
    // std::vector<int> fill(data_size - data.size(), 0);

    // std::cout << data.size() << ' ' << data_size << ' ' << block_count << '\n';

    int *dev_data, *dev_res;
    cudaCheck(cudaMalloc(&dev_data, sizeof(int) * data_size));
    cudaCheck(cudaMalloc(&dev_res, sizeof(int) * data_size));
    cudaCheck(cudaMemcpy(dev_data, data.data(), sizeof(int) * data.size(), cudaMemcpyHostToDevice));
    // cudaCheck(cudaMemcpy(dev_data + data.size(), fill.data(), sizeof(int) * fill.size(), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(dev_data + data.size(), 0, sizeof(int) * (data_size - data.size())));

    scan(dev_data, dev_res, data_size);

    std::vector<int> res(data.size());
    cudaCheck(cudaMemcpy(res.data(), dev_res, sizeof(int) * res.size(), cudaMemcpyDeviceToHost));
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

    std::vector<int> pr_data = scan(data);
    for (int i = 0; i < n; ++i) {
		std::cout << pr_data[i] << ' ';
	}
    std::cout << '\n';
}