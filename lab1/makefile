NVCC = nvcc
CXX = g++

CUDA_FLAGS = --std=c++11 -Werror cross-execution-space-call -lm
CFLAGS = --std=c++11 -Wall -Wextra -pedantic

.PHONY: all cuda bench

all: cuda bench

cuda: lab1.cu
	$(NVCC) $(CUDA_FLAGS) -o lab1 lab1.cu

bench: bench/lab1_cpu.cpp bench/lab1_gpu.cu
	$(NVCC) $(CUDA_FLAGS) -o bench/lab1_gpu bench/lab1_gpu.cu
	$(CXX) $(CFLAGS) -o bench/lab1_cpu bench/lab1_cpu.cpp
