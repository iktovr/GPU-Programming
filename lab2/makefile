NVCC = nvcc
CXX = g++

CUDA_FLAGS = --std=c++11 -Werror cross-execution-space-call -lm
CFLAGS = --std=c++11 -Wall -Wextra -pedantic

all: cuda cpu bench

cuda: lab2.cu
	$(NVCC) $(CUDA_FLAGS) -o lab2 lab2.cu

cpu: lab2_cpu.cpp
	$(CXX) $(CFLAGS) -o lab2_cpu lab2_cpu.cpp

bench: lab2_cpu.cpp lab2.cu
	$(NVCC) $(CUDA_FLAGS) -o bench/lab2_gpu -DTIME lab2.cu
	$(CXX) $(CFLAGS) -o bench/lab2_cpu -DTIME lab2_cpu.cpp

stbi: lab2_cpu.cpp
	$(CXX) $(CFLAGS) -o lab2_cpu -DSTBI lab2_cpu.cpp