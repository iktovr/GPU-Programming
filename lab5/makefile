NVCC = nvcc
CXX = g++

CUDA_FLAGS = --std=c++11 -Werror cross-execution-space-call -lm
CFLAGS = --std=c++11 -Wall -Wextra -pedantic

.PHONY: all main cuda cpu bench clean

all: main bench

main: cuda cpu

cuda: lab5.cu
	$(NVCC) $(CUDA_FLAGS) -o lab5 lab5.cu

cpu: lab5_cpu.cpp
	$(CXX) $(CFLAGS) -o lab5_cpu lab5_cpu.cpp

bench: lab5_cpu.cpp lab5.cu
	$(NVCC) $(CUDA_FLAGS) -o bench/lab5_gpu -DTIME lab5.cu
	$(CXX) $(CFLAGS) -o bench/lab5_cpu -DTIME lab5_cpu.cpp

clean:
	rm -f bench/lab5* lab5 lab5_cpu