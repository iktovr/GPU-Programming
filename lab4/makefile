NVCC = nvcc
CXX = g++

CUDA_FLAGS = --std=c++11 -Werror cross-execution-space-call -lm
CFLAGS = --std=c++11 -Wall -Wextra -pedantic

.PHONY: all main cuda cpu bench clean

all: main bench

main: cuda cpu

cuda: lab4.cu
	$(NVCC) $(CUDA_FLAGS) -o lab4 lab4.cu

cpu: lab4_cpu.cpp
	$(CXX) $(CFLAGS) -o lab4_cpu lab4_cpu.cpp

bench: lab4_cpu.cpp lab4.cu
	$(NVCC) $(CUDA_FLAGS) -o bench/lab4_gpu -DTIME lab4.cu
	$(CXX) $(CFLAGS) -o bench/lab4_cpu -DTIME lab4_cpu.cpp

clean:
	rm -f bench/lab4* lab4 lab4_cpu