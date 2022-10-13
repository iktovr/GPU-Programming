NVCC = nvcc
CXX = g++

CUDA_FLAGS = --std=c++11 -Werror cross-execution-space-call -lm
CFLAGS = --std=c++11 -Wall -Wextra -pedantic

.PHONY: all main cuda cpu bench stbi

all: main bench

main: cuda cpu

cuda: lab3.cu
	$(NVCC) $(CUDA_FLAGS) -o lab3 lab3.cu

cpu: lab3_cpu.cpp
	$(CXX) $(CFLAGS) -o lab3_cpu lab3_cpu.cpp

bench: lab3_cpu.cpp lab3.cu
	$(NVCC) $(CUDA_FLAGS) -o bench/lab3_gpu -DTIME lab3.cu
	$(CXX) $(CFLAGS) -o bench/lab3_cpu -DTIME lab3_cpu.cpp

stbi: lab3_cpu.cpp
	$(CXX) $(CFLAGS) -o lab3_cpu -DSTBI lab3_cpu.cpp

clean:
	rm -f bench/lab3* lab3 lab3_cpu