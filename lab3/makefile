NVCC = nvcc
CXX = g++

CUDA_FLAGS = --std=c++11 -Werror cross-execution-space-call -lm
CFLAGS = --std=c++11 -Wall -Wextra -pedantic

all: cuda cpu

cuda: lab3.cu
	$(NVCC) $(CUDA_FLAGS) -o lab3 lab3.cu

cpu: lab3_cpu.cpp
	$(CXX) $(CFLAGS) -o lab3_cpu lab3_cpu.cpp

clean:
	rm -f lab3 lab3_cpu