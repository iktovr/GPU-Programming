cpu: main.cu
	g++ --std=c++11 -g -O3 -Wall -Wextra -pedantic -x c++ -o main main.cu
gpu: main.cu
	nvcc -D_FORCE_INLINES --std=c++11 -O2 -o main main.cu