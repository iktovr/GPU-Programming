FLAGS=--std=c++11 -O2 -g -D_FORCE_INLINES

all: main.cu
	nvcc $(FLAGS) -o main main.cu

print_scene: print_scene.cu
	nvcc $(FLAGS) -o print_scene print_scene.cu