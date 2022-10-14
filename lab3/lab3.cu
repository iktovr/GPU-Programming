#include <vector>
#include <cmath>
#include <fstream>

#include "../common/error_checkers.hpp"
#include "matrix.hpp"

#ifdef TIME
#include "../common/cuda_timer.hpp"
#endif

using std::log;
using std::abs;

__constant__ double3 mean[32];
__constant__ double conv_inv[32][9];
__constant__ double conv_det[32];

__global__ void maximum_likelihood(uchar4* img, int length, int n_classes) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int offset = blockDim.x * gridDim.x;
	
	double max_l, l;
	uint8_t max_m;
	double3 v;
	
	for (; idx < length; idx += offset) {
		for (int m = 0; m < n_classes; ++m) {
			matrix3d* conv_inv_m = reinterpret_cast<matrix3d*>(conv_inv[m]);
			v = img[idx] - mean[m];
			l = -(v * (*conv_inv_m) * v) - conv_det[m];
			if (m == 0 || max_l < l) {
				max_l = l;
				max_m = m;
			}
		}
		img[idx].w = max_m;
	}
}

int main(int argc, char* argv[]) {
	std::ios::sync_with_stdio(false);
#ifdef TIME
	check(argc < 3, true, "Expected 2 arguments");
	char *end;
	dim3 grid_dim(std::strtol(argv[1], &end, 10)), 
	     block_dim(std::strtol(argv[2], &end, 10));
#else
	(void)argc; (void)argv;
	dim3 grid_dim(128), 
	     block_dim(128);
#endif

	std::string in_filename, out_filename;
	int n_classes;
	std::cin >> in_filename >> out_filename >> n_classes;

	int width, height, channels;
	uchar4 *img;
	double3 h_mean[32];
	matrix3d conv[32], h_conv_inv[32];
	double h_conv_det[32];

	std::ifstream in_file(in_filename, std::ios::binary);
	check(in_file.is_open(), false, "failed to open input file");

	in_file.read(reinterpret_cast<char*>(&width), sizeof(width));
	in_file.read(reinterpret_cast<char*>(&height), sizeof(height));
	channels = 4;

	std::vector<uchar4> imgv(width * height);
	imgv.shrink_to_fit();
	in_file.read(reinterpret_cast<char*>(imgv.data()), width * height * channels);
	img = imgv.data();
	
	int m, x, y;
	std::vector<std::pair<int, int>> coords;
	for (int i = 0; i < n_classes; ++i) {
		std::cin >> m;
		coords.clear();
		for (int j = 0; j < m; ++j) {
			std::cin >> x >> y;
			coords.push_back({x, y});
			h_mean[i] += img[y * width + x];
		}
		h_mean[i] /= m;

		for (int j = 0; j < m; ++j) {
			x = coords[j].first;
			y = coords[j].second;
			double3 v = img[y * width + x] - h_mean[i];
			conv[i] += matmul(v, v);
		}
		conv[i] /= (m-1);
		h_conv_det[i] = log(abs(conv[i].det()));
		h_conv_inv[i] = conv[i].invert();
	}

	cudaCheck(cudaMemcpyToSymbol(mean, h_mean, sizeof(double3) * 32));
	cudaCheck(cudaMemcpyToSymbol(conv_inv, h_conv_inv, sizeof(matrix3d) * 32));
	cudaCheck(cudaMemcpyToSymbol(conv_det, h_conv_det, sizeof(double) * 32));

	uchar4 *dev_img;
	cudaCheck(cudaMalloc(&dev_img, width * height * channels));
	cudaCheck(cudaMemcpy(dev_img, img, width * height * channels, cudaMemcpyHostToDevice));

#ifdef TIME
	cudaStartTimer();
#endif

	maximum_likelihood<<<grid_dim, block_dim>>>(dev_img, width * height, n_classes);

#ifdef TIME
	float t;
	cudaEndTimer(t);
	std::cout << t;
#else
	cudaCheck(cudaMemcpy(imgv.data(), dev_img, width * height * channels, cudaMemcpyDeviceToHost));

	std::ofstream out_file(out_filename, std::ios::binary);
	check(out_file.is_open(), false, "failed to open output file");

	out_file.write(reinterpret_cast<char*>(&width), sizeof(width));
	out_file.write(reinterpret_cast<char*>(&height), sizeof(height));
	out_file.write(reinterpret_cast<char*>(imgv.data()), width * height * channels);
#endif

	cudaCheck(cudaFree(dev_img));
	return 0;
}
