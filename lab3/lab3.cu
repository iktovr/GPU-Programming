#include <vector>
#include <cmath>

// #define STBI

// #ifdef STBI
// #define STB_IMAGE_IMPLEMENTATION
// #include "../external/stb/stb_image.h"
// #define STB_IMAGE_WRITE_IMPLEMENTATION
// #include "../external/stb/stb_image_write.h"
// #include <cassert>
// #else
// #include <fstream>
// #endif

#ifdef TIME
#include "../common/cuda_timer.hpp"
#endif

#include "../common/error_checkers.hpp"
#include "matrix.hpp"

using std::log;
using std::abs;

__constant__ double3 mean[32];
__constant__ matrix3d conv_inv[32];
__constant__ double conv_det[32];

__global__ void maximum_likelihood(uchar4* img, uchar4* res, int length, int class_num) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int offset = blockDim.x * gridDim.x;
	
	for (; idx < length; idx += offset) {
		double max_l, l;
		uint8_t max_m;
		for (int m = 0; m < class_num; ++m) {
			l = (mean[m] - img[idx]) * conv_inv[m] * (img[idx] - mean[m]) - conv_det[m];
			if (m == 0 || max_l < l) {
				max_l = l;
				max_m = m;
			}
		}
		res[idx] = make_uchar4(img[idx].x, img[idx].y, img[idx].z, max_m);
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
	int n;
	std::cin >> in_filename >> out_filename >> n;

	int width, height, channels;
	uchar4 *img;
	double3 host_mean[32];
	matrix3d conv[32], host_conv_inv[32];
	double host_conv_det[32];
	LUP3d lup;

// #ifdef STBI
// 	img = reinterpret_cast<uchar4*>(stbi_load(in_filename.c_str(), &width, &height, &channels, 0));
// 	assert(channels == 4);
// 	check(img, NULL, "error loading image");

// 	int m;
// 	uchar4 color;
// 	uint16_t tmp;
// 	std::vector<uchar4> colors;
// 	for (int i = 0; i < n; ++i) {
// 		std::cin >> m;
// 		colors.clear();
// 		for (int j = 0; j < m; ++j) {
// 			std::cin >> tmp; color.x = tmp;
// 			std::cin >> tmp; color.y = tmp;
// 			std::cin >> tmp; color.z = tmp;
// 			colors.push_back(color);
// 			host_mean[i] += color;
// 		}
// 		host_mean[i] /= m;
// 		// std::cout << matmul(color - host_mean[i], color - host_mean[i]) << '\n';

// 		for (int j = 0; j < m; ++j) {
// 			conv[i] += matmul(colors[j] - host_mean[i], colors[j] - host_mean[i]);
// 		}
// 		conv[i] /= m - 1;
// 		lup.assign(conv[i]);
// 		host_conv_det[i] = log(abs(lup.det()));
// 		host_conv_inv[i] = lup.invert();
// 		// std::cout << host_mean[i] << '\n' << conv[i] << '\n' << host_conv_inv[i] << '\n' << host_conv_det[i] << '\n';
// 	}
// #else
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
	for (int i = 0; i < n; ++i) {
		std::cin >> m;
		coords.clear();
		for (int j = 0; j < m; ++j) {
			std::cin >> x >> y;
			coords.push_back({x, y});
			host_mean[i] += img[y * width + x];
		}
		host_mean[i] /= m;

		for (int j = 0; j < m; ++j) {
			x = coords[j].first;
			y = coords[j].second;
			conv[i] += matmul(img[y * width + x] - host_mean[i], img[y * width + x] - host_mean[i]);
		}
		conv[i] /= (m-1);
		lup.assign(conv[i]);
		host_conv_det[i] = log(abs(lup.det()));
		host_conv_inv[i] = lup.invert();
		// std::cout << host_mean[i] << '\n' << conv[i] << '\n' << host_conv_inv[i] << '\n' << host_conv_det[i] << '\n';
	}
// #endif

	cudaCheck(cudaMemcpyToSymbol(mean, host_mean, sizeof(double3) * 32));
	cudaCheck(cudaMemcpyToSymbol(conv_inv, host_conv_inv, sizeof(matrix3d) * 32));
	cudaCheck(cudaMemcpyToSymbol(conv_det, host_conv_det, sizeof(double) * 32));

	uchar4 *dev_img, *dev_res;
	cudaCheck(cudaMalloc(&dev_img, width * height * channels));
	cudaCheck(cudaMalloc(&dev_res, width * height * channels));

#ifdef TIME
#endif

	maximum_likelihood<<<grid_dim, block_dim>>>(dev_img, dev_res, width * height, n);

// #ifdef TIME
// #else

// #ifdef STBI
// 	for (uchar4& px: res) {
// 		px.x = mean[px.w].x;
// 		px.y = mean[px.w].y;
// 		px.z = mean[px.w].z;
// 		px.w = 255;
// 	}
// 	stbi_write_png(out_filename.c_str(), width, height, channels, reinterpret_cast<uint8_t*>(res.data()), width * channels);
// #else
	std::ofstream out_file(out_filename, std::ios::binary);
	check(out_file.is_open(), false, "failed to open output file");

	out_file.write(reinterpret_cast<char*>(&width), sizeof(width));
	out_file.write(reinterpret_cast<char*>(&height), sizeof(height));
	out_file.write(reinterpret_cast<char*>(res), width * height * channels);
// #endif
// #endif

// #ifdef STBI
// 	stbi_image_free(reinterpret_cast<uint8_t*>(img));
// #endif
	cudaCheck(cudaFree(dev_img));
	cudaCheck(cudaFree(dev_res));
	return 0;
}
