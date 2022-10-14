#include <vector>
#include <cmath>

// #define STBI

#ifdef STBI
#define STB_IMAGE_IMPLEMENTATION
#include "../external/stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../external/stb/stb_image_write.h"
#include <cassert>
#else
#include <fstream>
#endif

#ifdef TIME
#include <chrono>
using namespace std::chrono;
#endif

#include "../common/error_checkers.hpp"
#include "matrix.hpp"

using std::log;
using std::abs;

#ifdef STBI
uint8_t pallete[39][3] = {
	{  0,   0,   0}, {128, 128, 128}, {192, 192, 192}, {255, 255, 255}, {255,   0, 255}, 
    {128,   0, 128}, {255,   0,   0}, {128,   0,   0}, {205,  92,  92}, {240, 128, 128}, 
    {250, 128, 114}, {233, 150, 122}, {205,  92,  92}, {240, 128, 128}, {250, 128, 114}, 
    {233, 150, 122}, {173, 255,  47}, {127, 255,   0}, {124, 252,   0}, {  0, 255,   0}, 
    { 50, 205,  50}, {152, 251, 152}, {144, 238, 144}, {  0, 250, 154}, {  0, 255, 127}, 
    { 60, 179, 113}, { 46, 139,  87}, { 34, 139,  34}, {  0, 128,   0}, {  0, 100,   0}, 
    {154, 205,  50}, {107, 142,  35}, {128, 128,   0}, { 85, 107,  47}, {102, 205, 170}, 
    {143, 188, 143}, { 32, 178, 170}, {  0, 139, 139}, {  0, 128, 128},
};
#endif

void maximum_likelihood(uchar4* img, int length, int n_classes, 
                        const double3 mean[32], const matrix3d conv_inv[32], const double conv_det[32]) {
	for (int i = 0; i < length; ++i) {
		double max_l, l;
		uint8_t max_m;
		double3 v;
		for (int m = 0; m < n_classes; ++m) {
			v = img[i] - mean[m];
			l = -(v * conv_inv[m] * v) - conv_det[m];
			if (m == 0 || max_l < l) {
				max_l = l;
				max_m = m;
			}
		}
		img[i].w = max_m;
	}
}

int main() {
	std::ios::sync_with_stdio(false);
	std::string in_filename, out_filename;
	int n_classes;
	std::cin >> in_filename >> out_filename >> n_classes;

	int width, height, channels;
	uchar4 *img;
	double3 mean[32];
	matrix3d conv[32], conv_inv[32];
	double conv_det[32];
	LUP3d lup;


#ifdef STBI
	img = reinterpret_cast<uchar4*>(stbi_load(in_filename.c_str(), &width, &height, &channels, 0));
	assert(channels == 4);
	check(img, NULL, "error loading image");

	int m;
	uchar4 color;
	uint16_t tmp;
	std::vector<uchar4> colors;
	for (int i = 0; i < n_classes; ++i) {
		std::cin >> m;
		colors.clear();
		for (int j = 0; j < m; ++j) {
			std::cin >> tmp; color.x = tmp;
			std::cin >> tmp; color.y = tmp;
			std::cin >> tmp; color.z = tmp;
			colors.push_back(color);
			mean[i] += color;
		}
		mean[i] /= m;
		// std::cout << matmul(color - mean[i], color - mean[i]) << '\n';

		for (int j = 0; j < m; ++j) {
			conv[i] += matmul(colors[j] - mean[i], colors[j] - mean[i]);
		}
		conv[i] /= m - 1;
		lup.assign(conv[i]);
		conv_det[i] = log(abs(lup.det()));
		conv_inv[i] = lup.invert();
		// std::cout << mean[i] << '\n' << conv[i] << '\n' << conv_inv[i] << '\n' << conv_det[i] << '\n';
	}
#else
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
			mean[i] += img[y * width + x];
		}
		mean[i] /= m;

		for (int j = 0; j < m; ++j) {
			x = coords[j].first;
			y = coords[j].second;
			double3 v = img[y * width + x] - mean[i];
			conv[i] += matmul(v, v);
		}
		conv[i] /= (m-1);
		lup.assign(conv[i]);
		conv_det[i] = log(abs(conv[i].det()));
		conv_inv[i] = conv[i].invert();
		// conv_det[i] = log(abs(lup.det()));
		// conv_inv[i] = lup.invert();
		// std::cout << mean[i] << '\n' << conv[i] << '\n' << conv_inv[i] << '\n' << conv_det[i] << '\n';
	}
#endif

#ifdef TIME
	steady_clock::time_point start = steady_clock::now();
#endif

	maximum_likelihood(img, width * height, n_classes, mean, conv_inv, conv_det);

#ifdef TIME
	steady_clock::time_point end = steady_clock::now();
	std::cout << duration_cast<nanoseconds>(end - start).count() / 1000000.0;
#else

#ifdef STBI
	for (int i = 0; i < width * height; ++i) {
		img[i].x = mean[img[i].w].x;
		img[i].y = mean[img[i].w].y;
		img[i].z = mean[img[i].w].z;
		img[i].w = 255;
	}
	stbi_write_png(out_filename.c_str(), width, height, channels, reinterpret_cast<uint8_t*>(img), width * channels);
#else
	std::ofstream out_file(out_filename, std::ios::binary);
	check(out_file.is_open(), false, "failed to open output file");

	out_file.write(reinterpret_cast<char*>(&width), sizeof(width));
	out_file.write(reinterpret_cast<char*>(&height), sizeof(height));
	out_file.write(reinterpret_cast<char*>(imgv.data()), width * height * channels);
#endif
#endif

#ifdef STBI
	stbi_image_free(reinterpret_cast<uint8_t*>(img));
#endif
	return 0;
}
