#include <iostream>
#include <string>
#include <vector>
#include <cmath>

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

using std::min;
using std::max;
using std::sqrt;

double rgb_to_luminance(uint8_t r, uint8_t g, uint8_t b) {
	return 0.299 * r +  0.587 * g + 0.114 * b;
}

void sobel_filter(uint8_t *img, int width, int height, int channels, uint8_t *res, int8_t Wx[][3], int8_t Wy[][3]) {
	double Gx, Gy, grad;
	for (int i = 0; i < width; ++i) {
		for (int j = 0; j < height; ++j) {
			Gx = 0;
			Gy = 0;
			for (int u = -1; u <= 1; ++u) {
				for (int v = -1; v <= 1; ++v) {
					int ip = max(min(i + u, width-1), 0),
					    jp = max(min(j + v, height-1), 0);
					double lum = rgb_to_luminance(img[(jp * width + ip) * channels],
					                             img[(jp * width + ip) * channels + 1],
					                             img[(jp * width + ip) * channels + 2]);
					Gx += Wx[u+1][v+1] * lum;
					Gy += Wy[u+1][v+1] * lum;
				}
			}
			grad = min(255., sqrt(Gx * Gx + Gy * Gy));

			res[(j * width + i) * channels] = static_cast<uint8_t>(grad);
			res[(j * width + i) * channels + 1] = static_cast<uint8_t>(grad);
			res[(j * width + i) * channels + 2] = static_cast<uint8_t>(grad);
			res[(j * width + i) * channels + 3] = img[(j * width + i) * channels + 3];
		}
	}
}

int main() {
	std::ios::sync_with_stdio(false);
	std::string in_filename, out_filename;
	std::cin >> in_filename >> out_filename;

	int width, height, channels;
	uint8_t *img;

#ifdef STBI
	img = stbi_load(in_filename.c_str(), &width, &height, &channels, 0);
	assert(channels == 4);
	check(img, NULL, "error loading image");
#else
	std::ifstream in_file(in_filename, std::ios::binary);
	check(in_file.is_open(), false, "failed to open input file");

	in_file.read(reinterpret_cast<char*>(&width), sizeof(width));
	in_file.read(reinterpret_cast<char*>(&height), sizeof(height));
	channels = 4;

	std::vector<uint8_t> imgv(width * height * channels);
	imgv.shrink_to_fit();
	in_file.read(reinterpret_cast<char*>(imgv.data()), width * height * channels);
	img = imgv.data();
#endif

	int8_t Wx[3][3] = {
		{-1, 0, 1},
		{-2, 0, 2},
		{-1, 0, 1}
	},
	Wy[3][3] = {
		{-1, -2, -1},
		{ 0,  0,  0},
		{ 1,  2,  1}
	};

	std::vector<uint8_t> res(width * height * channels);
	res.shrink_to_fit();

#ifdef TIME
	steady_clock::time_point start = steady_clock::now();
#endif

	sobel_filter(img, width, height, channels, res.data(), Wx, Wy);

#ifdef TIME
	steady_clock::time_point end = steady_clock::now();
	std::cout << duration_cast<nanoseconds>(end - start).count() / 1000000.0;
#else

#ifdef STBI
	stbi_write_png(out_filename.c_str(), width, height, channels, res.data(), width * channels);
#else
	std::ofstream out_file(out_filename, std::ios::binary);
	check(out_file.is_open(), false, "failed to open output file");

	out_file.write(reinterpret_cast<char*>(&width), sizeof(width));
	out_file.write(reinterpret_cast<char*>(&height), sizeof(height));
	out_file.write(reinterpret_cast<char*>(res.data()), width * height * channels);
#endif
#endif

#ifdef STBI
	stbi_image_free(img);
#endif
	return 0;
}