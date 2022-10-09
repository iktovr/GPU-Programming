#include <iostream>
#include <string>
#include <vector>
#include <cmath>

#include <fstream>

#include "../common/error_checkers.hpp"

__constant__ char Wx[3][3], Wy[3][3];

__global__ void sobel_filter(cudaTextureObject_t img, uchar4 *res, int width, int height) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;

	double Gx, Gy, grad, lum;
	uchar4 p;
	for(int y = idy; y < height; y += offsety)
		for(int x = idx; x < width; x += offsetx) {
			Gx = 0;
			Gy = 0;
			for (int u = -1; u <= 1; ++u) {
				for (int v = -1; v <= 1; ++v) {
					p = tex2D<uchar4>(img, x + u, y + v);
					lum = 0.299 * p.x +  0.587 * p.y + 0.114 * p.z;
					Gx += Wx[u+1][v+1] * lum;
					Gy += Wy[u+1][v+1] * lum;
				}
			}
			grad = min(255., sqrt(Gx * Gx + Gy * Gy));
			res[y * width + x] = make_uchar4(grad, grad, grad, p.w);
		}
}

int main() {
	std::ios::sync_with_stdio(false);

	char host_Wx[3][3] = {
		{-1, 0, 1},
		{-2, 0, 2},
		{-1, 0, 1}
	},
	host_Wy[3][3] = {
		{-1, -2, -1},
		{ 0,  0,  0},
		{ 1,  2,  1}
	};
	cudaCheck(cudaMemcpyToSymbol(Wx, host_Wx, 9));
	cudaCheck(cudaMemcpyToSymbol(Wy, host_Wy, 9));

	std::string in_filename, out_filename;
	std::cin >> in_filename >> out_filename;

	int width, height;
	std::ifstream in_file(in_filename, std::ios::binary);
	check(in_file.is_open(), "failed to open input file", false);

	in_file.read(reinterpret_cast<char*>(&width), sizeof(width));
	in_file.read(reinterpret_cast<char*>(&height), sizeof(height));

	std::vector<uchar4> img(width * height);
	img.shrink_to_fit();
	in_file.read(reinterpret_cast<char*>(img.data()), sizeof(uchar4) * width * height);

	cudaArray_t dev_img;
	cudaChannelFormatDesc ch_desc = cudaCreateChannelDesc<uchar4>();
	cudaCheck(cudaMallocArray(&dev_img, &ch_desc, width, height));
	cudaCheck(cudaMemcpy2DToArray(dev_img, 0, 0, img.data(), width * sizeof(uchar4), 
	                              width * sizeof(uchar4), height, cudaMemcpyHostToDevice));

	struct cudaResourceDesc res_desc;
	memset(&res_desc, 0, sizeof(res_desc));
	res_desc.resType = cudaResourceTypeArray;
	res_desc.res.array.array = dev_img;

	struct cudaTextureDesc tex_desc;
	memset(&tex_desc, 0, sizeof(tex_desc));
	tex_desc.addressMode[0] = cudaAddressModeClamp;
	tex_desc.addressMode[1] = cudaAddressModeClamp;
	tex_desc.filterMode = cudaFilterModePoint;
	tex_desc.readMode = cudaReadModeElementType;
	tex_desc.normalizedCoords = false;

	cudaTextureObject_t img_tex = 0;
    cudaCheck(cudaCreateTextureObject(&img_tex, &res_desc, &tex_desc, NULL));

    uchar4 *dev_res;
    cudaCheck(cudaMalloc(&dev_res, width * height * sizeof(uchar4)));

    sobel_filter<<<dim3(16, 16), dim3(32, 32)>>>(img_tex, dev_res, width, height);
    cudaCheck(cudaDeviceSynchronize());
	cudaCheckLastError();

	std::vector<uchar4> res(width * height);
	res.shrink_to_fit();
	cudaCheck(cudaMemcpy(res.data(), dev_res, sizeof(uchar4) * width * height, cudaMemcpyDeviceToHost));

	std::ofstream out_file(out_filename, std::ios::binary);
	check(out_file.is_open(), "failed to open output file", false);

	out_file.write(reinterpret_cast<char*>(&width), sizeof(width));
	out_file.write(reinterpret_cast<char*>(&height), sizeof(height));
	out_file.write(reinterpret_cast<char*>(res.data()), sizeof(uchar4) * width * height);

	cudaCheck(cudaDestroyTextureObject(img_tex));
	cudaCheck(cudaFreeArray(dev_img));
	cudaCheck(cudaFree(dev_res));
	return 0;
}