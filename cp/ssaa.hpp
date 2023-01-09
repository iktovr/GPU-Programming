#pragma once

#include <vector>
#include <cmath>

#include "../common/vec3.hpp"

namespace cpu {
	void ssaa(const std::vector<vec3f>& frame, std::vector<uchar4>& res_frame, int w, int h, int coeff) {
		vec3f color;
		for (int i = 0; i < w; ++i) {
			for (int j = 0; j < h; ++j) {
				color *= 0;
				for (int u = i * coeff; u < (i + 1) * coeff; ++u) {
					for (int v = j * coeff; v < (j + 1) * coeff; ++v) {
						color += frame[v * w * coeff + u];
					}
				}
				color /= (float)coeff * coeff;
				res_frame[j * w + i].x = (unsigned char)(std::min(color.x, 1.0f) * 255);
				res_frame[j * w + i].y = (unsigned char)(std::min(color.y, 1.0f) * 255);
				res_frame[j * w + i].z = (unsigned char)(std::min(color.z, 1.0f) * 255);
				res_frame[j * w + i].w = 255;
			}
		}
	}
}

namespace gpu {
	__global__ void ssaa(const vec3f *const frame, uchar4 *res_frame, int w, int h, int coeff) {
		int idx = blockDim.x * blockIdx.x + threadIdx.x;
		int idy = blockDim.y * blockIdx.y + threadIdx.y;
		int offsetx = gridDim.x * blockDim.x;
		int offsety = gridDim.y * blockDim.y;

		vec3f color;
		for (int i = idx; i < w; i += offsetx) {
			for (int j = idy; j < h; j += offsety) {
				color *= 0;
				for (int u = i * coeff; u < (i + 1) * coeff; ++u) {
					for (int v = j * coeff; v < (j + 1) * coeff; ++v) {
						color += frame[v * w * coeff + u];
					}
				}
				color /= (float)coeff * coeff;
				res_frame[j * w + i].x = (unsigned char)(min(color.x, 1.0f) * 255);
				res_frame[j * w + i].y = (unsigned char)(min(color.y, 1.0f) * 255);
				res_frame[j * w + i].z = (unsigned char)(min(color.z, 1.0f) * 255);
				res_frame[j * w + i].w = 255;
			}
		}
	}
}