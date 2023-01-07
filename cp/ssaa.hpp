#pragma once

#include <vector>
#include <cmath>

#include "../common/vec3.hpp"

void ssaa(const std::vector<vec3>& frame, std::vector<vec3c>& data, int w, int h, int coeff) {
	vec3 color;
	for (int i = 0; i < w; ++i) {
		for (int j = 0; j < h; ++j) {
			color *= 0;
			for (int u = i * coeff; u < (i + 1) * coeff; ++u) {
				for (int v = j * coeff; v < (j + 1) * coeff; ++v) {
					color += frame[v * w * coeff + u];
				}
			}
			color /= (double)coeff * coeff;
			data[j * w + i].x = (unsigned char)(std::min(std::max(color.x, 0.0), 1.0) * 255);
			data[j * w + i].y = (unsigned char)(std::min(std::max(color.y, 0.0), 1.0) * 255);
			data[j * w + i].z = (unsigned char)(std::min(std::max(color.z, 0.0), 1.0) * 255);
		}
	}
}