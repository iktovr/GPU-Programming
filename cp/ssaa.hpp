#pragma once

#include <vector>
#include <cmath>

#include "../common/vec3.hpp"

void ssaa(const std::vector<vec3f>& frame, std::vector<vec3c>& data, int w, int h, int coeff) {
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
			data[j * w + i].x = (unsigned char)(std::min(color.x, 1.0f) * 255);
			data[j * w + i].y = (unsigned char)(std::min(color.y, 1.0f) * 255);
			data[j * w + i].z = (unsigned char)(std::min(color.z, 1.0f) * 255);
		}
	}
}