#pragma once

#include <cmath>

#include "../common/vec3.hpp"
#include "../common/error_checkers.hpp"

#include "primitives.hpp"
#include "scene.hpp"

using std::max;
using std::min;
using std::pow;

__host__ __device__
vec3 phong_shade(const Material &material, const Light &light, const vec3 &intensity, const vec3 &ambient_light, const vec3 &point, const Ray &ray, const HitRecord &rec) {
	vec3 to_light = norm(light.pos - point);
	vec3 reflect_light = norm(reflect(-to_light, rec.normal));

	double diffuse = max(dot(to_light, rec.normal), 0.0);
	double specular = (diffuse > 0) ? pow(max(dot(reflect_light, -ray.dir), 0.0), material.p) : 0;

	return material.color * (material.Ka * ambient_light + intensity * (diffuse * material.Kd + specular * material.Ks)); 
}

namespace cpu {
	vec3 ray_color(const Ray& ray, const RawScene& scene, int max_depth, int depth = 0) {
		if (depth == max_depth) {
			return {0, 0, 0};
		}

		HitRecord rec;
		if (!scene.intersect(ray, rec))
			return {0, 0, 0};

		vec3 point = ray.at(rec.t * 0.99999);
		vec3 color{0, 0, 0};
		const Material &material = scene.materials[rec.material];

		for (int i = 0; i < scene.lights_count; ++i) {
			vec3 intensity = scene.light_intensity(i, rec, ray);
			color += phong_shade(material, scene.lights[i], intensity, scene.ambient_light, point, ray, rec);
		}

		if (material.reflection > 0) {
			Ray reflect_ray(point, reflect(ray.dir, rec.normal));
			color += material.reflection * ray_color(reflect_ray, scene, max_depth, depth + 1);
		}

		if (material.refraction > 0) {
			Ray refract_ray(ray.at(rec.t * 1.000001), ray.dir);
			color += material.refraction * ray_color(refract_ray, scene, max_depth, depth + 1);
		}

		return color;
	}

	void render(const RawScene &scene, const Camera &camera, std::vector<vec3> &frame, int w, int h, int max_depth) {
		double dw = 2.0 / (w - 1.0);
		double dh = 2.0 / (h - 1.0);
		double z = 1.0 / std::tan(camera.angle * PI / 360.0);
		vec3 bz = norm(camera.dir - camera.pos);
		vec3 bx = norm(cross(bz, {0.0, 0.0, 1.0}));
		vec3 by = norm(cross(bx, bz));
		Ray ray(camera.pos, {0, 0, 0});
		for(int i = 0; i < w; i++) {	
			for(int j = 0; j < h; j++) {
				vec3 v = {-1.0 + dw * i, (-1.0 + dh * j) * h / w, z};
				ray.dir = norm(matmul(bx, by, bz, v));
				frame[(h - 1 - j) * w + i] = ray_color(ray, scene, max_depth);
			}
		}
	}
}

#ifdef __CUDACC__

namespace gpu {
	__global__ void cast_rays(Ray *rays, const Camera &camera, int w, int h) {
		int idx = gridDim.x * blockDim.x + threadIdx.x;
		int idy = gridDim.y * blockDim.y + threadIdx.y;
		int offsetx = gridDim.x * blockDim.x;
		int offsety = gridDim.y * blockDim.y;

		double dw = 2.0 / (w - 1.0);
		double dh = 2.0 / (h - 1.0);
		double z = 1.0 / __tan(camera.angle * PI / 360.0);
		vec3 bz = norm(camera.dir - camera.pos);
		vec3 bx = norm(cross(bz, {0.0, 0.0, 1.0}));
		vec3 by = norm(cross(bx, bz));
		for(int i = idx; i < w; i += offsetx) {
			for(int j = idy; j < h; j += offsety) {
				vec3 v = {-1.0 + dw * i, (-1.0 + dh * j) * h / w, z};
				rays[(h - 1 - j) * w + i] = Ray(camera.pos, norm(matmul(bx, by, bz, v)), (h - 1 - j) * w + i);
			}
		}
	}

	__global__ void trace_rays(const RawScene &scene, Ray *in_rays, Ray *out_rays, int rays_count, int *out_rays_count, vec3 *frame) {
		int idx = gridDim.x * blockDim.x + threadIdx.x;
		int offset = gridDim.x * blockDim.x;

		HitRecord rec;
		vec3 point, color;
		int ind;
		for (; idx < rays_count; idx += offset) {
			if (!scene.intersect(in_rays[idx], rec)) {
				continue;
			}

			point = in_rays[idx].at(rec.t * 0.99999);

			for (int i = 0; i < scene.lights_count; ++i) {
				vec3 intensity = scene.light_intensity(i, rec, in_rays[idx]);
				color = in_rays[idx].attenuation * phong_shade(material, scene.lights[i], intensity, scene.ambient_light, point, in_rays[idx], rec);
				// TODO: atomicAdd(frame[in_rays[idx].px], color);
				atomicAdd(&frame[in_rays[idx].px].x, color.x);
				atomicAdd(&frame[in_rays[idx].px].y, color.y);
				atomicAdd(&frame[in_rays[idx].px].z, color.z);
			}

			if (scene.materials[rec.material].reflection > 0) {
				ind = atomicAdd(out_rays_count, 1);
				out_rays[ind] = Ray(point, reflect(in_rays[idx].dir, rec.normal), in_rays[idx].attenuation * scene.materials[rec.material].reflection);
			}

			if (scene.materials[rec.material].refraction > 0) {
				ind = atomicAdd(out_rays_count, 1);
				out_raus[ind] = Ray(in_rays[idx].at(rec.t * 1.000001), in_rays[idx].dir, in_rays[idx].attenuation * scene.materials[rec.material].refraction);
			}
		}
	}

	void render(const RawScene &scene, const Camera &camera, std::vector<vec3> &frame, int w, int h, int max_depth) {
		vec3 *dev_frame;
		cudaCheck(cudaMalloc(&dev_frame, sizeof(vec3) * frame.size()));
		cudaCheck(cudaMemset(dev_frame, 0, sizeof(vec3) * frame.size()));

		Ray *in_rays, *out_rays;
		int rays_count = w * h, in_rays_count = rays_count * 2, out_rays_count = in_rays_count;
		int *dev_rays_count;
		cudaCheck(cudaMalloc(&in_rays, sizeof(Ray) * in_rays_count));
		cudaCheck(cudaMalloc(&out_rays, sizeof(Ray) * out_rays_count));
		cudaCheck(cudaMalloc(&dev_rays_count, sizeof(int)));

		cast_rays<<<dim3(8, 8), dim3(8, 8)>>>(in_rays, camera, w, h);
		cudaCheck(cudaDeviceSynchronize());
		cudaCheckLastError();

		for (int i = 0; i < max_depth; ++i) {
			cudaCheck(cudaMemset(dev_rays_count, 0, sizeof(int)));

			trace_rays<<<1, 32>>>(scene, in_rays, out_rays, rays_count, dev_rays_count, frame);
			cudaCheck(cudaDeviceSynchronize());
			cudaCheckLastError();

			std::swap(in_rays, out_rays)
			std::swap(in_rays_count, out_rays_count);
			cudaCheck(cudaMalloc(&rays_count, dev_rays_count, sizeof(int), cudaMemcpyDeviceToHost));

			if (rays_count * 2 < out_rays_count) {
				cudaCheck(cudaFree(out_rays));
				out_rays_count = rays_count * 2;
				cudaCheck(cudaMalloc(&out_rays_count, sizeof(Ray) * out_rays_count));
			}
		}

		cudaCheck(cudaMemcpy(frame.data(), dev_frame, sizeof(vec3) * frame.size(), cudaMemcpyDeviceToHost));
		cudaCheck(cudaFree(dev_frame));
		cudaCheck(cudaFree(in_rays));
		cudaCheck(cudaFree(out_rays));
		cudaCheck(cudaFree(dev_rays_count));
	}
}

#endif