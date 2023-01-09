#pragma once

#include <cmath>

#include "../common/vec3.hpp"
#include "../common/error_checkers.hpp"

#include "primitives.hpp"
#include "scene.hpp"
#include "ssaa.hpp"

using std::max;
using std::min;
using std::pow;

__host__ __device__
vec3 phong_shade(const Material &material, const Light &light, const vec3 &intensity, const vec3 &ambient_light, const vec3 &point, const Ray &ray, const HitRecord &rec) {
	if (rec.material == EDGE_LIGHT_MTL) {
		return intensity * EDGE_LIGHT_INTENSITY;
	}
	
	vec3 to_light = norm(light.pos - point);
	vec3 reflect_light = norm(reflect(-to_light, rec.normal));

	double diffuse = max(dot(to_light, rec.normal), 0.0);
	double specular = (diffuse > 0) ? pow(max(dot(reflect_light, -ray.dir), 0.0), material.p) : 0;

	return material.color * (material.Ka * ambient_light + intensity * (diffuse * material.Kd + specular * material.Ks)); 
}

namespace cpu {
	vec3 ray_color(const Ray& ray, const RawScene& scene, int max_depth, int &total_rays_count, int depth = 0) {
		if (depth == max_depth) {
			return {0, 0, 0};
		}

		++total_rays_count;

		HitRecord rec;
		if (!scene.intersect(ray, rec))
			return {0, 0, 0};

		vec3 point = ray.at(rec.t * 0.99999);
		vec3 color{0, 0, 0};
		const Material &material = scene.materials[rec.material];

		for (int i = 0; i < scene.lights_count; ++i) {
			vec3 intensity = scene.light_intensity(i, rec, ray);
			if (rec.triangle >= 0 && scene.triangles[rec.triangle].texture >= 0) {
				vec3 texture_color = scene.get_texture_px(rec.triangle, point);
				color += texture_color * phong_shade(material, scene.lights[i], intensity, scene.ambient_light, point, ray, rec);
			} else {
				color += phong_shade(material, scene.lights[i], intensity, scene.ambient_light, point, ray, rec);
			}
		}

		if (material.reflection > 0) {
			Ray reflect_ray(point, reflect(ray.dir, rec.normal));
			color += material.reflection * ray_color(reflect_ray, scene, max_depth, total_rays_count, depth + 1);
		}

		if (material.refraction > 0) {
			Ray refract_ray(ray.at(rec.t * 1.000001), ray.dir);
			color += material.refraction * ray_color(refract_ray, scene, max_depth, total_rays_count, depth + 1);
		}

		return color;
	}

	int render(const RawScene &scene, const Camera &camera, std::vector<uchar4> &res_frame, int width, int height, int ssaa_coeff, int max_depth) {
		int w = width * ssaa_coeff, h = height * ssaa_coeff;
		std::vector<vec3f> frame(w * h); 
		double dw = 2.0 / (w - 1.0);
		double dh = 2.0 / (h - 1.0);
		double z = 1.0 / std::tan(camera.angle * PI / 360.0);
		vec3 bz = norm(camera.dir - camera.pos);
		vec3 bx = norm(cross(bz, {0.0, 0.0, 1.0}));
		vec3 by = norm(cross(bx, bz));
		Ray ray(camera.pos, {0, 0, 0});
		int total_rays_count = 0;
		for(int i = 0; i < w; i++) {	
			for(int j = 0; j < h; j++) {
				vec3 v = {-1.0 + dw * i, (-1.0 + dh * j) * h / w, z};
				ray.dir = norm(matmul(bx, by, bz, v));
				vec3 color = ray_color(ray, scene, max_depth, total_rays_count);
				frame[(h - 1 - j) * w + i].x = (float)color.x;
				frame[(h - 1 - j) * w + i].y = (float)color.y;
				frame[(h - 1 - j) * w + i].z = (float)color.z;
			}
		}

		cpu::ssaa(frame, res_frame, w, h, ssaa_coeff);
		return total_rays_count;
	}
}

namespace gpu {
	__global__ void cast_rays(Ray *rays, const Camera *const camera, int w, int h) {
		int idx = blockDim.x * blockIdx.x + threadIdx.x;
		int idy = blockDim.y * blockIdx.y + threadIdx.y;
		int offsetx = gridDim.x * blockDim.x;
		int offsety = gridDim.y * blockDim.y;

		double dw = 2.0 / (w - 1.0);
		double dh = 2.0 / (h - 1.0);
		double z = 1.0 / tan(camera->angle * PI / 360.0);
		vec3 bz = norm(camera->dir - camera->pos);
		vec3 bx = norm(cross(bz, {0.0, 0.0, 1.0}));
		vec3 by = norm(cross(bx, bz));
		for(int i = idx; i < w; i += offsetx) {
			for(int j = idy; j < h; j += offsety) {
				vec3 v = {-1.0 + dw * i, (-1.0 + dh * j) * h / w, z};
				rays[(h - 1 - j) * w + i] = {camera->pos, norm(matmul(bx, by, bz, v)), (h - 1 - j) * w + i, 1};
			}
		}
	}

	__global__ void trace_rays(const RawScene *const scene, Ray *in_rays, Ray *out_rays, int rays_count, int *out_rays_count, vec3f *frame) {
		int idx = blockDim.x * blockIdx.x + threadIdx.x;
		int offset = gridDim.x * blockDim.x;

		HitRecord rec;
		vec3 point, color;
		for (; idx < rays_count; idx += offset) {
			if (!scene->intersect(in_rays[idx], rec)) {
				continue;
			}

			point = in_rays[idx].at(rec.t * 0.99999);

			for (int i = 0; i < scene->lights_count; ++i) {
				vec3 intensity = scene->light_intensity(i, rec, in_rays[idx]);
				color = in_rays[idx].attenuation * phong_shade(scene->materials[rec.material], scene->lights[i], intensity, scene->ambient_light, point, in_rays[idx], rec);
				if (rec.triangle >= 0 && scene->triangles[rec.triangle].texture >= 0) {
					color *= scene->get_texture_px(rec.triangle, point);
				}

				atomicAdd(&frame[in_rays[idx].px].x, (float)color.x);
				atomicAdd(&frame[in_rays[idx].px].y, (float)color.y);
				atomicAdd(&frame[in_rays[idx].px].z, (float)color.z);
			}

			if (scene->materials[rec.material].reflection > 0) {
				out_rays[atomicAdd(out_rays_count, 1)] = {point, reflect(in_rays[idx].dir, rec.normal), in_rays[idx].px, in_rays[idx].attenuation * scene->materials[rec.material].reflection};
			}

			if (scene->materials[rec.material].refraction > 0) {
				out_rays[atomicAdd(out_rays_count, 1)] = {in_rays[idx].at(rec.t * 1.000001), in_rays[idx].dir, in_rays[idx].px, in_rays[idx].attenuation * scene->materials[rec.material].refraction};
			}
		}
	}

	int render(const RawScene &scene, const Camera &camera, std::vector<uchar4> &res_frame, int width, int height, int ssaa_coeff, int max_depth) {
		vec3f *dev_frame;
		cudaCheck(cudaMalloc(&dev_frame, sizeof(vec3f) * width * height * ssaa_coeff * ssaa_coeff));
		cudaCheck(cudaMemset(dev_frame, 0, sizeof(vec3f) * width * height * ssaa_coeff * ssaa_coeff));

		Camera *dev_camera;
		cudaCheck(cudaMalloc(&dev_camera, sizeof(Camera)));
		cudaCheck(cudaMemcpy(dev_camera, &camera, sizeof(Camera), cudaMemcpyHostToDevice));

		RawScene *dev_scene;
		cudaCheck(cudaMalloc(&dev_scene, sizeof(RawScene)));
		cudaCheck(cudaMemcpy(dev_scene, &scene, sizeof(RawScene), cudaMemcpyHostToDevice));

		Ray *in_rays, *out_rays;
		int rays_count = width * height * ssaa_coeff * ssaa_coeff,
		    in_rays_count = rays_count * 2, out_rays_count = rays_count * 2;
		int *dev_rays_count;
		cudaCheck(cudaMalloc(&in_rays, sizeof(Ray) * in_rays_count));
		cudaCheck(cudaMalloc(&out_rays, sizeof(Ray) * out_rays_count));
		cudaCheck(cudaMalloc(&dev_rays_count, sizeof(int)));

		cast_rays<<<dim3(32, 32), dim3(16, 32)>>>(in_rays, dev_camera, width * ssaa_coeff, height * ssaa_coeff);
		cudaCheck(cudaDeviceSynchronize());
		cudaCheckLastError();

		int total_rays_count = rays_count;

		for (int i = 0; i < max_depth; ++i) {
			cudaCheck(cudaMemset(dev_rays_count, 0, sizeof(int)));

			trace_rays<<<256, 128>>>(dev_scene, in_rays, out_rays, rays_count, dev_rays_count, dev_frame);
			cudaCheck(cudaDeviceSynchronize());
			cudaCheckLastError();

			std::swap(in_rays, out_rays);
			std::swap(in_rays_count, out_rays_count);
			cudaCheck(cudaMemcpy(&rays_count, dev_rays_count, sizeof(int), cudaMemcpyDeviceToHost));

			if (rays_count * 2 > out_rays_count) {
				cudaCheck(cudaFree(out_rays));
				out_rays_count = rays_count * 2;
				cudaCheck(cudaMalloc(&out_rays, sizeof(Ray) * out_rays_count));
			}
			
			if (i != max_depth - 1) {
				total_rays_count += rays_count;
			}
		}

		uchar4 *dev_res_frame;
		cudaCheck(cudaMalloc(&dev_res_frame, sizeof(uchar4) * res_frame.size()));

		gpu::ssaa<<<dim3(32, 32), dim3(16, 32)>>>(dev_frame, dev_res_frame, width, height, ssaa_coeff);
		cudaCheck(cudaDeviceSynchronize());
		cudaCheckLastError();
		
		cudaCheck(cudaMemcpy(res_frame.data(), dev_res_frame, sizeof(uchar4) * res_frame.size(), cudaMemcpyDeviceToHost));
		
		cudaCheck(cudaFree(dev_frame));
		cudaCheck(cudaFree(dev_res_frame));
		cudaCheck(cudaFree(in_rays));
		cudaCheck(cudaFree(out_rays));
		cudaCheck(cudaFree(dev_rays_count));
		cudaCheck(cudaFree(dev_camera));
		cudaCheck(cudaFree(dev_scene));

		return total_rays_count;
	}
}