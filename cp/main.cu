#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <vector>
#include <tuple>
#include <iomanip>
#include <cmath>
#include <string>

#define STBI
#ifdef STBI
#include "../common/stb_image_include.hpp"
#endif

#define TIME
#ifdef TIME
#include <chrono>
using namespace std::chrono;
#endif

#ifndef __CUDACC__
#define __host__
#define __device__
#endif

#include "../common/vec3.hpp"

#include "primitives.hpp"
#include "scene.hpp"
#include "render.hpp"
#include "ssaa.hpp"

int main() {
	// TODO: argv
	bool gpu = true;

	vec3 Ka(0.2), Kd(0.7), Ks(0.7);
	double p = 100;

	int frames;
	std::string path;
	int width, height;
	Camera camera;

	Material cube_material{vec3(0), Ka, Kd, Ks, p, 0.0, 0.0}, octahedron_material{vec3(0), Ka, Kd, Ks, p, 0.0, 0.0}, icosahedron_material{vec3(0), Ka, Kd, Ks, p, 0.0, 0.0};
	vec3 cube_origin, octahedron_origin, icosahedron_origin;
	double cube_scale, octahedron_scale, icosahedron_scale;
	int cube_lights, octahedron_lights, icosahedron_lights;

	Mesh floor(
		{{{5, 5, 0}, {0, 0, 1}}, {{-5, -5, 0}, {0, 0, 1}}, {{5, -5, 0}, {0, 0, 1}}, {{-5, 5, 0}, {0, 0, 1}}},
		{{0, 1, 2, 0}, {0, 3, 1, 0}}
	);
	std::string texture_path;
	Material floor_material{vec3(0), Ka, Kd, Ks, p, 0.0, 0.0};

	int lights_count;
	std::vector<Light> lights;

	int max_depth, ssaa_coeff;

	std::cin >> frames >> path >> width >> height >> camera
	         >> cube_origin >> cube_material.color >> cube_scale >> cube_material.reflection >> cube_material.refraction >> cube_lights 
	         >> octahedron_origin >> octahedron_material.color >> octahedron_scale >> octahedron_material.reflection >> octahedron_material.refraction >> octahedron_lights 
	         >> icosahedron_origin >> icosahedron_material.color >> icosahedron_scale >> icosahedron_material.reflection >> icosahedron_material.refraction >> icosahedron_lights 
	         >> floor.vertexes[0].point >> floor.vertexes[1].point >> floor.vertexes[2].point >> floor.vertexes[3].point
	         >> texture_path >> floor_material.color >> floor_material.reflection;

	std::cin >> lights_count;
	for (int i = 0; i < lights_count; ++i) {
		vec3 pos, color;
		std::cin >> pos >> color;
		lights.push_back({pos, color});
	}

	std::cin >> max_depth >> ssaa_coeff;

	Scene scene;
	scene.add_material({{0, 0, 0.6}, {0.2, 0.2, 0.2}, {0.7, 0.7, 0.7}, {0.7, 0.7, 0.7}, 100, 0, 0});
	scene.add_material(cube_material);
	scene.add_material(floor_material);

	int edge_mtl = 0, 
	    cube_mtl = 1,
	    floor_mtl = 2;

	scene.add_mesh(floor, {floor_mtl});
	Mesh cube("objects/cube.obj");
	scene.add_mesh(cube, {cube_mtl, edge_mtl}, cube_origin, cube_scale);

	scene.add_light(lights);
	scene.ambient_light = {1, 1, 1};

	// std::cout << scene;
	// return 0;

	RawScene raw_scene;
	if (gpu) {
		raw_scene = scene.get_gpu_raw_scene();
	} else {
		raw_scene = scene.get_raw_scene();
	}
	
	char buff[512];
	std::vector<vec3f> frame(width * height * ssaa_coeff * ssaa_coeff);
	std::vector<vec3c> data(width * height);

#ifdef TIME
	double frame_time = 0;
#endif

	double t;
	int field = std::floor(std::log10(frames - 1)) + 1;
	for(int k = 0; k < frames; k++) {
		t = 2 * PI / frames * k;
		camera.at(t);

#ifdef TIME
		steady_clock::time_point start = steady_clock::now();
#endif
		if (!gpu) {
			cpu::render(raw_scene, camera, frame, width * ssaa_coeff, height * ssaa_coeff, max_depth);
		} else {
			gpu::render(raw_scene, camera, frame, width * ssaa_coeff, height * ssaa_coeff, max_depth);
		}

#ifdef TIME
		steady_clock::time_point end = steady_clock::now();
		frame_time += duration_cast<nanoseconds>(end - start).count() / 1000000.0;
#endif

		// std::sprintf(buff, path.c_str(), field, k);
		std::cerr << "\rFrames remaining: " << std::setw(field) << std::setfill(' ') << (frames - k - 1);
		
		// ssaa(frame, data, width, height, ssaa_coeff);

		for (int i = 0; i < frame.size(); ++i) {
			data[i].x = (unsigned char)(std::min(frame[i].x, 1.0f) * 255);
			data[i].y = (unsigned char)(std::min(frame[i].y, 1.0f) * 255);
			data[i].z = (unsigned char)(std::min(frame[i].z, 1.0f) * 255);
		} 

		stbi_write_png(path.c_str(), width, height, 3, data.data(), width * 3);

		// std::ofstream out_file(path, std::ios::binary);
		// check(out_file.is_open(), false, "failed to open output file");

		// out_file.write(reinterpret_cast<char*>(&width), sizeof(width));
		// out_file.write(reinterpret_cast<char*>(&height), sizeof(height));
		// out_file.write(reinterpret_cast<char*>(data.data()), sizeof(vec3c) * width * height);
	}
	std::cerr << "\nConverting to gif...\n";
	// std::system("convert res/*.png res.gif");

#ifdef TIME
	frame_time /= frames;
	std::cout << "Frame time: " << frame_time << '\n';
#endif

	raw_scene.clear();

	return 0;
}