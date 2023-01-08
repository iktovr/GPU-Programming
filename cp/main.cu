#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <vector>
#include <tuple>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <string>

#define STBI
#ifdef STBI
#include "../common/stb_image_include.hpp"
#endif

#include "../common/vec3.hpp"
#include "../common/cuda_timer.hpp"

#include "primitives.hpp"
#include "scene.hpp"
#include "render.hpp"

int main(int argc, char *argv[]) {
	bool gpu = true;
	bool print_default = false;
	if (argc > 1) {
		if (std::strcmp(argv[1], "--cpu") == 0) {
			gpu = false;
		}
		if (std::strcmp(argv[1], "--default") == 0) {
			print_default = true;
		}
	}

	if (print_default) {
		std::cout << 
			"5\n"
			"res/%d.png\n"
			"1920 1080 120\n"
			"6 5 1.57079632679489 0 2 0 2 1 0 0\n"
			"3 0 4.71238898038469 0 0 0 0 1 0 0\n"
			"2.75 0.29 3\n"
			"0 0.3 0.3\n"
			"1.8\n"
			"0.4 0.4 4\n"
			"-1.5 -2 2.5\n"
			"0.3 0.3 0\n"
			"2\n"
			"0.4 0.4 5\n"
			"-1.48 2.6 4 \n"
			"0 0.3 0\n"
			"2.5\n"
			"0.4 0.4 3\n"
			"5 5 0 5 -5 0.5 -5 -5 0 -5 5 0.5\n"
			"texture.data\n"
			"0.8 0.8 0.8 0.5\n"
			"2\n"
			"5 -5 10 0.6 0.6 0.6\n"
			"5 5 10 0.6 0.6 0.6\n"
			"6 1\n";
		return 0;
	}


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
		{{{5, 5, 0}, {0, 0, 1}}, {{5, -5, 0}, {0, 0, 1}}, {{-5, -5, 0}, {0, 0, 1}}, {{-5, 5, 0}, {0, 0, 1}}},
		{{0, 1, 2, 0, 0, {1, 1}, {1, 0}, {0, 0}}, {2, 3, 0, 0, 0, {0, 0}, {0, 1}, {1, 1}}}
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
	int edge_mtl = scene.add_material({{0.2, 0.2, 0.2}, {0.2, 0.2, 0.2}, {0.7, 0.7, 0.7}, {0.7, 0.7, 0.7}, 100, 0, 0}),
	    cube_mtl = scene.add_material(cube_material),
	    octahedron_mtl = scene.add_material(octahedron_material),
	    icosahedron_mtl = scene.add_material(icosahedron_material),
	    floor_mtl = scene.add_material(floor_material);
	scene.load_texture(texture_path);

	scene.add_mesh(floor, {floor_mtl});
	Mesh cube("objects/cube.obj", cube_lights);
	scene.add_mesh(cube, {cube_mtl, edge_mtl}, {}, cube_origin, cube_scale);
	Mesh octahedron("objects/octahedron.obj", octahedron_lights);
	scene.add_mesh(octahedron, {octahedron_mtl, edge_mtl}, {}, octahedron_origin, octahedron_scale);
	Mesh icosahedron("objects/icosahedron.obj", icosahedron_lights);
	scene.add_mesh(icosahedron, {icosahedron_mtl, edge_mtl}, {}, icosahedron_origin, icosahedron_scale);

	scene.add_light(lights);
	scene.ambient_light = {1, 1, 1};

	RawScene raw_scene;
	if (gpu) {
		raw_scene = scene.get_gpu_raw_scene();
	} else {
		raw_scene = scene.get_raw_scene();
	}
	
	char buff[512];
	std::vector<uchar4> frame(width * height);

	double t;
	float time;
	int rays;
	for(int k = 0; k < frames; k++) {
		t = 2 * PI / frames * k;
		camera.at(t);

		cudaStartTimer();

		if (gpu) {
			rays = gpu::render(raw_scene, camera, frame, width, height, ssaa_coeff, max_depth);
		} else {
			rays = cpu::render(raw_scene, camera, frame, width, height, ssaa_coeff, max_depth);
		}

		cudaEndTimer(time);

		std::cout << k << '\t' << time << '\t' << rays << '\n';

		std::sprintf(buff, path.c_str(), k);

#ifdef STBI
		stbi_write_png(buff, width, height, 4, frame.data(), width * 4);
#else
		std::ofstream out_file(path, std::ios::binary);
		check(out_file.is_open(), false, "failed to open output file");

		out_file.write(reinterpret_cast<char*>(&width), sizeof(width));
		out_file.write(reinterpret_cast<char*>(&height), sizeof(height));
		out_file.write(reinterpret_cast<char*>(res_frame.data()), sizeof(uchar4) * width * height);
#endif
	}

	raw_scene.clear();

	return 0;
}