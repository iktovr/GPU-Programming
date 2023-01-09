#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <vector>
#include <tuple>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <string>

#include "../common/vec3.hpp"

#include "primitives.hpp"
#include "scene.hpp"
#include "render.hpp"

int main(int argc, char *argv[]) {
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

    std::cout << scene;

	double t;
	for(int k = 0; k < frames; k++) {
		t = 2 * PI / frames * k;
		camera.at(t);
		std::cout << camera.pos << '\n';
	}
	camera.at(0);
	std::cout << camera.pos << '\n';

    return 0;
}