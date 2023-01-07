#pragma once

#include <vector>
#include <cmath>
#include <fstream>
#include <string>
#include <iostream>

#include "../common/vec3.hpp"

const double PI = 2 * std::acos(0);

using std::sin;
using std::cos;

struct Camera {
	vec3 pos;
	vec3 dir;
	double angle;

	double r_c, z_c, phi_c, A_r_c, A_z_c, w_r_c, w_z_c, w_phi_c, p_r_c, p_z_c;
	double r_n, z_n, phi_n, A_r_n, A_z_n, w_r_n, w_z_n, w_phi_n, p_r_n, p_z_n;

	void at(double t) {
		double r = r_c + A_r_c * sin(w_r_c * t + p_r_c);
		double z = z_c + A_z_c * sin(w_z_c * t + p_z_c);
		double phi = phi_c + w_phi_c * t;
		pos = vec3{r * cos(phi), r * sin(phi), z};

		r = r_n + A_r_n * sin(w_r_n * t + p_r_n);
		z = z_n + A_z_n * sin(w_z_n * t + p_z_n);
		phi = phi_n + w_phi_n * t;
		dir = vec3{r * cos(phi), r * sin(phi), z};
	}
};

std::istream& operator>>(std::istream &is, Camera &c) {
	is >> c.angle >> c.r_c >> c.z_c >> c.phi_c >> c.A_r_c >> c.A_z_c >> c.w_r_c >> c.w_z_c >> c.w_phi_c >> c.p_r_c >> c.p_z_c
	   >> c.r_n >> c.z_n >> c.phi_n >> c.A_r_n >> c.A_z_n >> c.w_r_n >> c.w_z_n >> c.w_phi_n >> c.p_r_n >> c.p_z_n;
	return is;
}

struct Ray {
	vec3 pos, dir;
	int px;
	double attenuation = 1;

	__host__ __device__
	Ray(vec3 pos, vec3 dir, int px = -1): pos(pos), dir(dir), px(px) {}

	__host__ __device__
	vec3 at(const double t) const {
		return pos + t * dir;
	}
};

struct Vertex {
	vec3 point, normal;
};

struct Triangle {
	int a, b, c, material;

	// TODO: переписать самому
	__host__ __device__
	double intersect(const Vertex *const vertexes, const Ray &ray) {
		vec3 e1 = vertexes[b].point - vertexes[a].point;
		vec3 e2 = vertexes[c].point - vertexes[a].point;
		vec3 p = cross(ray.dir, e2);
		double div = dot(p, e1);
		if (std::fabs(div) < 1e-10)
			return -1;
		vec3 t = ray.pos - vertexes[a].point;
		double u = dot(p, t) / div;
		if (u < 0.0 || u > 1.0)
			return -1;
		vec3 q = cross(t, e1);
		double v = dot(q, ray.dir) / div;
		if (v < 0.0 || v + u > 1.0)
			return -1;
		double ts = dot(q, e2) / div; 
		if (ts < 0.0)
			return -1;
		return ts;
	}
};

struct HitRecord {
	double t = 0;
	int triangle = -1;
	int material = -1;
	vec3 normal;
};

struct Material {
	vec3 color;
	vec3 Ka, Kd, Ks;
	double p;
	double reflection, refraction;
};

struct Light {
	vec3 pos;
	vec3 intensity;
};

struct Sphere {
	double radius;
	vec3 center;

	__host__ __device__
	bool hit(const Ray &ray) {
		vec3 oc = ray.pos - center;
		double half_b = dot(oc, ray.dir);
		double c = oc.length_squared() - radius * radius;
		
		double discriminant = half_b * half_b - c;
		if (!(discriminant < 0)) {
			double sqrtd = sqrt(discriminant);
			double t = -half_b - sqrtd;
			if (t <= 0) {
				t = -half_b + sqrtd;
			}
			return t > 0;
		}
		return false;
	}
};

struct Mesh {
	std::vector<Vertex> vertexes;
	std::vector<Triangle> triangles;

	Mesh() = default;

	Mesh(std::vector<Vertex> vertexes, std::vector<Triangle> triangles) :
		vertexes(vertexes), triangles(triangles) {}

	// TODO: ручная триангуляция N-гонов, вынести в функцию
	Mesh(std::string obj) : vertexes(), triangles() {
		std::ifstream file(obj);

		std::vector<vec3> normals;
		std::string str;
		int v, vn;
		char tmp;
		vec3 vec;
		int material = 0;

		while (file >> str) {
			if (str == "v") {
				file >> vec;
				vertexes.push_back({vec, {0, 0, 0}});
			} else if (str == "vn") {
				file >> vec;
				normals.push_back(vec);
			} else if (str == "usemtl") {
				file >> material;
			} else if (str == "f") {
				Triangle triangle;
				file >> v >> tmp >> tmp >> vn;
				--v; --vn;
				triangle.a = v;
				vertexes[v].normal += normals[vn];
				
				file >> v >> tmp >> tmp >> vn;
				--v; --vn;
				triangle.b = v;
				vertexes[v].normal += normals[vn];
				
				file >> v >> tmp >> tmp >> vn;
				--v; --vn;
				triangle.c = v;
				vertexes[v].normal += normals[vn];

				triangle.material = material;
				triangles.push_back(triangle);
			} else {
				std::getline(file, str);
			}
		}
	}
};

