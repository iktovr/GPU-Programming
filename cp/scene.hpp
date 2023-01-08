#pragma once

#include <vector>

#include "../common/vec3.hpp"
#include "../common/error_checkers.hpp"

#include "primitives.hpp"

template <class T>
__host__ __device__
T barycentric_interpolation(const vec3 p, const vec3 a, const vec3 b, const vec3 c, T x, T y, T z) {
	double s = cross(b - a, c - a).length();
	double u = cross(b - p, c - p).length() / s;
	double v = cross(a - p, c - p).length() / s;
	double t = 1 - u - v;
	return u * x + v * y + t * z;
}

struct MeshInfo {
	int count;
	int lights_count;
	Sphere sphere;
};

struct RawScene {
	Material *materials;
	Vertex *vertexes;
	Triangle *triangles;
	Sphere *spheres;
	int meshes_count;
	MeshInfo *meshes;
	vec2i *textures;
	int *texture_sizes;
	vec3 *texture_data;
	int lights_count;
	Light *lights;
	vec3 ambient_light;
	bool gpu;

	__host__ __device__
	bool intersect(const Ray &ray, HitRecord& rec) const {
		rec.t = -1;
		int k;
		for (int mesh = 0, tr_i = 0, sp_i = 0; mesh < meshes_count; ++mesh) {
			if (meshes[mesh].sphere.radius > 0 && !meshes[mesh].sphere.hit(ray)) {
				tr_i += meshes[mesh].count;
				sp_i += meshes[mesh].lights_count;
				continue;
			}

			double t;
			k = sp_i;
			sp_i += meshes[mesh].lights_count;
			for (; k < sp_i; ++k) {
				t = spheres[k].intersect(ray);
				if (t > 0 && (t < rec.t || rec.t < 0)) {
					rec.t = t;
					rec.triangle = -k-1;
				}
			}

			k = tr_i;
			tr_i += meshes[mesh].count;
			for (; k < tr_i; ++k) {
				t = triangles[k].intersect(vertexes, ray);
				if (t > 0 && (t < rec.t || rec.t < 0)) {
					rec.t = t;
					rec.triangle = k;
				}
			}
		}

		if (rec.t < 0) {
			return false;
		}
		
		if (rec.triangle >= 0) {
			// rec.normal = norm(vertexes[triangles[rec.triangle].a].normal + vertexes[triangles[rec.triangle].b].normal + vertexes[triangles[rec.triangle].c].normal);
			rec.normal = norm(cross(vertexes[triangles[rec.triangle].c].point - vertexes[triangles[rec.triangle].a].point, vertexes[triangles[rec.triangle].b].point - vertexes[triangles[rec.triangle].a].point));
			// rec.normal = barycentric_interpolation(
			// 	ray.at(rec.t),
			// 	vertexes[triangles[rec.triangle].a].point, vertexes[triangles[rec.triangle].b].point, vertexes[triangles[rec.triangle].c].point,
			// 	vertexes[triangles[rec.triangle].a].normal, vertexes[triangles[rec.triangle].b].normal, vertexes[triangles[rec.triangle].c].normal
			// );
			
			rec.material = triangles[rec.triangle].material;
		} else {
			rec.normal = -ray.dir;
			rec.material = EDGE_LIGHT_MTL;
		}

		if (dot(rec.normal, ray.dir) > 0) {
			rec.normal *= -1;
		}

		return true;
	}

	__host__ __device__
	vec3 light_intensity(int light, HitRecord rec, Ray ray) const {
		vec3 intensity = lights[light].intensity;
		int target_triangle = rec.triangle;
		vec3 point = ray.at(rec.t);
		ray.pos = lights[light].pos;
		ray.dir = norm(point - ray.pos);

		while (intersect(ray, rec) && rec.triangle != target_triangle) {
			if (rec.triangle < 0) {
				ray.pos = ray.at(rec.t * 1.0001);
				continue;
			}
			intensity *= materials[rec.material].color * materials[rec.material].refraction;
			if (materials[rec.material].refraction < 1e-3) {
				break;
			}
			ray.pos = ray.at(rec.t * 1.0001);
		}
		return intensity;
	}

	__host__ __device__
	vec3 get_texture_py(int texture, vec2 uv) const {
		assert(texture >= 0);
		int x = uv.x * textures[texture].x;
		int y = uv.y * textures[texture].y;

		return texture_data[texture_sizes[texture] + y * textures[texture].x + x];
	}

	__host__ __device__
	vec3 get_texture_px(int triangle, const vec3 &point) const {
		assert(triangle >= 0);
		vec2 uv = barycentric_interpolation(
			point,
			vertexes[triangles[triangle].a].point, vertexes[triangles[triangle].b].point, vertexes[triangles[triangle].c].point,
			triangles[triangle].uv_a, triangles[triangle].uv_b, triangles[triangle].uv_c
		);
		return get_texture_py(triangles[triangle].texture, uv);
	}

	void clear() {
		if (gpu) {
			cudaCheck(cudaFree(materials));
			cudaCheck(cudaFree(vertexes));
			cudaCheck(cudaFree(triangles));
			cudaCheck(cudaFree(spheres));
			cudaCheck(cudaFree(meshes));
			cudaCheck(cudaFree(textures));
			cudaCheck(cudaFree(texture_sizes));
			cudaCheck(cudaFree(texture_data));
			cudaCheck(cudaFree(lights));
		}
	}
};

struct Scene {
	std::vector<Material> materials;
	std::vector<Vertex> vertexes;
	std::vector<Triangle> triangles;
	std::vector<Sphere> spheres;
	std::vector<MeshInfo> meshes;
	std::vector<vec2i> textures;
	std::vector<int> texture_sizes;
	std::vector<vec3> texture_data;
	std::vector<Light> lights;
	vec3 ambient_light;

	RawScene get_raw_scene() {
		return {
			materials.data(),
			vertexes.data(),
			triangles.data(),
			spheres.data(),
			static_cast<int>(meshes.size()),
			meshes.data(),
			textures.data(),
			texture_sizes.data(),
			texture_data.data(),
			static_cast<int>(lights.size()),
			lights.data(),
			ambient_light,
			false
		};
	}

	RawScene get_gpu_raw_scene() {
		Material *dev_materials;
		Vertex *dev_vertexes;
		Triangle *dev_triangles;
		Sphere *dev_spheres;
		MeshInfo *dev_meshes;
		vec2i *dev_textures;
		int *dev_texture_sizes;
		vec3 *dev_texture_data;
		Light *dev_lights;

		cudaCheck(cudaMalloc(&dev_materials, materials.size() * sizeof(Material)));
		cudaCheck(cudaMalloc(&dev_vertexes, vertexes.size() * sizeof(Vertex)));
		cudaCheck(cudaMalloc(&dev_triangles, triangles.size() * sizeof(Triangle)));
		cudaCheck(cudaMalloc(&dev_spheres, spheres.size() * sizeof(Sphere)));
		cudaCheck(cudaMalloc(&dev_meshes, meshes.size() * sizeof(MeshInfo)));
		cudaCheck(cudaMalloc(&dev_textures, textures.size() * sizeof(vec2i)));
		cudaCheck(cudaMalloc(&dev_texture_sizes, texture_sizes.size() * sizeof(int)));
		cudaCheck(cudaMalloc(&dev_texture_data, texture_data.size() * sizeof(vec3)));
		cudaCheck(cudaMalloc(&dev_lights, lights.size() * sizeof(Light)));

		cudaCheck(cudaMemcpy(dev_materials, materials.data(), materials.size() * sizeof(Material), cudaMemcpyHostToDevice));
		cudaCheck(cudaMemcpy(dev_vertexes, vertexes.data(), vertexes.size() * sizeof(Vertex), cudaMemcpyHostToDevice));
		cudaCheck(cudaMemcpy(dev_triangles, triangles.data(), triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice));
		cudaCheck(cudaMemcpy(dev_spheres, spheres.data(), spheres.size() * sizeof(Sphere), cudaMemcpyHostToDevice));
		cudaCheck(cudaMemcpy(dev_meshes, meshes.data(), meshes.size() * sizeof(MeshInfo), cudaMemcpyHostToDevice));
		cudaCheck(cudaMemcpy(dev_textures, textures.data(), textures.size() * sizeof(vec2i), cudaMemcpyHostToDevice));
		cudaCheck(cudaMemcpy(dev_texture_sizes, texture_sizes.data(), texture_sizes.size() * sizeof(int), cudaMemcpyHostToDevice));
		cudaCheck(cudaMemcpy(dev_texture_data, texture_data.data(), texture_data.size() * sizeof(vec3), cudaMemcpyHostToDevice));
		cudaCheck(cudaMemcpy(dev_lights, lights.data(), lights.size() * sizeof(Light), cudaMemcpyHostToDevice));
		
		return {
			dev_materials,
			dev_vertexes,
			dev_triangles,
			dev_spheres,
			static_cast<int>(meshes.size()),
			dev_meshes,
			dev_textures,
			dev_texture_sizes,
			dev_texture_data,
			static_cast<int>(lights.size()),
			dev_lights,
			ambient_light,
			true
		};
	}

	int add_material(Material material) {
		materials.push_back(material);
		return materials.size() - 1;
	}

	void load_texture(std::string path) {
		std::ifstream file(path, std::ios::binary);
		check(file.is_open(), false, "failed to open texture file");

		vec2i tex;
		file.read(reinterpret_cast<char*>(&tex.x), sizeof(int));
		file.read(reinterpret_cast<char*>(&tex.y), sizeof(int));

		std::vector<vec3> data(tex.x * tex.y);
		uchar4 color;
		for (int i = 0; i < tex.x * tex.y; ++i) {
			file.read(reinterpret_cast<char*>(&color), sizeof(uchar4));
			data[i] = {color.x / 255.0f, color.y / 255.0f, color.z / 255.0f};
		}

		if (texture_sizes.empty()) {
			texture_sizes.push_back(0);
		} else {
			texture_sizes.push_back(texture_sizes.back() + tex.x * tex.y);
		}

		textures.push_back(tex);
		texture_data.insert(texture_data.end(), data.begin(), data.end());
	}

	void add_light(Light light) {
		lights.push_back(light);
	}

	void add_light(const std::vector<Light> &light) {
		lights.insert(lights.end(), light.begin(), light.end());
	}

	void add_mesh(const Mesh &mesh, std::vector<int> materials, std::vector<int> textures = {}, vec3 origin = {0, 0, 0}, double scale = 1) {
		int vertexes_offset = 0;
		double radius = 0;

		if (!mesh.vertexes.empty()) {
			vertexes_offset = vertexes.size();
			for (auto vertex: mesh.vertexes) {
				radius = std::max(radius, vertex.point.length() * scale);
				vertex.point = vertex.point * scale + origin;
				vertexes.push_back(vertex);
			}
		}

		if (!mesh.spheres.empty()) {
			for (auto sphere: mesh.spheres) {
				sphere.center = sphere.center * scale + origin;
				spheres.push_back(sphere);
			}
		}

		meshes.push_back({static_cast<int>(mesh.triangles.size()), static_cast<int>(mesh.spheres.size()), {radius, origin}});
		for (auto triangle: mesh.triangles) {
			triangle.a += vertexes_offset;
			triangle.b += vertexes_offset;
			triangle.c += vertexes_offset;
			if (!materials.empty()) {
				triangle.material = materials[triangle.material];
			}
			if (!textures.empty()) {
				triangle.texture = textures[triangle.texture];
			}
			triangles.push_back(triangle);
		}
	}
};

std::ostream& operator<<(std::ostream &os, const Scene &scene) {
	for (int mesh = 0, i = 0; mesh < (int)scene.meshes.size(); ++mesh) {
		int k = i;
		i += scene.meshes[mesh].count;
		for (; k < i; ++k) {
			const Vertex &a = scene.vertexes[scene.triangles[k].a];
			const Vertex &b = scene.vertexes[scene.triangles[k].b];
			const Vertex &c = scene.vertexes[scene.triangles[k].c];
			vec3 center = (a.point + b.point + c.point) / 3.0; 
			vec3 normal = norm(a.normal + b.normal + c.normal);
			os << a.point << '\n' << b.point << '\n' << c.point << '\n' << a.point << "\n\n" <<
			      center << '\n' << center + normal << "\n\n";
		}
	}

	for (auto& sphere: scene.spheres) {
		os << (sphere.center - vec3{0, 0, sphere.radius}) << '\n' << (sphere.center + vec3{0, 0, sphere.radius}) << "\n\n" << 
		      (sphere.center - vec3{0, sphere.radius, 0}) << '\n' << (sphere.center + vec3{0, sphere.radius, 0}) << "\n\n" << 
		      (sphere.center - vec3{sphere.radius, 0, 0}) << '\n' << (sphere.center + vec3{sphere.radius, 0, 0}) << "\n\n";
	}

	for (auto &light: scene.lights) {
		os << light.pos << "\n\n";
	}

	return os;
}
