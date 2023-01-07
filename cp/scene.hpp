#pragma once

#include <vector>

#include "../common/vec3.hpp"
#include "../common/error_checkers.hpp"

#include "primitives.hpp"

struct MeshInfo {
	int count;
	Sphere sphere;
};

struct RawScene {
	Material *materials;
	Vertex *vertexes;
	Triangle *triangles;
	Sphere *spheres;
	int meshes_count;
	MeshInfo *meshes;
	int lights_count;
	Light *lights;
	vec3 ambient_light;
	bool gpu;

	__host__ __device__
	bool intersect(const Ray &ray, HitRecord& rec) const {
		rec.t = -1;
		int k;
		for (int mesh = 0, i = 0; mesh < meshes_count; ++mesh) {
			k = i;
			i += meshes[mesh].count;

			if (meshes[mesh].sphere.radius > 0 && !meshes[mesh].sphere.hit(ray)) {
				continue;
			}

			double t;
			for (; k < i; ++k) {
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

		// vec3 normal = norm(vertexes[triangles[rec.triangle].a].normal + vertexes[triangles[rec.triangle].b].normal + vertexes[triangles[rec.triangle].c].normal);
		rec.normal = norm(cross(vertexes[triangles[rec.triangle].c].point - vertexes[triangles[rec.triangle].a].point, vertexes[triangles[rec.triangle].b].point - vertexes[triangles[rec.triangle].a].point));
		
		if (dot(rec.normal, ray.dir) > 0) {
			rec.normal *= -1;
		}

		rec.material = triangles[rec.triangle].material;

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
			intensity *= materials[rec.material].color * materials[rec.material].refraction;
			if (materials[rec.material].refraction < 1e-3) {
				break;
			}
			ray.pos = ray.at(rec.t * 1.0001);
		}
		return intensity;
	}

	void clear() {
		if (gpu) {
			cudaCheck(cudaFree(materials));
			cudaCheck(cudaFree(vertexes));
			cudaCheck(cudaFree(triangles));
			// cudaCheck(cudaFree(spheres));
			cudaCheck(cudaFree(meshes));
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
		Light *dev_lights;

		cudaCheck(cudaMalloc(&dev_materials, materials.size() * sizeof(Material)));
		cudaCheck(cudaMalloc(&dev_vertexes, vertexes.size() * sizeof(Vertex)));
		cudaCheck(cudaMalloc(&dev_triangles, triangles.size() * sizeof(Triangle)));
		// cudaCheck(cudaMalloc(&dev_spheres, spheres.size() * sizeof(Sphere)));
		cudaCheck(cudaMalloc(&dev_meshes, meshes.size() * sizeof(Mesh)));
		cudaCheck(cudaMalloc(&dev_lights, lights.size() * sizeof(Light)));

		cudaCheck(cudaMemcpy(dev_materials, materials.data(), materials.size() * sizeof(Material), cudaMemcpyHostToDevice));
		cudaCheck(cudaMemcpy(dev_vertexes, vertexes.data(), vertexes.size() * sizeof(Vertex), cudaMemcpyHostToDevice));
		cudaCheck(cudaMemcpy(dev_triangles, triangles.data(), triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice));
		// cudaCheck(cudaMemcpy(dev_spheres, spheres.data(), spheres.size() * sizeof(Sphere), cudaMemcpyHostToDevice));
		cudaCheck(cudaMemcpy(dev_meshes, meshes.data(), meshes.size() * sizeof(Mesh), cudaMemcpyHostToDevice));
		cudaCheck(cudaMemcpy(dev_lights, lights.data(), lights.size() * sizeof(Light), cudaMemcpyHostToDevice));

		return {
			dev_materials,
			dev_vertexes,
			dev_triangles,
			dev_spheres,
			static_cast<int>(meshes.size()),
			dev_meshes,
			static_cast<int>(lights.size()),
			dev_lights,
			ambient_light,
			true
		};
	}

	void add_material(Material material) {
		materials.push_back(material);
	}

	void add_light(Light light) {
		lights.push_back(light);
	}

	void add_light(const std::vector<Light> &light) {
		lights.insert(lights.end(), light.begin(), light.end());
	}

	void add_mesh(const Mesh &mesh, std::vector<int> materials, vec3 origin = {0, 0, 0}, double scale = 1) {
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

		meshes.push_back({static_cast<int>(mesh.triangles.size()), {radius, origin}});
		for (auto triangle: mesh.triangles) {
			triangle.a += vertexes_offset;
			triangle.b += vertexes_offset;
			triangle.c += vertexes_offset;
			if (!materials.empty()) {
				triangle.material = materials[triangle.material];
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
