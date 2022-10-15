#pragma once

#include <vector>
#include <utility>

#ifndef __CUDACC__
struct double3 {
	double x = 0, y = 0, z = 0;

	double3() {}
	double3(double x, double y, double z) : x(x), y(y), z(z) {}
};

struct uchar4 {
	unsigned char x = 0, y = 0, z = 0, w = 0;

	uchar4() {}
	uchar4(unsigned char x, unsigned char y, unsigned char z, unsigned char w) : x(x), y(y), z(z), w(w) {}
};

#define __host__
#define __device__
#endif

inline double& get(double3& a, size_t i) {
	return (i == 0) ? a.x : ((i == 1) ? a.y : a.z);
}

inline const double& get(const double3& a, size_t i) {
	return (i == 0) ? a.x : ((i == 1) ? a.y : a.z);
}

__host__ __device__
double3 operator-(const uchar4& a, const double3& b) {
	return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__host__ __device__
double operator*(const double3& a, const double3& b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

double3& operator+=(double3& a, const uchar4& b) {
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	return a;
}

double3& operator/=(double3& a, int n) {
	a.x /= n;
	a.y /= n;
	a.z /= n;
	return a;
}

__host__ __device__
struct matrix3d {
	double data[3][3];

	matrix3d() : data{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}} {}

	__host__ __device__
	matrix3d(double a11, double a12, double a13, 
             double a21, double a22, double a23, 
             double a31, double a32, double a33) : 
			 data{{a11, a12, a13}, {a21, a22, a23}, {a31, a32, a33}} {}

	__host__ __device__
	double* operator[](size_t i) {
		return data[i];
	}

	__host__ __device__
	const double* operator[](size_t i) const {
		return data[i];
	}

	inline size_t size() const {
		return 3;
	}

	double det() {
		return data[0][0] * data[1][1] * data[2][2] + 
		       data[0][1] * data[1][2] * data[2][0] + 
		       data[0][2] * data[1][0] * data[2][1] - 
		       data[0][2] * data[1][1] * data[2][0] - 
		       data[0][0] * data[1][2] * data[2][1] - 
		       data[0][1] * data[1][0] * data[2][2];
	}

	matrix3d invert() {
		double invdet = 1. / det();
		return {
			(data[1][1] * data[2][2] - data[2][1] * data[1][2]) * invdet,
			(data[0][2] * data[2][1] - data[0][1] * data[2][2]) * invdet,
			(data[0][1] * data[1][2] - data[0][2] * data[1][1]) * invdet,
			(data[1][2] * data[2][0] - data[1][0] * data[2][2]) * invdet,
			(data[0][0] * data[2][2] - data[0][2] * data[2][0]) * invdet,
			(data[1][0] * data[0][2] - data[0][0] * data[1][2]) * invdet,
			(data[1][0] * data[2][1] - data[2][0] * data[1][1]) * invdet,
			(data[2][0] * data[0][1] - data[0][0] * data[2][1]) * invdet,
			(data[0][0] * data[1][1] - data[1][0] * data[0][1]) * invdet
		};
	}
};

matrix3d matmul(const double3& a, const double3& b) {
	return {
		a.x * b.x, a.x * b.y, a.x * b.z,
		a.y * b.x, a.y * b.y, a.y * b.z,
		a.z * b.x, a.z * b.y, a.z * b.z
	};
}

__host__ __device__
double3 operator*(const double3& a, const matrix3d& b) {
	return {
		a.x * b[0][0] + a.y * b[1][0] + a.z * b[2][0],
		a.x * b[0][1] + a.y * b[1][1] + a.z * b[2][1],
		a.x * b[0][2] + a.y * b[1][2] + a.z * b[2][2]
	};
}

template <class T>
matrix3d& operator/=(matrix3d& a, T n) {
	for (size_t i = 0; i < a.size(); ++i) {
		for (size_t j = 0; j < a.size(); ++j) {
			a[i][j] /= n;
		}
	}
	return a;
}

matrix3d& operator+=(matrix3d& a, const matrix3d& b) {
	for (size_t i = 0; i < a.size(); ++i) {
		for (size_t j = 0; j < a.size(); ++j) {
			a[i][j] += b[i][j];
		}
	}
	return a;
}
