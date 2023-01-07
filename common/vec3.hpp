#pragma once

#include <iostream>
#include <cmath>

#ifndef __CUDACC__
#define __host__
#define __device__
#endif

template <class T>
struct vec3_t;

using vec3 = vec3_t<double>;
using vec3f = vec3_t<float>;
using vec3i = vec3_t<int>;
using vec3c = vec3_t<unsigned char>;

template <class T>
struct vec3_t {
	T x;
	T y;
	T z;

	__host__ __device__
	vec3_t() : x(0), y(0), z(0) {}
	__host__ __device__
	vec3_t(T x, T y, T z): x(x), y(y), z(z) {}
	__host__ __device__
	vec3_t(T a): x(a), y(a), z(a) {}

	__host__ __device__
	vec3_t<T> operator-() const { return vec3_t<T>(-x, -y, -z); }

	__host__ __device__
	template <class U>
	vec3_t<T>& operator+=(const vec3_t<U> &v) {
		x += v.x;
		y += v.y;
		z += v.z;
		return *this;
	}

	__host__ __device__
	template <class U>
	vec3_t<T>& operator*=(const vec3_t<U> &v) {
		x *= v.x;
		y *= v.y;
		z *= v.z;
		return *this;
	}

	__host__ __device__
	template <class U>
	vec3_t<T>& operator*=(const U t) {
		x *= t;
		y *= t;
		z *= t;
		return *this;
	}

	__host__ __device__
	template <class U>
	vec3_t<T>& operator/=(const U t) {
		return *this *= 1/t;
	}

	__host__ __device__
	void set_length(const T l) {
		T old_l = length();
		if (old_l < 1e-6) {
			return;
		}
		x *= l / old_l;
		y *= l / old_l;
		z *= l / old_l;
	}

	__host__ __device__
	inline T length() const {
		return std::sqrt(length_squared());
	}

	__host__ __device__
	inline T length_squared() const {
		return x * x + y * y + z * z;
	}
};

template <class T>
inline std::istream& operator>>(std::istream &is, vec3_t<T> &v) {
	return is >> v.x >> v.y >> v.z;
}

template <class T>
inline std::ostream& operator<<(std::ostream &os, const vec3_t<T> &v) {
	return os << v.x << ' ' << v.y << ' ' << v.z;
}

__host__ __device__
template <class T, class U>
inline vec3_t<T> operator+(const vec3_t<T> &u, const vec3_t<U> &v) {
	return {u.x + v.x, u.y + v.y, u.z + v.z};
}

__host__ __device__
template <class T, class U>
inline vec3_t<T> operator-(const vec3_t<T> &u, const vec3_t<U> &v) {
	return {u.x - v.x, u.y - v.y, u.z - v.z};
}

__host__ __device__
template <class T, class U>
inline vec3_t<T> operator*(const vec3_t<T> &u, const vec3_t<U> &v) {
	return {u.x * v.x, u.y * v.y, u.z * v.z};
}

__host__ __device__
template <class T, class U>
inline vec3_t<T> operator*(const T t, const vec3_t<U> &v) {
	return {t * v.x, t * v.y, t * v.z};
}

__host__ __device__
template <class T, class U>
inline vec3_t<T> operator*(const vec3_t<T> &v, const U t) {
	return {t * v.x, t * v.y, t * v.z};
}


__host__ __device__
template <class T, class U>
inline vec3_t<T> operator+(const vec3_t<T> &v, const U t) {
	return {t + v.x, t + v.y, t + v.z};
}

__host__ __device__
template <class T, class U>
inline vec3_t<T> operator-(const vec3_t<T> &v, const U t) {
	return {t - v.x, t - v.y, t - v.z};
}

__host__ __device__
template <class T, class U>
inline vec3_t<T> operator/(const vec3_t<T> &u, const vec3_t<U> &v) {
	return {u.x / v.x, u.y / v.y, u.z / v.z};
}

__host__ __device__
template <class T, class U>
inline vec3_t<T> operator/(const vec3_t<T> &v, const U t) {
	return v * (1/t);
}

__host__ __device__
template <class T>
inline T dot(const vec3_t<T> &v, const vec3_t<T> &u) {
	return u.x * v.x + u.y * v.y + u.z * v.z;
}

__host__ __device__
template <class T>
inline vec3_t<T> cross(const vec3_t<T> &v, const vec3_t<T> &u) {
	return {u.y * v.z - u.z * v.y,
	        u.z * v.x - u.x * v.z,
	        u.x * v.y - u.y * v.x};
}

__host__ __device__
template <class T>
inline vec3_t<T> norm(const vec3_t<T> &v) {
	return v / v.length();
}

__host__ __device__
template <class T>
inline vec3_t<T> matmul(vec3_t<T> a, vec3_t<T> b, vec3_t<T> c, vec3_t<T> v) {
	return {a.x * v.x + b.x * v.y + c.x * v.z,
	        a.y * v.x + b.y * v.y + c.y * v.z,
	        a.z * v.x + b.z * v.y + c.z * v.z};
}

__host__ __device__
template <class T>
inline vec3_t<T> project(vec3_t<T> a, vec3_t<T> b) {
	return dot(a, b) / b.length_squared() * b;
}

__host__ __device__
template <class T>
inline vec3_t<T> reflect(vec3_t<T> a, vec3_t<T> n) {
	return a - T(2) * project(a, n);
}