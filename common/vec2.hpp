#pragma once

#include <iostream>
#include <cmath>

#ifndef __CUDACC__
#define __host__
#define __device__
#endif

template <class T>
struct vec2_t;

using vec2 = vec2_t<double>;
using vec2f = vec2_t<float>;
using vec2i = vec2_t<int>;
using vec2c = vec2_t<unsigned char>;

template <class T>
struct vec2_t {
	T x;
	T y;

	__host__ __device__
	vec2_t() : x(0), y(0) {}
	__host__ __device__
	vec2_t(T x, T y): x(x), y(y) {}
	__host__ __device__
	vec2_t(T a): x(a), y(a) {}

	__host__ __device__
	vec2_t<T> operator-() const { return vec2_t<T>(-x, -y); }

	template <class U>
	__host__ __device__
	vec2_t<T>& operator+=(const vec2_t<U> &v) {
		x += v.x;
		y += v.y;
		return *this;
	}

	template <class U>
	__host__ __device__
	vec2_t<T>& operator*=(const vec2_t<U> &v) {
		x *= v.x;
		y *= v.y;
		return *this;
	}

	template <class U>
	__host__ __device__
	vec2_t<T>& operator*=(const U t) {
		x *= t;
		y *= t;
		return *this;
	}

	template <class U>
	__host__ __device__
	vec2_t<T>& operator/=(const U t) {
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
	}

	__host__ __device__
	inline T length() const {
		return std::sqrt(length_squared());
	}

	__host__ __device__
	inline T length_squared() const {
		return x * x + y * y;
	}
};

template <class T>
inline std::istream& operator>>(std::istream &is, vec2_t<T> &v) {
	return is >> v.x >> v.y;
}

template <class T>
inline std::ostream& operator<<(std::ostream &os, const vec2_t<T> &v) {
	return os << v.x << ' ' << v.y;
}

template <class T, class U>
__host__ __device__
inline vec2_t<T> operator+(const vec2_t<T> &u, const vec2_t<U> &v) {
	return {u.x + v.x, u.y + v.y};
}

template <class T, class U>
__host__ __device__
inline vec2_t<T> operator-(const vec2_t<T> &u, const vec2_t<U> &v) {
	return {u.x - v.x, u.y - v.y};
}

template <class T, class U>
__host__ __device__
inline vec2_t<T> operator*(const vec2_t<T> &u, const vec2_t<U> &v) {
	return {u.x * v.x, u.y * v.y};
}

template <class T, class U>
__host__ __device__
inline vec2_t<T> operator*(const T t, const vec2_t<U> &v) {
	return {t * v.x, t * v.y};
}

template <class T, class U>
__host__ __device__
inline vec2_t<T> operator*(const vec2_t<T> &v, const U t) {
	return {t * v.x, t * v.y};
}

template <class T, class U>
__host__ __device__
inline vec2_t<T> operator+(const vec2_t<T> &v, const U t) {
	return {t + v.x, t + v.y};
}

template <class T, class U>
__host__ __device__
inline vec2_t<T> operator-(const vec2_t<T> &v, const U t) {
	return {t - v.x, t - v.y};
}

template <class T, class U>
__host__ __device__
inline vec2_t<T> operator/(const vec2_t<T> &u, const vec2_t<U> &v) {
	return {u.x / v.x, u.y / v.y};
}

template <class T, class U>
__host__ __device__
inline vec2_t<T> operator/(const vec2_t<T> &v, const U t) {
	return v * (1/t);
}

template <class T>
__host__ __device__
inline T dot(const vec2_t<T> &v, const vec2_t<T> &u) {
	return u.x * v.x + u.y * v.y;
}

template <class T>
__host__ __device__
inline vec2_t<T> norm(const vec2_t<T> &v) {
	return v / v.length();
}

template <class T>
__host__ __device__
inline vec2_t<T> project(vec2_t<T> a, vec2_t<T> b) {
	return dot(a, b) / b.length_squared() * b;
}

template <class T>
__host__ __device__
inline vec2_t<T> reflect(vec2_t<T> a, vec2_t<T> n) {
	return a - T(2) * project(a, n);
}