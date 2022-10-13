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
double3 operator-(const double3& a, const uchar4& b) {
	return {a.x - b.x, a.y - b.y, a.z - b.z};
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

	matrix3d transpose() {
		matrix3d result;
		for (size_t i = 0; i < size(); ++i) {
			for (size_t j = 0; j < size(); ++j) {
				result[i][j] = data[j][i];
			}
		}
		return result;
	}

	void swap_rows(size_t i, size_t j) {
		for (size_t k = 0; k < size(); ++k) {
			std::swap(data[i][k], data[j][k]);
		}
	}

	void swap_rows(std::vector<std::pair<size_t, size_t>>& P) {
		for (const std::pair<size_t, size_t>& p: P) {
			swap_rows(p.first, p.second);
		}
	}

	void swap_columns(size_t i, size_t j) {
		for (size_t k = 0; k < size(); ++k) {
			std::swap(data[k][i], data[k][j]);
		}
	}

	void decompose(matrix3d& L, matrix3d& U, std::vector<std::pair<size_t, size_t>>& P) const {
		P.clear();
		for (size_t i = 0; i < size(); ++i) {
			for (size_t j = 0; j < size(); ++j) {
				L[i][j] = 0;
				U[i][j] = data[i][j];
			}
			L[i][i] = 1;
		}

		for (size_t k = 0; k < size()-1; ++k) {
			double max = U[k][k];
			size_t max_ind = k;
			for (size_t i = k+1; i < size(); ++i) {
				if (U[i][k] > max) {
					max = U[i][k];
					max_ind = i;
				}
			}

			if (max_ind != k) {
				P.push_back(std::make_pair(k, max_ind));
				U.swap_rows(k, max_ind);
				L.swap_rows(k, max_ind);
				L.swap_columns(k, max_ind);
			}

			for (size_t i = k+1; i < size(); ++i) {
				L[i][k] = U[i][k] / U[k][k];

				for (size_t j = 0; j < size(); ++j) {
					U[i][j] -= L[i][k] * U[k][j];
				}
			}
		}
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

struct LUP3d {
	matrix3d L;
	matrix3d U;
	std::vector<std::pair<size_t, size_t>> P;

	LUP3d() {}

	LUP3d(const matrix3d& matrix) {
		matrix.decompose(L, U, P);
	}

	void assign(const matrix3d& matrix) {
		matrix.decompose(L, U, P);
	}

	double det() {
		double res = 1;
		for (size_t i = 0; i < U.size(); ++i) {
			res *= U[i][i];
		}
		if (P.size() & 1) {
			res = -res;
		}
		return res;
	}

	double3 solve(double3 b) {
		for (const std::pair<size_t, size_t>& p: P) {
			std::swap(get(b, p.first), get(b, p.second));
		}

		double3 x, z;
		for (size_t i = 0; i < L.size(); ++i) {
			get(z, i) = get(b, i);
			for (size_t j = 0; j < i; ++j) {
				get(z, i) -= L[i][j] * get(z, j);
			}
		}

		for (int i = L.size()-1; i >= 0; --i) {
			get(x, i) = get(z, i);
			for (int j = L.size()-1; j > i; --j) {
				get(x, i) -= U[i][j] * get(x, j);
			}
			get(x, i) /= U[i][i];
		}

		return x;
	}

	matrix3d invert() {
		double3 e;
		matrix3d inv;

		for (size_t i = 0; i < L.size(); ++i) {
			get(e, i) = 1;
			double3 b = solve(e);
			for (size_t j = 0; j < L.size(); ++j) {
				inv[j][i] = get(b, j);
			}
			get(e, i) = 0;
		}
		
		return inv;
	}
};

#include <iostream>

std::ostream& operator<<(std::ostream& os, const double3& a) {
	os << a.x << ' ' << a.y << ' ' << a.z;
	return os;
}

std::ostream& operator<<(std::ostream& os, const matrix3d& a) {
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			os << a[i][j] << ' ';
		}
		os << '\n';
	}
	return os;
}