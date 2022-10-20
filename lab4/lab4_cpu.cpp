#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

#ifdef TIME
#include <chrono>
using namespace std::chrono;
#endif

using std::abs;

int leading_row(std::vector<std::vector<double>>& matrix, int row, int n) {
	int max_row = row;
	for (int i = row+1; i < n; ++i) {
		if (abs(matrix[i][row]) > abs(matrix[max_row][row])) {
			max_row = i;
		}
	}
	return max_row;
}

void swap_rows(std::vector<std::vector<double>>& matrix, int i, int j, int n) {
	for (int k = 0; k < n+1; ++k) {
		std::swap(matrix[i][k], matrix[j][k]);
	}
} 

void gaussian_solver_step(std::vector<std::vector<double>>& matrix, int row, int n) {
	for (int i = row+1; i < n; ++i) {
		for (int j = row+1; j < n+1; ++j) {
			matrix[i][j] -= matrix[row][j] / matrix[row][row] * matrix[i][row];
		}
	}
}

int main() {
	std::ios::sync_with_stdio(false);
	int n;
	std::cin >> n;
	std::vector<std::vector<double>> matrix(n, std::vector<double>(n+1));
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			std::cin >> matrix[i][j];
		}
	}

	for (int i = 0; i < n; ++i) {
		std::cin >> matrix[i][n];
	}

#ifdef TIME
	steady_clock::time_point start = steady_clock::now();
#endif

	for (int row = 0; row < n; ++row) {
		int j = leading_row(matrix, row, n);
		if (j != row) {
			swap_rows(matrix, row, j, n);
		}
		gaussian_solver_step(matrix, row, n);
	}

#ifdef TIME
	steady_clock::time_point end = steady_clock::now();
	std::cout << duration_cast<nanoseconds>(end - start).count() / 1000000.0;
#else

	std::vector<double> x(n);
	for (int i = n-1; i >= 0; --i) {
		x[i] = matrix[i][n];
		for (int j = i+1; j < n; ++j) {
			x[i] -= x[j] * matrix[i][j];
		}
		x[i] /= matrix[i][i];
	}

	// std::cout << std::setprecision(10) << std::fixed;
	for (int i = 0; i < n; ++i) {
		std::cout << ((abs(x[i]) < 1e-7) ? 0 : x[i]) << ' ';
	}
	std::cout << '\n';
#endif
	return 0;
}