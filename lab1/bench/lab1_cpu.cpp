#include <iostream>
#include <vector>
#include <iomanip>
#include <chrono>

using namespace std::chrono;

void reverse(std::vector<double> &data, int n) {
	double p;
	for (int i = 0; i <= n / 2 - 1; ++i) {
		p = data[i];
		data[i] = data[data.size() - 1 - i];
		data[data.size() - 1 - i] = p;
	}
}

int main() {
	std::ios::sync_with_stdio(false);
	int n;
	std::cin >> n;
	std::vector<double> data(n);
	data.shrink_to_fit();
	for (int i = 0; i < n; ++i) {
		std::cin >> data[i];
	}

	steady_clock::time_point start = steady_clock::now();

	reverse(data, n);

	steady_clock::time_point end = steady_clock::now();

	std::cout << duration_cast<milliseconds>(end - start).count() << " ms\n";
}