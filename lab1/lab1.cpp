#include <iostream>
#include <vector>
#include <iomanip>

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

	reverse(data, n);

	std::cout << std::setprecision(10) << std::fixed;
	for (int i = 0; i < n; ++i) {
		std::cout << data[i] << ' ';
	}
	std::cout << '\n';
}