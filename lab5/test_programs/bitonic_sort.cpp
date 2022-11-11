#include <vector>
#include <iostream>

#include "../utils.hpp"

std::vector<int> bitonic_sort(std::vector<int> data) {
	for (size_t m = 2; m <= data.size(); m <<= 1) {
		for (size_t b = m; b >= 2; b >>= 1) {
			for (size_t i = 0; i < data.size(); ++i) {
				if ((i & (b - 1)) < (b >> 1)) {
					size_t k = i + (b >> 1);
					if (((i & m) && (data[i] < data[k])) || (!(i & m) && (data[i] > data[k]))) {
						std::swap(data[i], data[k]);
					}
				}
			}
		}
	}
	return data;
}

int main() {
	int n;
	std::cin >> n;
	std::vector<int> data(n);
	for (int i = 0; i < n; ++i) {
		std::cin >> data[i];
	}

    std::vector<int> res = bitonic_sort(data);
    for (size_t i = 0; i < res.size(); ++i) {
		std::cout << res[i] << ' ';
	}
    std::cout << '\n';
}