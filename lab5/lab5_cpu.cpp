#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstring>

#define __device__

#include "utils.hpp"

template <class T>
void bitonic_sort(std::vector<T>& data) {
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
}

template <class T>
void bitonic_sort(T* data, size_t size) {
	size_t pad_size = ceil_2_pow(size);
	std::vector<T> pad_data(pad_size, std::numeric_limits<T>::max());

	std::memcpy(pad_data.data(), data, sizeof(T) * size);
	bitonic_sort(pad_data);
	std::memcpy(data, pad_data.data(), sizeof(T) * size);
}

const int SPLIT_SIZE = 32;
const int BUCKET_SIZE = 1024;
const float SPLIT_COEF = 1. - 1e-6;
const float EPS = 1e-8;
using index_t = uint32_t;

template <class T>
void bucket_sort(T* data, size_t size) {
	T min = std::numeric_limits<T>::max();
	T max = std::numeric_limits<T>::min();
	for (size_t i = 0; i < size; ++i) {
		if (data[i] < min) {
			min = data[i];
		}
		if (data[i] > max) {
			max = data[i];
		}
	}
	// std::cout << min << ' ' << max << '\n';

	if (std::abs(min - max) < EPS) {
		return;
	}

	// std::cout << "splits\n";
	size_t n_splits = (size - 1) / SPLIT_SIZE + 1;
	// std::cout << n_splits << '\n';
	std::vector<index_t> splits(size);
	for (size_t i = 0; i < splits.size(); ++i) {
		splits[i] = SPLIT_COEF * (data[i] - min) / (max - min) * n_splits;
		// std::cout << splits[i] << ' ';
	}
	// std::cout << '\n';

	std::vector<index_t> hist(n_splits);
	for (index_t i: splits) {
		++hist[i];
	}

	// std::cout << "hist\n";
	// for (auto a: hist) {
	// 	std::cout << a << ' ';
	// }
	// std::cout << '\n';

	for (size_t i = 1; i < hist.size(); ++i) {
		hist[i] += hist[i-1];
	}
	for (int i = hist.size()-2; i > 0; --i) {
		hist[i+1] = hist[i];
	}
	hist[0] = 0;

	// std::cout << "hist scan\n";
	// for (auto a: hist) {
	// 	std::cout << a << ' ';
	// }
	// std::cout << '\n';

	std::vector<index_t> groups(hist);
	groups.push_back(size);

	std::vector<T> group_data(size);
	for (size_t i = 0; i < size; ++i) {
		group_data[hist[splits[i]]++] = data[i];
	}

	// std::cout << "group data\n";
	// for (auto a: group_data) {
	// 	std::cout << a << ' ';
	// }
	// std::cout << '\n';

	size_t len, start;
    for (size_t i = 0; i < n_splits; ++i) {
        start = i;
        len = 0;
        while (i < n_splits && len + (groups[i + 1] - groups[i]) < BUCKET_SIZE) {
            len += groups[i + 1] - groups[i];
            ++i;
        }

        if (start == i) {
        	// std::cout << groups[start] << ' ' << groups[i + 1] - groups[i] << " 1\n";
            bucket_sort(group_data.data() + groups[start], groups[i + 1] - groups[i]);
        } else {
        	// std::cout << groups[start] << ' ' << len << " 2\n";
            --i;
            bitonic_sort(group_data.data() + groups[start], len);
        }
    }

    std::memcpy(data, group_data.data(), sizeof(T) * size);
}

template <class T>
void bucket_sort(std::vector<T>& data) {
	bucket_sort(data.data(), data.size());
}

int main() {
	int n;
	std::cin >> n;
	std::vector<int> data(n);
	for (int i = 0; i < n; ++i) {
		std::cin >> data[i];
	}

	bucket_sort(data);
    for (size_t i = 0; i < data.size(); ++i) {
		std::cout << data[i] << ' ';
	}
    std::cout << '\n';
}