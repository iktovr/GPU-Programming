#pragma once

template <class T>
inline T get_block_count(T size, T block_size, T log_block_size) {
	return (size >> (log_block_size + 1)) + ((size & ((block_size << 1) - 1)) > 0);
}

template <class T>
T log2(T a) {
	T log = 0;
	while (a > 1) {
		++log;
		a >>= 1;
	}
	return log;
}

template <class T>
T ceil_2_pow(T a) {
    if (a & (a - 1) == 0) {
        return a;
    }
	while ((a & (a - 1)) != 0) {
		a &= a - 1;
	}
	return a << 1;
}


#define sh_index(i) ((i) + ((i) >> 5))

const size_t BLOCK_SIZE = 1024;
const size_t BLOCK_COUNT = 1024;

template <class T>
using func_pointer = T (*) (T, T);