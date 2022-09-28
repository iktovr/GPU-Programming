#!/usr/bin/env bash
./test_gen.py tests 1000 10000 100000 1000000 10000000
../benchmark.py --gpu bench/lab1_gpu --cpu bench/lab1_cpu -r 5 --test-dir tests "1 32" "32 32" "64 64" "128 128" "256 256" "512 512" "1024 1024"