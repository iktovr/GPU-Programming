#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path
import subprocess


def print_table(table, headers=None, hsep='-', vsep='|'):
    assert headers is None or len(table[0]) == len(headers)

    column_width = []
    for i, col in enumerate(zip(*table)):
        column_width.append(max([len(str(i)) for i in col]))
        if headers is not None:
            column_width[-1] = max(column_width[-1], len(str(headers[i])))

    if headers is not None:
        print(vsep, vsep.join([' {:%d} ' % i for i in column_width]).format(*headers), vsep, sep='')
        print(vsep, vsep.join([hsep * (i + 2) for i in column_width]), vsep, sep='')

    for row in table:
        print(vsep, vsep.join([' {:%d} ' % i for i in column_width]).format(*row), vsep, sep='')


def benchmark(gpu, cpu, tests, test_dir, kernels):
    if tests is None:
        tests = []
    if test_dir is not None:
        for test in test_dir.iterdir():
            if test.is_file():
                tests.append(test)
    tests.sort()

    headers = ['name', *[i.name for i in tests]]
    table = []

    if cpu is not None:
        table.append(['cpu'])
        for test_file in tests:
            with open(str(test_file), 'r') as test:
                try:
                    result = subprocess.run([str(cpu)], stdin=test, stdout=subprocess.PIPE)
                except OSError as e:
                    print("Execution failed: %s" % e)
                    sys.exit()
                if result.returncode != 0:
                    print("%s with test %s return %d" % (cpu.name, test_file.name, result.returncode))
                    sys.exit()
                table[-1].append(float(result.stdout))

    if gpu is not None:
        for kernel in kernels:
            table.append([0 for i in range(len(tests)+1)])
            table[-1][0] = 'x'.join(kernel)

        for i, test_file in enumerate(tests):
            with open(str(test_file), 'r') as test:
                for j, kernel in enumerate(kernels):
                    try:
                        result = subprocess.run([str(gpu), *kernel], stdin=test, stdout=subprocess.PIPE)
                    except OSError as e:
                        print("Execution failed: %s" % e)
                        sys.exit()
                    if result.returncode != 0:
                        print("%s with test %s return %d" % (gpu.name, test_file.name, result.returncode))
                        sys.exit()
                    table[j+1][i+1] = float(result.stdout)
                test.seek(0)

    print_table(table, headers=headers)


def grid_dim(value):
    return [str(int(i)) for i in value.split(' ')]


parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=Path, default=None)
parser.add_argument("--cpu", type=Path, default=None)
parser.add_argument("--tests", "-t", nargs="*", type=Path)
parser.add_argument("--test-dir", type=Path, default=None)
parser.add_argument("kernels", nargs="*", type=grid_dim)

if __name__ == "__main__":
    args = parser.parse_args()
    benchmark(**vars(args))
