#!/usr/bin/env python3
# Написан для python 3.5

import argparse
import sys
from pathlib import Path
import subprocess
import re


def print_table(table, headers=None, hsep='-', vsep='|'):
    assert headers is None or len(table[0]) == len(headers)

    column_width = []
    for i, col in enumerate(zip(*table)):
        column_width.append(max([len(('{:.6f}' if type(i) is float else '{}').format(i)) for i in col]))
        if headers is not None:
            column_width[-1] = max(column_width[-1], len(str(headers[i])))

    if headers is not None:
        print(vsep, vsep.join([' {:%d} ' % i for i in column_width]).format(*headers), vsep, sep='')
        print(vsep, vsep.join([hsep * (i + 2) for i in column_width]), vsep, sep='')

    for row in table:
        print(vsep, 
              vsep.join([(' {:%d.6f} ' if type(row[i]) is float else ' {:%d} ') % w 
                         for i, w in enumerate(column_width)]).format(*row), 
              vsep, sep='')


def benchmark(gpu, cpu, tests, kernels, repeat, pattern):
    if pattern is not None:
        file_re = re.compile(pattern)

    test_files = []
    for test_file in tests:
        if test_file.is_dir():
            for test in test_file.iterdir():
                if test.is_file() and (pattern is None or file_re.fullmatch(test.name) is not None):
                    test_files.append(test)
        elif pattern is None or file_re.fullmatch(test_file.name) is not None:
            test_files.append(test_file)
    test_files.sort()

    headers = ['name', *[i.name for i in test_files]]
    table = []
    time = [0 for i in range(repeat)]

    if cpu is not None:
        table.append(['cpu'])
        for test_file in test_files:
            with open(str(test_file), 'rt') as test:
                for i in range(repeat):
                    try:
                        result = subprocess.run([str(cpu)], stdin=test, stdout=subprocess.PIPE)
                    except OSError as e:
                        print("%s execution failed: %s" % (cpu.name, e), file=sys.stderr)
                        sys.exit()
                    if result.returncode != 0:
                        print("%s with test %s return %d" % (cpu.name, test_file.name, result.returncode), 
                              file=sys.stderr)
                        sys.exit()
                    time[i] = float(result.stdout)
                    test.seek(0)
                table[-1].append(sum(time) / repeat)

    if gpu is not None:
        for kernel in kernels:
            table.append([0 for i in range(len(test_files)+1)])
            table[-1][0] = 'x'.join(kernel)

        for i, test_file in enumerate(test_files):
            with open(str(test_file), 'r') as test:
                for j, kernel in enumerate(kernels):
                    for k in range(repeat):
                        try:
                            result = subprocess.run([str(gpu), *kernel], stdin=test, stdout=subprocess.PIPE)
                        except OSError as e:
                            print("%s execution failed: %s" % (gpu.name + ' '.join(kernel), e), file=sys.stderr)
                            sys.exit()
                        if result.returncode != 0:
                            print("%s with test %s return %d" % 
                                  (gpu.name + ' '.join(kernel), test_file.name, result.returncode), file=sys.stderr)
                            sys.exit()
                        time[k] = float(result.stdout)
                        test.seek(0)
                    table[j+1][i+1] = sum(time) / repeat

    print_table(table, headers=headers)


def grid_dim(value):
    return [str(int(i)) for i in value.split(' ')]


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--gpu", type=Path, default=None, help="Исполняемый файл для CUDA")
parser.add_argument("--cpu", type=Path, default=None, help="Исполняемый файл для CPU")
parser.add_argument("--tests", "-t", nargs="*", type=Path, help="Тестовые файлы (директории)")
parser.add_argument("--repeat", "-r", type=int, default=1, help="Число запусков каждой конфигурации")
parser.add_argument("--pattern", "-p", type=str, default=None, help="Регулярное выражения имен тестов")
parser.add_argument("kernels", nargs="*", type=grid_dim, help="Конфигурации ядра в формате \"DIM1 DIM2 ...\"")

if __name__ == "__main__":
    args = parser.parse_args()
    benchmark(**vars(args))
