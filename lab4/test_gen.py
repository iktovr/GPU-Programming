#!/usr/bin/env python3
# Написан для python 3.5

from random import uniform, randrange, choice
from pathlib import Path
import argparse
import sys
import numpy as np


def random_system(size):
    matrix = np.eye(size)
    matrix[0, 0] = uniform(0.1, 10)
    np.random.shuffle(matrix)
    for _ in range(randrange(size, size * 5)):
        a = uniform(0.1, 5) * (randrange(0, 2) * 2 - 1)
        i = randrange(size)
        if i == 0:
            j = randrange(1, size)
        elif i == size-1:
            j = randrange(size-1)
        else:
            j = choice([randrange(i), randrange(i+1, size)])
        matrix[j] += matrix[i] * a
    x = np.random.randint(-10, 11, size)
    b = matrix @ x
    return matrix, b, x


def test_gen(answers, tests_dir, sizes):
    if tests_dir.exists() and not tests_dir.is_dir():
        print("Tests dir is not a dir", file=sys.stderr)
        sys.exit()
    
    if not tests_dir.exists():
        try:
            tests_dir.mkdir()
        except FileNotFoundError:
            print("Tests dir is unreachable")

    for s in sizes:
        if 'x' not in s:
            size = int(s)
            count = 1
        else:
            size, count = map(int, s.split('x'))

        count_len = len(str(count))
        for i in range(count):
            matrix, b, x = random_system(size)
            if count > 1:
                testname = Path("test%d_%0*d.in" % (size, count_len, i))
            else:
                testname = Path("test%d.in" % (size))
            with open(str(tests_dir / testname), "wt") as test:
                test.write(str(size) + '\n')
                test.write('\n'.join([' '.join(map(str, matrix[i])) for i in range(size)]) + '\n')
                test.write(' '.join(map(str, b)) + '\n')

            if answers:
                with open(str(tests_dir / testname.with_suffix('.out')), 'wt') as test:
                    test.write(' '.join(map(str, x)) + " \n")


parser = argparse.ArgumentParser()
parser.add_argument("--answers", "-a", action='store_true')
parser.add_argument("tests_dir", type=Path)
parser.add_argument("sizes", type=str, nargs="*", help="sizes of square matrices in format SIZE[xTEST_COUNT]")

if __name__ == "__main__":
    args = parser.parse_args()
    test_gen(**vars(args))
