#!/usr/bin/env python3
# Написан для python 3.5

from random import randrange
from pathlib import Path
import argparse
import struct
import sys


def test_gen(resolution, tests_dir):
    if tests_dir.exists() and not tests_dir.is_dir():
        print("Tests dir is not a dir", file=sys.stderr)
        sys.exit()
    
    if not tests_dir.exists():
        try:
            tests_dir.mkdir()
        except FileNotFoundError:
            print("Tests dir is unreachable")

    for width, height, n_classes, count in [map(int, res.split('x')) for res in resolution]:
        testname = Path("test%dx%d_%d.data" % (width, height, n_classes))
        with open(str(tests_dir / testname), "wb") as test:
            test.write(struct.pack('ii', width, height))
            for _ in range(width * height):
                test.write(struct.pack('BBBB', randrange(256), randrange(256), randrange(256), 0))

        classes = [[(randrange(width), randrange(height)) for _ in range(count)] for _ in range(n_classes)]
        with open(str(tests_dir / testname.with_suffix('.in')), 'wt') as test:
            test.write(str(tests_dir / testname) + '\n' + str(tests_dir / testname.with_suffix('.ans')))
            test.write('\n' + str(n_classes))
            for c in classes:
                test.write('\n' + str(len(c)))
                for x, y in c:
                    test.write(" %d %d" % (x, y))


parser = argparse.ArgumentParser(description="Примечание: не генерирует ответы к тестам")
parser.add_argument("tests_dir", type=Path)
parser.add_argument("resolution", type=str, nargs="*", 
                    help="resolutions in format WIDTHxHEIGHTxN_CLASSESxCOUNT_PER_CLASS")

if __name__ == "__main__":
    args = parser.parse_args()
    test_gen(**vars(args))
