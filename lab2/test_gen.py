#!/usr/bin/env python3
# Написан для python 3.5

from random import randrange
from pathlib import Path
import argparse
import struct
import sys


def test_gen(resolution, tests_dir):
    assert tests_dir.is_dir() or not tests_dir.exists()
    
    if not tests_dir.exists():
        try:
            tests_dir.mkdir()
        except FileNotFoundError:
            print('Директория недостижима', file=sys.stderr)
            sys.exit()
    for width, height in [map(int, res.split('x')) for res in resolution]:
        testname = Path("test%dx%d.data" % (width, height))
        with open(str(tests_dir / testname), "wb") as test:
            test.write(struct.pack('ii', width, height))
            for _ in range(width * height):
                test.write(struct.pack('BBBB', randrange(256), randrange(256), randrange(256), randrange(256)))
        with open(str(tests_dir / testname.with_suffix('.in')), 'wt') as test:
            test.write(str(tests_dir / testname) + '\n' + str(tests_dir / testname.with_suffix('.out')))


parser = argparse.ArgumentParser()
parser.add_argument("tests_dir", type=Path)
parser.add_argument("resolution", type=str, nargs="*", help="resolutions in format WIDTHxHEIGHT")

if __name__ == "__main__":
    args = parser.parse_args()
    test_gen(**vars(args))
