#!/usr/bin/env python3
# Написан для python 3.5

from random import randrange
from pathlib import Path
import argparse
import struct


def test_gen(resolution, tests_dir):
    assert tests_dir.is_dir() or not tests_dir.exists()
    
    if not tests_dir.exists():
        tests_dir.mkdir()
    for width, height in [map(int, res.split('x')) for res in resolution]:
        with open(str(tests_dir / ("test%dx%d" % (width, height))), "wb") as test:
            test.write(struct.pack('ii', width, height))
            for _ in range(width * height):
                test.write(struct.pack('BBBB', randrange(256), randrange(256), randrange(256), randrange(256)))


parser = argparse.ArgumentParser()
parser.add_argument("tests_dir", type=Path)
parser.add_argument("resolution", type=str, nargs="*", help="resolutions in format WIDTHxHEIGHT")

if __name__ == "__main__":
    args = parser.parse_args()
    test_gen(**vars(args))
