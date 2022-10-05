#!/usr/bin/env python3
# Написан для python 3.5

from random import randint
from pathlib import Path
import argparse


def test_gen(start, stop, tests_dir, counts):
    assert tests_dir.is_dir()
    
    if not tests_dir.exists():
        tests_dir.mkdir()
    for count in counts:
        nums = list()
        for i in range(count):
            nums.append(randint(start, stop))
        with open(str(tests_dir / ("test%d.in" % count)), "wt") as test:
            test.write("%d\n" % count)
            for num in nums:
                test.write("%d " % num)
        with open(str(tests_dir / ("test%d.out" % count)), "wt") as ans:
            for num in nums[::-1]:
                ans.write("%.10f " % num)
            ans.write('\n')


parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--stop", type=int, default=1e6)
parser.add_argument("tests_dir", type=Path)
parser.add_argument("counts", type=int, nargs="*")

if __name__ == "__main__":
    args = parser.parse_args()
    test_gen(**vars(args))
