#!/usr/bin/env python3
# Написан для python 3.5

from random import randint
from pathlib import Path
import argparse
import sys


def test_gen(start, stop, tests_dir, counts):
    if tests_dir.exists() and not tests_dir.is_dir():
        print("Tests dir is not a dir", file=sys.stderr)
        sys.exit()
    
    if not tests_dir.exists():
        try:
            tests_dir.mkdir()
        except FileNotFoundError:
            print("Tests dir is unreachable")

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


parser = argparse.ArgumentParser(description="Примечание: также генерирует ответы к тестам")
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--stop", type=int, default=1e6)
parser.add_argument("tests_dir", type=Path)
parser.add_argument("counts", type=int, nargs="*")

if __name__ == "__main__":
    args = parser.parse_args()
    test_gen(**vars(args))
