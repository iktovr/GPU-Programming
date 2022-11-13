#!/usr/bin/env python3
# Написан для python 3.5

from random import uniform
from pathlib import Path
import argparse
import sys


def test_gen(start, stop, answer, tests_dir, counts):
    if tests_dir.exists() and not tests_dir.is_dir():
        print("Tests dir is not a dir", file=sys.stderr)
        sys.exit()
    
    if not tests_dir.exists():
        try:
            tests_dir.mkdir()
        except FileNotFoundError:
            print("Tests dir is unreachable", file=sys.stderr)

    for count in counts:
        nums = list()
        if answer == 'scan':
            scan_nums = [0]
        for _ in range(count):
            nums.append(int(uniform(start, stop)))
            if answer == 'scan':
                scan_nums.append(scan_nums[-1] + nums[-1])
        
        with open(str(tests_dir / ("test%d.in" % count)), "wt") as test:
            test.write("%d\n" % count)
            for num in nums:
                test.write("%d " % num)
        
        if answer != 'none':
            with open(str(tests_dir / ("test%d.out" % count)), "wt") as ans:
                if answer == 'sum':
                    ans.write("%d\n" % sum(nums))
                elif answer == 'max':
                    ans.write("%d\n" % max(nums))
                elif answer == 'min':
                    ans.write("%d\n" % min(nums))
                elif answer == 'minmax':
                    ans.write("%d %d\n" % (min(nums), max(nums)))
                elif answer == 'scan':
                    ans.write(' '.join([str(i) for i in scan_nums[:-1]]) + ' \n')
                elif answer == 'sort':
                    ans.write(' '.join([str(i) for i in sorted(nums)]) + ' \n')


parser = argparse.ArgumentParser(description="Примечание: также генерирует ответы к тестам")
parser.add_argument("--start", type=int, default=-1e9)
parser.add_argument("--stop", type=int, default=1e9)
parser.add_argument("--answer", "-a", type=str, choices=['none', 'minmax', 'sum', 'max', 'min', 'scan', 'sort'], default='none')
parser.add_argument("tests_dir", type=Path)
parser.add_argument("counts", type=int, nargs="*")

if __name__ == "__main__":
    args = parser.parse_args()
    test_gen(**vars(args))
