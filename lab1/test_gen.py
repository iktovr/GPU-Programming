from random import randint
from pathlib import Path
import argparse


def test_gen(start, stop, tests_dir, counts):
    if not tests_dir.exists():
        tests_dir.mkdir()
    for count in counts:
        with open(tests_dir / ("test%d" % count), "wt") as test:
            test.write("%d\n" % count)
            for i in range(count):
                test.write("%d " % randint(start, stop))


parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--stop", type=int, default=1e6)
parser.add_argument("tests_dir", type=Path)
parser.add_argument("counts", type=int, nargs="*")

if __name__ == "__main__":
    args = parser.parse_args()
    test_gen(**vars(args))
