from random import randint
from pathlib import Path
from sys import argv


assert len(argv) >= 3
tests_dir = Path(argv[1])
assert tests_dir.is_dir()
counts = [int(i) for i in argv[2:]]
start = 0
stop = 1e6

for count in counts:
    with open(tests_dir / "test{}".format(count), "wt") as test:
        test.write(str(count) + "\n")
        for i in range(count):
            test.write(str(randint(start, stop)) + " ")
