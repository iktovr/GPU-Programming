import click
from random import randint
from pathlib import Path


@click.command()
@click.option("--start", type=int, default=0)
@click.option("--stop", type=int, default=1e6)
@click.argument("tests_dir", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.argument("counts", type=int, nargs=-1)
def test_gen(start, stop, tests_dir, counts):
    for count in counts:
        with open(tests_dir / f"test{count}", "wt") as test:
            test.write(f"{count}\n")
            for i in range(count):
                test.write(f"{randint(start, stop)} ")


if __name__ == "__main__":
    test_gen()
