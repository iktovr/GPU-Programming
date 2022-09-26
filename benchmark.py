import click
import sys
from pathlib import Path
import subprocess
from tabulate import tabulate


class GridDim(click.ParamType):
    def convert(self, value, param, ctx):
        try:
            _ = [int(i) for i in value.split(' ')]
        except Exception as e:
            self.fail(str(e), param, ctx)
        else:
            return value.split(' ')


@click.command()
@click.option("--gpu", type=Path, default=None)
@click.option("--cpu", type=Path, default=None)
@click.option("--test", "-t", "single_tests", multiple=True, 
              type=click.Path(exists=True, path_type=Path))
@click.option("--test-dir", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path), default=None)
@click.argument("kernels", nargs=-1, type=GridDim())
def benchmark(gpu, cpu, single_tests, test_dir, kernels):
    tests = list(single_tests)
    if test_dir is not None:
        for test in test_dir.iterdir():
            if test.is_file():
                tests.append(test)

    headers = ['name', *[i.name for i in tests]]
    table = []

    if gpu is not None:
        table = [[0 for i in range(len(tests))] for i in range(len(kernels)+1)]
        for j, kernel in enumerate(kernels):
            table[j][0] = 'x'.join(kernel)

        for i, test_file in enumerate(tests):
            with open(test_file, 'r') as test:
                for j, kernel in enumerate(kernels):
                    try:
                        result = subprocess.run([gpu, *kernel], stdin=test, capture_output=True)
                    except OSError as e:
                        click.echo(f"Execution failed: {e}")
                        sys.exit()
                    if result.returncode != 0:
                        click.echo(f"{gpu.name} with test {test_file.name} return {result.returncode}")
                        sys.exit()
                    table[j][i+1].append(int(result.stdout))
                test.seek(0)

    if cpu is not None:
        table.append(['cpu'])
        for test_file in tests:
            with open(test_file, 'r') as test:
                try:
                    result = subprocess.run([cpu], stdin=test, capture_output=True)
                except OSError as e:
                    click.echo(f"Execution failed: {e}")
                    sys.exit()
                if result.returncode != 0:
                    click.echo(f"{cpu.name} with test {test_file.name} return {result.returncode}")
                    sys.exit()
                table[-1].append(int(result.stdout))

    click.echo(tabulate(table, headers=headers))


if __name__ == "__main__":
    benchmark()
