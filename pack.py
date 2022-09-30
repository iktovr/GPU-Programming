# Допущения:
#   в заголовочных файлах нет вызовов других пользовательских заголовочных файлов
#   пути к заголовочным файлам относительные

import click
from pathlib import Path
import re
import os
from tempfile import TemporaryDirectory
import tarfile
import shutil

header_re = re.compile("^#include \"(.*?([^/]*))\"$", re.M)
makefile = """\
CC = /usr/local/cuda/bin/nvcc
CFLAGS = --std=c++11 -Werror cross-execution-space-call -lm
SOURSES = {main}
BIN = {bin}
all:
\t$(CC) $(CFLAGS) -o $(BIN) $(SOURSES)
"""


def replace_header(match):
    return match.group(0).replace(match.group(1), match.group(2))


@click.command()
@click.argument("main_file", type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path))
@click.argument("name", type=str, default="")
def pack(main_file, name):
    if name == "":
        name = main_file.stem
    main_file_dir = main_file.parent
    files = list()
    with TemporaryDirectory() as tmpdirname:
        tar_root = Path(tmpdirname) / name
        tar_root.mkdir()

        with open(main_file, 'rt') as program:
            for line in program:
                match = header_re.match(line)
                if match is None:
                    continue

                file = Path(match.group(1))
                file = (main_file_dir / file).resolve()
                files.append(file)

        with open(tar_root / main_file.name, 'wt') as new_program:
            with open(main_file, 'rt') as program:
                new_program.write(header_re.sub(replace_header, program.read()))

        for file in files:
            shutil.copy(file, tar_root)

        with open(tar_root / 'makefile', 'wt') as file:
            file.write(makefile.format(main=main_file.name, bin=name))
        
        cwd = os.getcwd()
        with tarfile.open(name + '.tar', 'w') as archive:
            os.chdir(tmpdirname)
            archive.add(tar_root.name)
        os.chdir(cwd)

        os.system('gpg -ab ' + name + '.tar')


if __name__ == "__main__":
    pack()
