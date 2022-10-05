import click
from pathlib import Path, PurePosixPath
import re
import os
import sys
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


def search_headers(file, root):
    files = set()
    with open(file, 'rt') as program:
        for line in program:
            match = header_re.match(line)
            if match is None:
                continue

            header = Path(match.group(1))
            try:
                header = (file.parent.resolve() / header).resolve().relative_to(root)
            except ValueError:
                print(f"File {header} unreachable from root {root}", file=sys.stderr)
            files.add(header)
    return files


@click.command(context_settings={'show_default': True})
@click.option("--name", type=str, default="", help="Имя бинарного файла. По умолчанию - имя первого файла")
@click.option("--root", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path),
              default='./', help="Родительская директория по отношению ко всем возможным файлам проекта")
@click.option("-o", "output", type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path),
              default='./', help="Директория для архива и подписи")
@click.option("--sign/--no-sign", default=True, help="Подписывать или нет получившийся архив")
@click.argument("source_files", nargs=-1, type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path),
                required=True)
def pack(source_files, name, root, output, sign):
    """\
    SOURCE_FILES - Исходные файлы проекта

    Упаковывает SOURCE_FILES, необходимые им заголовочные файлы и сгенерированный makefile в tar архив,
    с сохранением структуры директорий. Вызывает gpg для подписи получившегося архива (если не указано обратное).
    
    \b
    Допущения:
      * все пользовательские заголовочные файлы упоминаются в исходных файлах
        (рекурсивный поиск не реализован) 
      * пути к заголовочным файлам относительные
    
    \b
    Пример:
    pack.py --no-sign lab1/lab1.cu
    Создаст архив:
    lab1.tar
    └───lab1
        │   makefile
        ├───common
        │       error_checkers.hpp
        └───lab1
                lab1.cu
    """
    
    if name == "":
        name = source_files[0].stem
    
    output = output.resolve()
    root = root.resolve()
    files = set()
    sources = []
    for file in source_files:
        try:
            files.add(file.resolve().relative_to(root))
        except ValueError:
            print(f"File {file} unreachable from root {root}", file=sys.stderr)
        sources.append(str(PurePosixPath(file)))
        headers = search_headers(file, root)
        files.update(headers)

    with TemporaryDirectory() as tmpdirname:
        tar_root = Path(tmpdirname) / name
        tar_root.mkdir()

        for file in files:
            for parent in file.parents[::-1]:
                if parent == '.' or parent == '..':
                    continue
            (tar_root / parent).mkdir()
            shutil.copy(file, tar_root / file)

        with open(tar_root / 'makefile', 'wt') as file:
            file.write(makefile.format(main=' '.join(sources), bin=name))

        os.chdir(output)
        with tarfile.open(name + '.tar', 'w') as archive:
            os.chdir(tmpdirname)
            archive.add(tar_root.name)
        os.chdir(output)

    if sign:
        os.system('gpg -ab ' + name + '.tar')


if __name__ == "__main__":
    pack()
