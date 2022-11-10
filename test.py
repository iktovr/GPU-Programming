#!/usr/bin/env python3
# Написан для python 3.5

import argparse
import sys
from pathlib import Path
import subprocess


def test(program, tests_dir, in_ext, ans_ext, out_ext):
    assert program.is_file() and tests_dir.is_dir()

    tests = []
    for file in tests_dir.iterdir():
        if not file.is_file():
            continue
        if file.suffix == in_ext:
            if not file.with_suffix(ans_ext).exists():
                print("Missing answer for test", str(file), file=sys.stderr)
            else:
                tests.append(file)
    tests.sort(key=lambda s: (len(s.name), s))

    for test in tests:
        if out_ext is None:
            out_file = open(str(test.with_suffix(".log")), "wt")
        else:
            out_file = None

        with open(str(test), 'rt') as test_file:
            try:
                result = subprocess.run(['./' + str(program)], stdin=test_file, stdout=out_file, stderr=subprocess.PIPE)
            except OSError as e:
                print("%s execution failed: %s" % (program.name, e), file=sys.stderr)
                sys.exit()
            if result.returncode != 0:
                print("%s with test %s return %d" % (program.name, test.name, result.returncode), 
                      file=sys.stderr)
                sys.exit()
            if result.stderr.startswith(b"ERROR:"):
                print("%s %s" % (program.name, result.stderr.decode('utf-8')), file=sys.stderr)
                sys.exit()

        if out_ext is None:
            out = Path(out_file.name)
            out_file.close()
        else:
            out = test.with_suffix(out_ext)
        ans = test.with_suffix(ans_ext)

        try:
            diff = subprocess.run(["diff", "-q", str(out), str(ans)], stdout=subprocess.PIPE)
        except OSError as e:
            print("diff execution failed: ", e, file=sys.stderr)
            sys.exit()

        print(test.stem, ':', sep='', end=' ')
        if len(diff.stdout) == 0:
            print("OK")
            out.unlink()
        else:
            print("WA")


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description="Тестирует исполняемый файл program на всех тестах из директории tests_dir. \
                 Предполагается что для файла с тестом testIN_EXT существует файл с ответом testANS_EXT, \
                 а программа пишет результат в stdout или в файл testOUT_EXT. Если в stdout, \
                 то неправильный ответ сохранится в файле test.log")
parser.add_argument("program", type=Path)
parser.add_argument("tests_dir", type=Path)
parser.add_argument("--in-ext", "-i", type=str, default=".in", help="Расширение файла с тестом")
parser.add_argument("--ans-ext", "-a", type=str, default=".out", help="Расширение файла с ответом на тест")
parser.add_argument("--out-ext", "-o", type=str, default=None, 
                    help="Расширение файла в который программа пишет ответ (если пишет)")
# parser.add_argument("--keep-wa")

if __name__ == "__main__":
    args = parser.parse_args()
    test(**vars(args))
