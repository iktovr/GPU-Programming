import click
from PIL import Image
from pathlib import Path, PurePosixPath


@click.command()
@click.argument('exec_path', type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=Path))
@click.argument('images', nargs=-1, type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path))
def convert(exec_path, images):
    """\
        Конвертирует изображения во входной файл для программы. На изображении должны быть прямоугольники из 
        прозрачных пикселей. Каждый прямоугольник интерпретируется как класс, выборка класса составляется из 
        пикселей прямоугольника (включая границы). Само изображение используется как исходное для программы,
        так как она не использует альфа канал для классификации.

        В код зашито, что имя изображения оканчивается на _test, а выходной файл теста - то же имя, 
        оканчивающееся на _clusters

        Прямоугольники можно нарисовать, например, при помощи GIMP (https://stackoverflow.com/a/8097548).
        При экспорте необходимо убедится, что у прозрачных пикселей сохраняется цвет.
    """
    exec_path = exec_path.resolve()
    cwd = Path.cwd().relative_to(exec_path)
    for image_file in images:
        classes = []
        with Image.open(image_file, formats=['png']) as img:
            width, height = img.size
            pix = img.load()
            for i in range(width):
                for j in range(height):
                    if pix[i, j][3] == 0:
                        classes.append([])
                        ax, ay = i, j
                        bx, by = 0, 0
                        for k in range(i+1, width):
                            if pix[k, j][3] != 0:
                                bx = k-1
                                break
                        else:
                            bx = width-1

                        for k in range(j+1, height):
                            if pix[i, k][3] != 0:
                                by = k-1
                                break
                        else:
                            by = height-1

                        for x in range(ax, bx+1):
                            for y in range(ay, by+1):
                                classes[-1].append((x, y))
                                pix[x, y] = (*pix[x, y][:3], 255)

        with open(image_file.with_suffix('.in'), 'wt') as test:
            test.write(f"{PurePosixPath(cwd / image_file)}\n")
            test.write(f"{str(PurePosixPath(cwd / image_file)).replace('_test.png', '_clusters.png', 1)}\n")
            test.write(f'{len(classes)}\n')
            for c in classes:
                test.write(f'{len(c)}\n' + ' '.join([' '.join(map(str, i)) for i in c]) + '\n')

        print(f'{image_file}: {len(classes)} classes')


if __name__ == '__main__':
    convert()
