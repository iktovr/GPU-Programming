import sys
from pathlib import Path
import numpy as np

for filename in map(Path, sys.argv[1:]):
    with open(filename, 'rt') as file:
        lines = list(map(float, file.readlines()))
        if filename.name.startswith('cuda'):
            lines = lines[1:]
        print(f'{filename.name}: {np.mean(lines)}')
input()
