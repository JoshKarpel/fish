from pathlib import Path
import itertools

import fish

if __name__ == '__main__':
    IN = Path.cwd() / 'data'
    OUT = Path.cwd() / 'out'

    dimensions = [5]
    clusters = [4]
    for dims, clus in itertools.product(dimensions, clusters):
        fish.label_movie(
            input_movie = IN / 'control.avi',
            output_path = OUT / f'control__dims={dims}_clusters={clus}',
            pca_dimensions = dims,
            clusters = clus,
        )