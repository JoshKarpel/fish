from pathlib import Path
import itertools

from tqdm import tqdm
import numpy as np

import fish

if __name__ == '__main__':
    IN = Path.cwd() / 'data'
    OUT = Path.cwd() / 'out'

    dimensions = [10]
    clusters = [2]
    for dims, clus in itertools.product(dimensions, clusters):
        fish.label_movie(
            input_movie = IN / 'control.avi',
            output_path = OUT / f'test',
            pca_dimensions = dims,
            clusters = clus,
            remove_background = True,
            # skip_frames = 1500,
            make_vector = fish.sorted_ravel,
            chunk_size = 64,
            clustering_algorithm = 'gmm',
        )
