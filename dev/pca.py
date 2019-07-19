from pathlib import Path
import itertools

from tqdm import tqdm
import numpy as np

import fish


def concatenate_sorted_chunk_differences(frames, chunk_size):
    prev_chunks = {}
    for frame_number, frame in enumerate(tqdm(frames, desc = 'Building vectors from frames')):
        for (v, h), chunk in fish.iterate_over_chunks(fish.frame_to_chunks(frame, horizontal_chunk_size = chunk_size, vertical_chunk_size = chunk_size)):
            foo = fish.sorted_ravel(chunk)
            if (v, h) in prev_chunks:
                bar = fish.sorted_ravel(chunk - prev_chunks[v, h])
            else:
                bar = np.zeros_like(foo)

            yield np.concatenate((foo, bar))

            prev_chunks[v, h] = chunk


if __name__ == '__main__':
    IN = Path.cwd() / 'data'
    OUT = Path.cwd() / 'out'

    dimensions = [5, 10]
    clusters = [2, 4, 6, 8]
    for dims, clus in itertools.product(dimensions, clusters):
        fish.label_movie(
            input_movie = IN / 'control.avi',
            output_path = OUT / f'control__dims={dims}_clusters={clus}',
            pca_dimensions = dims,
            clusters = clus,
            remove_background = True,
            make_vectors = concatenate_sorted_chunk_differences,
        )
