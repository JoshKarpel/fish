#!/usr/bin/env python3

import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage.util.shape import view_as_blocks
from tqdm import tqdm

import fish


def frames_to_chunks(frames, horizontal_chunk_size = 64, vertical_chunk_size = 64):
    for frame_number, frame in enumerate(frames):
        chunks = view_as_blocks(frame, block_shape = (vertical_chunk_size, horizontal_chunk_size))
        for v, h in itertools.product(range(chunks.shape[0]), range(chunks.shape[1])):
            yield (frame_number, v, h), chunks[v, h]


def chunk_to_sorted_array(chunk):
    return np.sort(chunk, axis = None)


if __name__ == '__main__':
    IN = Path.cwd() / 'data'
    OUT = Path.cwd() / 'out'

    frames = fish.load_or_read(IN / 'control')

    for (frame_number, v, h), chunk in frames_to_chunks(frames):
        vec = chunk_to_sorted_array(chunk)

        fig, (left, right) = plt.subplots(1, 2, squeeze = True)
        right.plot(vec)
        right.set_title(f'Sorted Intensities')
        left.imshow(chunk, cmap = 'Greys_r')
        left.set_title(f'Chunk {frame_number}:{v}-{h}')
        plt.show()
