#!/usr/bin/env python3

import itertools
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from skimage.util.shape import view_as_blocks
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tqdm import tqdm

import fish


def frames_to_chunks(frames, horizontal_chunk_size = 64, vertical_chunk_size = 64):
    for frame_number, frame in enumerate(tqdm(frames, desc = 'Converting frames to chunks')):
        chunks = view_as_blocks(frame, block_shape = (vertical_chunk_size, horizontal_chunk_size))
        for v, h in itertools.product(range(chunks.shape[0]), range(chunks.shape[1])):
            yield (frame_number, v, h), chunks[v, h]


def chunk_to_sorted_vector(chunk):
    return np.sort(chunk, axis = None)


def stack_vectors(vectors):
    return np.vstack(vectors)


def visualize_chunks_and_vectors(frames):
    for (frame_number, v, h), chunk in frames_to_chunks(frames):
        vec = chunk_to_sorted_vector(chunk)

        fig, (left, right) = plt.subplots(1, 2, squeeze = True)
        right.plot(vec)
        right.set_title(f'Sorted Intensities')
        left.imshow(chunk, cmap = 'Greys_r')
        left.set_title(f'Chunk {frame_number}:{v}-{h}')
        plt.show()


if __name__ == '__main__':
    IN = Path.cwd() / 'data'
    OUT = Path.cwd() / 'out'

    # frames = fish.load_or_read(IN / 'control')
    # stacked = stack_vectors(tuple(chunk_to_sorted_vector(chunk) for _, chunk in frames_to_chunks(frames)))
    # fish.save(IN / 'stacked_control_vectors.npy', stacked)

    stacked = fish.load(IN / 'stacked_control_vectors.npy')

    print(stacked.shape)

    stacked = stacked[:50000]

    pca = PCA(n_components = 10, copy = False)
    fit_result = pca.fit_transform(stacked)
    print(pca)
    print(fit_result.shape)

    kmeans = KMeans(n_clusters = 2)
    means = kmeans.fit(fit_result)
    # print(means.labels_)
    # print(means.cluster_centers_)
