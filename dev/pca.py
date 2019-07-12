#!/usr/bin/env python3

import itertools
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.util.shape import view_as_blocks
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, IncrementalPCA
from tqdm import tqdm

import fish

from dev.playground import process_frames

COLORS = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02', '#a6761d', '#666666']
RGB_COLORS = [(27, 158, 119), (217, 95, 2,), (117, 112, 179), (231, 41, 138), (102, 166, 30), (230, 171, 2), (166, 118, 29), (102, 102, 102)]
BRG_COLORS = [(B, R, G) for R, G, B in RGB_COLORS]


def BRG_fractions(b, r, g):
    tot = b + r + g
    return np.array([b / tot, r / tot, g / tot])


BRG_FRACTIONS = [BRG_fractions(b, r, g) for b, r, g in BRG_COLORS]


def frame_to_chunks(frame, horizontal_chunk_size = 64, vertical_chunk_size = 64):
    return view_as_blocks(frame, block_shape = (vertical_chunk_size, horizontal_chunk_size))


def iterate_over_chunks(chunks):
    for v, h in itertools.product(range(chunks.shape[0]), range(chunks.shape[1])):
        yield (v, h), chunks[v, h]


def chunk_to_sorted_vector(chunk):
    return np.sort(chunk, axis = None)


def stack_vectors(vectors):
    return np.vstack(vectors)


def vectors_from_frames(frames):
    for frame_number, frame in enumerate(tqdm(frames, desc = 'Building vectors from frames')):
        for (v, h), chunk in iterate_over_chunks(frame_to_chunks(frame)):
            yield chunk_to_sorted_vector(chunk)


def visualize_chunks_and_vectors(frames):
    for frame_number, frame in enumerate(frames):
        for (v, h), chunk in iterate_over_chunks(frame_to_chunks(frame)):
            vec = chunk_to_sorted_vector(chunk)

            fig, (left, right) = plt.subplots(1, 2, squeeze = True)
            right.plot(vec)
            right.set_title(f'Sorted Intensities')
            left.imshow(chunk, cmap = 'Greys_r')
            left.set_title(f'Chunk {frame_number}:{v}-{h}')
            plt.show()


def batches_of_vectors_for_pca(stacked, batch_size = 100):
    full, last = divmod(len(stacked), batch_size)
    yield full + (1 if last != 0 else 0)

    for i in range(0, full):
        yield stacked[i * batch_size: (i + 1) * batch_size]
    if last != 0:
        yield stacked[((i + 1) * batch_size):]


if __name__ == '__main__':
    IN = Path.cwd() / 'data'
    OUT = Path.cwd() / 'out'

    # visualize_chunks_and_vectors(frames)

    # stacked = stack_vectors(tuple(vectors_from_frames(frames)))
    # fish.save(IN / 'stacked_control_vectors.npy', stacked)

    stacked = fish.load(IN / 'stacked_control_vectors.npy')

    print(stacked.shape)

    pca_stack = stacked[100_000:200_000]
    print(pca_stack.shape)

    n = len(RGB_COLORS)
    pca = IncrementalPCA(n_components = n)
    batches = batches_of_vectors_for_pca(pca_stack, batch_size = 100)
    num_batches = next(batches)
    for batch in tqdm(batches, total = num_batches, desc = 'Performing PCA'):
        pca.partial_fit(batch)

    # print(pca)
    # print(pca.get_covariance())

    transformed = pca.transform(pca_stack)

    kmeans = KMeans(n_clusters = 2)
    means = kmeans.fit(transformed)
    # print(means.labels_)
    # print(means.cluster_centers_)

    # for x_dim, y_dim, z_dim in itertools.combinations(range(n), r = 3):
    #     fig = plt.figure(figsize = (12, 8))
    #     ax = fig.add_subplot(111, projection = '3d')
    #
    #     for label, color in zip(sorted(set(means.labels_)), COLORS):
    #         ax.scatter(
    #             transformed[means.labels_ == label, x_dim],
    #             transformed[means.labels_ == label, y_dim],
    #             transformed[means.labels_ == label, z_dim],
    #             color = color,
    #             label = label,
    #             linewidths = .1,
    #         )
    #
    #     ax.set_xlabel(f'{x_dim}')
    #     ax.set_ylabel(f'{y_dim}')
    #     ax.set_zlabel(f'{z_dim}')
    #
    #     plt.show()

    frames = fish.load_or_read(IN / 'control')
    frames_minus_bgnd = process_frames(frames, threshold = 1)
    # frames = frames[:100]


    # @profile
    def modify_frames(frames, pca, kmeans):
        horizontal_chunk_size = 64
        vertical_chunk_size = 64
        color_fractions = None
        for frame in frames:
            if color_fractions is None:
                color_fractions = np.empty(frame.shape + (3,), dtype = np.float64)

            coords = []
            vecs = []
            for (v, h), chunk in iterate_over_chunks(frame_to_chunks(frame, horizontal_chunk_size = horizontal_chunk_size, vertical_chunk_size = vertical_chunk_size)):
                coords.append((v, h))
                vecs.append(chunk_to_sorted_vector(chunk).reshape((1, -1)))

            vec_stack = stack_vectors(vecs)
            transformed = pca.transform(vec_stack)
            labels = kmeans.predict(transformed)

            for (v, h), label in zip(coords, labels):
                vslice = slice((v * vertical_chunk_size), ((v + 1) * vertical_chunk_size))
                hslice = slice((h * horizontal_chunk_size), ((h + 1) * horizontal_chunk_size))

                color_fractions[vslice, hslice] = BRG_FRACTIONS[label]

            yield (color_fractions * frame[..., np.newaxis]).astype(np.uint8)


    fish.make_movie(OUT / 'test_pca_no_bgnd', modify_frames(frames_minus_bgnd, pca, kmeans), num_frames = len(frames))
