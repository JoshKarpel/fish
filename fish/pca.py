#!/usr/bin/env python3

import itertools

import numpy as np
import matplotlib.pyplot as plt
from skimage.util.shape import view_as_blocks
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm

from . import io, bgnd, colors


def frame_to_chunks(frame, horizontal_chunk_size = 64, vertical_chunk_size = 64):
    return view_as_blocks(frame, block_shape = (vertical_chunk_size, horizontal_chunk_size))


def iterate_over_chunks(chunks):
    for v, h in itertools.product(range(chunks.shape[0]), range(chunks.shape[1])):
        yield (v, h), chunks[v, h]


def sorted_ravel(frame_idx, v, h, chunk, prev_chunks):
    return np.sort(chunk, axis = None)


def sorted_ravel_with_diff(frame_idx, v, h, chunk, prev_chunks):
    foo = np.sort(chunk, axis = None)
    if frame_idx != 0:
        bar = np.sort(chunk - prev_chunks[frame_idx - 1, v, h], axis = None)
    else:
        bar = np.zeros_like(foo)

    return np.concatenate((foo, bar))


def stack_vectors(vectors):
    return np.vstack(vectors)


def make_vectors_from_frames(frames, chunk_size, make_vector = sorted_ravel):
    prev_chunks = {}
    for frame_idx, frame in enumerate(frames):
        for (v, h), chunk in iterate_over_chunks(frame_to_chunks(frame, horizontal_chunk_size = chunk_size, vertical_chunk_size = chunk_size)):
            yield frame_idx, v, h, make_vector(frame_idx, v, h, chunk, prev_chunks)

            prev_chunks[frame_idx, v, h] = chunk


def make_batches_of_vectors(stacked, batch_size = 100):
    full, last = divmod(len(stacked), batch_size)
    return full + (1 if last != 0 else 0), _make_batches(stacked, batch_size, full, last)


def _make_batches(stacked, batch_size, full, last):
    for i in range(0, full):
        yield stacked[i * batch_size: (i + 1) * batch_size]
    if last != 0:
        yield stacked[((i + 1) * batch_size):]


def do_pca(vector_stack, pca_dimensions):
    num_batches, batches = make_batches_of_vectors(vector_stack, batch_size = 100)
    pca = IncrementalPCA(n_components = pca_dimensions)
    for batch in tqdm(batches, total = num_batches, desc = 'Performing PCA'):
        pca.partial_fit(batch)

    return pca


# anything from here will work
# https://scikit-learn.org/stable/modules/clustering.html
# todo: switch on these from higher-level code
def do_clustering(vector_stack, pca, clusters):
    num_batches, batches = make_batches_of_vectors(vector_stack, batch_size = 100)

    kmeans = MiniBatchKMeans(n_clusters = clusters)
    for batch in tqdm(batches, total = num_batches, desc = 'Performing Clustering'):
        kmeans.partial_fit(pca.transform(batch))

    return kmeans


def label_chunks_in_frames(frames, pca, clusterer, make_vector, chunk_size, label_colors, corner_blocks = 5):
    color_fractions = None
    for frame_idx, frame in enumerate(frames):
        if color_fractions is None:
            color_fractions = np.empty(frame.shape + (3,), dtype = np.float64)

        coords = []
        vecs = []
        for _, v, h, vec in make_vectors_from_frames(frame[np.newaxis, ...], make_vector = make_vector, chunk_size = chunk_size):
            coords.append((v, h))
            vecs.append(vec.reshape((1, -1)))

        # it's more efficient to stack up all the vectors, then transform them
        vec_stack = stack_vectors(vecs)
        transformed = pca.transform(vec_stack)
        labels = clusterer.predict(transformed)

        for (v, h), label in zip(coords, labels):
            vslice = slice((v * chunk_size), ((v + 1) * chunk_size))
            hslice = slice((h * chunk_size), ((h + 1) * chunk_size))

            frame[(v * chunk_size):(v * chunk_size) + corner_blocks, (h * chunk_size): (h * chunk_size) + corner_blocks] = 255
            color_fractions[vslice, hslice] = colors.fractions(*label_colors[label])

        yield (color_fractions * frame[..., np.newaxis]).astype(np.uint8)


def label_movie(
    input_movie,
    output_path,
    pca_dimensions: int,
    clusters: int,
    remove_background = False,
    background_threshold = 0,
    skip_frames = 0,
    chunk_size = 64,
    make_vector = sorted_ravel,
):
    try:
        label_colors = colors.COLOR_SCHEMES[clusters]
    except KeyError:
        raise ValueError(f'no suitable color scheme for {clusters} clusters')

    frames = io.read(input_movie)[skip_frames:]
    if remove_background:
        frames = np.stack(list(bgnd.remove_background(frames, threshold = background_threshold)), axis = 0)

    vector_stack = stack_vectors(list(vec for *_, vec in make_vectors_from_frames(frames, chunk_size, make_vector)))

    pca = do_pca(vector_stack, pca_dimensions)
    clusterer = do_clustering(vector_stack, pca, clusters)

    labelled_frames = label_chunks_in_frames(frames, pca, clusterer, make_vector, chunk_size, label_colors = label_colors)

    io.make_movie(
        output_path,
        labelled_frames,
        num_frames = len(frames),
    )
