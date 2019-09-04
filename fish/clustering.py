import logging
from typing import List, Iterable, Union, Callable, Collection

import itertools
import collections

import numpy as np
from skimage.util.shape import view_as_blocks
from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import GaussianMixture

import cv2 as cv

from tqdm import tqdm

from . import io, bgnd, colors, vectorize

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def frame_to_chunks(frame, horizontal_chunk_size=64, vertical_chunk_size=64):
    return view_as_blocks(
        frame, block_shape=(vertical_chunk_size, horizontal_chunk_size)
    )


def iterate_over_chunks(chunks):
    for v, h in itertools.product(range(chunks.shape[0]), range(chunks.shape[1])):
        yield (v, h), chunks[v, h]


def stack_vectors(vectors):
    return np.stack(vectors, axis=0)


def make_vectors_from_frames(frames, chunk_size, vectorizer):
    prev_chunks = {}
    for frame_idx, frame in enumerate(frames):
        for (v, h), chunk in iterate_over_chunks(
            frame_to_chunks(
                frame, horizontal_chunk_size=chunk_size, vertical_chunk_size=chunk_size
            )
        ):
            yield frame_idx, v, h, vectorizer(frame_idx, v, h, chunk, prev_chunks)

            prev_chunks[frame_idx, v, h] = chunk


def make_all_vectors_from_frames(frames, chunk_size, vectorizers):
    prev_chunks = {}
    for frame_idx, frame in enumerate(frames):
        chunks_for_frame = []
        for (v, h), chunk in iterate_over_chunks(
            frame_to_chunks(
                frame, horizontal_chunk_size=chunk_size, vertical_chunk_size=chunk_size
            )
        ):
            chunks_for_frame.append(
                (
                    (v, h),
                    [vec(frame_idx, v, h, chunk, prev_chunks) for vec in vectorizers],
                )
            )
            prev_chunks[frame_idx, v, h] = chunk
        yield frame_idx, chunks_for_frame


def make_batches_of_vectors(stacked, batch_size=100):
    full, last = divmod(len(stacked), batch_size)
    return (
        full + (1 if last != 0 else 0),
        _make_batches(stacked, batch_size, full, last),
    )


def _make_batches(stacked, batch_size, full, last):
    for i in range(0, full):
        yield stacked[i * batch_size : (i + 1) * batch_size]
    if last != 0:
        yield stacked[((i + 1) * batch_size) :]


def make_all_batches(vector_stacks, batch_size=100):
    num_batches = []
    batches_for_each_vectorizer = []
    for vector_stack in vector_stacks:
        num, batches = make_batches_of_vectors(vector_stack, batch_size=batch_size)
        num_batches.append(num)
        batches_for_each_vectorizer.append(batches)

    if not len(set(num_batches)) == 1:
        raise Exception("Number of batches must match!")

    return num_batches[0], batches_for_each_vectorizer


def do_pca(
    vector_stacks: List[np.ndarray],
    pca_dimensions: Union[int, Iterable[int]],
    batch_size: int = 100,
) -> List[IncrementalPCA]:
    if isinstance(pca_dimensions, int):
        pca_dimensions = itertools.repeat(pca_dimensions)

    num_batches, batches_for_each_vectorizer = make_all_batches(
        vector_stacks, batch_size=batch_size
    )
    pcas = [
        IncrementalPCA(n_components=dims)
        for _, dims in zip(range(len(vector_stacks)), pca_dimensions)
    ]
    for idx, (pca, batches) in enumerate(zip(pcas, batches_for_each_vectorizer)):
        for batch in tqdm(
            batches, total=num_batches, desc=f"Performing PCA {idx + 1}/{len(pcas)}"
        ):
            pca.partial_fit(batch)

    return pcas


def normalized_pca_transform(pca: IncrementalPCA, vector: np.ndarray) -> np.ndarray:
    return pca.transform(vector) / pca.singular_values_[0]


def _do_cluster_via_kmeans(vector_stacks, pcas, clusters, batch_size=100):
    logger.debug(f"Created KMeans clusterer")
    kmeans = MiniBatchKMeans(n_clusters=clusters)

    for batch in _get_transformed_batches_for_clustering(
        vector_stacks, pcas, batch_size
    ):
        kmeans.partial_fit(batch)

    return kmeans


def _do_clustering_via_gmm(vector_stacks, pcas, clusters, batch_size=100):
    logger.debug(f"Created GMM clusterer")
    gmm = GaussianMixture(n_components=clusters, warm_start=True)

    for batch in _get_transformed_batches_for_clustering(
        vector_stacks, pcas, batch_size
    ):
        gmm.fit(batch)

    return gmm


def _get_transformed_batches_for_clustering(vector_stacks, pcas, batch_size=100):
    num_batches, batch_iterators = make_all_batches(
        vector_stacks, batch_size=batch_size
    )
    for batches in tqdm(
        zip(*batch_iterators), total=num_batches, desc="Performing clustering"
    ):
        transformed_batches = [
            normalized_pca_transform(pca, batch) for pca, batch in zip(pcas, batches)
        ]

        yield np.concatenate(transformed_batches, axis=1)


def do_clustering(vector_stack, pca, clusters, clustering_algorithm):
    return _CLUSTERING_ALGORITHMS[clustering_algorithm](vector_stack, pca, clusters)


_CLUSTERING_ALGORITHMS = {
    "kmeans": _do_cluster_via_kmeans,
    "gmm": _do_clustering_via_gmm,
}


def label_chunks_in_frames(
    frames,
    pcas,
    clusterer,
    vectorizers: Collection[Callable],
    chunk_size,
    corner_blocks=5,
):
    color_fractions = None

    label_counters = []
    for frame_idx, coords_and_vecs in make_all_vectors_from_frames(
        frames, chunk_size=chunk_size, vectorizers=vectorizers
    ):
        vec_stacks = [[] for _ in pcas]
        coords = []
        for chunk_coords, chunk_vecs in coords_and_vecs:
            coords.append(chunk_coords)
            for vec, stack in zip(chunk_vecs, vec_stacks):
                stack.append(vec)

        transformed_stacks = []
        for pca, vec_stack in zip(pcas, vec_stacks):
            vec_stack = stack_vectors(vec_stack)
            transformed_stacks.append(normalized_pca_transform(pca, vec_stack))

        cat_stacks = np.concatenate(transformed_stacks, axis=1)
        labels = clusterer.predict(cat_stacks)

        frame = frames[frame_idx]
        if color_fractions is None:
            color_fractions = np.empty(frame.shape + (3,), dtype=np.float64)

        for (v, h), label in zip(coords, labels):
            vslice = slice((v * chunk_size), ((v + 1) * chunk_size))
            hslice = slice((h * chunk_size), ((h + 1) * chunk_size))

            frame[
                (v * chunk_size) : (v * chunk_size) + corner_blocks,
                (h * chunk_size) : (h * chunk_size) + corner_blocks,
            ] = 255
            color_fractions[vslice, hslice] = colors.fractions(
                *colors.BGR_COLORS[label]
            )

        # apply label colors to frame
        frame = (color_fractions * frame[..., np.newaxis]).astype(np.uint8)

        # count number of each label in frame
        label_counter = collections.Counter(labels)
        label_counters.append(label_counter)

        # draw legend box
        legend_width = 130
        legend_height = (len(clusterer.means_) * 40) + 10
        cv.rectangle(
            frame, (0, 0), (legend_width, legend_height), color=(0, 0, 0), thickness=-1
        )
        cv.rectangle(
            frame,
            (0, 0),
            (legend_width, legend_height),
            color=(255, 255, 255),
            thickness=1,
        )

        # draw legend text
        just = len(str(max(label_counter.values())))
        for i, label in enumerate(range(len(clusterer.means_)), start=1):
            cv.putText(
                frame,
                f"{label}: {label_counter[label]: >{just}d}",
                (10, (i * 40)),
                fontFace=cv.FONT_HERSHEY_DUPLEX,
                fontScale=1,
                color=255 * colors.fractions(*colors.BGR_COLORS[label]),
                thickness=1,
                lineType=cv.LINE_AA,
            )

        yield frame


def label_movie(
    input_movie,
    output_path,
    pca_dimensions: int,
    clusters: int,
    remove_background: bool = True,
    background_threshold: Union[int, float] = 0,
    include_frames: slice = None,
    chunk_size: int = 64,
    vectorizers: Iterable[Callable] = (vectorize.sorted_ravel,),
    clustering_algorithm: str = "kmeans",
):
    frames = io.load_or_read(input_movie)
    if include_frames is not None:
        frames = frames[include_frames]
    if remove_background:
        frames = np.stack(
            list(bgnd.remove_background(frames, threshold=background_threshold)), axis=0
        )
    logger.debug(f"Frame stack shape (number of frames, height, width): {frames.shape}")

    vector_stacks = [
        stack_vectors(
            [
                vec
                for *_, vec in make_vectors_from_frames(frames, chunk_size, vectorizer)
            ]
        )
        for vectorizer in vectorizers
    ]
    logger.debug(
        f"Vector stack shapes (number of vectors, vector length): {[vs.shape for vs in vector_stacks]}"
    )

    pcas = do_pca(vector_stacks, pca_dimensions)
    clusterer = do_clustering(vector_stacks, pcas, clusters, clustering_algorithm)

    plot_clusters(output_path, vector_stacks, pcas, clusterer)

    labelled_frames = label_chunks_in_frames(
        frames, pcas, clusterer, vectorizers, chunk_size
    )

    io.make_movie(output_path, labelled_frames, num_frames=len(frames))


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_clusters(output_path, vector_stacks, pcas, clusterer):
    transformed_vectors = []
    for pca, vector_stack in zip(pcas, vector_stacks):
        transformed_vectors.append(normalized_pca_transform(pca, vector_stack))
    transformed = np.concatenate(transformed_vectors, axis=1)

    labels = clusterer.predict(transformed)
    c = [colors.HTML_COLORS[i] for i in labels]

    if isinstance(clusterer, MiniBatchKMeans):
        centers = clusterer.cluster_centers_
    elif isinstance(clusterer, GaussianMixture):
        centers = clusterer.means_
    center_kwargs = dict(c="black", s=200, alpha=0.5, marker="x")

    fig = plt.figure(figsize=(8, 8))
    if transformed.shape[1] == 1:
        ax = fig.add_subplot(111)
        ax.scatter(transformed[:, 0], np.ones_like(transformed[:, 0]), s=1, c=c)
        ax.scatter(centers[:, 0], np.ones_like(centers[:, 0]), **center_kwargs)
        ax.set_ylim(0.9, 1.1)
    elif transformed.shape[1] == 2:
        ax = fig.add_subplot(111)
        ax.scatter(transformed[:, 0], transformed[:, 1], s=1, c=c)
        ax.scatter(centers[:, 0], centers[:, 1], **center_kwargs)
    elif transformed.shape[1] == 3:
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(transformed[:, 0], transformed[:, 1], transformed[:, 2], s=1, c=c)
        ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], **center_kwargs)
    else:
        logger.debug("Skipped making plot")
        return

    fig.tight_layout()
    op = str(output_path.with_suffix(".png"))
    logger.debug(f"Writing cluster plot to {op}")
    plt.savefig(op, dpi=600)
