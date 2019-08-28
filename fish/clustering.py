import logging
from typing import List, Iterable, Union, Callable

import itertools

import numpy as np
from skimage.util.shape import view_as_blocks
from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import GaussianMixture

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
    kmeans = MiniBatchKMeans(n_clusters=clusters)

    for batch in _get_transformed_batches_for_clustering(
        vector_stacks, pcas, batch_size
    ):
        kmeans.partial_fit(batch)

    return kmeans


def _do_clustering_via_gmm(vector_stacks, pcas, clusters, batch_size=100):
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
    vectorizers: Iterable[Callable],
    chunk_size,
    label_colors,
    corner_blocks=5,
):
    color_fractions = None
    for frame_idx, frame in enumerate(frames):
        if color_fractions is None:
            color_fractions = np.empty(frame.shape + (3,), dtype=np.float64)

        transformed_stacks = []
        coords = []
        for idx, (pca, mv) in enumerate(zip(pcas, vectorizers)):
            vectors_for_frame = []
            for _, v, h, vec in make_vectors_from_frames(
                frame[np.newaxis, ...], vectorizer=mv, chunk_size=chunk_size
            ):
                # only need to build coords on first vectorizer
                if idx == 0:
                    coords.append((v, h))
                vectors_for_frame.append(vec)

            # it's more efficient to stack up all the vectors, then transform them
            vec_stack = stack_vectors(vectors_for_frame)
            transformed_stacks.append(normalized_pca_transform(pca, vec_stack))

        labels = clusterer.predict(np.concatenate(transformed_stacks, axis=1))

        for (v, h), label in zip(coords, labels):
            vslice = slice((v * chunk_size), ((v + 1) * chunk_size))
            hslice = slice((h * chunk_size), ((h + 1) * chunk_size))

            frame[
                (v * chunk_size) : (v * chunk_size) + corner_blocks,
                (h * chunk_size) : (h * chunk_size) + corner_blocks,
            ] = 255
            color_fractions[vslice, hslice] = colors.fractions(*label_colors[label])

        yield (color_fractions * frame[..., np.newaxis]).astype(np.uint8)


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
    try:
        label_colors = colors.BRG_COLOR_SCHEMES[clusters]
    except KeyError:
        raise ValueError(f"no suitable color scheme for {clusters} clusters")

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
        frames, pcas, clusterer, vectorizers, chunk_size, label_colors=label_colors
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

    fig = plt.figure(figsize=(8, 8))

    if transformed.shape[1] == 2:
        ax = fig.add_subplot(111)
        ax.scatter(transformed[:, 0], transformed[:, 1], s=1, c=c)
    if transformed.shape[1] == 3:
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(transformed[:, 0], transformed[:, 1], transformed[:, 2], s=1, c=c)
    else:
        return

    fig.tight_layout()
    plt.savefig(str(output_path.with_suffix(".png")), dpi=600)
