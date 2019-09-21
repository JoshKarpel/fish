import logging

from pathlib import Path
import itertools

import fish

logging.basicConfig()

if __name__ == "__main__":
    IN = Path(__file__).parent.parent / "data"
    OUT = Path(__file__).parent / "out" / Path(__file__).stem
    OUT.mkdir(exist_ok=True)

    for (
        dims,
        clusters,
        cluster_alg,
        vectorizers,
        draw,
        cutoff_quantile,
    ) in itertools.product(
        [1, 2, 3],
        [3, 4, 6],
        ["gmm"],
        [
            [fish.sorted_ravel, fish.sorted_diff],
            [fish.sorted_ravel, fish.sorted_diff, fish.sorted_ds],
        ],
        [True, False],
        [0.0, 0.8, 0.9, 0.95],
    ):
        print(dims, clusters, cluster_alg, vectorizers, cutoff_quantile)
        op = (
            OUT
            / f"test_dims={dims}_clusters={clusters}_cluster={cluster_alg}_vectorizer={'+'.join(v.__name__ for v in vectorizers)}__cutoff={round(cutoff_quantile * 100)}__draw_on_original={draw}"
        )
        print(op)
        fish.label_movie(
            input_movie=IN / "drug.avi",
            output_path=op,
            pca_dimensions=dims,
            clusters=clusters,
            background_threshold=0,
            include_frames=slice(200, 700),
            vectorizers=vectorizers,
            chunk_size=32,
            cutoff_quantile=cutoff_quantile,
            clustering_algorithm=cluster_alg,
            make_cluster_plot=True,
            draw_on_original=draw,
        )
