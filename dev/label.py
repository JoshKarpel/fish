import logging

from pathlib import Path
import itertools

import fish

logging.basicConfig()

if __name__ == "__main__":
    IN = Path(__file__).parent.parent / "data"
    OUT = Path(__file__).parent / "out" / Path(__file__).stem
    OUT.mkdir(exist_ok=True)

    for dims, clusters, cluster_alg, vectorizer, draw in itertools.product(
        [1, 2],
        [3, 4, 6],
        ["gmm"],
        [
            [fish.sorted_ravel, fish.sorted_diff],
            [fish.sorted_ravel, fish.sorted_diff, fish.sorted_ds],
        ],
        [True, False],
        [0.8, 0.9, 0.95],
    ):
        print(dims, clusters, cluster_alg, vectorizer)
        fish.label_movie(
            input_movie=IN / "drug.avi",
            output_path=OUT
            / f"test_dims={dims}_clusters={clusters}_cluster={cluster_alg}_vectorizer={'+'.join(v.__name__ for v in vectorizer)}__draw_on_original={draw}",
            pca_dimensions=dims,
            clusters=clusters,
            background_threshold=0,
            include_frames=slice(-600, -400),
            vectorizers=vectorizer,
            chunk_size=32,
            cutoff_quantile=0.95,
            clustering_algorithm=cluster_alg,
            make_cluster_plot=True,
            draw_on_original=draw,
        )
