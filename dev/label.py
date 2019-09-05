import logging

from pathlib import Path
import itertools

import fish

logging.basicConfig()

if __name__ == "__main__":
    IN = Path(__file__).parent.parent / "data"
    OUT = Path(__file__).parent / "out"

    for dims, clusters, cluster_alg, vectorizer in itertools.product(
        [1, 2], [3, 4, 6], ["kmeans", "gmm"], [[fish.sorted_ravel, fish.sorted_diff]]
    ):
        print(dims, clusters, cluster_alg, vectorizer)
        fish.label_movie(
            input_movie=IN / "drug.avi",
            output_path=OUT
            / f"test_dims={dims}_clusters={clusters}_cluster={cluster_alg}_vectorizer={'+'.join(v.__name__ for v in vectorizer)}",
            pca_dimensions=dims,
            clusters=clusters,
            remove_background=True,
            background_threshold=0,
            include_frames=slice(-500, -400),
            vectorizers=vectorizer,
            chunk_size=32,
            clustering_algorithm=cluster_alg,
            make_cluster_plot=True,
        )
