import logging

from pathlib import Path
import itertools

import fish

logging.basicConfig()

if __name__ == "__main__":
    IN = Path(__file__).parent.parent / "data"
    OUT = Path(__file__).parent / "out"

    for dims, method, vectorizer in itertools.product(
        [2, 3], ["kmeans", "gmm"], [fish.sorted_ravel, fish.sorted_diff]
    ):
        fish.label_movie(
            input_movie=IN / "control.avi",
            output_path=OUT
            / f"test_dims={dims}_method={method}_vectorizer={vectorizer.__name__}",
            pca_dimensions=dims,
            clusters=2,
            remove_background=True,
            background_threshold=0,
            include_frames=slice(100, None),
            vectorizers=[vectorizer],
            chunk_size=64,
            clustering_algorithm=method,
        )
