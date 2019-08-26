import logging

from pathlib import Path

import fish

logging.basicConfig()

if __name__ == "__main__":
    IN = Path(__file__).parent.parent / "data"
    OUT = Path(__file__).parent / "out"

    fish.label_movie(
        input_movie=IN / "control.avi",
        output_path=OUT / f"test",
        pca_dimensions=2,
        clusters=2,
        remove_background=True,
        background_threshold=0,
        include_frames=slice(-100, None),
        vectorizers=[fish.sorted_ravel, fish.sorted_diff],
        chunk_size=31,
        chunk_skip=16,
        clustering_algorithm="gmm",
    )
