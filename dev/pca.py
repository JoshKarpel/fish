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
        pca_dimensions=10,
        clusters=4,
        remove_background=True,
        include_frames=range(-500, -400),
        make_vector=fish.sorted_ravel,
        chunk_size=64,
        clustering_algorithm="kmeans",
    )
