from pathlib import Path
import itertools
import sys

import numpy as np
from tqdm import tqdm

import fish
import htmap

import fish.vectorize


@htmap.mapped(map_options=htmap.MapOptions(request_disk="10GB", request_memory="16GB"))
def label_movie(
    movie,
    dimensions,
    clusters,
    remove_background,
    include_frames,
    chunk_size,
    vectorizers,
):
    parts = [
        f"dims={dimensions}",
        f"clusters={clusters}",
        f"chunk={chunk_size}",
        f"vecs={'+'.join(v.__name__ for v in vectorizers)}.mp4",
    ]
    op = Path.cwd() / f"{movie}__{'_'.join(parts)}"

    fish.label_movie(
        input_movie=f"{movie}.avi",
        output_path=op,
        pca_dimensions=dimensions,
        clusters=clusters,
        remove_background=remove_background,
        include_frames=include_frames,
        chunk_size=chunk_size,
        vectorizers=vectorizers,
    )

    htmap.transfer_output_files(op)


if __name__ == "__main__":
    tag_prefix = sys.argv[1]
    docker_version = sys.argv[2]

    htmap.settings["DOCKER.IMAGE"] = f"maventree/fish:{docker_version}"

    movies = ["control"]
    dimensions = [2, 4, 6, 10, 20]
    clusters = [2, 3, 4, 6, 8]
    vectorizer_sets = [
        [fish.vectorize.sorted_ravel],
        [fish.vectorize.sorted_ravel, fish.sorted_diff],
    ]
    chunk_sizes = [32, 64]

    for movie in movies:
        with label_movie.build_map(
            tag=f"{tag_prefix}-{movie}",
            map_options=htmap.MapOptions(
                fixed_input_files=[
                    f"http://proxy.chtc.wisc.edu/SQUID/karpel/{movie}.avi"
                ]
            ),
        ) as mb:
            for dim, clu, vecs, chunk_size in itertools.product(
                dimensions, clusters, vectorizer_sets, chunk_sizes
            ):
                mb(
                    movie,
                    dimensions=dim,
                    clusters=clu,
                    chunk_size=chunk_size,
                    vectorizers=vecs,
                    remove_background=True,
                    include_frames=slice(100, None),
                )

        map = mb.map

        print(f"Submitted {map} with {len(map)} jobs")
