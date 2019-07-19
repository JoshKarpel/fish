from pathlib import Path
import itertools
import sys

import numpy as np
from tqdm import tqdm

import fish
import htmap


@htmap.mapped(map_options = htmap.MapOptions(
    request_disk = '10GB',
    request_memory = '8GB',
))
def label_movie(movie, dimensions, clusters, remove_background = False, skip_frames = 0, make_vectors = fish.chunk_to_sorted_vector):
    op = f'{movie}__dims={dimensions}_clusters={clusters}_rmv={remove_background}_skip={skip_frames}_vecs={make_vectors.__name__}.mp4'

    fish.label_movie(
        input_movie = f'{movie}.avi',
        output_path = op,
        pca_dimensions = dimensions,
        clusters = clusters,
        remove_background = remove_background,
        skip_frames = skip_frames,
        make_vectors = make_vectors,
    )

    htmap.transfer_output_files(op)


def concatenate_sorted_chunk_differences(frames):
    prev_chunks = {}
    for frame_number, frame in enumerate(tqdm(frames, desc = 'Building vectors from frames')):
        for (v, h), chunk in fish.iterate_over_chunks(fish.frame_to_chunks(frame)):
            foo = fish.chunk_to_sorted_vector(chunk)
            if (v, h) in prev_chunks:
                bar = fish.chunk_to_sorted_vector(chunk - prev_chunks[v, h])
            else:
                bar = np.zeros_like(foo)

            yield np.concatenate((foo, bar))

            prev_chunks[v, h] = chunk


if __name__ == '__main__':
    tag_prefix = sys.argv[1]
    docker_version = sys.argv[2]

    htmap.settings['DOCKER.IMAGE'] = f'maventree/fish:{docker_version}'

    movies = ['control', 'drug']
    dimensions = [2, 5, 10, 20]
    clusters = [2, 3, 4, 6, 8]
    remove_bgnds = [False, True]
    skips = [0, 100]
    vector_makers = [fish.chunk_to_sorted_vector, concatenate_sorted_chunk_differences]

    for movie in movies:
        with label_movie.build_map(
            tag = f'{tag_prefix}-{movie}',
            map_options = htmap.MapOptions(
                fixed_input_files = [f'http://proxy.chtc.wisc.edu/SQUID/karpel/{movie}.avi']
            )
        ) as mb:
            for dim, clu, rmv, skip, mv in itertools.product(dimensions, clusters, remove_bgnds, skips, vector_makers):
                mb(movie, dim, clu, remove_background = rmv, skip_frames = skip, make_vectors = mv)

        map = mb.map
        print(f'Submitted {map} with {len(map)} jobs')
