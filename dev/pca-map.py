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
def label_movie(movie, dimensions, clusters, remove_background = False, skip_frames = 0, chunk_size = 64, make_vectors = fish.sorted_ravel):
    op = f'{movie}__dims={dimensions}_clusters={clusters}_rmv={remove_background}_skip={skip_frames}_chunk={chunk_size}_vecs={make_vectors.__name__}.mp4'

    fish.label_movie(
        input_movie = f'{movie}.avi',
        output_path = op,
        pca_dimensions = dimensions,
        clusters = clusters,
        remove_background = remove_background,
        skip_frames = skip_frames,
        chunk_size = chunk_size,
        make_vectors = make_vectors,
    )

    htmap.transfer_output_files(op)


def concatenate_sorted_chunk_differences(frames, chunk_size):
    prev_chunks = {}
    for frame_number, frame in enumerate(tqdm(frames, desc = 'Building vectors from frames')):
        for (v, h), chunk in fish.iterate_over_chunks(fish.frame_to_chunks(frame, horizontal_chunk_size = chunk_size, vertical_chunk_size = chunk_size)):
            foo = fish.sorted_ravel(chunk)
            if (v, h) in prev_chunks:
                bar = fish.sorted_ravel(chunk - prev_chunks[v, h])
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
    vector_makers = [fish.sorted_ravel, concatenate_sorted_chunk_differences]
    chunk_sizes = [32, 64]

    for movie in movies:
        with label_movie.build_map(
            tag = f'{tag_prefix}-{movie}',
            map_options = htmap.MapOptions(
                fixed_input_files = [f'http://proxy.chtc.wisc.edu/SQUID/karpel/{movie}.avi']
            )
        ) as mb:
            for dim, clu, rmv, skip, mv, chunk_size in itertools.product(dimensions, clusters, remove_bgnds, skips, vector_makers, chunk_sizes):
                mb(movie, dim, clu, remove_background = rmv, skip_frames = skip, chunk_size = chunk_size, make_vectors = mv)

        map = mb.map
        print(f'Submitted {map} with {len(map)} jobs')
