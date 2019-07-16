from pathlib import Path
import itertools
import sys

import fish
import htmap


@htmap.mapped(map_options = htmap.MapOptions(
    request_disk = '10GB',
    request_memory = '8GB',
))
def label_movie(movie, dimensions, clusters, remove_background = False):
    op = f'{movie}__dims={dimensions}_clusters={clusters}_rmv={remove_background}.mp4'

    fish.label_movie(
        input_movie = f'{movie}.avi',
        output_path = op,
        pca_dimensions = dimensions,
        clusters = clusters,
        remove_background = remove_background,
    )

    htmap.transfer_output_files(op)


if __name__ == '__main__':
    tag_prefix = sys.argv[1]
    docker_version = sys.argv[2]

    htmap.settings['DOCKER.IMAGE'] = f'maventree/fish:{docker_version}'

    movies = ['control', 'drug']
    dimensions = [2, 5, 10]
    clusters = [2, 4, 8]
    remove_bgnds = [False, True]

    for movie in movies:
        with label_movie.build_map(
            tag = f'{tag_prefix}-{movie}',
            map_options = htmap.MapOptions(
                fixed_input_files = [f'http://proxy.chtc.wisc.edu/SQUID/karpel/{movie}.avi']
            )
        ) as mb:
            for dim, clu, rmv in itertools.product(dimensions, clusters, remove_bgnds):
                mb(movie, dim, clu, remove_background = rmv)

        map = mb.map
        print(f'Submitted {map} with {len(map)} jobs')
