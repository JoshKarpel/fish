from pathlib import Path
import itertools

import fish
import htmap


@htmap.mapped(map_options = htmap.MapOptions(
    request_disk = '10GB',
    request_memory = '4GB',
))
def label_movie(movie, dimensions, clusters):
    op = f'{movie}__dims={dimensions}_clusters={clusters}.mp4'

    fish.label_movie(
        input_movie = f'{movie}.avi',
        output_path = op,
        pca_dimensions = dimensions,
        clusters = clusters,
    )

    htmap.transfer_output(op)


if __name__ == '__main__':
    movies = ['control', 'drug']
    dimensions = [2, 5, 10]
    clusters = [2, 4, 8]

    for movie in movies:
        with label_movie.map_builder(
            map_options = htmap.MapOptions(
                shared_input_files = [f'http://proxy.chtc.wisc.edu/SQUID/karpel/{movie}.avi']
            )
        ) as mb:
            for dim, clu in itertools.product(dimensions, clusters):
                mb(movie, dim, clu)

        map = mb.map
        print(f'Submitted {map} with {len(map)} jobs')
