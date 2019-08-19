import logging

from pathlib import Path

import htmap

import fish


logging.basicConfig()


def make_frames(frames, lower, upper, smoothing):
    for frame in frames:
        edges = fish.get_edges(frame, lower, upper, smoothing)
        yield fish.draw_bounding_circles(frame, edges)


@htmap.mapped(map_options=htmap.MapOptions(request_disk="10GB", request_memory="16GB"))
def make_movie(movie, lower, upper, smoothing):
    frames = fish.load(movie)[100:]

    op = fish.make_movie(
        Path.cwd() / f"lower={lower}_upper={upper}_smoothing={smoothing}",
        frames=make_frames(fish.remove_background(frames), lower, upper, smoothing),
        num_frames=len(frames),
    )

    htmap.transfer_output_files(op)


if __name__ == "__main__":
    IN = Path(__file__).parent.parent / "data"
    OUT = Path(__file__).parent / "out"

    frames = fish.load_or_read(IN / "control")[100:]

    with make_movie.build_map() as mb:
        for lower in range(60, 200, 20):
            for upper in range(max(lower + 20, 100), 240, 20):
                for smoothing in (3, 5, 7):
                    mb("control", lower, upper, smoothing)

    m = mb.map
    print(m.tag)
