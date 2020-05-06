import logging

from pathlib import Path
import itertools
import collections
import shutil

from tqdm import tqdm

import numpy as np
import cv2 as cv
from scipy.stats import mode
import matplotlib.pyplot as plt

import fish
import fish.io

ISOLATED_DISTANCE_CUTOFF = 100


def find_isolated_blobs(blobs_path, movies_dir, out_dir=None):
    if out_dir is None:
        out_dir = Path.cwd()

    blobs_by_frame = fish.io.load_object(blobs_path)

    isolated_blobs_by_frame = filter_isolated_blobs_from_frames(blobs_by_frame)

    isolated_blobs = list(itertools.chain(*isolated_blobs_by_frame.values()))

    print(len(isolated_blobs))

    areas = np.array([blob.area for blob in isolated_blobs])
    perimeters = np.array([blob.perimeter for blob in isolated_blobs])

    fig, (left, right) = plt.subplots(nrows=1, ncols=2, figsize=(12, 8), dpi=100)

    bins = 100

    left.hist(areas, bins=bins)
    left.set_xlabel("area")
    left.set_ylabel("counts")

    right.hist(perimeters, bins=bins)
    right.set_xlabel("perimeter")
    right.set_ylabel("counts")

    fig.tight_layout()

    fish.save_figure(fig, out_dir / "hist.png")

    left.set_yscale("log")
    right.set_yscale("log")

    fish.save_figure(fig, out_dir / "hist-log.png")

    rough_singleton_blobs_by_area = fish.find_blobs_with_one_unit_rough(
        isolated_blobs, "area"
    )
    rough_singleton_blobs_by_perim = fish.find_blobs_with_one_unit_rough(
        isolated_blobs, "perimeter"
    )

    rough_area = np.mean([b.area for b in rough_singleton_blobs_by_area])
    rough_perimeter = np.mean([b.perimeter for b in rough_singleton_blobs_by_perim])

    # expand to include not just "exact matches" but also blobs with nearly the right area or perimeter
    singleton_blobs_by_area = [
        blob for blob in isolated_blobs if np.rint(blob.area / rough_area) == 1
    ]
    singleton_blobs_by_perim = [
        blob
        for blob in isolated_blobs
        if np.rint(blob.perimeter / rough_perimeter) == 1
    ]

    # this is our final estimate of the blob area
    blob_area = np.mean([b.area for b in singleton_blobs_by_area])
    blob_perimeter = np.mean([b.perimeter for b in singleton_blobs_by_perim])

    singleton_blobs = list(
        filter(
            lambda b: np.rint(b.area / blob_area) == 1
            and np.rint(b.perimeter / blob_perimeter) == 1,
            isolated_blobs,
        )
    )

    print(len(singleton_blobs))

    # blob_dir = out_dir / "blobs"
    # shutil.rmtree(blob_dir, ignore_errors=True)
    # for blob in tqdm(singleton_blobs, desc="Writing blob domain brightnesses"):
    #     fish.write_image(
    #         blob.domain_brightness,
    #         blob_dir
    #         / f"{blob.movie_stem}_frame={blob.frame_idx}_blob={blob.label}_brightness.png",
    #     )

    isolated_blobs_by_frame = collections.defaultdict(list)
    for blob in isolated_blobs:
        isolated_blobs_by_frame[blob.frame_idx].append(blob)

    fish.make_movie(
        out_dir / f"{blobs_path.stem}_labelled.mp4",
        make_labelled_movie(
            blobs_by_frame,
            isolated_blobs_by_frame,
            singleton_blobs,
            isolated_blobs[0].movie_path(movies_dir),
        ),
        num_frames=len(isolated_blobs_by_frame),
        fps=1,
    )

    # fish.save_object(isolated_blobs, out_dir / "isolated.blobs")

    return isolated_blobs


def make_labelled_movie(
    blobs_by_frame, isolated_blobs_by_frame, singleton_blobs, movie_path
):
    movie_frames = fish.cached_read(movie_path)
    for frame_idx, blobs in sorted(blobs_by_frame.items()):
        movie_frame = fish.convert_colorspace(
            movie_frames[frame_idx], cv.COLOR_GRAY2BGR
        )
        label_frame = np.zeros_like(movie_frame)

        if blobs is None:
            continue

        for blob in blobs:
            if isinstance(blob, fish.VelocityBlob):
                continue

            if blob in singleton_blobs:
                color = fish.CYAN
            elif blob in isolated_blobs_by_frame[frame_idx]:
                color = fish.MAGENTA
            else:
                color = fish.RED

            label_frame[blob.points_in_label] = color

        yield fish.overlay_image(movie_frame, label_frame)


def filter_isolated_blobs_from_frames(blobs_by_frame):
    isolated_blobs_by_frame = collections.defaultdict(list)

    for frame_idx, blobs in sorted(blobs_by_frame.items()):
        if blobs is None:
            continue

        for blob in blobs:
            if isinstance(blob, fish.VelocityBlob):
                continue

            # blob must be lonely
            if any(
                blob is not other
                and blob.distance_to(other) <= ISOLATED_DISTANCE_CUTOFF
                for other in blobs
            ):
                continue

            isolated_blobs_by_frame[frame_idx].append(blob)

    return isolated_blobs_by_frame


if __name__ == "__main__":
    HERE = Path(__file__).absolute().parent
    DATA = HERE.parent / "data"
    OUT = HERE / "out" / Path(__file__).stem

    # blobs_paths = sorted(DATA.glob("*.blobs"))
    blobs_paths = [DATA / "D1-1.blobs"]

    for blobs_path in blobs_paths:
        find_isolated_blobs(blobs_path, movies_dir=DATA, out_dir=OUT)
