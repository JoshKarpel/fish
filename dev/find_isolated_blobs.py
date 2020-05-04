import logging

from pathlib import Path

import itertools
from tqdm import tqdm

import numpy as np
import cv2 as cv

import fish
import fish.io

ISOLATED_DISTANCE_CUTOFF = 100


def find_isolated_blobs(blobs_paths, out_dir=None):
    if out_dir is None:
        out_dir = Path.cwd()

    isolated_blobs = list(
        itertools.chain.from_iterable(
            tqdm([_yield_isolated_blobs(p) for p in blobs_paths])
        )
    )

    print(len(isolated_blobs))

    for blob in tqdm(isolated_blobs, desc="Writing blob domain brightnesses"):
        fish.write_image(
            blob.domain_brightness,
            out_dir
            / "blobs"
            / f"{blob.movie_stem}_frame={blob.frame_idx}_blob={blob.label}_brightness.png",
        )

    fish.save_object(isolated_blobs, out_dir / "isolated.blobs")


def _yield_isolated_blobs(blobs_path):
    blobs_by_frame = fish.io.load_object(blobs_path)

    for frame_idx, blobs in sorted(blobs_by_frame.items()):
        if blobs is None:
            continue

        for blob in blobs:
            if any(
                blob is not other
                and blob.distance_to(other) <= ISOLATED_DISTANCE_CUTOFF
                for other in blobs
            ):
                continue

            yield blob


if __name__ == "__main__":
    HERE = Path(__file__).absolute().parent
    DATA = HERE.parent / "data"
    OUT = HERE / "out" / Path(__file__).stem

    blobs_paths = sorted(DATA.glob("*.blobs"))
    # blobs_paths = [DATA / "D1-1.blobs"]

    find_isolated_blobs(blobs_paths, OUT)
