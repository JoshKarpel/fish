import logging

from pathlib import Path
import itertools
import csv

import numpy as np
import scipy as sp
import scipy.signal as sig
import cv2 as cv

import matplotlib.pyplot as plt

from tqdm import tqdm, trange

import fish

logging.basicConfig()


def show(path):
    frames = fish.cached_read(path)

    bgnd = fish.background_via_min(frames)

    cleaned_frame = fish.clean_frame_for_hough_transform(bgnd)
    circles = fish.find_circles_via_hough_transform(cleaned_frame)
    dish = fish.decide_dish(circles, cleaned_frame)

    last_pipette_frame = fish.find_last_pipette_frame(
        frames, background=bgnd, dish=dish,
    )

    for frame_idx, frame in enumerate(frames[: last_pipette_frame * 2]):
        # produce the movie frame that we'll actually write out to disk
        img = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)

        img = cv.circle(img, (dish.x, dish.y), dish.r, color=fish.RED, thickness=2)

        if frame_idx <= last_pipette_frame:
            pip = np.zeros_like(img)
            pip[:, :] = fish.MAGENTA
            img = fish.overlay_image(img, pip)

        yield img


if __name__ == "__main__":
    HERE = Path(__file__).absolute().parent
    DATA = HERE.parent / "data"
    OUT = HERE / "out" / Path(__file__).stem

    movies = [p for p in DATA.iterdir() if p.suffix == ".hsv"]

    for path in movies:
        output_frames = show(path)

        op = fish.make_movie(
            OUT / f"{path.stem}__pp.mp4", frames=output_frames, fps=10,
        )
