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


def overlay_petri(frames, path):
    bgnd = fish.background_via_min(frames)

    dish = fish.find_dish(bgnd)

    # produce the movie frame that we'll actually write out to disk
    img = fish.bw_to_bgr(frames[0])

    img = cv.circle(img, (dish.x, dish.y), dish.r, color=fish.RED, thickness=1)

    fish.save_frame(path, img)


if __name__ == "__main__":
    HERE = Path(__file__).absolute().parent
    DATA = HERE.parent / "data"
    OUT = HERE / "out" / Path(__file__).stem

    movies = [f"D1-{n}" for n in range(1, 13)] + [f"C-{n}" for n in range(1, 4)]

    for movie in movies:
        input_frames = fish.cached_read((DATA / f"{movie}.hsv"))[100:]

        overlay_petri(input_frames, OUT / f"{movie}__dish.png")
