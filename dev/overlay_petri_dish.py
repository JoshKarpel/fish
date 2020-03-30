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


def overlay_petri(frames, path, background_training_iterations=5):
    backsub = fish.train_background_subtractor(
        frames, iterations=background_training_iterations
    )

    dish = fish.decide_dish(fish.find_circles(backsub.getBackgroundImage()))

    # produce the movie frame that we'll actually write out to disk
    img = cv.cvtColor(frames[0], cv.COLOR_GRAY2BGR)

    img = cv.circle(img, (dish.x, dish.y), dish.r, color=fish.RED, thickness=1)

    fish.save_frame(path, img)


if __name__ == "__main__":
    HERE = Path(__file__).absolute().parent
    DATA = HERE.parent / "data"
    OUT = HERE / "out" / Path(__file__).stem

    movies = [f"D1-{n}" for n in range(1, 13)] + [f"C-{n}" for n in range(1, 4)]

    for movie in movies:
        input_frames = fish.read((DATA / f"{movie}.hsv"))[100:]

        overlay_petri(input_frames, OUT / f"{movie}__dish.png", background_training_iterations=1)
