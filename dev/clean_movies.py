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


def clean(frames, background_training_iterations=5, dish_radius_buffer=10):
    backsub = fish.train_background_subtractor(
        frames, iterations=background_training_iterations
    )

    dish = fish.decide_dish(fish.find_circles(backsub.getBackgroundImage()))
    dish_mask = np.zeros_like(backsub.getBackgroundImage(), np.uint8)
    dish_mask = cv.circle(
        dish_mask, (dish.x, dish.y), dish.r + dish_radius_buffer, 1, thickness=-1
    )

    for frame_idx, frame in enumerate(frames):
        # produce the movie frame that we'll actually write out to disk
        img = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)

        bgnd_mask = fish.apply_background_subtraction(backsub, frame)
        img = cv.bitwise_and(img, img, mask=bgnd_mask)

        img = cv.bitwise_and(img, img, mask=dish_mask)

        yield img


if __name__ == "__main__":
    HERE = Path(__file__).absolute().parent
    DATA = HERE.parent / "data"
    OUT = HERE / "out" / Path(__file__).stem

    movies = [f"D1-{n}" for n in range(1, 13)] + [f"C-{n}" for n in range(1, 4)]

    for movie in movies:
        input_frames = fish.read((DATA / f"{movie}.hsv"))[100:]

        output_frames = clean(
            input_frames, background_training_iterations=1, dish_radius_buffer=5
        )

        op = fish.make_movie(
            OUT / f"{movie}__cleaned.mp4",
            frames=output_frames,
            num_frames=len(input_frames),
            fps=10,
        )
