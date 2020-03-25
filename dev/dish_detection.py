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


def draw_dish(frames):
    backsub = fish.train_background_subtractor(frames, iterations=1)
    circles = fish.find_circles(backsub.getBackgroundImage())
    dish = fish.decide_dish(circles)

    for frame_idx, frame in enumerate(frames):
        # produce the "modified" frame that we actually perform tracking on
        mod = frame.copy()
        # mod = fish.apply_background_subtraction(backsub, frame)

        # produce the movie frame that we'll actually write out to disk
        img = cv.cvtColor(mod, cv.COLOR_GRAY2BGR)

        for idx, circle in enumerate(circles):
            img = cv.circle(img, (circle.x, circle.y), circle.r, fish.GREEN, 2)
            img = cv.putText(
                img,
                str(idx),
                (circle.x + circle.r, circle.y),
                cv.FONT_HERSHEY_SIMPLEX,
                1,
                fish.GREEN,
                1,
                cv.LINE_AA,
            )
            img = cv.circle(img, (circle.x, circle.y), 2, fish.GREEN, 2)
            img = cv.putText(
                img,
                str(idx),
                (circle.x + 10, circle.y),
                cv.FONT_HERSHEY_SIMPLEX,
                1,
                fish.GREEN,
                1,
                cv.LINE_AA,
            )

        img = cv.circle(img, (dish.x, dish.y), dish.r + 10, fish.RED, 2)
        img = cv.putText(
            img,
            "D",
            (dish.x + dish.r + 10, dish.y),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            fish.RED,
            1,
            cv.LINE_AA,
        )
        img = cv.circle(img, (dish.x, dish.y), 2, fish.RED, 2)
        img = cv.putText(
            img,
            "D",
            (dish.x + 10, dish.y),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            fish.RED,
            1,
            cv.LINE_AA,
        )

        img_center = np.array(frame.shape)[::-1].T // 2
        img = cv.circle(img, (img_center[0], img_center[1]), 2, fish.YELLOW, 2)

        yield img


if __name__ == "__main__":
    HERE = Path(__file__).absolute().parent
    DATA = HERE.parent / "data"
    OUT = HERE / "out" / Path(__file__).stem

    movies = [f"D1-{n}" for n in range(1, 13)] + [f"C-{n}" for n in range(1, 4)]

    for movie in movies:
        input_frames = fish.read((DATA / f"{movie}.hsv"))[100:]

        tracker = fish.ObjectTracker()

        output_frames = draw_dish(input_frames)

        op = fish.make_movie(
            OUT / f"{movie}__dish.mp4",
            frames=output_frames,
            num_frames=len(input_frames),
            fps=10,
        )
