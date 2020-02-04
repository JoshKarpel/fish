import logging

from pathlib import Path
import itertools

import numpy as np
import scipy as sp
import scipy.signal as sig
import cv2 as cv

import matplotlib.pyplot as plt

from tqdm import tqdm, trange

import fish

logging.basicConfig()

OPEN_KERNEL = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
CLOSE_KERNEL = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))


def make_frames(frames, edge_options, draw_on_original=True):
    backsub = train_background(frames, iterations=5)

    tracker = fish.ObjectTracker()

    for frame_idx, frame in enumerate(frames):
        # produce the "modified" frame that we actually perform tracking on
        mod = apply_background_subtraction(backsub, frame)

        mod = cv.morphologyEx(mod, cv.MORPH_OPEN, OPEN_KERNEL)
        mod = cv.morphologyEx(mod, cv.MORPH_CLOSE, CLOSE_KERNEL)

        # find edges, and from edges, contours
        edges = fish.get_edges(mod, **edge_options)
        contours = fish.get_contours(edges, area_cutoff=30)

        tracker.update_tracks(contours, frame_idx)
        tracker.check_for_locks(frame_idx)

        # produce the movie frame that we'll actually write out to disk
        img = cv.cvtColor((frame if draw_on_original else mod), cv.COLOR_GRAY2BGR)
        img = fish.draw_bounding_rectangles(img, contours)
        img = fish.draw_live_object_tracks(img, tracker)
        yield img


def train_background(frames, iterations=10, seed=1):
    rnd = np.random.RandomState(seed)

    backsub = cv.createBackgroundSubtractorMOG2(
        detectShadows=False, history=len(frames) * iterations
    )

    shuffled = frames.copy()

    for iteration in trange(iterations, desc="Training background model"):
        rnd.shuffle(shuffled)

        for frame in tqdm(
            shuffled,
            desc=f"Training background model (iteration {iteration + 1})",
            leave=False,
        ):
            backsub.apply(frame)

    return backsub


def apply_background_subtraction(background_model, frame):
    return background_model.apply(frame, learningRate=0)


if __name__ == "__main__":
    HERE = Path(__file__).absolute().parent
    DATA = HERE.parent / "data"
    OUT = HERE / "out" / Path(__file__).stem

    movies = ["D1-1"]
    draws = [True, False]
    lowers = [25, 50]
    uppers = [200, 225]
    smoothings = [3, 5, 7]

    for movie, draw, lower, upper, smoothing in itertools.product(
        movies, draws, lowers, uppers, smoothings
    ):
        input_frames = fish.read((DATA / f"{movie}.mp4"))[100:]

        output_frames = make_frames(
            input_frames,
            edge_options={
                "lower_threshold": lower,
                "upper_threshold": upper,
                "smoothing": smoothing,
            },
            draw_on_original=draw,
        )

        op = fish.make_movie(
            OUT
            / f"{movie}__lower={lower}_upper={upper}_smoothing={smoothing}_draw_on_original={draw}.mp4",
            frames=output_frames,
            num_frames=len(input_frames),
            fps=1,
        )
