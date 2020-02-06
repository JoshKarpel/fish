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


def diamond(n):
    a = np.arange(n)
    b = np.minimum(a, a[::-1])
    return ((b[:, None] + b) >= (n - 1) // 2).astype(np.uint8)


KERNEL_3 = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
KERNEL_5 = diamond(5)


def make_frames(frames, tracker, edge_options, draw_on_original=True, draw_tracks=True):
    backsub = train_background_subtractor(frames, iterations=5)

    for frame_idx, frame in enumerate(frames):
        # produce the "modified" frame that we actually perform tracking on
        mod = apply_background_subtraction(backsub, frame)

        mod = cv.morphologyEx(mod, cv.MORPH_CLOSE, KERNEL_5)
        mod = cv.morphologyEx(mod, cv.MORPH_OPEN, KERNEL_3)

        # find edges, and from edges, contours
        edges = fish.get_edges(mod, **edge_options)
        contours = fish.get_contours(edges, area_cutoff=30)

        tracker.update_tracks(contours, frame_idx)
        tracker.check_for_locks(frame_idx)

        # produce the movie frame that we'll actually write out to disk
        img = cv.cvtColor((frame if draw_on_original else mod), cv.COLOR_GRAY2BGR)
        img = fish.draw_bounding_rectangles(img, contours)
        if draw_tracks:
            img = fish.draw_live_object_tracks(img, tracker, history=100)
        yield img


def train_background_subtractor(frames, iterations=10, seed=1):
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

    movies = [f"D1-{n}" for n in range(1, 13)]
    lowers = [25]
    uppers = [200]
    smoothings = [3]

    originals = [False]
    tracks = [False]

    for movie, lower, upper, smoothing in itertools.product(
        movies, lowers, uppers, smoothings
    ):
        for original, track in itertools.product(originals, tracks):
            input_frames = fish.read((DATA / f"{movie}.mp4"))[100:]

            tracker = fish.ObjectTracker()

            output_frames = make_frames(
                input_frames,
                tracker,
                edge_options={
                    "lower_threshold": lower,
                    "upper_threshold": upper,
                    "smoothing": smoothing,
                },
                draw_on_original=original,
                draw_tracks=track,
            )

            for frame in tqdm(output_frames, desc="Processing movie..."):
                pass

            # op = fish.make_movie(
            #     OUT
            #     / f"{movie}__lower={lower}_upper={upper}_smoothing={smoothing}__original={original}_tracks={track}.mp4",
            #     frames=output_frames,
            #     num_frames=len(input_frames),
            #     fps=1,
            # )

            with (
                OUT
                / f"{movie}__lower={lower}_upper={upper}_smoothing={smoothing}__centroids.csv"
            ).open(mode="w", newline="") as f:
                writer = csv.DictWriter(f, ["frame", "x", "y", "area", "perimeter"])

                for frame_index, contours in tracker.raw.items():
                    for contour in contours:
                        x, y = contour.centroid
                        writer.writerow(
                            {
                                "frame": frame_index,
                                "x": x,
                                "y": y,
                                "area": contour.area,
                                "perimeter": contour.perimeter,
                            }
                        )
