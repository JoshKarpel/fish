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


def track_objects(
    frames,
    tracker,
    edge_options,
    draw_on_original=True,
    draw_bounding_rectangles=True,
    draw_tracks=True,
):
    backsub = train_background_subtractor(frames, iterations=5)

    objects_by_frame = {}
    for frame_idx, frame in enumerate(tqdm(frames, desc="Detecting objects")):
        # produce the "modified" frame that we actually perform tracking on
        mod = frame.copy()
        mod = apply_background_subtraction(backsub, frame)

        mod = cv.morphologyEx(mod, cv.MORPH_CLOSE, KERNEL_5)
        mod = cv.morphologyEx(mod, cv.MORPH_OPEN, KERNEL_3)

        # find edges, and from edges, contours
        edges = fish.get_edges(mod, **edge_options)
        contours = fish.detect_objects(edges, area_cutoff=30)

        tracker.update_tracks(contours, frame_idx)

        # produce the movie frame that we'll actually write out to disk
        img = cv.cvtColor((frame if draw_on_original else mod), cv.COLOR_GRAY2BGR)
        if draw_bounding_rectangles:
            img = fish.draw_bounding_rectangles(img, tracker)
        if draw_tracks:
            img = fish.draw_object_tracks(
                img, tracker, track_length=100, live_only=True
            )
        yield img

    for oid, track in tracker.tracks.items():
        print(oid, track)

    for pid, path in tracker.paths.items():
        print(pid, path)

    return objects_by_frame


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

    # input_frames = fish.read((OUT.parent / "fake" / f"fake.mp4"))
    input_frames = fish.read((DATA / f"D1-1.hsv"))[100:]

    tracker = fish.ObjectTracker()

    output_frames = track_objects(
        input_frames,
        tracker=tracker,
        edge_options={"lower_threshold": 25, "upper_threshold": 200, "smoothing": 3,},
    )

    op = fish.make_movie(
        OUT / f"tracked.mp4", frames=output_frames, num_frames=len(input_frames), fps=1,
    )
