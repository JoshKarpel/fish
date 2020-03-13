import logging

from pathlib import Path
import itertools
import csv

import numpy as np
import cv2 as cv

from tqdm import tqdm, trange

import fish

logging.basicConfig()


def diamond(n):
    a = np.arange(n)
    b = np.minimum(a, a[::-1])
    return ((b[:, None] + b) >= (n - 1) // 2).astype(np.uint8)


KERNEL_3 = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
KERNEL_5 = diamond(5)


def find_points(frames, edge_options):
    backsub = train_background_subtractor(frames, iterations=5)

    objects_by_frame = {}
    for frame_idx, frame in enumerate(tqdm(frames, desc="Detecting objects")):
        # produce the "modified" frame that we actually perform tracking on
        mod = apply_background_subtraction(backsub, frame)

        mod = cv.morphologyEx(mod, cv.MORPH_CLOSE, KERNEL_5)
        mod = cv.morphologyEx(mod, cv.MORPH_OPEN, KERNEL_3)

        # find edges, and from edges, contours
        edges = fish.get_edges(mod, **edge_options)
        contours = fish.detect_objects(edges, area_cutoff=30)

        objects_by_frame[frame_idx] = contours

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


def write_points(objects_by_frame, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open(mode="w", newline="") as f:
        writer = csv.DictWriter(f, ["frame", "x", "y", "area", "perimeter"])

        for frame_index, contours in objects_by_frame.items():
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


if __name__ == "__main__":
    HERE = Path(__file__).absolute().parent
    DATA = HERE.parent / "data"
    OUT = HERE / "out" / Path(__file__).stem

    movies = [f"D1-{n}" for n in range(1, 13)] + [f"C-{n}" for n in range(1, 4)]
    # movies = [f"D1-1"]
    lowers = [25]
    uppers = [200]
    smoothings = [3]

    for movie, lower, upper, smoothing in itertools.product(
        movies, lowers, uppers, smoothings
    ):
        input_frames = fish.read((DATA / f"{movie}.hsv"))[100:]

        objects = find_points(
            input_frames,
            edge_options={
                "lower_threshold": lower,
                "upper_threshold": upper,
                "smoothing": smoothing,
            },
        )

        write_points(objects, (OUT / f"{movie}__objects.csv"))
