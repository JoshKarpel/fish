import logging

from pathlib import Path
import itertools
import csv
import getpass
import shutil

import numpy as np
import cv2 as cv

import fish

import htmap


def diamond(n):
    a = np.arange(n)
    b = np.minimum(a, a[::-1])
    return ((b[:, None] + b) >= (n - 1) // 2).astype(np.uint8)


def find_objects(frames, edge_options):
    backsub = train_background_subtractor(frames, iterations=5)

    KERNEL_3 = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    KERNEL_5 = diamond(5)

    objects_by_frame = {}
    for frame_idx, frame in enumerate(frames):
        # produce the "modified" frame that we actually perform tracking on
        mod = apply_background_subtraction(backsub, frame)

        mod = cv.morphologyEx(mod, cv.MORPH_CLOSE, KERNEL_5)
        mod = cv.morphologyEx(mod, cv.MORPH_OPEN, KERNEL_3)

        # find edges, and from edges, contours
        edges = fish.get_edges(mod, **edge_options)
        contours = fish.get_contours(edges, area_cutoff=30)

        objects_by_frame[frame_idx] = [
            make_object_output(frame_idx, c) for c in contours
        ]

    return objects_by_frame


def train_background_subtractor(frames, iterations=10, seed=1):
    rnd = np.random.RandomState(seed)

    backsub = cv.createBackgroundSubtractorMOG2(
        detectShadows=False, history=len(frames) * iterations
    )

    shuffled = frames.copy()

    for iteration in range(iterations):
        rnd.shuffle(shuffled)

        for frame in shuffled:
            backsub.apply(frame)

    return backsub


def apply_background_subtraction(background_model, frame):
    return background_model.apply(frame, learningRate=0)


def make_object_output(frame_index, contour):
    x, y = contour.centroid
    return {
        "frame": frame_index,
        "x": x,
        "y": y,
        "area": contour.area,
        "perimeter": contour.perimeter,
    }


def run_object_detector(movie, lower, upper, smoothing):
    local_path = Path(f"{movie}.hsv")

    input_frames = fish.read(local_path)[100:]

    objects_by_frame = find_objects(
        input_frames,
        edge_options={
            "lower_threshold": lower,
            "upper_threshold": upper,
            "smoothing": smoothing,
        },
    )

    return dict(
        movie=movie,
        lower=lower,
        upper=upper,
        smoothing=smoothing,
        objects=objects_by_frame,
    )


if __name__ == "__main__":
    tag = input("Map tag? ")

    if len(tag) < 1 or tag in htmap.get_tags():
        raise ValueError(f"tag {tag} is already in use")

    staging_path = Path("/staging") / getpass.getuser() / "fish"
    print(f"Staging path is {staging_path}")

    movies = [f"D1-{n}" for n in range(1, 13)] + [f"C-{n}" for n in range(1, 4)]

    lowers = eval(input("Edge detector lower thresholds? "))
    uppers = eval(input("Edge detector upper thresholds? "))
    smoothings = eval(input("Edge detector smoothings? "))

    kwargs = [
        dict(movie=movie, lower=lower, upper=upper, smoothing=smoothing)
        for movie, lower, upper, smoothing in itertools.product(
            movies, lowers, uppers, smoothings
        )
    ]

    htmap.settings["DOCKER.IMAGE"] = input("Docker Image? ")
    mo = htmap.MapOptions(
        input_files=[
            [f"file://{p.as_posix()}.hsv"]
            for p in (staging_path / kw["movie"] for kw in kwargs)
        ],
        request_memory="6GB",
        request_disk="2GB",
        max_idle="100",
        requirements="(Target.HasCHTCStaging == true)",
    )

    map = htmap.starmap(run_object_detector, kwargs=kwargs, map_options=mo, tag=tag)
    print(map.tag)
