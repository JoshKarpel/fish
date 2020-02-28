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

logging.basicConfig()


def diamond(n):
    a = np.arange(n)
    b = np.minimum(a, a[::-1])
    return ((b[:, None] + b) >= (n - 1) // 2).astype(np.uint8)


KERNEL_3 = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
KERNEL_5 = diamond(5)


def find_objects(frames, tracker, edge_options, background_subtractor_iterations=5):
    backsub = train_background_subtractor(
        frames, iterations=background_subtractor_iterations
    )

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

    return tracker


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


def write_points(tracker, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open(mode="w", newline="") as f:
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
    return out


def object_rows(tracker):
    for frame_index, contours in tracker.raw.items():
        for contour in contours:
            x, y = contour.centroid
            yield {
                "frame": frame_index,
                "x": x,
                "y": y,
                "area": contour.area,
                "perimeter": contour.perimeter,
            }


def run_object_detector(movie, lower, upper, smoothing):
    staged_path = staging_path / f"{movie}.hsv"
    local_path = Path(f"{movie}.hsv")

    print(f"Copying from {staged_path} -> {local_path}")
    shutil.copy2(staged_path, local_path)
    print(f"Copy succeeded!")

    input_frames = fish.read(local_path)[100:]

    tracker = fish.ObjectTracker()

    tracker = find_objects(
        input_frames,
        tracker,
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
        objects=list(object_rows(tracker)),
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

    htmap.settings["DOCKER.IMAGE"] = input("Docker Image? ")
    mo = htmap.MapOptions(
        transfer_input_files=[
            f"file://{p}" for p in (staging_path / m for m in movies)
        ],
        request_memory="1GB",
        request_disk="2GB",
        requirements="(Target.HasCHTCStaging == true)",
    )

    with htmap.build_map(run_object_detector, map_options=mo, tag=tag) as builder:
        for movie, lower, upper, smoothing in itertools.product(
            movies, lowers, uppers, smoothings
        ):
            builder(movie, lower, upper, smoothing)

    map = builder.map
    print(map.tag)
