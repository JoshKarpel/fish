import logging

from pathlib import Path

import numpy as np
import cv2 as cv

import fish

logging.basicConfig()


def make_frames(frames, lower, upper, smoothing, draw_on_original=True):
    open_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (4, 4))
    close_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    backsub = cv.createBackgroundSubtractorKNN()
    oj = fish.ObjectTracker()

    for frame_idx, frame in enumerate(frames):
        mod = backsub.apply(frame)
        mod = cv.morphologyEx(mod, cv.MORPH_OPEN, open_kernel)
        mod = cv.morphologyEx(mod, cv.MORPH_CLOSE, close_kernel)
        edges = fish.get_edges(mod, lower, upper, smoothing)
        contours = fish.get_contours(edges)
        if frame_idx > 10:
            fish.track_objects(oj, contours, frame_idx)
            oj.clean(frame_idx)

        img = cv.cvtColor((frame if draw_on_original else mod), cv.COLOR_GRAY2BGR)
        img = fish.draw_contours(img, contours)
        img = fish.draw_objects(img, oj)
        yield img


if __name__ == "__main__":
    IN = Path(__file__).parent.parent / "data"
    OUT = Path(__file__).parent / "out" / Path(__file__).stem
    OUT.mkdir(exist_ok=True)

    for movie in ["drug", "control"]:
        frames = fish.load_or_read(IN / movie)[100:]

        # frames = fish.remove_background(frames, threshold=0)

        for d in [True, False]:
            op = fish.make_movie(
                OUT / f"edge_test__{movie}__draw_on_original={d}",
                frames=make_frames(frames, 100, 200, 5, draw_on_original=d),
                num_frames=len(frames),
                fps=5,
            )
