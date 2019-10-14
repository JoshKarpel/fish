import logging

from pathlib import Path
import itertools

import numpy as np
import cv2 as cv

import fish

logging.basicConfig()


def make_frames(frames, lower, upper, smoothing, draw_on_original=True):
    open_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (4, 4))
    close_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    backsub = cv.createBackgroundSubtractorKNN()
    tracker = fish.ObjectTracker()

    for frame_idx, frame in enumerate(frames):
        # produce the "modified" frame that we actually perform tracking on
        mod = backsub.apply(frame)
        mod = cv.morphologyEx(mod, cv.MORPH_OPEN, open_kernel)
        mod = cv.morphologyEx(mod, cv.MORPH_CLOSE, close_kernel)

        # find edges, and from edges, contours
        edges = fish.get_edges(mod, lower, upper, smoothing)
        contours = fish.get_contours(edges)

        if frame_idx > 10:
            tracker.update_tracks(contours, frame_idx)
            tracker.clean(frame_idx)

        # produce the movie frame that we'll actually write out to disk
        img = cv.cvtColor((frame if draw_on_original else mod), cv.COLOR_GRAY2BGR)
        img = fish.draw_bounding_rectangles(img, contours)
        img = fish.draw_live_object_tracks(img, tracker)
        yield img


if __name__ == "__main__":
    IN = Path(__file__).parent.parent / "data"
    OUT = Path(__file__).parent / "out" / Path(__file__).stem
    OUT.mkdir(exist_ok=True)

    for movie, draw in itertools.product(["drug", "control"], (True, False)):
        frames = fish.load_or_read(IN / movie)[100:]

        op = fish.make_movie(
            OUT / f"edge_test__{movie}__draw_on_original={draw}",
            frames=make_frames(frames, 100, 200, 5, draw_on_original=draw),
            num_frames=len(frames),
            fps=5,
        )
