import logging

from pathlib import Path

import numpy as np
import cv2 as cv

import fish


logging.basicConfig()


def make_frames(frames, lower, upper, smoothing):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    curves = []
    for frame in frames:
        frame = cv.morphologyEx(frame, cv.MORPH_CLOSE, kernel)
        edges = fish.get_edges(frame, lower, upper, smoothing)
        yield fish.draw_bounding_circles(frame, edges, curves)

        curves = [curve[-100:] for curve in curves]


if __name__ == "__main__":
    IN = Path(__file__).parent.parent / "data"
    OUT = Path(__file__).parent / "out"

    frames = fish.load_or_read(IN / "control")[100:]

    op = fish.make_movie(
        OUT / f"edge_test",
        frames=make_frames(fish.remove_background(frames, threshold=0), 60, 170, 3),
        num_frames=len(frames),
    )
