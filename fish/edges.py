import logging
from typing import Optional

import numpy as np
import cv2 as cv

from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def get_edges(frame: np.array, lower_threshold, upper_threshold, smoothing):
    return cv.Canny(
        frame, lower_threshold, upper_threshold, apertureSize=smoothing, L2gradient=True
    )


def draw_bounding_circles(frame: np.array, edges: np.array):
    image, contours, hierarchy = cv.findContours(
        edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE
    )

    img = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)

    for c in contours:
        (x, y), radius = cv.minEnclosingCircle(c)
        center = (int(x), int(y))
        radius = int(radius)
        img = cv.circle(img, center, radius, (0, 255, 0), 1)

    return img
