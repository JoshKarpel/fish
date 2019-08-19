import logging
from typing import Optional

import numpy as np
import cv2 as cv

from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def get_edges(frame: np.array, lower_threshold, upper_threshold, smoothing):
    return cv.Canny(
        frame,
        threshold1=lower_threshold,
        threshold2=upper_threshold,
        apertureSize=smoothing,
        L2gradient=True,
    )


def vector_length(a, b) -> float:
    return np.linalg.norm(a - b)


def draw_bounding_circles(frame: np.array, edges: np.array, curves):
    image, contours, hierarchy = cv.findContours(
        edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE
    )

    img = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)

    img = cv.drawContours(img, contours, -1, color=(255, 0, 0), thickness=1)

    for c in contours:
        moments = cv.moments(c)
        area = moments["m00"]
        if area < 16:
            continue

        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        pos = np.array([cx, cy])

        rect = cv.minAreaRect(c)
        box = cv.boxPoints(rect)
        box = np.int0(box)
        cv.drawContours(img, [box], 0, (0, 255, 0), 1)

        if len(curves) == 0:
            curves.append([pos])

        nearest_curve = min(curves, key=lambda curve: vector_length(curve[-1], pos))
        if vector_length(nearest_curve[-1], pos) < 50:
            nearest_curve.append(pos)
        else:
            curves.append([pos])

    cv.polylines(
        img,
        [np.vstack(curve) for curve in curves],
        color=(0, 0, 255),
        thickness=1,
        isClosed=False,
    )

    return img
