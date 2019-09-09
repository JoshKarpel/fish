import logging
from typing import Optional

import numpy as np
import cv2 as cv

from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def get_edges(frame: np.ndarray, lower_threshold, upper_threshold, smoothing):
    return cv.Canny(
        frame,
        threshold1=lower_threshold,
        threshold2=upper_threshold,
        apertureSize=smoothing,
        L2gradient=True,
    )


def vector_length(a, b) -> float:
    return np.linalg.norm(a - b)


def get_contours(edges: np.ndarray):
    image, contours, hierarchy = cv.findContours(
        edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE
    )

    return contours


BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)


def draw_contours(frame, contours):
    frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)

    centers = []
    areas = []
    boxes = []
    for c in contours:
        moments = cv.moments(c, binaryImage=True)
        area = moments["m00"]
        if area <= 10:
            continue
        areas.append(area)

        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        pos = np.array([cx, cy])
        centers.append(pos)

        rect = cv.minAreaRect(c)
        box = cv.boxPoints(rect)
        box = np.int0(box)
        boxes.append(box)

    frame = cv.drawContours(
        frame, contours, -1, color=RED, thickness=1, lineType=cv.LINE_AA
    )
    cv.drawContours(frame, boxes, -1, color=GREEN, thickness=2, lineType=cv.LINE_AA)
    for (x, y), area in zip(centers, areas):
        cv.rectangle(
            frame,
            (x - 1, y - 1),
            (x + 1, y + 1),
            color=BLUE,
            thickness=-1,
            lineType=cv.LINE_AA,
        )
        cv.putText(
            frame,
            str(area),
            (x + 15, y - 15),
            fontFace=cv.FONT_HERSHEY_DUPLEX,
            fontScale=1,
            color=BLUE,
            thickness=1,
            lineType=cv.LINE_AA,
        )

    return frame
