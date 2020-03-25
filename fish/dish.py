import dataclasses

import cv2 as cv
import numpy as np

from tqdm import tqdm

from . import utils


def find_circles(frame):
    circles = cv.HoughCircles(
        cv.GaussianBlur(frame, (7, 7), 3),
        cv.HOUGH_GRADIENT,
        dp=1,
        minDist=100,
        param1=150,
        param2=35,
        minRadius=250,
        maxRadius=0,
    )[0]

    return [Circle(*map(int, circle)) for circle in circles]


def decide_dish(circles):
    # trust the Hough transform voting
    return circles[0]

    # # closest to the center?
    # img_center = np.array(frame_shape)[::-1] // 2
    # return min(
    #     circles,
    #     key = lambda circle: circle.distance_to(img_center),
    # )


@dataclasses.dataclass(frozen=True)
class Circle:
    x: int
    y: int
    r: int
