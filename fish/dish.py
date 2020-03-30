import dataclasses

import cv2 as cv
import numpy as np

from tqdm import tqdm

from . import utils


def show_frame(frame):
    cv.imshow("image", frame)
    cv.waitKey(0)
    cv.destroyAllWindows()


def save_frame(path, frame):
    cv.imwrite(str(path), frame)


def remove_components_below_cutoff_area(frame, cutoff):
    modified = frame.copy()
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(frame, 4)
    for label in range(num_labels):
        if stats[label, cv.CC_STAT_AREA] < cutoff:
            modified[labels == label] = 0
    return modified


def find_circles(frame):
    blurred = cv.GaussianBlur(frame, (7, 7), 3)
    edges = cv.Canny(blurred, 3, 7, L2gradient = True)
    filtered = remove_components_below_cutoff_area(edges, 100)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (31, 31))
    closed = cv.morphologyEx(filtered, cv.MORPH_CLOSE, kernel)

    circles = cv.HoughCircles(
        closed,
        cv.HOUGH_GRADIENT,
        dp = 1,
        minDist = 100,
        param1 = 150,
        param2 = 35,
        minRadius = 250,
        maxRadius = 0,
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


@dataclasses.dataclass(frozen = True)
class Circle:
    x: int
    y: int
    r: int
