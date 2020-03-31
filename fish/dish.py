import dataclasses
import itertools

import cv2 as cv
import numpy as np

from tqdm import tqdm

from . import utils, colors


CIRCLE_CLOSING_KERNEL = cv.getStructuringElement(cv.MORPH_ELLIPSE, (31, 31))


def clean_frame_for_hough_transform(frame):
    blurred = cv.GaussianBlur(frame, (7, 7), 3)
    edges = cv.Canny(blurred, 3, 7, L2gradient=True)

    # remove small edges
    filtered = remove_components_below_cutoff_area(edges, 100)

    # close the filtered edges with a big kernel to form big chunky shapes
    closed = cv.morphologyEx(filtered, cv.MORPH_CLOSE, CIRCLE_CLOSING_KERNEL)

    return closed


def remove_components_below_cutoff_area(frame, cutoff):
    modified = frame.copy()
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(frame, 4)
    for label in range(num_labels):
        if stats[label, cv.CC_STAT_AREA] < cutoff:
            modified[labels == label] = 0
    return modified


def find_circles_via_hough_transform(cleaned_frame):
    circles = cv.HoughCircles(
        cleaned_frame,
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


@dataclasses.dataclass
class Circle:
    x: int
    y: int
    r: int


def draw_circles(frame, circles, mark_centers=False, label=False):
    img = colors.bw_to_bgr(frame)

    for idx, (circle, color) in enumerate(
        zip(circles, itertools.cycle(colors.BGR_COLORS))
    ):
        img = cv.circle(img, (circle.x, circle.y), circle.r, color, 2)

        if mark_centers:
            img = cv.circle(img, (circle.x, circle.y), 2, color, 2)

        if label:
            img = cv.putText(
                img,
                str(idx),
                (circle.x + circle.r, circle.y),
                cv.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                1,
                cv.LINE_AA,
            )
            img = cv.putText(
                img,
                str(idx),
                (circle.x + 10, circle.y),
                cv.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                1,
                cv.LINE_AA,
            )

    return img
