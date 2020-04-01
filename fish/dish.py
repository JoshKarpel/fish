import dataclasses
import itertools

import cv2 as cv
import numpy as np

from . import colors, utils


def find_dish(frame):
    cleaned = clean_frame_for_hough_transform(frame)
    circles = find_circles_via_hough_transform(cleaned)
    dish = decide_dish(circles, cleaned)

    return dish


KERNEL_SIZE = 31
CIRCLE_CLOSING_KERNEL = cv.getStructuringElement(
    cv.MORPH_ELLIPSE, (KERNEL_SIZE, KERNEL_SIZE)
)
CANNY_KWARGS = dict(threshold1=1, threshold2=64, apertureSize=3, L2gradient=True)
AREA_CUTOFF = 5000


def clean_frame_for_hough_transform(frame):
    edges = cv.Canny(frame, **CANNY_KWARGS)

    # close the filtered edges with a big kernel to form big chunky shapes
    closed = cv.morphologyEx(edges, cv.MORPH_CLOSE, CIRCLE_CLOSING_KERNEL)

    # remove small blobs
    filtered = remove_components_below_cutoff_area(closed, AREA_CUTOFF)

    return filtered


CONNECTIVITY = 8


def remove_components_below_cutoff_area(frame, cutoff):
    modified = frame.copy()
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(frame, CONNECTIVITY)
    for label in range(num_labels):
        if stats[label, cv.CC_STAT_AREA] < cutoff:
            modified[labels == label] = 0
    return modified


def find_circles_via_hough_transform(cleaned_frame):
    circles = cv.HoughCircles(
        cleaned_frame,
        cv.HOUGH_GRADIENT,
        dp=1,
        minDist=1,
        param1=128,
        param2=10,
        minRadius=350,
        maxRadius=450,
    )[0]

    return [Circle(*map(int, circle)) for circle in circles]


def decide_dish(circles, cleaned_frame):
    return min(circles[:5], key=lambda c: area_ratio(c, cleaned_frame))

    # # trust the Hough transform voting
    # return circles[0]

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

    def draw_on(self, array, color=255):
        return cv.circle(array, (self.x, self.y), self.r, color, thickness=1)

    def mask_like(self, array):
        mask = np.zeros_like(array, dtype=np.uint8)
        mask = cv.circle(mask, (self.x, self.y), self.r, 1, thickness=-1)

        return mask

    @property
    def area(self):
        return np.pi * (self.r ** 2)


def area_ratio(circle, frame):
    area = np.sum(utils.apply_mask(frame, circle.mask_like(frame))) / 255
    return area / circle.area


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
