from typing import Optional
import logging

import dataclasses
import itertools

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

    filtered = []
    for c in contours:
        moments = cv.moments(c, binaryImage=True)
        area = moments["m00"]
        if area <= 10:
            continue

        filtered.append(c)

    return filtered


BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)

# must be able to lock an object when it deregisters
# after not being seen for a while
# to stop very old tracks from picking up new entries
# but still keep around the track


class ObjectTracker:
    def __init__(self):
        self.objects = {}
        self._id_counter = itertools.count()

    def _next_id(self):
        return next(self._id_counter)

    def register(self, centroid):
        id = self._next_id()
        self.objects[id] = [centroid]

    def update(self, centroids):
        if len(self.objects) == 0:
            for centroid in centroids:
                self.register(centroid)

        centroids = centroids[:]
        for object_id, positions in list(self.objects.items()):
            if len(centroids) == 0:
                return
            last_position = positions[-1]
            centroid_idx, closest_centroid = min(
                enumerate(centroids), key=lambda c: vector_length(c[1], last_position)
            )

            if vector_length(closest_centroid, last_position) <= 20:
                positions.append(closest_centroid)
                centroids.pop(centroid_idx)

        for centroid in centroids:
            self.register(centroid)


def track_objects(object_tracker, contours):
    centroids = []
    for c in contours:
        moments = cv.moments(c, binaryImage=True)
        area = moments["m00"]
        cx = int(moments["m10"] / area)
        cy = int(moments["m01"] / area)
        pos = np.array([cx, cy])
        centroids.append(pos)

    object_tracker.update(centroids)


def draw_contours(frame, contours):
    centroids = []
    areas = []
    boxes = []
    for c in contours:
        moments = cv.moments(c, binaryImage=True)
        area = moments["m00"]
        areas.append(area)

        cx = int(moments["m10"] / area)
        cy = int(moments["m01"] / area)
        pos = np.array([cx, cy])
        centroids.append(pos)

        rect = cv.minAreaRect(c)
        box = cv.boxPoints(rect)
        box = np.int0(box)
        boxes.append(box)

    cv.drawContours(frame, boxes, -1, color=GREEN, thickness=2, lineType=cv.LINE_AA)
    for (x, y), area in zip(centroids, areas):
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


def draw_objects(frame, object_tracker):
    cv.polylines(
        frame,
        [np.vstack(curve) for curve in object_tracker.objects.values()],
        isClosed=False,
        color=RED,
        thickness=2,
        lineType=cv.LINE_AA,
    )

    return frame
