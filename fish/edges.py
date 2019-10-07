from typing import Optional, List
import logging

import dataclasses
import itertools

import numpy as np
import cv2 as cv

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

Frame = np.ndarray


def get_edges(
    frame: Frame, lower_threshold: float, upper_threshold: float, smoothing: int
):
    return cv.Canny(
        frame,
        threshold1=lower_threshold,
        threshold2=upper_threshold,
        apertureSize=smoothing,
        L2gradient=True,
    )


def vector_length(a: np.ndarray, b: np.ndarray) -> float:
    return np.linalg.norm(a - b)


def get_contours(edges: np.ndarray, area_cutoff: int = 10):
    image, contours, hierarchy = cv.findContours(
        edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE
    )

    contours = map(Contour, contours)
    contours = [c for c in contours if c.area > area_cutoff]

    return contours


BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)


class ObjectTracker:
    def __init__(
        self,
        snap_to: int = 20,
        lock_after: int = 10,
        max_track_length: Optional[int] = None,
    ):
        self.snap_to = snap_to
        self.lock_after = lock_after
        self.max_track_length = max_track_length

        self.objects = {}
        self._id_counter = itertools.count()
        self.last_updated_frame = {}
        self.locked = {}

    def _next_id(self):
        return next(self._id_counter)

    def update(self, centroids, frame_idx):
        if len(self.objects) == 0:
            for centroid in centroids:
                self.register(centroid, frame_idx)

        centroids = centroids[:]
        for object_id, positions in (
            (oid, pos) for oid, pos in self.objects.items() if not self.locked[oid]
        ):
            if len(centroids) == 0:
                return
            last_position = positions[-1]
            centroid_idx, closest_centroid = min(
                enumerate(centroids), key=lambda c: vector_length(c[1], last_position)
            )

            if vector_length(closest_centroid, last_position) <= self.snap_to:
                positions.append(closest_centroid)
                centroids.pop(centroid_idx)
                self.last_updated_frame[object_id] = frame_idx

        for centroid in centroids:
            self.register(centroid, frame_idx)

    def register(self, centroid, frame_idx):
        id = self._next_id()
        self.objects[id] = [centroid]
        self.last_updated_frame[id] = frame_idx
        self.locked[id] = False

    def clean(self, frame_idx):
        if self.max_track_length is not None:
            self.objects = {
                oid: track[-self.max_track_length :]
                for oid, track in self.objects.items()
            }
        for oid in self.objects:
            if self.last_updated_frame[oid] + self.lock_after < frame_idx:
                self.locked[oid] = True

    def track(self, contours, frame_idx):
        centroids = [np.array([*c.xy]) for c in contours]

        self.update(centroids, frame_idx)


@dataclasses.dataclass(frozen=True)
class Contour:
    contour: np.array

    @property
    def moments(self):
        return cv.moments(self.contour, binaryImage=True)

    @property
    def area(self):
        return self.moments["m00"]

    @property
    def xy(self):
        a = self.area
        return self.moments["m10"] / a, self.moments["m01"] / a

    @property
    def xy_ints(self):
        x, y = self.xy
        return int(x), int(y)

    @property
    def perimeter(self):
        return cv.arcLength(self.contour, closed=True)

    @property
    def bounding_rectangle(self):
        return cv.minAreaRect(self.contour)


def draw_bounding_rectangles(
    frame: Frame,
    contours: List[Contour],
    mark_centroid: bool = False,
    display_area: bool = False,
    display_perimeter: bool = False,
):
    boxes = [cv.boxPoints(c.bounding_rectangle).astype(np.int0) for c in contours]
    cv.drawContours(frame, boxes, -1, color=GREEN, thickness=2, lineType=cv.LINE_AA)

    for c in contours:
        x, y = c.xy_ints
        if mark_centroid:
            cv.rectangle(
                frame,
                (x - 1, y - 1),
                (x + 1, y + 1),
                color=BLUE,
                thickness=-1,
                lineType=cv.LINE_AA,
            )
        if display_area:
            cv.putText(
                frame,
                f"{int(c.area)}",
                (x + 15, y - 15),
                fontFace=cv.FONT_HERSHEY_DUPLEX,
                fontScale=1,
                color=BLUE,
                thickness=1,
                lineType=cv.LINE_AA,
            )
        if display_perimeter:
            cv.putText(
                frame,
                f"{int(c.perimeter)}",
                (x - 15, y - 15),
                fontFace=cv.FONT_HERSHEY_DUPLEX,
                fontScale=1,
                color=GREEN,
                thickness=1,
                lineType=cv.LINE_AA,
            )

    return frame


def draw_live_object_tracks(frame: Frame, object_tracker: ObjectTracker):
    object_id_to_tracks = {
        oid: np.vstack(curve).astype(np.int0)
        for oid, curve in object_tracker.objects.items()
        if not object_tracker.locked[oid]
    }
    cv.polylines(
        frame,
        list(object_id_to_tracks.values()),
        isClosed=False,
        color=RED,
        thickness=2,
        lineType=cv.LINE_AA,
    )

    for oid, curve in object_id_to_tracks.items():
        cv.putText(
            frame,
            f"{oid}",
            (curve[-1, 0] + 15, curve[-1, 1] + 15),
            fontFace=cv.FONT_HERSHEY_DUPLEX,
            fontScale=1,
            color=YELLOW,
            thickness=1,
            lineType=cv.LINE_AA,
        )

    return frame
