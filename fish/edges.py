from typing import Optional, List, Iterable, Tuple, Dict
import logging

import dataclasses
import itertools
import collections

import numpy as np
import cv2 as cv

from . import utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

Frame = np.ndarray

OBJECT_COUNTER = itertools.count()


def object_counter():
    return next(OBJECT_COUNTER)


@dataclasses.dataclass(frozen=True)
class Object:
    contour: np.array
    id: int = dataclasses.field(default_factory=object_counter)

    def __hash__(self):
        return id(self)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.id})"

    @property
    def moments(self) -> Dict[str, float]:
        return cv.moments(self.contour, binaryImage=True)

    @property
    def area(self) -> float:
        return self.moments["m00"]

    @property
    def centroid(self) -> np.array:
        a = self.area
        return np.array([self.moments["m10"] / a, self.moments["m01"] / a])

    @property
    def centroid_ints(self) -> np.array:
        return self.centroid.astype(np.int0)

    @property
    def perimeter(self) -> float:
        return cv.arcLength(self.contour, closed=True)

    @property
    def bounding_rectangle(self):
        return cv.minAreaRect(self.contour)


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


def detect_objects(edges: np.ndarray, area_cutoff: float):
    res = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    try:
        contours, hierarchy = res
    except ValueError:  # compat hack for opencv 3 (below) vs 4 (above)
        _, contours, hierarchy = res

    objects = map(Object, contours)
    objects = [c for c in objects if c.area > area_cutoff]

    return objects


def distance_between(a: np.ndarray, b: np.ndarray) -> float:
    return np.linalg.norm(a - b)


BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)


@dataclasses.dataclass
class ObjectTrack:
    id: int
    frame_idxs: List[int] = dataclasses.field(default_factory=list)
    positions: List[np.array] = dataclasses.field(default_factory=list)
    objects: List[Object] = dataclasses.field(default_factory=list)
    _is_locked: bool = False

    def __repr__(self):
        return f"{self.__class__.__name__}({self.id}, objects = {[o.id for o in self.objects]})"

    def __hash__(self):
        return hash((self.__class__, self.id))

    @property
    def is_alive(self):
        return not self._is_locked

    @property
    def is_locked(self):
        return self._is_locked

    def lock(self):
        self._is_locked = True

    def update(self, frame_idx: int, contour):
        self.frame_idxs.append(frame_idx)
        self.positions.append(contour.centroid)
        self.objects.append(contour)

    @property
    def last_updated_frame(self) -> int:
        return self.frame_idxs[-1]

    def predict(self):
        return (2 * self.positions[-1]) - self.positions[-2]


#
# PATH_COUNTER = itertools.count()
#
#
# def path_counter():
#     return next(PATH_COUNTER)
#
#
# @dataclasses.dataclass
# class ObjectPath:
#     id: int = dataclasses.field(default_factory=path_counter)
#     tracks: List[ObjectTrack] = dataclasses.field(default_factory=list)
#
#     def __repr__(self):
#         return f"{self.__class__.__name__}(tracks = {[t.id for t in self.tracks]})"
#


class ObjectTracker:
    def __init__(self, snap_to: int = 20, lock_after: int = 5):
        self.snap_to = snap_to
        self.lock_after = lock_after

        self.tracks = {}
        self.paths = {}
        self._id_counter = itertools.count()

    def _next_id(self):
        return next(self._id_counter)

    def live_tracks(self):
        return {oid: track for oid, track in self.tracks.items() if track.is_alive}

    def update_tracks(self, contours, frame_idx):
        print("frame_idx", frame_idx)
        # if we have no tracks yet, just make everything a track
        if len(self.tracks) == 0:
            for contour in contours:
                self.register(frame_idx, contour)
            return

        print(contours)

        assigned_contours = set()
        for oid, track in self.live_tracks().items():
            if len(track.positions) > 2:
                predicted = track.predict()
            else:
                predicted = track.positions[-1]

            closest = min(
                contours, key=lambda c: distance_between(c.centroid, predicted),
            )

            if distance_between(closest.centroid, predicted) > self.snap_to:
                continue

            print(f"assigning {closest} to {track}")

            track.update(frame_idx, closest)
            assigned_contours.add(closest)

        # register leftover centroids as new objects
        for contour in set(contours) - assigned_contours:
            print(f"registering new track for unassigned contour {contour}")
            self.register(frame_idx, contour)

    def register(self, frame_idx: int, contour):
        id = self._next_id()
        track = ObjectTrack(id)
        track.update(frame_idx, contour)
        self.tracks[id] = track

    def check_for_locks(self, frame_idx: int):
        by_position = collections.defaultdict(list)
        for oid, track in self.tracks.items():
            if track.last_updated_frame + self.lock_after < frame_idx:
                track.lock()

            by_position[tuple(track.positions[-1])].append(track)

        for tracks in by_position.values():
            if len(tracks) > 1:
                for track in tracks:
                    track.lock()


def total_dist(points):
    return sum(distance_between(a, b) for a, b in utils.window(points, 2))


def draw_bounding_rectangles(
    frame: Frame,
    tracker,
    mark_centroid: bool = False,
    display_area: bool = False,
    display_perimeter: bool = False,
    mark_slow: bool = False,
):
    contours = [
        track.objects[-1] for track in tracker.tracks.values() if track.is_alive
    ]
    boxes = [cv.boxPoints(c.bounding_rectangle).astype(np.int0) for c in contours]

    if mark_slow:
        recent_track = [
            [c.centroid for c in track.objects[-11:-1]]
            for track in tracker.tracks.values()
            if track.is_alive
        ]
        moving = [len(rt) > 1 and total_dist(rt) > 3 * len(rt) for rt in recent_track]

        for box, is_moving in zip(boxes, moving):
            cv.drawContours(
                frame,
                [box],
                -1,
                color=GREEN if is_moving else RED,
                thickness=1,
                lineType=cv.LINE_AA,
            )
    else:
        cv.drawContours(frame, boxes, -1, color=GREEN, thickness=1, lineType=cv.LINE_AA)

    for c in contours:
        x, y = c.centroid_ints
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


def draw_object_tracks(
    frame: Frame,
    object_tracker: ObjectTracker,
    track_length=100,
    display_id=True,
    live_only=True,
):
    object_id_to_track = {
        oid: np.vstack(track.positions).astype(np.int0)[-track_length:]
        for oid, track in object_tracker.tracks.items()
        if track.is_alive or not live_only
    }

    cv.polylines(
        frame,
        list(object_id_to_track.values()),
        isClosed=False,
        color=YELLOW,
        thickness=1,
        lineType=cv.LINE_AA,
    )

    if display_id:
        for oid, curve in object_id_to_track.items():
            cv.putText(
                frame,
                f"{oid}",
                (curve[-1, 0] + 15, curve[-1, 1] + 15),
                fontFace=cv.FONT_HERSHEY_DUPLEX,
                fontScale=0.5,
                color=YELLOW,
                thickness=1,
                lineType=cv.LINE_AA,
            )

    return frame
