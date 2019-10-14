from typing import Optional, List, Iterable, Tuple, Dict
import logging

import dataclasses
import itertools

import numpy as np
import cv2 as cv

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

Frame = np.ndarray


@dataclasses.dataclass(frozen=True)
class Contour:
    contour: np.array

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


def distance_between(a: np.ndarray, b: np.ndarray) -> float:
    return np.linalg.norm(a - b)


def get_contours(edges: np.ndarray, area_cutoff: float = 10):
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


@dataclasses.dataclass
class ObjectTrack:
    frame_idxs: List[int] = dataclasses.field(default_factory=list)
    positions: List[np.array] = dataclasses.field(default_factory=list)
    areas: List[float] = dataclasses.field(default_factory=list)
    is_locked: bool = False

    def update(self, frame_idx: int, contour):
        self.frame_idxs.append(frame_idx)
        self.positions.append(contour.centroid)
        self.areas.append(contour.area)

    @property
    def last_updated_frame(self) -> int:
        return self.frame_idxs[-1]


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

        self.tracks = {}
        self._id_counter = itertools.count()

    def _next_id(self):
        return next(self._id_counter)

    def update_tracks(self, contours, frame_idx):
        # if we have no tracks yet, just make everything a track
        if len(self.tracks) == 0:
            for contour in contours:
                self.register(frame_idx, contour)
            return

        # copy the list of centroids, since we're about to mutate it
        contours = contours[:]

        # try to assign a contour to each unlocked object track
        for track in (track for track in self.tracks.values() if not track.is_locked):
            if len(contours) == 0:
                return
            last_position = track.positions[-1]
            closest_centroid_idx, closest_centroid = min(
                enumerate(contours),
                key=lambda c: distance_between(c[1].centroid, last_position),
            )

            if (
                distance_between(closest_centroid.centroid, last_position)
                <= self.snap_to
            ):
                track.update(frame_idx, closest_centroid)
                contours.pop(closest_centroid_idx)

        # register leftover centroids as new objects
        for contour in contours:
            self.register(frame_idx, contour)

    def register(self, frame_idx: int, contour):
        id = self._next_id()
        track = ObjectTrack()
        track.update(frame_idx, contour)
        self.tracks[id] = track

    def clean(self, frame_idx: int):
        if self.max_track_length is not None:
            self.tracks = {
                oid: track[-self.max_track_length :]
                for oid, track in self.tracks.items()
            }
        for oid, track in self.tracks.items():
            if track.last_updated_frame + self.lock_after < frame_idx:
                track.is_locked = True


def draw_bounding_rectangles(
    frame: Frame,
    contours: Iterable[Contour],
    mark_centroid: bool = False,
    display_area: bool = False,
    display_perimeter: bool = False,
):
    contours = list(contours)

    boxes = [cv.boxPoints(c.bounding_rectangle).astype(np.int0) for c in contours]
    cv.drawContours(frame, boxes, -1, color=GREEN, thickness=2, lineType=cv.LINE_AA)

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


def draw_live_object_tracks(frame: Frame, object_tracker: ObjectTracker):
    object_id_to_track = {
        oid: np.vstack(track.positions).astype(np.int0)
        for oid, track in object_tracker.tracks.items()
        if not track.is_locked
    }
    cv.polylines(
        frame,
        list(object_id_to_track.values()),
        isClosed=False,
        color=RED,
        thickness=2,
        lineType=cv.LINE_AA,
    )

    for oid, curve in object_id_to_track.items():
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
