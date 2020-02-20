from pathlib import Path
import csv
from pprint import pprint
import collections
import itertools
from copy import deepcopy
from multiprocessing import Pool

from tqdm import tqdm, trange

import numpy as np

from scipy import ndimage as ndi

import networkx as nx
from dataclasses import dataclass

import matplotlib.pyplot as plt

import skimage
import skimage.feature
import skimage.filters
import skimage.morphology
import skimage.segmentation as seg
import skimage.measure

import cv2 as cv

import fish

THIS_DIR = Path.cwd()
ROOT_DIR = THIS_DIR.parent
DATA_DIR = ROOT_DIR / "data"
OUT_DIR = THIS_DIR / "out" / Path(__file__).stem


@dataclass(frozen=True)
class Point:
    index: int
    frame: int
    x: float
    y: float
    area: float
    perimeter: float

    def distance_to(self, p):
        return np.sqrt(((self.x - p.x) ** 2) + ((self.y - p.y) ** 2))


def load_points(path):
    points = []
    with path.open(newline="") as f:
        spamreader = csv.reader(f, delimiter=",")
        for idx, (frame, x, y, area, perimeter) in enumerate(spamreader):
            points.append(
                Point(
                    idx, int(frame), float(x), float(y), float(area), float(perimeter)
                )
            )
    return points


def make_point_array(points):
    arr = []
    for point in points:
        arr.append([point.frame, point.x, point.y])
    return np.array(arr)


def find_mean_point(point_array):
    center = np.mean(point_array, axis=0)[1:]
    center_point = Point(-1, -1, x=center[0], y=center[1], area=-1, perimeter=-1)

    return center_point


def distances_from_center(point_array, center):
    return np.linalg.norm(point_array[:, 1:] - center, axis=-1)


def filter_points_by_distance(points, center_point, cutoff):
    return [p for p in points if p.distance_to(center_point) < cutoff]


def window(seq, n):
    it = iter(seq)
    result = tuple(itertools.islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def group_points_by_frame(points):
    groups = collections.defaultdict(list)

    for p in points:
        groups[p.frame].append(p)

    return dict(groups.items())


def make_graph_from_points(points_by_frame, max_distance):
    g = nx.DiGraph()

    for (curr_index, curr_points), (next_index, next_points) in window(
        tqdm(list(sorted(points_by_frame.items(), key=lambda x: x[0]))), 2
    ):
        edges = (
            (a, b, a.distance_to(b))
            for a, b in itertools.product(curr_points, next_points)
        )
        edges = ((a, b, d) for a, b, d in edges if d < max_distance)
        g.add_weighted_edges_from(edges)

    return g


BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)

LOOK = 50

LINE_OPTS = dict(isClosed=False, thickness=1, lineType=cv.LINE_AA)


def original_with_paths(frames, paths):
    paths = [p.coordinates.astype(np.int0) for p in paths]

    for frame_index, frame in enumerate(frames):
        frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)

        frame = cv.polylines(
            frame,
            [p[max(frame_index - LOOK, 0) : frame_index] for p in paths],
            color=RED,
            **LINE_OPTS,
        )
        frame = cv.polylines(
            frame,
            [p[frame_index : frame_index + LOOK] for p in paths],
            color=GREEN,
            **LINE_OPTS,
        )

        for idx, p in enumerate(paths):
            x, y = p[frame_index]
            frame = cv.putText(
                frame,
                str(idx),
                (x + 15, y),
                fontFace=cv.FONT_HERSHEY_DUPLEX,
                fontScale=1,
                color=YELLOW,
                thickness=1,
                lineType=cv.LINE_AA,
            )

        yield frame


@dataclass(frozen=True)
class Path:
    points: list

    @property
    def coordinates(self):
        return np.array([[point.x, point.y] for point in self.points])

    @property
    def pairs(self):
        yield from window(self.points, 2)

    def cost(self, graph):
        return sum(graph[a][b]["weight"] for a, b in self.pairs)

    def __str__(self):
        return f'Path({"->".join(str(p.index) for p in points)})'


def find_shortest_paths(g, start):
    try:
        return start, nx.shortest_path(g, source=start, weight="weight")
    except Exception as e:
        print(e)
        return None


def shortest_paths_between(g, starts, ends):
    shortest_paths = {}

    results = []
    for start in tqdm(starts):
        rv = find_shortest_paths(g, start)
        if rv is None:
            continue
        results.append(rv)

    ends_set = set(ends)
    for start, paths in results:
        for end, path in paths.items():
            if end not in ends_set:
                continue
            shortest_paths[start, end] = Path(path)

    return shortest_paths


def find_paths_basic(graph, starts, ends):
    possible_paths = shortest_paths_between(graph, starts, ends)

    paths = {}
    while len(possible_paths) > 0:
        (start, end), path = min(
            possible_paths.items(), key=lambda kv: kv[1].cost(graph)
        )
        paths[start, end] = path
        possible_paths = {
            (s, e): p
            for (s, e), p in possible_paths.items()
            if s is not start and e is not end
        }

    return paths


def find_paths_increasing_weights(graph, starts, ends):
    graph = deepcopy(graph)
    starts = starts.copy()
    ends = ends.copy()

    possible_paths = shortest_paths_between(graph, starts, ends)

    paths = {}
    while len(possible_paths) > 0:
        (start, end), path = min(
            possible_paths.items(), key=lambda kv: kv[1].cost(graph)
        )
        paths[start, end] = path
        starts.remove(start)
        ends.remove(end)

        for a, b in path.pairs:
            graph[a][b]["weight"] += 10

        possible_paths = shortest_paths_between(graph, starts, ends)

    return paths


def make_movie(out, frames, paths):
    f = frames.copy()
    fish.make_movie(
        out, frames=original_with_paths(f, paths.values()), num_frames=len(f), fps=5
    )


def make_span_plot(out, points_by_frame, paths):
    num_points = []
    used_points = []
    for frame_index, points in points_by_frame.items():
        num_points.append(len(points))
        used_points.append(
            len(set(path.points[frame_index] for path in paths.values()))
        )

    num_points = np.array(num_points, dtype=np.float64)
    used_points = np.array(used_points, dtype=np.float64)

    span_percent = 100 * used_points / num_points

    fig, ax_raw = plt.subplots()

    ax_raw.plot(num_points, color="blue", linestyle="--", label="# total points")
    ax_raw.plot(used_points, color="blue", label="# used points")

    ax_raw.set_xlim(0, len(num_points))
    ax_raw.set_ylim(0, np.max(num_points))

    ax_raw.set_xlabel("frame #")
    ax_raw.set_ylabel("#", color="blue")
    ax_raw.tick_params(axis="y", labelcolor="blue")

    ax_frac = ax_raw.twinx()
    ax_frac.plot(span_percent, color="red", label="% used points")

    ax_frac.set_ylim(0, 100)

    ax_frac.set_ylabel("%", color="red")
    ax_frac.tick_params(axis="y", labelcolor="red")

    ax_frac.grid()

    fig.legend(loc="upper left")

    fig.tight_layout()
    plt.savefig(str(out))


if __name__ == "__main__":
    prefix = "plus_10"

    points = load_points(
        THIS_DIR
        / "out"
        / "edges"
        / "D1-1__lower=25_upper=200_smoothing=3__centroids.csv"
    )

    point_array = make_point_array(points)

    mean_point = find_mean_point(point_array)

    points = filter_points_by_distance(points, center_point=mean_point, cutoff=430)

    points_by_frame = group_points_by_frame(points)
    g = make_graph_from_points(points_by_frame, max_distance=50)

    last_frame_index = max(points_by_frame.keys())
    starts = points_by_frame[0]
    ends = points_by_frame[last_frame_index]

    paths = find_paths_increasing_weights(g, starts, ends)

    make_span_plot(OUT_DIR / f"{prefix}__span.png", points_by_frame, paths)

    frames = fish.read(DATA_DIR / "D1-1.hsv")[100:]
    make_movie(OUT_DIR / f"{prefix}__test.mp4", frames, paths)
