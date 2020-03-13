from pathlib import Path
import csv
from pprint import pprint
import collections
import itertools
from copy import deepcopy
from multiprocessing import Pool
from dataclasses import dataclass
import os
import pickle
from typing import List, Mapping

from tqdm import tqdm, trange

import numpy as np

import networkx as nx

import matplotlib.pyplot as plt

import cv2 as cv

import fish


@dataclass(frozen=True)
class Object:
    index: int
    frame: int
    x: float
    y: float
    area: float
    perimeter: float

    def distance_to(self, p):
        return np.sqrt(((self.x - p.x) ** 2) + ((self.y - p.y) ** 2))


def load_objects(path):
    points = []
    with path.open(newline="") as f:
        spamreader = csv.reader(f, delimiter=",")
        for idx, (frame, x, y, area, perimeter) in enumerate(spamreader):
            points.append(
                Object(
                    idx, int(frame), float(x), float(y), float(area), float(perimeter)
                )
            )
    return points


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
        tqdm(
            list(sorted(points_by_frame.items(), key=lambda x: x[0])),
            desc="Making graph from objects",
        ),
        2,
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

LOOK = 20

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
class ObjectPath:
    points: list

    def __len__(self):
        return len(self.points)

    @property
    def coordinates(self):
        return np.array([[point.x, point.y] for point in self.points])

    @property
    def pairs(self):
        yield from window(self.points, 2)

    def cost(self, graph):
        return sum(graph[a][b]["weight"] for a, b in self.pairs)

    def __str__(self):
        return f'Path({"->".join(str(p.index) for p in self.points)})'

    @property
    def velocity(self):
        return np.diff(self.coordinates, axis=0)

    @property
    def speed(self):
        return np.linalg.norm(self.velocity, axis=1)


def find_shortest_paths(g, start):
    try:
        return start, nx.shortest_path(g, source=start, weight="weight")
    except Exception as e:
        print(e)
        return None


def shortest_paths_between(g, starts, ends):
    shortest_paths = {}
    ends_set = set(ends)

    for start in tqdm(starts, desc="Finding shortest paths"):
        rv = find_shortest_paths(g, start)

        start, paths = rv
        for end, path in paths.items():
            if end not in ends_set:
                continue
            shortest_paths[start, end] = ObjectPath(path)

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


def find_paths_increasing_weights_by_fixed_quantity(graph, starts, ends, quantity=10):
    graph = deepcopy(graph)
    starts = starts.copy()
    ends = ends.copy()

    paths = {}

    # TODO: why does the len of possible paths go to zero before we run out of starts and ends?
    while True:
        possible_paths = shortest_paths_between(graph, starts, ends)

        if len(possible_paths) == 0:
            break

        (start, end), path = min(
            possible_paths.items(), key=lambda kv: kv[1].cost(graph)
        )

        paths[start, end] = path
        starts.remove(start)
        ends.remove(end)

        for a, b in path.pairs:
            graph[a][b]["weight"] += quantity

    return paths


def make_movie(out, frames, paths):
    f = frames.copy()
    fish.make_movie(
        out, frames=original_with_paths(f, paths.values()), num_frames=len(f), fps=10
    )


def make_span_plot(out, points_by_frame, paths, reported=None):
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

    fig = plt.figure(figsize=(12, 8), dpi=600)
    ax_raw = fig.add_subplot(111)

    ax_raw.plot(num_points, color="blue", linestyle="--", label="# total points")
    ax_raw.plot(used_points, color="blue", label="# used points")

    ax_raw.set_xlim(0, len(num_points))
    ax_raw.set_ylim(0, int(max(np.nanquantile(num_points, 0.95), reported or 0) * 1.1))

    ax_raw.set_xlabel("frame #")
    ax_raw.set_ylabel("#", color="blue")
    ax_raw.tick_params(axis="y", labelcolor="blue")

    ax_frac = ax_raw.twinx()
    ax_frac.plot(span_percent, color="red", label="% used points")

    ax_frac.set_ylim(0, 100)

    ax_frac.set_ylabel("%", color="red")
    ax_frac.tick_params(axis="y", labelcolor="red")

    ax_frac.grid()

    if reported is not None:
        ax_raw.axhline(y=reported, color="black", linestyle=":", linewidth=2)

    fig.legend(loc="lower left")

    fig.tight_layout()
    plt.savefig(str(out))
    print(f"saved span plot to {out}")


def save_paths(out, paths):
    with out.open(mode="wb") as f:
        pickle.dump(paths, f)

    return out


def load_paths(path):
    with path.open(mode="rb") as f:
        return pickle.load(f)


@dataclass(frozen=True)
class HandCounted:
    movie: str
    times_to_counts: Mapping[int, int]
    total: int

    @property
    def times(self):
        return np.array(list(self.times_to_counts.keys()))

    @property
    def counts(self):
        return np.array(list(self.times_to_counts.values()))


def load_hand_data(path):
    data = []
    with path.open(mode="r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(
                HandCounted(
                    movie=row.pop("Movie"),
                    total=int(row.pop("Total")),
                    times_to_counts={int(k): int(v) for k, v in row.items()},
                )
            )
    return data


def moving_average(arr, width):
    return np.convolve(arr, np.ones(width), "valid") / width


def make_single_comparison_plot(
    paths, hand_counted, out, speed_thresholds=None, fps=10
):
    if speed_thresholds is None:
        speed_thresholds = []

    fig = plt.figure(figsize=(12, 8), dpi=600)
    ax = fig.add_subplot(111)

    comparison_plot_on_axis(
        ax, paths, hand_counted, speed_thresholds=speed_thresholds, fps=fps
    )

    fig.legend()

    fig.tight_layout()
    plt.savefig(str(out))
    print(f"saved span plot to {out}")


def make_tiled_comparison_plot(paths, hand_counted, out, speed_thresholds=None, fps=10):
    fig = plt.figure(figsize=(16, 9), dpi=600)
    axs = fig.subplots(nrows=4, ncols=4)

    for idx, (ax, movie) in enumerate(zip(axs.flatten(), hand_counted.keys())):
        comparison_plot_on_axis(
            ax,
            paths[movie],
            hand_counted[movie],
            speed_thresholds,
            title=movie,
            fps=fps,
            labels=idx == 0,
        )

    axs.flatten()[-1].axis("off")
    for ax in axs.flatten():
        for item in ax.get_xticklabels() + ax.get_yticklabels():
            item.set_fontsize(8)

    fig.legend(loc="lower right")

    fig.tight_layout()
    plt.savefig(str(out))
    print(f"saved span plot to {out}")


def comparison_plot_on_axis(
    ax, paths, hand_counted, speed_thresholds, title="", fps=10, labels=True,
):
    ax.axhline(
        hand_counted.total,
        color="black",
        linestyle="--",
        linewidth=3,
        label="Total (Hand)" if labels else None,
    )
    ax.step(
        x=hand_counted.times * fps,
        y=hand_counted.counts,
        color="black",
        linewidth=3,
        label="Paralyzed (Hand)" if labels else None,
    )

    ax.axhline(
        len(paths),
        color="C0",
        linestyle="--",
        linewidth=2,
        label="Total (Paths)" if labels else None,
    )

    all_speeds = np.stack([path.speed for path in paths.values()]).flatten()
    quantiles = np.quantile(all_speeds, [0.1, 0.5, 0.9])
    print(quantiles)
    num_frames = len(next(iter(paths.values()))) - 1
    num_not_moving = {threshold: np.zeros(num_frames) for threshold in speed_thresholds}
    for (start, end), path in paths.items():
        speed = path.speed
        for threshold, arr in num_not_moving.items():
            arr += speed < threshold

    COLORS = ["C1", "C2", "C3", "C4"]
    for (threshold, arr), color in zip(num_not_moving.items(), COLORS):
        ax.plot(arr, color=color, alpha=0.1)
        ax.plot(
            moving_average(arr, 10 * fps),
            color=color,
            label=f"Speed < {threshold} px/frame" if labels else None,
        )

    ax.set_xlim(0, num_frames)

    ax.set_title(title)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Count")


if __name__ == "__main__":
    THIS_DIR = Path(__file__).absolute().parent
    ROOT_DIR = THIS_DIR.parent
    DATA_DIR = ROOT_DIR / "data"
    OUT_DIR = THIS_DIR / "out" / Path(__file__).stem

    prefix = "plus_10"

    movies = [f"D1-{n}" for n in range(1, 13)] + [f"C-{n}" for n in range(1, 4)]

    hand_by_movie = {hc.movie: hc for hc in load_hand_data(DATA_DIR / "counts.csv")}
    paths_by_movie = {
        movie: load_paths(OUT_DIR / f"{prefix}__{movie}.paths") for movie in movies
    }

    make_tiled_comparison_plot(
        paths_by_movie,
        hand_by_movie,
        OUT_DIR / f"{prefix}__tiled__comparison.png",
        speed_thresholds=[0.1, 0.5, 1, 5],
    )

    # for movie, hc in hand_by_movie.items():
    #     print(movie)
    #
    #     paths = load_paths(OUT_DIR / f"{prefix}__{movie}.paths")
    #     print(len(paths))
    #
    #     make_single_comparison_plot(
    #         paths=paths,
    #         hand_counted=hc,
    #         speed_thresholds=[1, 5, 10],
    #         out=OUT_DIR / f"{prefix}__{movie}__comparison.png",
    #     )

    # def do(movie, reported_count):
    #     points = load_objects(THIS_DIR / "out" / "paths" / f"{movie}__objects.csv")
    #
    #     points_by_frame = group_points_by_frame(points)
    #     g = make_graph_from_points(points_by_frame, max_distance=50)
    #
    #     # the largest connected component of the graph should be the main dish
    #     # because the graph is directed, we want "weak connection", which is equivalent to normal connection for undirected graphs
    #     g = max((g.subgraph(c) for c in nx.weakly_connected_components(g)), key=len)
    #
    #     last_frame_index = max(points_by_frame.keys())
    #
    #     starts = [n for n in g.nodes if n.frame == 0]
    #     ends = [n for n in g.nodes if n.frame == last_frame_index]
    #
    #     path_file = OUT_DIR / f"{prefix}__{movie}.paths"
    #     if path_file.exists():
    #         paths = load_paths(path_file)
    #     else:
    #         paths = find_paths_increasing_weights_by_fixed_quantity(
    #             g, starts, ends, quantity=10
    #         )
    #         save_paths(path_file, paths)
    #
    #     OUT_DIR.mkdir(parents=True, exist_ok=True)
    #
    #     make_span_plot(
    #         OUT_DIR / f"{prefix}__{movie}__span.png",
    #         points_by_frame,
    #         paths,
    #         reported=reported_count,
    #     )
    #
    #     frames = fish.read(DATA_DIR / f"{movie}.hsv")[100:]
    #     make_movie(OUT_DIR / f"{prefix}__{movie}__test.mp4", frames, paths)
    #
    # for movie, reported_count in zip(movies, reported_counts):
    #     do(movie, reported_count)

    # with Pool(processes=os.cpu_count() - 1) as p:
    #     p.starmap(do, zip(movies, reported_counts))
