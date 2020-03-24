import logging

from pathlib import Path
import itertools
import csv
import getpass
import shutil
import json
import dataclasses
import collections

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

import fish

HERE = Path(__file__).absolute().parent
DATA = HERE.parent / "data"
OUT = HERE / "out" / Path(__file__).stem


def read_detections(path):
    for line in path.open(mode="r"):
        yield json.loads(line)


def group_by_key(dicts, key):
    groups = collections.defaultdict(list)
    for d in dicts:
        groups[d[key]].append(d)
    return groups


def make_object_count_by_frame_comparison_plot(detections, expected, out):
    fig = plt.figure(figsize=(12, 8), dpi=600)

    ax = fig.add_subplot(111)

    counts = np.array([det["object_counts"] for det in detections]).T
    print(counts.shape)

    avg = np.mean(counts, axis=1)

    ax.plot(counts, color="C0", alpha=0.01)
    ax.plot(avg, color="black")
    ax.axhline(expected, color="C1", linestyle="--")

    ax.set_xlim(0, counts.shape[0])

    ax.set_xlabel("Frame #")
    ax.set_ylabel("# of Detected Objects")

    fig.tight_layout()

    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out))
    print(f"Saved plot to {out}")


if __name__ == "__main__":
    path = DATA / "scan-area__objects.json"
    detections = list(read_detections(path))

    by_movie = group_by_key(detections, "movie")
    expected_counts = dict(
        zip(
            [f"D1-{n}" for n in range(1, 13)] + [f"C-{n}" for n in range(1, 4)],
            [34, 37, 38, 26, 21, 24, 39, 34, 22, 36, 34, 42, 52, 52, 60],
        )
    )

    for movie, group in by_movie.items():
        print(f"Making plot for {movie}")
        out = OUT / f"{movie}__count_by_frame_comparison.png"
        make_object_count_by_frame_comparison_plot(group, expected_counts[movie], out)
