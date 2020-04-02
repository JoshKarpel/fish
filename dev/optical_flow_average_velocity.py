import logging

from pathlib import Path
import itertools
import csv

import numpy as np
import scipy as sp
import scipy.signal as sig
import cv2 as cv

import matplotlib.pyplot as plt

from tqdm import tqdm, trange

import fish

logging.basicConfig()


def do_it(movie, path):
    frames = fish.cached_read((DATA / f"{movie}.hsv"))
    flows = np.array(list(fish.average_velocity_per_frame(frames)))
    print(flows)

    fig = plt.Figure(figsize=(12, 8), dpi=600)
    ax = fig.add_subplot(111)

    ax.plot(flows)

    ax.set_xlabel("frame #")
    ax.set_ylabel("avg velocity")

    fig.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(str(path), bbox_inches="tight")


if __name__ == "__main__":
    HERE = Path(__file__).absolute().parent
    DATA = HERE.parent / "data"
    OUT = HERE / "out" / Path(__file__).stem

    movies = [f"D1-{n}" for n in range(1, 13)] + [f"C-{n}" for n in range(1, 4)]

    for movie in movies:
        do_it(movie, OUT / f"{movie}__avg_velocity.png")
