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

HERE = Path(__file__).absolute().parent
DATA = HERE.parent / "data"
OUT = HERE / "out" / Path(__file__).stem


class Object:
    def __init__(self, size, start, velocity):
        self.size = size
        self.start = start
        self.velocity = velocity

    def position(self, frame_index):
        return self.start + (self.velocity * frame_index)


def add_objects(frame, frame_index, objects):
    frame = frame.copy()
    for object in objects:
        x, y = object.position(frame_index)
        frame[x : x + object.size, y : y + object.size] = 255
    return frame


if __name__ == "__main__":
    base_frame = np.zeros((500, 500), dtype=np.uint8)
    length = 30

    objects = [
        Object(size=10, start=np.array([250, 200]), velocity=np.array([0, 5])),
        Object(size=10, start=np.array([200, 250]), velocity=np.array([5, 0])),
        Object(size=10, start=np.array([180, 250]), velocity=np.array([7, 0])),
        # Object(size=10, start=np.array([200, 200]), velocity=np.array([5, 5])),
        # Object(size=10, start=np.array([50, 50]), velocity=np.array([5, 0])),
        # Object(size=10, start=np.array([230, 230]), velocity=np.array([0, 5])),
    ]

    fish.make_movie(
        OUT / "fake.mp4",
        (
            add_objects(base_frame, frame_index, objects)
            for frame_index in range(length)
        ),
        fps=1,
    )
