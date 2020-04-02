from typing import Tuple

import datetime
import time
import itertools

import cv2 as cv
import numpy as np


class BlockTimer:
    """A context manager that times the code in the ``with`` block. Print the :class:`BlockTimer` after exiting the block to see the results."""

    __slots__ = (
        "wall_time_start",
        "wall_time_end",
        "wall_time_elapsed",
        "proc_time_start",
        "proc_time_end",
        "proc_time_elapsed",
    )

    def __init__(self):
        self.wall_time_start = None
        self.wall_time_end = None
        self.wall_time_elapsed = None

        self.proc_time_start = None
        self.proc_time_end = None
        self.proc_time_elapsed = None

    def __enter__(self):
        self.wall_time_start = datetime.datetime.now()
        self.proc_time_start = time.process_time()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.proc_time_end = time.process_time()
        self.wall_time_end = datetime.datetime.now()

        self.wall_time_elapsed = self.wall_time_end - self.wall_time_start
        self.proc_time_elapsed = datetime.timedelta(
            seconds=self.proc_time_end - self.proc_time_start
        )

    def __str__(self):
        if self.wall_time_end is None:
            return f"{self.__class__.__name__} started at {self.wall_time_start} and is still running"

        return f"{self.__class__.__name__} started at {self.wall_time_start}, ended at {self.wall_time_end}. Elapsed time: {self.wall_time_elapsed}. Process time: {self.proc_time_elapsed}."


def chunk(iterable, n, fill=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fill)


def window(seq, n):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(itertools.islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def distance_between(a: np.ndarray, b: np.ndarray) -> float:
    return np.linalg.norm(a - b)


def moving_average(array, width):
    return np.convolve(array, np.ones(width), "valid") / width


def _shape_to_half_width(width):
    return (width - 1) // 2


def domain_indices(array: np.ndarray, point: Tuple[int, ...], shape: Tuple[int, ...]):
    """
    Return indices for a view of the "domain" for the point defined at the indices ``point``.

    Parameters
    ----------
    array
    point
    shape

    Returns
    -------

    """
    half_widths = map(_shape_to_half_width, shape)
    return tuple(
        slice(max(c - h, 0), min(c + h + 1, array.shape[idx]))
        for idx, (c, h) in enumerate(zip(point, half_widths))
    )


def iter_indices(array: np.ndarray):
    yield from np.ndindex(*array.shape)


def iter_domain_indices(array: np.ndarray, half_widths: Tuple[int, ...]):
    for idxs in iter_indices(array):
        yield idxs, domain_indices(array, idxs, half_widths)


def apply_mask(frame, mask):
    mask = np.where(mask != 0, 1, 0).astype(np.uint8)
    return cv.bitwise_and(frame, frame, mask=mask)
