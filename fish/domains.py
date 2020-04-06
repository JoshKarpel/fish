from typing import Tuple

import itertools

import numpy as np
from scipy.spatial.transform import Rotation
import cv2 as cv


def domain(center, widths, points):
    vecs = [np.linspace(-w / 2, w / 2, num = n) + c for c, w, n in zip(center, widths, points)]
    # print(vecs)
    return np.meshgrid(*vecs, indexing = 'xy')


def iter_domain_indices(mesh):
    yield from itertools.product(*[range(n) for n in mesh.shape])


def rotate_domain_xy(x, y, angle = 0, degrees = False, center = None):
    if degrees:
        angle = np.deg2rad(angle)
    if center is None:
        center = (np.mean(x), np.mean(y))
    c_x, c_y = center

    x, y = x - c_x, y - c_y

    c, s = np.cos(angle), np.sin(angle)

    rot_x = c * x + s * y
    rot_y = -s * x + c * y

    rot_x, rot_y = rot_x + c_x, rot_y + c_y

    return rot_x, rot_y
