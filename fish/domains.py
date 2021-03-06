from typing import Tuple

import itertools

import numpy as np
import scipy as sp
import cv2 as cv


def domain(center, widths, points=None):
    if points is None:
        points = map(int, widths)

    vecs = [
        np.linspace(-w / 2, w / 2, num=n) + c for c, w, n in zip(center, widths, points)
    ]

    return np.meshgrid(*vecs, indexing="xy")


def domain_circle(center, r, r_points=10, theta_points=10):
    vecs = [
        np.linspace(0.5, r, r_points),
        np.linspace(0, 2 * np.pi, theta_points, endpoint=False),
    ]

    r, theta = np.meshgrid(*vecs, indexing="xy")
    x, y = r * np.cos(theta), -r * np.sin(theta)

    c_x, c_y = center
    x += c_x
    y += c_y

    return x, y


# TODO: consider returning the arrays pre-stacked; depends on exact use cases


def iter_domain_indices(mesh):
    yield from itertools.product(*[range(n) for n in mesh.shape])


def rotate_domain_xy(x, y, angle=0, degrees=False, center=None):
    if degrees:
        angle = np.deg2rad(angle)

    if center is None:
        center = (np.mean(x), np.mean(y))
    c_x, c_y = center
    # y coordinates are all negative, because they start from the top of the image, not the bottom
    x, y = x - c_x, c_y - y

    c, s = np.cos(angle), np.sin(angle)

    rot_x = c * x - s * y
    rot_y = s * x + c * y

    rot_x, rot_y = rot_x + c_x, -rot_y + c_y

    return rot_x, rot_y


def interpolate_frame(frame, **kwargs):
    """
    Return an (x, y) interpolator for the frame data in (x, y) coordinates.

    Parameters
    ----------
    frame

    Returns
    -------

    """
    y = np.arange(frame.shape[0])
    x = np.arange(frame.shape[1])

    return sp.interpolate.RegularGridInterpolator((x, y), frame.T, **kwargs)


def evaluate_interpolation(x, y, interpolator):
    points = np.reshape(np.stack((x, y), axis=-1), (-1, 2))

    return np.reshape(interpolator(points), x.shape)
