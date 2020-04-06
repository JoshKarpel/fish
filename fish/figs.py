import logging

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from . import colors


def show_frame(frame):
    plt.close()

    fig = plt.figure(figsize=(6, 6), dpi=300)
    ax = fig.add_subplot(111)

    kwargs = {}
    if len(frame.shape) == 2:
        kwargs.update(dict(cmap="gray", vmin=0, vmax=255))

    ax.imshow(frame, **kwargs)

    ax.axis("off")

    fig.tight_layout()

    fig.show()


def save_frame(path, frame):
    cv.imwrite(str(path), frame)
    logger.debug(f"Wrote image to {str(path)}")


def draw_text(
    frame, position, text, font=cv.FONT_HERSHEY_SIMPLEX, size=1, color=colors.WHITE
):
    return cv.putText(
        frame, str(text), tuple(int(p) for p in position), font, size, color=color
    )


def draw_arrow(frame, start, end, color=colors.WHITE, thickness=1):
    return cv.arrowedLine(
        frame,
        tuple(int(p) for p in start),
        tuple(int(p) for p in end),
        color,
        thickness,
    )


def draw_circle(frame, center, radius, color=colors.WHITE, thickness=1):
    return cv.circle(
        frame, tuple(int(p) for p in center), int(radius), color, thickness=thickness
    )


def draw_ellipse(frame, center, axes, rotation, color=colors.WHITE, thickness=1):
    return cv.ellipse(
        frame,
        tuple(int(p) for p in center),
        tuple(int(p) for p in axes),
        angle=rotation,
        startAngle=0,
        endAngle=360,
        color=color,
        thickness=thickness,
    )


def draw_rectangle(frame, corner, opposite, color=colors.WHITE, thickness=1):
    return cv.rectangle(frame, corner, opposite, color=color, thickness=thickness)
