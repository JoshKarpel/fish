import logging

import itertools

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from . import colors


def show_image(image):
    plt.close()

    fig = plt.figure(figsize=(6, 6), dpi=300)
    ax = fig.add_subplot(111)

    kwargs = {}
    if len(image.shape) == 2:
        kwargs.update(dict(cmap="gray", vmin=0, vmax=255))
    elif len(image.shape) == 3:
        image = colors.convert_colorspace(image, cv.COLOR_BGR2RGB)

    ax.imshow(image, **kwargs)

    ax.axis("off")

    fig.tight_layout()

    return


def save_figure(figure, path):
    figure.savefig(path)

    return path


def write_image(image, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv.imwrite(str(path), image)


def color_labels(image, label_range, labels):
    show_markers = np.zeros_like(image)

    for idx, (label, color) in enumerate(
        zip(label_range, itertools.cycle(colors.BGR_COLORS_8))
    ):
        show_markers[label == labels] = color

    return show_markers


def overlay_image(image_1, image_2, alpha=0.5):
    return cv.addWeighted(image_1, alpha, image_2, 1 - alpha, 0)


def draw_text(
    frame,
    position,
    text,
    font=cv.FONT_HERSHEY_SIMPLEX,
    size=1,
    thickness=1,
    color=colors.WHITE,
):
    return cv.putText(
        frame,
        str(text),
        tuple(int(p) for p in position),
        font,
        size,
        color=color,
        thickness=thickness,
    )


def draw_line(frame, start, end, color=colors.WHITE, thickness=1):
    return cv.line(
        frame,
        tuple(int(p) for p in start),
        tuple(int(p) for p in end),
        color,
        thickness,
        lineType=cv.LINE_AA,
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
    return cv.rectangle(
        frame,
        tuple(int(p) for p in corner),
        tuple(int(p) for p in opposite),
        color=color,
        thickness=thickness,
    )
