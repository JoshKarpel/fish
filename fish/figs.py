import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


def show_frame(frame):
    plt.close()

    fig = plt.Figure(figsize=(6, 6), dpi=300)
    ax = fig.add_subplot(111)

    kwargs = {}
    if len(frame.shape) == 2:
        kwargs.update(dict(cmap="gray", vmin=0, vmax=255))

    ax.imshow(frame, **kwargs)

    ax.axis("off")

    fig.tight_layout()

    return fig


def save_frame(path, frame):
    cv.imwrite(str(path), frame)
