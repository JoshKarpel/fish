import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


def show_frame(frame):
    plt.close()

    fig = plt.Figure()
    ax = fig.add_subplot(111)

    kwargs = {}
    if len(frame.shape) == 2:
        kwargs.update(dict(cmap="gray", vmin=0, vmax=255))

    fig.tight_layout()

    ax.imshow(frame, **kwargs)


def save_frame(path, frame):
    cv.imwrite(str(path), frame)
