import numpy as np
import cv2 as cv


def html_to_rgb(html):
    """Convert #RRGGBB to an (R, G, B) tuple """
    html = html.strip().lstrip("#")
    if len(html) != 6:
        raise ValueError(f"input {html} is not in #RRGGBB format")

    r, g, b = (int(n, 16) for n in (html[:2], html[2:4], html[4:]))
    return r, g, b


def rgb_to_bgr(rgb):
    r, g, b = rgb
    return b, g, r


def fractions(x, y, z):
    tot = x + y + z
    return np.array([x / tot, y / tot, z / tot])


HTML_COLORS = [
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#ffff33",
    "#a65628",
    "#f781bf",
]
RGB_COLORS = [html_to_rgb(c) for c in HTML_COLORS]
BGR_COLORS = [rgb_to_bgr(rgb) for rgb in RGB_COLORS]

BGR_FRACTIONS = [fractions(*bgr) for bgr in BGR_COLORS]

GRAY = "#666666"
BGR_GRAY_FRACTIONS = fractions(*rgb_to_bgr(html_to_rgb(GRAY)))

BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)


def bw_to_bgr(frame):
    return cv.cvtColor(frame, cv.COLOR_GRAY2BGR)


def bw_to_rgb(frame):
    return cv.cvtColor(frame, cv.COLOR_GRAY2RGB)


def bgr_to_rgb(frame):
    return cv.cvtColor(frame, cv.COLOR_BGR2RGB)


def rgb_to_bgr(frame):
    return cv.cvtColor(frame, cv.COLOR_RGB2BGR)
