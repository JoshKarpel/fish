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


HTML_COLORS_8 = [
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#ffff33",
    "#a65628",
    "#f781bf",
]
HTML_COLORS_12 = [
    "#8dd3c7",
    "#ffffb3",
    "#bebada",
    "#fb8072",
    "#80b1d3",
    "#fdb462",
    "#b3de69",
    "#fccde5",
    "#d9d9d9",
    "#bc80bd",
    "#ccebc5",
    "#ffed6f",
]
RGB_COLORS_8 = [html_to_rgb(c) for c in HTML_COLORS_8]
RGB_COLORS_12 = [html_to_rgb(c) for c in HTML_COLORS_12]
BGR_COLORS_8 = [rgb_to_bgr(rgb) for rgb in RGB_COLORS_8]
BGR_COLORS_12 = [rgb_to_bgr(rgb) for rgb in RGB_COLORS_12]

BGR_FRACTIONS = [fractions(*bgr) for bgr in BGR_COLORS_8]

GRAY = "#666666"
BGR_GRAY_FRACTIONS = fractions(*rgb_to_bgr(html_to_rgb(GRAY)))

BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
CYAN = (255, 255, 0)
MAGENTA = (255, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def convert_colorspace(image, converter):
    return cv.cvtColor(image, converter)
