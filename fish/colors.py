import numpy as np

from . import utils


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
    "#1b9e77",
    "#d95f02",
    "#7570b3",
    "#e7298a",
    "#66a61e",
    "#e6ab02",
    "#a6761d",
]
RGB_COLORS = [html_to_rgb(c) for c in HTML_COLORS]
BGR_COLORS = [rgb_to_bgr(rgb) for rgb in RGB_COLORS]

BGR_FRACTIONS = [fractions(*bgr) for bgr in BGR_COLORS]

GRAY = "#666666"
BGR_GRAY_FRACTIONS = fractions(*rgb_to_bgr(html_to_rgb(GRAY)))
