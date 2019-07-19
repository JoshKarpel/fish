import numpy as np

from . import utils


def html_to_rgb(html):
    """Convert #RRGGBB to an (R, G, B) tuple """
    html = html.strip().lstrip('#')
    if len(html) != 6:
        raise ValueError(f"input {html} is not in #RRGGBB format")

    r, g, b = (int(n, 16) for n in (html[:2], html[2:4], html[4:]))
    return r, g, b


def rgb_to_brg(rgb):
    r, g, b = rgb
    return b, r, g


def fractions(x, y, z):
    tot = x + y + z
    return np.array([x / tot, y / tot, z / tot])


HTML_COLORS = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02', '#a6761d', '#666666']
RGB_COLORS = [html_to_rgb(c) for c in HTML_COLORS]
BRG_COLORS = [rgb_to_brg(rgb) for rgb in RGB_COLORS]

COLOR_SCHEMES = {}
COLOR_SCHEMES.update({i: BRG_COLORS[:i] for i in range(1, len(BRG_COLORS) + 1)})
