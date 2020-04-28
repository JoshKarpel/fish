import logging
from typing import Optional

from pathlib import Path
import os

import numpy as np
import cv2 as cv
from tqdm import tqdm
import scipy.ndimage as ndi

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from . import bgnd, utils
from . import dish as dish_


def find_last_pipette_frame(
    frames,
    background=None,
    dish=None,
    area_cutoff_ratio=2,
    pipette_sector=(0, 90),
    extra_frames=20,
    upper_bound=200,
):
    """

    Parameters
    ----------
    frames
    background
    dish
    area_cutoff_ratio
        Look for areas-touching-the-dish that are at least this many times larger
        than the average across the entire movie.
    pipette_sector
        The angle (in degrees) to look for objects touching the dish in.
    extra_frames
        An additional number of frames to delay by.

    Returns
    -------

    """
    if background is None:
        background = bgnd.background_via_min(frames)

    if dish is None:
        dish = dish_.find_dish(background)

    areas = np.array(
        [
            _area_touching_dish(
                bgnd.subtract_background(frame, background), dish, pipette_sector
            )
            for frame in tqdm(
                frames[:upper_bound],
                desc="Calculating area of objects touching the disk",
            )
        ]
    )

    thresh = areas > area_cutoff_ratio * np.mean(areas)
    closed = ndi.binary_closing(thresh, structure=np.array([1, 1, 1]), iterations=3)

    labels, num_features = ndi.measurements.label(closed)
    slices = [
        s[0] for s in ndi.measurements.find_objects(labels)
    ]  # 1d, so we just take the first slice
    pipette_slice = max(
        slices, key=lambda s: s.stop - s.start
    )  # the pipette slice is (hopefully) the longest)

    return pipette_slice.stop + extra_frames


def _area_touching_dish(frame, dish, pipette_sector):
    mask = dish.mask_like(frame)

    masked = utils.apply_mask(frame, mask)
    thresh, thresholded = cv.threshold(
        masked, thresh=0, maxval=255, type=cv.THRESH_OTSU
    )

    with_edge = dish.draw_on(
        thresholded, start_angle=min(pipette_sector), end_angle=max(pipette_sector)
    )
    inverted = cv.bitwise_not(with_edge)

    h, w = inverted.shape
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    flood_mask[1:-1, 1:-1] = inverted

    _, flooded, b, _ = cv.floodFill(
        inverted,
        mask=flood_mask,
        seedPoint=(dish.x + dish.r, dish.y),
        newVal=127,
        flags=8,
    )

    touching_disk = np.where(flooded == 127, 1, 0).astype(np.uint8)

    return np.sum(touching_disk)
