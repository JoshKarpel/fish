import logging

from pathlib import Path

from tqdm import tqdm

import numpy as np
import cv2 as cv

import matplotlib.pyplot as plt

import fish

logging.basicConfig()

FLOW_CLOSE_KERNEL_SIZE = 5
FLOW_CLOSE_KERNEL = cv.getStructuringElement(
    cv.MORPH_ELLIPSE, (FLOW_CLOSE_KERNEL_SIZE, FLOW_CLOSE_KERNEL_SIZE)
)
FRAME_CLOSE_KERNEL_SIZE = 5
FRAME_CLOSE_KERNEL = cv.getStructuringElement(
    cv.MORPH_ELLIPSE, (FRAME_CLOSE_KERNEL_SIZE, FRAME_CLOSE_KERNEL_SIZE)
)


def find_blobs(movie_path, out_dir):
    out_dir.mkdir(parents = True, exist_ok = True)

    frames = fish.cached_read(movie_path)
    movie_name = movie_path.stem

    bgnd = fish.background_via_min(frames)

    dish = fish.find_dish(bgnd)
    dish_mask = dish.mask_like(bgnd)

    start_frame = fish.find_last_pipette_frame(frames, background = bgnd, dish = dish)

    bgnd = fish.background_via_min(frames[start_frame:])

    velocity = None
    for frame_idx, frame in enumerate(tqdm(frames, desc = "Finding blobs")):
        blob_path = out_dir / f'{movie_name}__frame={frame_idx}.blobs'

        if frame_idx < start_frame:
            fish.save_blobs(blob_path, None)
            continue

        frame_masked = fish.apply_mask(frame, dish_mask)
        frame_masked_no_bgnd = fish.subtract_background(frame_masked, bgnd)

        brightness_interp = fish.interpolate_frame(frame_masked_no_bgnd)

        prev_frame = frames[frame_idx - 1]
        prev_frame_masked = fish.apply_mask(prev_frame, dish_mask)
        prev_frame_masked_no_bgnd = fish.subtract_background(prev_frame_masked, bgnd)

        # OBJECT CALCULATIONS

        frame_thresh, frame_thresholded = cv.threshold(
            frame_masked_no_bgnd, thresh = 30, maxval = 255, type = cv.THRESH_BINARY
        )

        frame_closed = cv.morphologyEx(
            frame_thresholded, cv.MORPH_CLOSE, FLOW_CLOSE_KERNEL
        )

        velocity = fish.optical_flow(
            prev_frame_masked_no_bgnd, frame_masked_no_bgnd, velocity,
        )

        flow_norm = np.linalg.norm(velocity, axis = -1)
        flow_norm_image = (flow_norm * 255 / np.max(flow_norm)).astype(np.uint8)

        flow_thresh, flow_norm_thresholded = cv.threshold(
            flow_norm_image, thresh = 0, maxval = 255, type = cv.THRESH_OTSU
        )

        flow_norm_closed = cv.morphologyEx(
            flow_norm_thresholded, cv.MORPH_CLOSE, FLOW_CLOSE_KERNEL
        )

        vel_x = velocity[..., 0]
        vel_y = velocity[..., 1]
        vel_x_interp = fish.interpolate_frame(vel_x)
        vel_y_interp = fish.interpolate_frame(vel_y)

        # BLOBS

        brightness_labels, brightness_blobs = fish.find_brightness_blobs(
            movie_name,
            frame_idx,
            frame_closed,
            brightness_interp,
            vel_x_interp,
            vel_y_interp,
        )
        flow_labels, velocity_blobs = fish.find_velocity_blobs(
            movie_name,
            frame_idx,
            flow_norm_closed,
            brightness_interp,
            vel_x_interp,
            vel_y_interp,
        )

        blobs = brightness_blobs + velocity_blobs

        fish.save_blobs(blob_path, blobs)


if __name__ == "__main__":
    HERE = Path(__file__).absolute().parent
    DATA = HERE.parent / "data"
    OUT = HERE / "out" / Path(__file__).stem

    # movies = [p for p in DATA.iterdir() if p.suffix == ".hsv"]
    movies = [DATA / "D1-1.hsv"]

    for path in movies:
        find_blobs(path, out_dir = OUT)
