import logging

from pathlib import Path

from tqdm import tqdm

import numpy as np
import cv2 as cv

import fish

logging.basicConfig()


def find_blobs(movie_path, blobs_path):
    frames = fish.cached_read(movie_path)

    bgnd = fish.background_via_min(frames)

    dish = fish.find_dish(bgnd)
    dish_mask = dish.mask_like(bgnd)

    start_frame = fish.find_last_pipette_frame(frames, background=bgnd, dish=dish)

    bgnd = fish.background_via_min(frames[start_frame:])

    velocity = None
    blobs_by_frame = {}
    for frame_idx, frame in enumerate(tqdm(frames, desc="Finding blobs")):
        if frame_idx < start_frame:
            blobs_by_frame[frame_idx] = None
            continue

        frame_masked = fish.apply_mask(frame, dish_mask)
        frame_masked_no_bgnd = fish.subtract_background(frame_masked, bgnd)

        brightness_interp = fish.interpolate_frame(frame_masked_no_bgnd)

        prev_frame = frames[frame_idx - 1]
        prev_frame_masked = fish.apply_mask(prev_frame, dish_mask)
        prev_frame_masked_no_bgnd = fish.subtract_background(prev_frame_masked, bgnd)

        # OBJECT CALCULATIONS

        frame_closed = fish.threshold_and_close(
            frame_masked_no_bgnd, threshold=30, type=cv.THRESH_BINARY
        )

        velocity = fish.optical_flow(
            prev_frame_masked_no_bgnd, frame_masked_no_bgnd, velocity,
        )

        flow_norm = np.linalg.norm(velocity, axis=-1)
        flow_norm_image = (flow_norm * 255 / np.max(flow_norm)).astype(np.uint8)

        flow_norm_closed = fish.threshold_and_close(flow_norm_image)

        vel_x = velocity[..., 0]
        vel_y = velocity[..., 1]
        vel_x_interp = fish.interpolate_frame(vel_x)
        vel_y_interp = fish.interpolate_frame(vel_y)

        # BLOBS

        brightness_blobs = fish.find_brightness_blobs(
            movie_path.name,
            frame_idx,
            frame_closed,
            brightness_interp,
            vel_x_interp,
            vel_y_interp,
        )
        velocity_blobs = fish.find_velocity_blobs(
            movie_path.name,
            frame_idx,
            flow_norm_closed,
            brightness_interp,
            vel_x_interp,
            vel_y_interp,
        )

        blobs_by_frame[frame_idx] = brightness_blobs + velocity_blobs

    fish.save_blobs(blobs_path, blobs_by_frame)


if __name__ == "__main__":
    HERE = Path(__file__).absolute().parent
    DATA = HERE.parent / "data"
    OUT = HERE / "out" / Path(__file__).stem

    # movies = [p for p in DATA.iterdir() if p.suffix == ".hsv"]
    movies = [DATA / "D1-1.hsv"]

    for movie_path in movies:
        blobs_path = fish.blobs_path(movie_path, OUT)
        find_blobs(movie_path, blobs_path)
