from pathlib import Path
import subprocess
import sys

from tqdm import tqdm

import numpy as np
import cv2 as cv
import pickle

import fish

import htmap

FLOW_CLOSE_KERNEL_SIZE = 5
FLOW_CLOSE_KERNEL = cv.getStructuringElement(
    cv.MORPH_ELLIPSE, (FLOW_CLOSE_KERNEL_SIZE, FLOW_CLOSE_KERNEL_SIZE)
)
FRAME_CLOSE_KERNEL_SIZE = 5
FRAME_CLOSE_KERNEL = cv.getStructuringElement(
    cv.MORPH_ELLIPSE, (FRAME_CLOSE_KERNEL_SIZE, FRAME_CLOSE_KERNEL_SIZE)
)


def find_blobs(movie_name):
    movie_path = Path.cwd() / movie_name

    tmp = Path.cwd() / "tmp-blob-store"

    with tmp.open(mode="wb") as f:
        for item in yield_blobs(movie_path):
            pickle.dump(item, f)
        pickle.dump(None, f)

    return dict(load_tmp_blobs(tmp))


def load_tmp_blobs(tmp_blobs_path):
    with tmp_blobs_path.open(mode="rb") as f:
        item = pickle.load(f)

        if item is None:
            return

        yield item


def yield_blobs(movie_path):
    frames = fish.read(movie_path)

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

        blobs = brightness_blobs + velocity_blobs

        yield frame_idx, blobs


if __name__ == "__main__":
    docker_image, s3_url, mc_alias, s3_bucket, tag = sys.argv[1:]

    movie_names = [
        line.split()[-1]
        for line in subprocess.run(
            ["mc", "ls", f"{mc_alias}/{s3_bucket}"], capture_output=True, text=True
        ).stdout.splitlines()
    ]
    movie_paths = [[f"s3://{s3_url}/{s3_bucket}/{movie}"] for movie in movie_names]

    htmap.settings["DOCKER.IMAGE"] = docker_image

    s3_keys_root = Path.home() / ".chtc_s3"

    m = htmap.map(
        find_blobs,
        movie_names,
        tag=tag,
        map_options=htmap.MapOptions(
            request_memory="4GB",
            request_disk="2GB",
            input_files=movie_paths,
            aws_access_key_id_file=(s3_keys_root / "access.key").as_posix(),
            aws_secret_access_key_file=(s3_keys_root / "secret.key").as_posix(),
        ),
    )

    print(m, len(m))
