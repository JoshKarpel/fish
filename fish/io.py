import gzip
import logging
import pickle
from typing import Optional, Mapping, List, Any

from pathlib import Path
import os

import numpy as np
import cv2 as cv
from tqdm import tqdm

from fish.blobs import Blob

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def read(path: Path, slice_=None):
    if slice_ is None:
        slice_ = slice(None, None, None)
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No file at {path}")

    cap = cv.VideoCapture(str(path))

    num_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))

    frame_indexes = list(range(num_frames))[slice_]

    # each frame is an rgb image
    # BUT R == G == B, because its grayscale
    # so we can lose the RGB dimension
    frames = np.empty((len(frame_indexes), height, width), dtype=np.uint8)
    for current_frame in tqdm(frame_indexes, desc=f"Reading frames from {path}"):
        _, frame = cap.read()
        frames[current_frame] = frame[:, :, 0]  # grab first channel

    cap.release()

    return frames


def save(path: os.PathLike, array):
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".working")
    with tmp_path.open(mode="wb") as file:
        np.save(file, array)

    tmp_path.rename(path)

    logger.debug(f"Saved frames to {path}")


def load(path: os.PathLike):
    with Path(path).open(mode="rb") as file:
        return np.load(file)


def cached_read(path):
    cached_path = path.with_suffix(path.suffix + ".cached")
    if cached_path.exists():
        logger.debug(f"Cached read from {cached_path}")
        return load(cached_path)

    data = read(path)
    save(cached_path, data)

    return data


def display(frames, wait: int = 0):
    cv.namedWindow("movie")

    for idx, frame in enumerate(frames):
        cv.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
        cv.putText(frame, str(idx), (15, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        cv.imshow("movie", frame)

        key = cv.waitKey(wait)
        if key == ord("q"):
            break


def make_movie(path, frames, num_frames: Optional[int] = None, fps=30):
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = path.with_suffix(".working" + path.suffix)

    writer = None
    for frame in tqdm(frames, total=num_frames, desc=f"Writing frames to {path}"):
        if writer is None:
            height, width = frame.shape[:2]
            color = len(frame.shape) == 3
            fourcc = cv.VideoWriter_fourcc(*"mp4v")  # make sure to use lower case
            writer = cv.VideoWriter(
                str(tmp_path), fourcc, float(fps), (width, height), isColor=color
            )

        writer.write(frame)

    writer.release()

    try:
        path.unlink()
    except FileNotFoundError:
        pass
    tmp_path.rename(path)

    return path


def int_to_float(array):
    return array.astype(np.float64) / 255


def float_to_int(array):
    return array.astype(np.uint8) * 8


def save_object(object: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(path, mode="wb") as f:
        pickle.dump(object, f)

    return path


def load_object(path: Path) -> Any:
    with gzip.open(path, mode="rb") as f:
        return pickle.load(f)
