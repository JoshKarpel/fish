from typing import Optional

from pathlib import Path
import os

import numpy as np
import cv2 as cv
from tqdm import tqdm


# opencv color order is BGR

def read(path: os.PathLike):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'No file at {path}')

    cap = cv.VideoCapture(str(path))

    num_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))

    # each frame is an rgb image
    # BUT R == G == B, because its grayscale
    # so we can lose the RGB dimension
    frames = np.empty((num_frames, height, width), dtype = np.uint8)
    for current_frame in tqdm(range(num_frames), desc = f'Reading frames from {path}'):
        _, frame = cap.read()
        frames[current_frame] = frame[:, :, 0]  # grab first channel

    cap.release()

    return frames


def save(path: os.PathLike, frames):
    with Path(path).open(mode = 'wb') as file:
        np.save(file, frames)


def load(path: os.PathLike):
    with Path(path).open(mode = 'rb') as file:
        return np.load(file)


def load_or_read(path: os.PathLike):
    path = Path(path)

    try:
        frames = load(path.with_suffix('.npy'))
    except FileNotFoundError:
        frames = read(path.with_suffix('.avi'))
        save(path.with_suffix('.npy'), frames)
    return frames


def display(frames, wait: int = 0):
    cv.namedWindow('movie')

    for idx, frame in enumerate(frames):
        cv.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
        cv.putText(frame, str(idx), (15, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        cv.imshow('movie', frame)

        key = cv.waitKey(wait)
        if key == ord('q'):
            break


def make_movie(path, frames, num_frames: Optional[int] = None, fps = 30,):
    path = Path(path)
    path.parent.mkdir(parents = True, exist_ok = True)

    tmp_path = path.with_suffix('.working' + path.suffix)

    writer = None
    for frame in tqdm(frames, total = num_frames, desc = f'Writing frames to {path}'):
        if writer is None:
            height, width = frame.shape[:2]
            fourcc = cv.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
            writer = cv.VideoWriter(str(tmp_path), fourcc, float(fps), (width, height), isColor = False)

        writer.write(frame)

    writer.release()

    tmp_path.rename(path)
