from pathlib import Path

import numpy as np
import cv2 as cv
from tqdm import tqdm


# opencv color order is BGR

def read(path):
    cap = cv.VideoCapture(path)

    num_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))

    # each frame is an rgb image
    # BUT R == G == B, because its grayscale
    # so we can lose the RGB dimension
    frames = np.empty((num_frames, height, width), dtype = np.uint8)
    for current_frame in tqdm(range(num_frames)):
        _, frame = cap.read()
        frames[current_frame] = frame[:, :, 0]  # grab first channel

    cap.release()

    return frames


def save(path, frames):
    with Path(path).open(mode = 'wb') as file:
        np.save(file, frames)


def load(path):
    with Path(path).open(mode = 'rb') as file:
        return np.load(file)


def display(frames, wait = 0):
    cv.namedWindow('movie')

    display_frame = None
    for idx, frame in enumerate(frames):
        # if display_frame is None:
        #     height, width = frame.shape
        #     display_frame = np.empty((height, width, 3))
        #
        # display_frame[:, :, 0] = frame

        display_frame = frame

        cv.rectangle(display_frame, (10, 2), (100, 20), (255, 255, 255), -1)
        cv.putText(display_frame, str(idx), (15, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        cv.imshow('movie', display_frame)

        key = cv.waitKey(wait)
        if key == ord('q'):
            break


def make_movie(path, frames):
    writer = None

    for frame in tqdm(frames, desc = 'Writing frames for movie'):
        if writer is None:
            height, width = frame.shape
            fourcc = cv.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
            writer = cv.VideoWriter(str(path), fourcc, 10.0, (width, height), isColor = False)

        writer.write(frame)

    writer.release()


def streaming_std(frames):
    avg_over_frames = np.mean(frames, axis = 0)

    std_squared = np.zeros_like(frames[0], dtype = np.float64)
    for frame in tqdm(frames, desc = 'Calculating standard deviation'):
        std_squared += np.abs(frame - avg_over_frames) ** 2

    return np.sqrt(std_squared / len(frames)).astype(np.uint8)


def process_frames(frames):
    avg_over_frames = np.mean(frames, axis = 0)
    std_over_frames = streaming_std(frames)

    avg_plus_std = avg_over_frames + (2 * std_over_frames)
    frame_minus_background = None
    for frame in frames:
        if frame_minus_background is None:
            frame_minus_background = np.empty_like(frame)

        frame_minus_background[:] = frame - avg_over_frames
        frame_minus_background[:] = np.where(
            frame >= avg_plus_std,
            frame_minus_background,
            0
        )

        yield frame_minus_background


if __name__ == '__main__':
    # frames = read('data/control.avi')
    # save('data/control.npy', frames)

    frames = load('data/control.npy')

    # display(frames_for_display(frames), wait = 10)
    make_movie('test.mp4', process_frames(frames))
