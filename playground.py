from pathlib import Path

import numpy as np
import cv2
from tqdm import tqdm


def read(path):
    cap = cv2.VideoCapture(path)

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # each frame is an rgb image
    # BUT R == G == B, because its grayscale
    # so we can lose the RGB dimension
    frames = np.empty((num_frames, height, width), dtype = np.dtype('uint8'))
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


if __name__ == '__main__':
    # frames = read('data/control.avi')
    # save('data/control.npy', frames)

    frames = load('data/control.npy')

    print(frames.shape)
    print(frames.nbytes / (1024 ** 3))

    # for idx, frame in enumerate(frames):
    #     print(idx)
    #
    #     R = frame[:, :, 0]
    #     G = frame[:, :, 1]
    #     B = frame[:, :, 2]
    #
    #     print(np.array_equal(R, G))
    #     print(np.array_equal(R, B))
    #     print(np.array_equal(G, B))
    #
    #     print()
