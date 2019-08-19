import numpy as np


def sorted_ravel(frame_idx, v, h, chunk, prev_chunks):
    return np.sort(chunk, axis=None)


def sorted_diff(frame_idx, v, h, chunk, prev_chunks):
    if frame_idx != 0:
        return np.sort(chunk - prev_chunks[frame_idx - 1, v, h], axis=None)
    else:
        return np.zeros(chunk.size)
