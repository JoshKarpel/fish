import numpy as np


def sorted_ravel(frame_idx, v, h, chunk, prev_chunks):
    return np.sort(chunk, axis=None)


def sorted_diff(frame_idx, v, h, chunk, prev_chunks):
    if frame_idx != 0:
        return np.sort(
            np.abs(
                chunk.astype(np.int32)
                - prev_chunks[frame_idx - 1, v, h].astype(np.int32)
            ),
            axis=None,
        ).astype(np.uint8)
    else:
        return np.zeros(chunk.size, dtype=np.uint8)
