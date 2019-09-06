import numpy as np


def sorted_ravel(window, frame_idx, coords, windows):
    return np.sort(window, axis=None)


def sorted_diff(window, frame_idx, coords, windows):
    if frame_idx != 0:
        return np.sort(
            np.abs(
                window.astype(np.int32)
                - windows[frame_idx - 1][coords].astype(np.int32)
            ),
            axis=None,
        ).astype(np.uint8)
    else:
        return None
