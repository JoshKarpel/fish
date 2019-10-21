import cv2 as cv
import numpy as np


from tqdm import tqdm


def _streaming_std(frames):
    avg_over_frames = np.mean(frames, axis=0)

    std_squared = np.zeros_like(frames[0], dtype=np.float64)
    for frame in tqdm(frames, desc="Calculating standard deviation"):
        std_squared += np.abs(frame - avg_over_frames) ** 2

    return np.sqrt(std_squared / len(frames)).astype(np.uint8)


def remove_background(frames, threshold=0):
    avg_over_frames = np.mean(frames, axis=0)

    pixel_threshold = avg_over_frames
    if threshold > 0:
        std_over_frames = _streaming_std(frames)
        pixel_threshold += threshold * std_over_frames

    # open_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (4, 4))
    # close_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))

    # pixel_threshold = cv.morphologyEx(pixel_threshold, cv.MORPH_OPEN, open_kernel)
    # pixel_threshold = cv.morphologyEx(pixel_threshold, cv.MORPH_CLOSE, close_kernel)

    for frame in frames:
        yield np.where(frame >= pixel_threshold, frame - avg_over_frames, 0).astype(
            np.uint8
        )


def remove_background2(frames):
    fgbg = cv.createBackgroundSubtractorMOG2()

    for frame in frames:
        blurred = cv.GaussianBlur(frame, (5, 5), 0)
        yield np.where(fgbg.apply(blurred), frame, 0)
