import cv2 as cv
import numpy as np

from tqdm import tqdm, trange


def train_background_subtractor(frames, iterations=5, seed=1):
    rnd = np.random.RandomState(seed)

    backsub = cv.createBackgroundSubtractorMOG2(
        detectShadows=False, history=len(frames) * iterations
    )

    shuffled = frames.copy()

    for iteration in range(iterations):
        rnd.shuffle(shuffled)

        for frame in tqdm(
            shuffled, desc=f"Training background model (iteration {iteration + 1})"
        ):
            backsub.apply(frame)

    return backsub


def apply_background_subtraction(background_model, frame):
    return background_model.apply(frame, learningRate=0)


def background_via_min(frames):
    bgnd = np.ones_like(frames[0]) * 255

    for frame in tqdm(frames, desc="Calculating background"):
        bgnd = np.minimum(bgnd, cv.blur(frame, ksize=(5, 5)))

    return bgnd


def subtract_background(frame, background):
    return np.where(frame > background, frame - background, 0)
