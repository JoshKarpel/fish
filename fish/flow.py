import dataclasses
import itertools

import cv2 as cv
import numpy as np

from tqdm import tqdm

from . import colors, utils


def optical_flow(prev, next, prev_flow=None):
    return cv.calcOpticalFlowFarneback(
        prev,
        next,
        prev_flow,
        pyr_scale=0.5,
        levels=3,
        winsize=11,
        iterations=5,
        poly_n=5,
        poly_sigma=1.1,
        flags=cv.OPTFLOW_FARNEBACK_GAUSSIAN
        if prev_flow is None
        else cv.OPTFLOW_USE_INITIAL_FLOW,
    )


KERNEL_SIZE = 5
KERNEL = cv.getStructuringElement(cv.MORPH_ELLIPSE, (KERNEL_SIZE, KERNEL_SIZE))


def average_velocity_per_frame(frames):
    prev_flow = None

    for frame_idx, frame in tqdm(
        enumerate(frames[:-1]),
        desc="Calculating average velocity of objects...",
        total=len(frames),
    ):
        next_frame = frames[frame_idx + 1]

        flow = optical_flow(frame, next_frame, prev_flow=prev_flow)
        prev_flow = flow

        flow_norm = np.linalg.norm(flow, axis=-1)
        flow_norm_image = (flow_norm * 255 / np.max(flow_norm)).astype(np.uint8)

        thresh, thresholded = cv.threshold(
            flow_norm_image, thresh=0, maxval=255, type=cv.THRESH_OTSU
        )

        closed_thresholded = cv.morphologyEx(thresholded, cv.MORPH_CLOSE, KERNEL)

        num_labels, labels, stats, _ = cv.connectedComponentsWithStats(
            closed_thresholded, 8
        )

        label_areas = {}
        for label in range(num_labels):
            area = stats[label, cv.CC_STAT_AREA]
            if 100 < area < 10_000:
                label_areas[label] = area

        label_avg_velocity = {}
        for label, area in label_areas.items():
            velocity_in_label = utils.apply_mask(flow_norm, labels == label)
            label_avg_velocity[label] = np.sum(velocity_in_label) / area

        avg_velocity = np.mean(list(label_avg_velocity.values()))

        yield avg_velocity


def total_velocity_per_frame(frames):
    prev_flow = None

    for frame_idx, frame in tqdm(
        enumerate(frames[:-1]),
        desc="Calculating average velocity of objects...",
        total=len(frames),
    ):
        next_frame = frames[frame_idx + 1]

        flow = optical_flow(frame, next_frame, prev_flow=prev_flow)
        prev_flow = flow

        flow_norm = np.linalg.norm(flow, axis=-1)
        flow_norm_image = (flow_norm * 255 / np.max(flow_norm)).astype(np.uint8)

        thresh, thresholded = cv.threshold(
            flow_norm_image, thresh=0, maxval=255, type=cv.THRESH_OTSU
        )

        closed_thresholded = cv.morphologyEx(thresholded, cv.MORPH_CLOSE, KERNEL)

        num_labels, labels, stats, _ = cv.connectedComponentsWithStats(
            closed_thresholded, 8
        )

        label_areas = {}
        for label in range(num_labels):
            area = stats[label, cv.CC_STAT_AREA]
            if 100 < area < 10_000:
                label_areas[label] = area

        label_avg_velocity = {}
        for label, area in label_areas.items():
            velocity_in_label = utils.apply_mask(flow_norm, labels == label)
            label_avg_velocity[label] = np.sum(velocity_in_label) / area

        total_velocity = np.sum(list(label_avg_velocity.values()))

        yield total_velocity
