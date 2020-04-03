import logging

from pathlib import Path
import itertools
import csv

import numpy as np
import scipy as sp
import cv2 as cv

import matplotlib.pyplot as plt

from tqdm import tqdm, trange

import fish

logging.basicConfig()

FLOW_CLOSE_KERNEL_SIZE = 5
FLOW_CLOSE_KERNEL = cv.getStructuringElement(
    cv.MORPH_ELLIPSE, (FLOW_CLOSE_KERNEL_SIZE, FLOW_CLOSE_KERNEL_SIZE)
)
FRAME_CLOSE_KERNEL_SIZE = 5
FRAME_CLOSE_KERNEL = cv.getStructuringElement(
    cv.MORPH_ELLIPSE, (FRAME_CLOSE_KERNEL_SIZE, FRAME_CLOSE_KERNEL_SIZE)
)

LOW_AREA_BRIGHTNESS = 100
LOW_AREA_FLOW = 100


def do_optical_flow(frames):
    bgnd = fish.background_via_min(frames)

    dish = fish.find_dish(bgnd)
    dish_mask = dish.mask_like(bgnd)

    flow = None
    for frame_idx, frame in enumerate(frames[1:], start = 1):
        frame_masked = fish.apply_mask(frame, dish_mask)
        frame_masked_no_bgnd = fish.subtract_background(frame_masked, bgnd)

        prev_frame = frames[frame_idx - 1]
        prev_frame_masked = fish.apply_mask(prev_frame, dish_mask)
        prev_frame_masked_no_bgnd = fish.subtract_background(prev_frame_masked, bgnd)

        # OBJECT CALCULATIONS

        frame_thresh, frame_thresholded = cv.threshold(
            frame_masked_no_bgnd, thresh = 0, maxval = 255, type = cv.THRESH_OTSU
        )

        frame_closed = cv.morphologyEx(
            frame_thresholded, cv.MORPH_CLOSE, FLOW_CLOSE_KERNEL
        )

        (
            brightness_num_labels,
            brightness_labels,
            brightness_stats,
            brightness_centroids,
        ) = cv.connectedComponentsWithStats(frame_closed, 8)

        brightness_blobs = []
        for label in range(1, brightness_num_labels):
            area = brightness_stats[label, cv.CC_STAT_AREA]
            centroid = brightness_centroids[label]

            if area > LOW_AREA_BRIGHTNESS:
                brightness_blobs.append(
                    BrightnessBlob(label = label, area = area, centroid = centroid, )
                )

        # FLOW CALCULATIONS

        flow = cv.calcOpticalFlowFarneback(
            prev_frame_masked_no_bgnd,
            frame_masked_no_bgnd,
            flow,
            pyr_scale = 0.5,
            levels = 3,
            winsize = 11,
            iterations = 5,
            poly_n = 5,
            poly_sigma = 1.1,
            flags = cv.OPTFLOW_FARNEBACK_GAUSSIAN
            if flow is None
            else cv.OPTFLOW_USE_INITIAL_FLOW,
        )

        flow_norm = np.linalg.norm(flow, axis = -1)
        flow_norm_image = (flow_norm * 255 / np.max(flow_norm)).astype(np.uint8)

        flow_thresh, flow_norm_thresholded = cv.threshold(
            flow_norm_image, thresh = 0, maxval = 255, type = cv.THRESH_OTSU
        )

        flow_norm_closed = cv.morphologyEx(
            flow_norm_thresholded, cv.MORPH_CLOSE, FLOW_CLOSE_KERNEL
        )

        y = np.arange(flow.shape[0])
        x = np.arange(flow.shape[1])
        flow_x = flow[..., 0]
        flow_y = flow[..., 1]

        # must flip the flow because it's normally in y-x order, but we store coordinates in x-y order
        flow_x_interp = sp.interpolate.RegularGridInterpolator((x, y), flow_x.T)
        flow_y_interp = sp.interpolate.RegularGridInterpolator((x, y), flow_y.T)

        (
            flow_num_labels,
            flow_labels,
            flow_stats,
            flow_centroids,
        ) = cv.connectedComponentsWithStats(flow_norm_closed, 8)

        velocity_blobs = []
        for label in range(1, flow_num_labels):
            area = flow_stats[label, cv.CC_STAT_AREA]
            centroid = flow_centroids[label]

            if area > LOW_AREA_FLOW:
                velocity_blobs.append(
                    VelocityBlob(
                        label = label,
                        area = area,
                        centroid = centroid,
                        centroid_velocity = np.array(
                            [flow_x_interp(centroid), flow_y_interp(centroid)]
                        ),
                    )
                )

        # DISPLAY

        img = fish.bw_to_bgr(frame)

        img_flow = (
            fish.bw_to_bgr(flow_norm_closed) * fish.fractions(*fish.YELLOW)
        ).astype(np.uint8)
        img_objects = (
            fish.bw_to_bgr(frame_closed) * fish.fractions(*fish.GREEN)
        ).astype(np.uint8)
        shadows = cv.addWeighted(img_flow, 0.5, img_objects, 0.5, 0)
        img = cv.addWeighted(img, 0.6, shadows, 0.4, 0)

        for blob in brightness_blobs:
            img = fish.draw_text(
                img, (blob.x, blob.y + 30), blob.label, color = fish.GREEN, size = 0.5
            )
            img = fish.draw_circle(
                img, (blob.x, blob.y), radius = 2, thickness = -1, color = fish.GREEN
            )

        ARROW_SCALE = 7
        ELLIPSE_SCALE = 7

        for blob in velocity_blobs:
            img = fish.draw_text(
                img, (blob.x + 20, blob.y), blob.label, color = fish.RED, size = 0.5,
            )
            img = fish.draw_circle(
                img, (blob.x, blob.y), radius = 2, thickness = -1, color = fish.RED
            )
            img = fish.draw_arrow(
                img,
                (blob.x, blob.y),
                (blob.x + blob.v_x * ARROW_SCALE, blob.y + blob.v_y * ARROW_SCALE),
                color = fish.RED,
            )
            img = fish.draw_arrow(
                img,
                (blob.x, blob.y),
                (blob.x + blob.v_t_x * ARROW_SCALE, blob.y + blob.v_t_y * ARROW_SCALE),
                color = fish.BLUE,
            )

            blow_flow = flow[flow_labels == blob.label]
            v_rel = np.dot(blow_flow, blob.v_unit)
            v_rel_mean = np.mean(v_rel)
            v_rel_std = np.std(v_rel)

            v_t_rel = np.dot(blow_flow, blob.v_t_unit)
            v_t_rel_mean = np.sum(v_t_rel[v_t_rel != 0])
            v_t_rel_std = np.std(v_t_rel[v_t_rel != 0])

            if not np.isnan(v_rel_mean):
                img = fish.draw_ellipse(
                    img,
                    (blob.x, blob.y),
                    (v_rel_std * ELLIPSE_SCALE, v_t_rel_std * ELLIPSE_SCALE),
                    rotation = np.rad2deg(np.arctan2(blob.v_y, blob.v_x)),
                    color = fish.YELLOW,
                )

        yield img


class Blob:
    def __init__(self, label, area, centroid):
        self.label = label
        self.area = area
        self.centroid = centroid

    @property
    def x(self):
        return self.centroid[0]

    @property
    def y(self):
        return self.centroid[1]


class BrightnessBlob(Blob):
    pass


class VelocityBlob(Blob):
    def __init__(self, label, area, centroid, centroid_velocity):
        super().__init__(label, area, centroid)
        self.centroid_velocity = centroid_velocity
        self.centroid_velocity_tangent = np.array([self.v_y, -self.v_x])

    @property
    def v_x(self):
        return self.centroid_velocity[0]

    @property
    def v_y(self):
        return self.centroid_velocity[1]

    @property
    def v_t_x(self):
        return self.centroid_velocity_tangent[0]

    @property
    def v_t_y(self):
        return self.centroid_velocity_tangent[1]

    @property
    def v_unit(self):
        return self.centroid_velocity / np.linalg.norm(self.centroid_velocity)

    @property
    def v_t_unit(self):
        return self.centroid_velocity_tangent / np.linalg.norm(
            self.centroid_velocity_tangent
        )


if __name__ == "__main__":
    HERE = Path(__file__).absolute().parent
    DATA = HERE.parent / "data"
    OUT = HERE / "out" / Path(__file__).stem

    movies = [f"D1-{n}" for n in range(1, 13)] + [f"C-{n}" for n in range(1, 4)]

    for movie in movies[:1]:
        input_frames = fish.cached_read((DATA / f"{movie}.hsv"))[300:400]

        frames = do_optical_flow(input_frames)

        fish.make_movie(
            OUT / f"{movie}__optical_flow.mp4",
            frames,
            num_frames = len(input_frames),
            fps = 1,
        )
