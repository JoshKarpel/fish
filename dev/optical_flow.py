import logging

import abc
from pathlib import Path
import itertools
import csv

import numpy as np
import scipy as sp
import sklearn.decomposition as decomp
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


def do_optical_flow(frames, plot_out, hand_counted):
    # start_frame = fish.find_last_pipette_frame(frames)
    start_frame = 0

    bgnd = fish.background_via_min(frames[start_frame:])

    dish = fish.find_dish(bgnd)
    dish_mask = dish.mask_like(bgnd)

    flow = None
    count_moving_by_frame = []
    count_not_moving_by_frame = []
    count_total_by_frame = []
    for frame_idx, frame in enumerate(frames[start_frame:], start=start_frame):
        # print("frame_index", frame_idx)
        frame_masked = fish.apply_mask(frame, dish_mask)
        frame_masked_no_bgnd = fish.subtract_background(frame_masked, bgnd)

        brightness_interp = fish.interpolate_frame(frame_masked_no_bgnd)

        prev_frame = frames[frame_idx - 1]
        prev_frame_masked = fish.apply_mask(prev_frame, dish_mask)
        prev_frame_masked_no_bgnd = fish.subtract_background(prev_frame_masked, bgnd)

        # OBJECT CALCULATIONS

        frame_thresh, frame_thresholded = cv.threshold(
            frame_masked_no_bgnd, thresh=30, maxval=255, type=cv.THRESH_BINARY
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
                points = np.transpose((label == brightness_labels).nonzero())

                pca = decomp.PCA(n_components=2)
                pca.fit(points)

                y, x = pca.components_[0]

                # -y because y points down the screen
                angle = np.arctan2(-y, x)

                b = BrightnessBlob(
                    label=label,
                    points_in_label=brightness_labels == label,
                    area=area,
                    centroid=centroid,
                    angle=angle,
                )
                brightness_blobs.append(b)

        # FLOW CALCULATIONS

        flow = cv.calcOpticalFlowFarneback(
            prev_frame_masked_no_bgnd,
            frame_masked_no_bgnd,
            flow,
            pyr_scale=0.5,
            levels=3,
            winsize=11,
            iterations=5,
            poly_n=5,
            poly_sigma=1.1,
            flags=cv.OPTFLOW_FARNEBACK_GAUSSIAN
            if flow is None
            else cv.OPTFLOW_USE_INITIAL_FLOW,
        )

        flow_norm = np.linalg.norm(flow, axis=-1)
        flow_norm_image = (flow_norm * 255 / np.max(flow_norm)).astype(np.uint8)

        flow_thresh, flow_norm_thresholded = cv.threshold(
            flow_norm_image, thresh=0, maxval=255, type=cv.THRESH_OTSU
        )

        flow_norm_closed = cv.morphologyEx(
            flow_norm_thresholded, cv.MORPH_CLOSE, FLOW_CLOSE_KERNEL
        )

        flow_x = flow[..., 0]
        flow_y = flow[..., 1]
        flow_x_interp = fish.interpolate_frame(flow_x)
        flow_y_interp = fish.interpolate_frame(flow_y)

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
                v = VelocityBlob(
                    label=label,
                    points_in_label=flow_labels == label,
                    area=area,
                    centroid=centroid,
                    centroid_velocity=np.array(
                        [flow_x_interp(centroid), flow_y_interp(centroid)]
                    ),
                )
                velocity_blobs.append(v)

        # todo: move blob merging up here
        # the idea is build up a feature vector for each blob with the same structure,
        # where some entries might be zero because there's no velocity, or whatever

        # BRIGHTNESS FEATURES

        brightness_feature_vectors = []
        for blob in brightness_blobs:
            feature_vector = blob.feature_vector(
                brightness_interpolation=brightness_interp,
                flow_x_interpolation=flow_x_interp,
                flow_y_interpolation=flow_y_interp,
            )

            brightness_feature_vectors.append(feature_vector)

        brightness_feature_vectors = np.row_stack(brightness_feature_vectors)

        # VELOCITY FEATURES

        velocity_feature_vectors = []
        for blob in velocity_blobs:
            feature_vector = blob.feature_vector(
                brightness_interpolation=brightness_interp,
                flow_x_interpolation=flow_x_interp,
                flow_y_interpolation=flow_y_interp,
            )

            velocity_feature_vectors.append(feature_vector)

        velocity_feature_vectors = np.row_stack(velocity_feature_vectors)

        # print("velocity_feature_vectors")
        # print(velocity_feature_vectors.shape)
        # print(velocity_feature_vectors)

        # COUNTING

        # flow_dist_transform = cv.distanceTransform(flow_norm_closed, distanceType = cv.DIST_L2, maskSize = 5)
        # brightness_dist_transform = cv.distanceTransform(frame_closed)

        # if it's moving, it's probably a fish
        count_moving = len(velocity_blobs)
        # add brightness blobs that aren't moving
        # i.e., the centroid of the brightness blob is not inside a velocity blob
        count_not_moving = len(
            list(
                blob
                for blob in brightness_blobs
                if flow_norm_closed[int(blob.y), int(blob.x)] == 0
            )
        )
        count_total = count_moving + count_not_moving

        count_moving_by_frame.append(count_moving)
        count_not_moving_by_frame.append(count_not_moving)
        count_total_by_frame.append(count_total)

        # DISPLAY

        ARROW_SCALE = 7
        ELLIPSE_SCALE = 7

        img = fish.convert_colorspace(frame, cv.COLOR_GRAY2BGR)

        img_flow = (
            fish.convert_colorspace(flow_norm_closed, cv.COLOR_GRAY2BGR)
            * fish.fractions(*fish.YELLOW)
        ).astype(np.uint8)
        img_objects = (
            fish.convert_colorspace(frame_closed, cv.COLOR_GRAY2BGR)
            * fish.fractions(*fish.GREEN)
        ).astype(np.uint8)
        shadows = cv.addWeighted(img_flow, 0.5, img_objects, 0.5, 0)
        img = cv.addWeighted(img, 0.6, shadows, 0.4, 0)

        displays = [
            f"# T HC: {hand_counted.total}",
            f"# T Br: {len(brightness_blobs)}",
            f"# T Ve: {len(velocity_blobs)}",
            f"# T: {count_total}",
            f"# Tavg: {round(np.mean(count_total_by_frame), 1)}",
            f"# P: {count_not_moving}",
        ]

        img = fish.draw_rectangle(
            img, (0, 0), (270, 400), color=fish.BLACK, thickness=-1
        )
        for offset, disp in enumerate(displays):
            img = fish.draw_text(img, (30, 30 + 35 * offset), disp, color=fish.WHITE)

        for blob in brightness_blobs:
            img = fish.draw_text(
                img, (blob.x, blob.y + 30), blob.label, color=fish.GREEN, size=0.5
            )
            img = fish.draw_circle(
                img, (blob.x, blob.y), radius=2, thickness=-1, color=fish.GREEN
            )

            pointing = blob.centroid + (blob.u * 50)
            pointing_t = blob.centroid + (blob.v * 50)
            img = fish.draw_arrow(
                img, (blob.x, blob.y), (pointing[0], pointing[1]), color=fish.RED,
            )
            img = fish.draw_arrow(
                img, (blob.x, blob.y), (pointing_t[0], pointing_t[1]), color=fish.GREEN,
            )

            # domain_x, domain_y = blob.domain(widths=(20, 10), points=(3, 3))
            # for y_idx, x_idx in fish.iter_domain_indices(domain_x):
            #     x = domain_x[y_idx, x_idx]
            #     y = domain_y[y_idx, x_idx]
            #     img = fish.draw_circle(
            #         img, center=(x, y), radius=3, color=fish.RED, thickness=-1,
            #     )
            # img = fish.draw_circle(
            #     img,
            #     center=(domain_x[0, 0], domain_y[0, 0]),
            #     radius=3,
            #     color=fish.GREEN,
            #     thickness=-1,
            # )

        # for blob in velocity_blobs:
        #     img = fish.draw_text(
        #         img, (blob.x + 20, blob.y), blob.label, color=fish.RED, size=0.5,
        #     )
        #     img = fish.draw_circle(
        #         img, (blob.x, blob.y), radius=2, thickness=-1, color=fish.RED
        #     )
        #     img = fish.draw_arrow(
        #         img,
        #         (blob.x, blob.y),
        #         (blob.x + blob.v_x * ARROW_SCALE, blob.y + blob.v_y * ARROW_SCALE),
        #         color=fish.RED,
        #     )
        #     img = fish.draw_arrow(
        #         img,
        #         (blob.x, blob.y),
        #         (blob.x + blob.v_t_x * ARROW_SCALE, blob.y + blob.v_t_y * ARROW_SCALE),
        #         color=fish.BLUE,
        #     )
        #
        #     blob_flow = flow[flow_labels == blob.label]
        #     v_rel = np.dot(blob_flow, blob.u)
        #     v_rel_mean = np.mean(v_rel)
        #     v_rel_std = np.std(v_rel)
        #
        #     v_t_rel = np.dot(blob_flow, blob.v)
        #     v_t_rel_mean = np.sum(v_t_rel[v_t_rel != 0])
        #     v_t_rel_std = np.std(v_t_rel[v_t_rel != 0])
        #
        #     if not np.isnan(v_rel_mean):
        #         img = fish.draw_ellipse(
        #             img,
        #             (blob.x, blob.y),
        #             (v_rel_std * ELLIPSE_SCALE, v_t_rel_std * ELLIPSE_SCALE),
        #             rotation=np.rad2deg(blob.angle),
        #             color=fish.YELLOW,
        #         )
        #
        #     # domain_x, domain_y = blob.domain(widths = (20, 10), points = (5, 5))
        #     # for y_idx, x_idx in fish.iter_domain_indices(domain_x):
        #     #     x = domain_x[y_idx, x_idx]
        #     #     y = domain_y[y_idx, x_idx]
        #     #     img = fish.draw_circle(
        #     #         img, center = (x, y), radius = 3, color = fish.RED, thickness = -1,
        #     #     )
        #     # img = fish.draw_circle(
        #     #     img,
        #     #     center = (domain_x[0, 0], domain_y[0, 0]),
        #     #     radius = 3,
        #     #     color = fish.GREEN,
        #     #     thickness = -1,
        #     # )

        yield img

    plt.close()

    x = np.arange(start=start_frame, stop=len(frames))

    fig = plt.figure(figsize=(12, 8), dpi=600)
    ax = fig.add_subplot(111)

    ax.axvline(start_frame, color="pink", linestyle="--")

    ax.axhline(
        hand_counted.total, label="Hand-Counted Total", color="black", linestyle="--"
    )
    # * 10 to convert from time to frames at 10 fps
    ax.plot(
        (hand_counted.times * 10) + start_frame,
        hand_counted.counts,
        label="Hand-Counted Paralyzed",
        color="black",
    )

    ax.plot(x, count_not_moving_by_frame, label="Not Moving")
    ax.plot(x, count_moving_by_frame, label="Moving")
    ax.plot(x, count_total_by_frame, label="Moving + Not Moving")

    ax.legend(loc="upper left")

    ax.set_xlim(0, len(frames))

    ax.set_xlabel("Frame #")
    ax.set_ylabel("Count")
    ax.set_title(plot_out.stem)

    plt.savefig(str(plot_out))


class Blob(metaclass=abc.ABCMeta):
    def __init__(self, label, points_in_label, area, centroid):
        self.label = label
        self.points_in_label = points_in_label
        self.area = area
        self.centroid = centroid

    @property
    def x(self):
        return self.centroid[0]

    @property
    def y(self):
        return self.centroid[1]

    @property
    @abc.abstractmethod
    def u(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def v(self):
        raise NotImplementedError

    def domain(self, widths, points=None):
        return fish.rotate_domain_xy(
            *fish.domain(center=self.centroid, widths=widths, points=points),
            angle=self.angle,
        )

    def feature_vector(
        self, *, brightness_interpolation, flow_x_interpolation, flow_y_interpolation,
    ):
        feature_vector = []

        domain_x, domain_y = self.domain(widths=(20, 20), points=(10, 10))

        domain_brightness = fish.evaluate_interpolation(
            domain_x, domain_y, brightness_interpolation
        )
        feature_vector += [*domain_brightness.ravel()]

        domain_flow_x = fish.evaluate_interpolation(
            domain_x, domain_y, flow_x_interpolation
        )
        domain_flow_y = fish.evaluate_interpolation(
            domain_x, domain_y, flow_y_interpolation
        )

        domain_flow_u, domain_flow_v = self._domain_flow_uv_from_xy(
            domain_flow_x, domain_flow_y
        )

        feature_vector += [*domain_flow_u.ravel(), *domain_flow_v.ravel()]

        domain_divergence, domain_curl = self._domain_flow_div_and_curl(
            domain_flow_u, domain_flow_v
        )

        feature_vector += [*domain_divergence.ravel(), *domain_curl.ravel()]

        return np.array(feature_vector)

    def _domain_flow_uv_from_xy(self, domain_flow_x, domain_flow_y):
        domain_flow = np.stack((domain_flow_x, domain_flow_y), axis=-1)
        domain_flow_u = np.dot(domain_flow, self.u).squeeze()
        domain_flow_v = np.dot(domain_flow, self.v).squeeze()

        return domain_flow_u, domain_flow_v

    def _relative_velocity_mean_and_std(self, flow):
        blob_flow = flow[self.points_in_label]

        v_rel = np.dot(blob_flow, self.u)
        v_rel_mean = np.mean(v_rel)
        v_rel_std = np.std(v_rel)

        v_t_rel = np.dot(blob_flow, self.v)
        v_t_rel_mean = np.mean(v_t_rel)
        v_t_rel_std = np.std(v_t_rel)

        return [v_rel_mean, v_rel_std, v_t_rel_mean, v_t_rel_std]

    def _domain_flow_div_and_curl(self, domain_flow_u, domain_flow_v):
        # 1st index is "x", 2nd index is "y"
        domain_flow_grad_u_v, domain_flow_grad_u_u = np.gradient(domain_flow_u)
        domain_flow_grad_v_v, domain_flow_grad_v_u = np.gradient(domain_flow_v)
        domain_divergence = domain_flow_grad_u_u + domain_flow_grad_v_v
        domain_curl = domain_flow_grad_v_u - domain_flow_grad_u_v

        return domain_divergence, domain_curl


class BrightnessBlob(Blob):
    def __init__(self, label, points_in_label, area, centroid, angle):
        super().__init__(label, points_in_label, area, centroid)
        self.angle = angle

    @property
    def u(self):
        """Unit vector along the pointing angle."""
        # TODO: probably a minus sign on the y component
        return np.array([np.cos(self.angle), -np.sin(self.angle)])

    @property
    def v(self):
        """Unit vector of the orthogonal vector to the pointing angle."""
        x, y = self.u
        return np.array([-y, x])


class VelocityBlob(Blob):
    def __init__(self, label, points_in_label, area, centroid, centroid_velocity):
        super().__init__(label, points_in_label, area, centroid)
        self.centroid_velocity = centroid_velocity
        self.centroid_velocity_tangent = np.array([self.v_y, -self.v_x])

    @property
    def v_x(self):
        """The x component of the centroid velocity."""
        return self.centroid_velocity[0]

    @property
    def v_y(self):
        """The y component of the centroid velocity."""
        return self.centroid_velocity[1]

    @property
    def v_t_x(self):
        """The x component of the orthogonal vector to the centroid velocity."""
        return self.centroid_velocity_tangent[0]

    @property
    def v_t_y(self):
        """The y component of the orthogonal vector to the centroid velocity."""
        return self.centroid_velocity_tangent[1]

    @property
    def u(self):
        """Unit vector along the centroid velocity."""
        return self.centroid_velocity / np.linalg.norm(self.centroid_velocity)

    @property
    def v(self):
        """Unit vector orthogonal to the centroid velocity."""
        return self.centroid_velocity_tangent / np.linalg.norm(
            self.centroid_velocity_tangent
        )

    @property
    def angle(self):
        """The angle of the centroid velocity."""
        # -v_y because positive v_y points down on the screen
        return np.arctan2(-self.v_y, self.v_x)


if __name__ == "__main__":
    HERE = Path(__file__).absolute().parent
    DATA = HERE.parent / "data"
    OUT = HERE / "out" / Path(__file__).stem

    # movies = [p for p in DATA.iterdir() if p.suffix == ".hsv"]
    movies = [DATA / "D1-1.hsv"]
    hand_by_movie = {
        hc.movie: hc for hc in fish.load_hand_counted_data(DATA / "counts.csv")
    }

    for path in movies:
        movie = path.stem

        input_frames = fish.cached_read(path)[500:600]

        frames = do_optical_flow(
            input_frames,
            OUT / f"{movie}__counts.png",
            hand_counted=hand_by_movie[movie],
        )

        fish.make_movie(
            OUT / f"{movie}__optical_flow.mp4",
            frames,
            num_frames=len(input_frames),
            fps=1,
        )
