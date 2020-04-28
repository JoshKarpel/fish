from typing import List, Mapping

import abc
import pickle
from pathlib import Path
import gzip

import cv2 as cv
import numpy as np
from sklearn import decomposition as decomp

import fish

LOW_AREA_BRIGHTNESS = 100
LOW_AREA_VELOCITY = 100


def threshold_and_close(image, threshold=128, type=cv.THRESH_OTSU, kernel_size=5):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
    thresh, thresholded = cv.threshold(image, thresh=threshold, maxval=255, type=type)

    closed = cv.morphologyEx(thresholded, cv.MORPH_CLOSE, kernel)

    return closed


class Blob(metaclass=abc.ABCMeta):
    def __init__(
        self,
        *,
        movie,
        frame_idx,
        label,
        points_in_label,
        area,
        centroid,
        brightness_interpolation,
        velocity_x_interpolation,
        velocity_y_interpolation,
    ):
        self.movie = movie
        self.frame_idx = frame_idx

        self.label = label
        self.points_in_label = points_in_label
        self.area = area
        self.centroid = centroid

        domain_x, domain_y = self.domain(widths=(20, 20), points=(10, 10))

        self.domain_brightness = fish.evaluate_interpolation(
            domain_x, domain_y, brightness_interpolation
        )

        domain_flow_x = fish.evaluate_interpolation(
            domain_x, domain_y, velocity_x_interpolation
        )
        domain_flow_y = fish.evaluate_interpolation(
            domain_x, domain_y, velocity_y_interpolation
        )

        self.domain_flow_u, self.domain_flow_v = self._domain_flow_uv_from_xy(
            domain_flow_x, domain_flow_y
        )

    def movie_path(self, movies_dir: Path) -> Path:
        return movies_dir / self.movie

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

    def feature_vector(self,):
        domain_divergence, domain_curl = self._domain_flow_div_and_curl(
            self.domain_flow_u, self.domain_flow_v
        )

        feature_vector = [
            *self.domain_brightness.ravel(),
            *self.domain_flow_u.ravel(),
            *self.domain_flow_v.ravel(),
            *domain_divergence.ravel(),
            *domain_curl.ravel(),
        ]

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
    def __init__(self, *, angle, **kwargs):
        self.angle = angle
        super().__init__(**kwargs)

    @property
    def u(self):
        """Unit vector along the pointing angle."""
        return np.array([np.cos(self.angle), -np.sin(self.angle)])

    @property
    def v(self):
        """Unit vector of the orthogonal vector to the pointing angle."""
        x, y = self.u
        return np.array([-y, x])


class VelocityBlob(Blob):
    def __init__(self, *, centroid_velocity, **kwargs):
        self.centroid_velocity = centroid_velocity
        self.centroid_velocity_tangent = np.array([self.v_y, -self.v_x])
        super().__init__(**kwargs)

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


def find_brightness_blobs(
    movie,
    frame_idx,
    frame_closed,
    brightness_interpolation,
    velocity_x_interpolation,
    velocity_y_interpolation,
):
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

        if area < LOW_AREA_BRIGHTNESS:
            continue

        points_in_label = np.where(brightness_labels == label)

        pca = decomp.PCA(n_components=2)
        pca.fit(np.transpose(points_in_label))
        y, x = pca.components_[0]
        # -y because y points down the screen
        angle = np.arctan2(-y, x)

        b = BrightnessBlob(
            movie=movie,
            frame_idx=frame_idx,
            label=label,
            points_in_label=points_in_label,
            area=area,
            centroid=centroid,
            angle=angle,
            brightness_interpolation=brightness_interpolation,
            velocity_x_interpolation=velocity_x_interpolation,
            velocity_y_interpolation=velocity_y_interpolation,
        )
        brightness_blobs.append(b)

    return brightness_blobs


def find_velocity_blobs(
    movie,
    frame_idx,
    velocity_norm_closed,
    brightness_interpolation,
    velocity_x_interpolation,
    velocity_y_interpolation,
):
    (
        velocity_num_labels,
        velocity_labels,
        velocity_stats,
        velocity_centroids,
    ) = cv.connectedComponentsWithStats(velocity_norm_closed, 8)

    velocity_blobs = []
    for label in range(1, velocity_num_labels):
        area = velocity_stats[label, cv.CC_STAT_AREA]
        centroid = velocity_centroids[label]

        if area < LOW_AREA_VELOCITY:
            continue

        v = VelocityBlob(
            movie=movie,
            frame_idx=frame_idx,
            label=label,
            points_in_label=np.where(velocity_labels == label),
            area=area,
            centroid=centroid,
            centroid_velocity=np.array(
                [
                    velocity_x_interpolation(centroid),
                    velocity_y_interpolation(centroid),
                ]
            ),
            brightness_interpolation=brightness_interpolation,
            velocity_x_interpolation=velocity_x_interpolation,
            velocity_y_interpolation=velocity_y_interpolation,
        )
        velocity_blobs.append(v)

    return velocity_blobs


def save_blobs(path: Path, blobs: Mapping[int, List[Blob]]):
    path.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(path, mode="wb") as f:
        pickle.dump(blobs, f)


def load_blobs(path: Path) -> Mapping[int, List[Blob]]:
    with gzip.open(path, mode="rb") as f:
        return pickle.load(f)


def blobs_path(movie_path: Path, out_dir: Path) -> Path:
    return out_dir / f"{movie_path.stem}.blobs"
