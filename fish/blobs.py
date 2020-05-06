import abc
from pathlib import Path

import cv2 as cv
import numpy as np
from scipy import stats
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
        domain_widths=(35, 20),
        domain_points=None,
    ):
        self.movie = movie
        self.frame_idx = frame_idx

        self.label = label
        self.points_in_label = points_in_label
        self.area = area
        self.centroid = centroid

        self.domain_widths = domain_widths
        self.domain_points = domain_points

        domain_x, domain_y = self.domain()

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
    def movie_stem(self):
        return Path(self.movie).stem

    @property
    def x(self):
        return self.centroid[0]

    @property
    def y(self):
        return self.centroid[1]

    def distance_to(self, other: "Blob"):
        return np.linalg.norm(self.centroid - other.centroid)

    @property
    @abc.abstractmethod
    def u(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def v(self):
        raise NotImplementedError

    @property
    def contour(self):
        y, x = self.points_in_label

        img = np.zeros((np.max(y) + 10, np.max(x) + 10), dtype=np.uint8)
        img[self.points_in_label] = 1

        contours, hierarchy = cv.findContours(
            img, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE
        )
        return contours[0]

    @property
    def perimeter(self):
        return cv.arcLength(self.contour, closed=True)

    def domain(self,):
        return fish.rotate_domain_xy(
            *fish.domain(
                center=self.centroid,
                widths=self.domain_widths,
                points=self.domain_points,
            ),
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


def blobs_path(movie_path: Path, out_dir: Path) -> Path:
    return out_dir / f"{movie_path.stem}.blobs"


def find_blobs_with_one_unit_rough(blobs, measure):
    getter = lambda blob: getattr(blob, measure)

    all_measures = np.array([getter(blob) for blob in blobs])

    # for each measure, count up the number of blobs that have (roughly) that measure
    num_matching_measure_ratios = {}
    for blob in blobs:
        ratios = all_measures / getter(blob)
        rounded = np.rint(ratios)
        num_matching_measure_ratios[blob] = np.count_nonzero(rounded == 1)

    # assuming that most seeds are alone, the most common sum is the one where
    # the measure we divided by was roughly the measure of a single seed
    most_common_sum = stats.mode(list(num_matching_measure_ratios.values()))
    mode = most_common_sum.mode[0]

    return [blob for blob, sum in num_matching_measure_ratios.items() if sum == mode]
