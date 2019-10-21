import logging

from pathlib import Path
import itertools

import numpy as np
import scipy as sp
import scipy.signal as sig
import cv2 as cv

import matplotlib.pyplot as plt

from tqdm import tqdm

import fish

logging.basicConfig()


def make_frames(
    frames,
    lower,
    upper,
    smoothing,
    draw_on_original=True,
    areas_out=None,
    velocities_out=None,
    velocity_histogram_out=None,
    do_backsub=True,
):
    open_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (4, 4))
    close_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    backsub = cv.createBackgroundSubtractorKNN()
    tracker = fish.ObjectTracker()

    for frame_idx, frame in enumerate(frames):
        # produce the "modified" frame that we actually perform tracking on
        mod = frame.copy()
        if do_backsub:
            mod = backsub.apply(mod)
        mod = cv.morphologyEx(mod, cv.MORPH_OPEN, open_kernel)
        mod = cv.morphologyEx(mod, cv.MORPH_CLOSE, close_kernel)

        # find edges, and from edges, contours
        edges = fish.get_edges(mod, lower, upper, smoothing)
        contours = fish.get_contours(edges)

        if frame_idx > 10:
            tracker.update_tracks(contours, frame_idx)
            tracker.check_for_locks(frame_idx)

        # produce the movie frame that we'll actually write out to disk
        img = cv.cvtColor((frame if draw_on_original else mod), cv.COLOR_GRAY2BGR)
        img = fish.draw_bounding_rectangles(img, contours)
        img = fish.draw_live_object_tracks(img, tracker)
        yield img
        # yield None

    make_velocities_over_time(velocities_out, tracker)
    make_areas_over_time(areas_out, tracker)

    make_velocity_histogram_over_time(
        velocity_histogram_out, tracker, num_frames=frame_idx
    )


def make_velocity_histogram_over_time(out, tracker, num_frames):
    padded_velocities = []
    for oid, track in tracker.tracks.items():
        idxs = np.array(track.frame_idxs)
        areas = np.array(track.areas)

        if len(idxs) < 10:
            # if len(idxs) < 10 or not np.all(areas > 50):
            continue

        velocities = np.linalg.norm(np.diff(np.vstack(track.positions), axis=0), axis=1)

        padded = np.zeros(num_frames) * np.NaN
        padded[idxs[:-1]] = velocities
        padded_velocities.append(padded)

    velocities = np.vstack(padded_velocities)
    percentiles = [10, 25, 50, 75, 90]
    percentile_style = {10: ":", 25: "--", 50: "-", 75: "--", 90: ":"}
    velocity_percentiles = np.nanpercentile(velocities, percentiles, axis=0)
    rng = (0, np.nanmax(velocities))

    num_bins = 10
    velocity_hist_per_frame = [
        np.histogram(v[~np.isnan(v)], bins=num_bins, range=rng)[0]
        for v in velocities.transpose()
    ]
    velocity_hist_per_frame = np.vstack(velocity_hist_per_frame)

    fig = plt.figure(figsize=(12, 8), dpi=600)
    ax = fig.add_subplot(111)

    frame_array = np.array(list(range(num_frames)))
    vel_array = np.linspace(*rng, num_bins)
    frame_mesh, vel_mesh = np.meshgrid(frame_array, vel_array, indexing="ij")

    mesh = ax.pcolormesh(
        frame_mesh,
        vel_mesh,
        velocity_hist_per_frame,
        norm=plt.Normalize(vmin=0, vmax=10),
    )

    plt.colorbar(mappable=mesh, ax=ax)

    for percentile, vel_per in zip(percentiles, velocity_percentiles):
        vel_per[np.isnan(vel_per)] = 0
        smoothed = sig.savgol_filter(vel_per, window_length=51, polyorder=3)
        ax.plot(
            frame_array, smoothed, color="white", linestyle=percentile_style[percentile]
        )

    ax.set_ylim(*rng)

    ax.set_xlabel("frame index")
    ax.set_ylabel("velocity bin")

    plt.savefig(out)


def make_velocities_over_time(out, tracker):
    fig = plt.figure(figsize=(8, 8), dpi=600)
    ax = fig.add_subplot(111)

    for oid, track in tracker.tracks.items():
        idxs = np.array(track.frame_idxs)
        areas = np.array(track.areas)

        if len(idxs) < 10:
            # if len(idxs) < 10 or not np.all(areas > 100):
            continue

        velocities = np.linalg.norm(np.diff(np.vstack(track.positions), axis=0), axis=1)

        ax.plot(idxs[:-1], velocities, color=np.random.rand(3))
    ax.set_ylim(0, None)

    ax.set_xlabel("frame index")
    ax.set_ylabel("velocity")

    plt.savefig(out)


def make_areas_over_time(out, tracker):
    fig = plt.figure(figsize=(8, 8), dpi=600)
    ax = fig.add_subplot(111)

    for oid, track in tracker.tracks.items():
        idxs = np.array(track.frame_idxs)
        areas = np.array(track.areas)

        if len(idxs) < 10:
            # if len(idxs) < 10 or not np.all(areas > 100):
            continue

        ax.plot(idxs, areas, color=np.random.rand(3))

    ax.set_ylim(0, None)

    ax.set_xlabel("frame index")
    ax.set_ylabel("contour area")

    plt.savefig(out)


if __name__ == "__main__":
    IN = Path(__file__).parent.parent / "data"
    OUT = Path(__file__).parent / "out" / Path(__file__).stem
    OUT.mkdir(exist_ok=True)

    for movie, draw in itertools.product(["control", "drug"], [True, False]):
        frames = fish.load_or_read(IN / movie)[100:]

        frames = list(fish.remove_background(frames))

        it = make_frames(
            frames,
            100,
            200,
            5,
            draw_on_original=draw,
            areas_out=OUT / f"{movie}_areas.png",
            velocities_out=OUT / f"{movie}_velocities.png",
            velocity_histogram_out=OUT / f"{movie}_velocity_histogram.png",
            do_backsub=False,
        )

        op = fish.make_movie(
            OUT / f"{movie}__edge_test__draw_on_original={draw}",
            frames=it,
            num_frames=len(frames),
            fps=5,
        )

    # for frame in tqdm(it, total=len(frames)):
    #     pass
