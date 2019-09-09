import logging

import pickle
from pathlib import Path

import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import GaussianMixture


import fish

logging.basicConfig(
    format=f"[%(asctime)s] [%(levelname)s] %(filename)s:%(lineno)d ~ %(message)s"
)

IN = Path(__file__).parent.parent / "data"
OUT = Path(__file__).parent / "out"

frames = fish.load_or_read(IN / "control.avi")
frames = frames[500:505]
mod = np.stack(list(fish.remove_background(frames)), axis=0)
print(mod.shape)

pcas_to_vectorizers = {
    IncrementalPCA(n_components=3): fish.sorted_diff,
    # IncrementalPCA(n_components=3): fish.sorted_ravel,
}

windows = fish.make_windows_from_frames(mod, window_radius=20, window_step=5)

fish.train_pcas(pcas_to_vectorizers, windows)

clusterer = GaussianMixture(n_components=4, warm_start=True)
fish.train_clusterer(clusterer, pcas_to_vectorizers, windows)


for name, base_frames in zip(("base", "mod"), [frames, mod]):
    labelled = fish.label_chunks_in_frames(
        base_frames=base_frames,
        windows=windows,
        pcas_to_vectorizers=pcas_to_vectorizers,
        clusterer=clusterer,
    )

    fish.make_movie(
        OUT / f"new_test__{name}.mp4", frames=labelled, num_frames=len(mod), fps=10
    )
