import logging

from pathlib import Path

import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import GaussianMixture

import fish

logging.basicConfig(
    format=f"[%(asctime)s] [%(levelname)s] %(filename)s:%(lineno)d ~ %(message)s"
)

frames = fish.load_or_read(Path(__file__).parent.parent / "data" / "drug.avi")
frames = frames[100:105]
print(frames.shape)

pcas_to_vectorizers = {
    IncrementalPCA(n_components=3): fish.sorted_diff,
    IncrementalPCA(n_components=3): fish.sorted_ravel,
}

windows = fish.make_windows_from_frames(frames, window_radius=25, window_step=20)

fish.train_pcas(pcas_to_vectorizers, windows)

clusterer = GaussianMixture(n_components=4, warm_start=True)
fish.train_clusterer(clusterer, pcas_to_vectorizers, windows)
