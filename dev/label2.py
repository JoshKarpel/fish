import logging

from pathlib import Path

import numpy as np
from sklearn.decomposition import IncrementalPCA

import fish

logging.basicConfig(format = f'[%(asctime)s] [%(levelname)s] %(filename)s:%(lineno)d ~ %(message)s')

frames = fish.load_or_read(Path(__file__).parent.parent / 'data' / 'drug.avi')
frames = frames[100:110]
print(frames.shape)

pcas_to_vectorizers = {
    IncrementalPCA(n_components = 3): fish.sorted_diff,
    IncrementalPCA(n_components = 3): fish.sorted_ravel,
}

windows = fish.make_windows_from_frames(frames, window_radius = 25, window_step = 10)
# for frame_idx, coords_to_windows in windows.items():
#     print(frame_idx, len(coords_to_windows))

fish.train_pcas(pcas_to_vectorizers, windows)
