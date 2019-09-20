import logging

from pathlib import Path
import itertools

import numpy as np
import matplotlib.pyplot as plt

import fish

logging.basicConfig()

if __name__ == "__main__":
    IN = Path(__file__).parent.parent / "data"
    OUT = Path(__file__).parent / "out" / Path(__file__).stem
    OUT.mkdir(exist_ok=True)

    frames = fish.load_or_read(IN / "control.avi")
    frames = frames[100:200]
    frames = fish.remove_background(frames)

    chunk_size = 32
    max_counts_per_chunk = (chunk_size ** 2) * 255

    vectors = list(
        v
        for *_, v in fish.make_vectors_from_frames(
            frames, chunk_size=chunk_size, vectorizer=fish.sorted_ravel
        )
    )
    print(vectors[0].shape)

    vectors = fish.stack_vectors(vectors)
    print(vectors.shape)

    total_counts = np.sum(vectors, axis=1).astype(np.float64) / max_counts_per_chunk
    print(total_counts.shape)

    # fig = plt.figure(figsize=(8, 8))
    # ax = fig.add_subplot(111)
    # ax.hist(total_counts, bins=50, density=True)
    # ax.set_yscale("log")
    # ax.set_xlim(0, np.max(total_counts))
    # ax.set_ylim(1e-9, 1e-3)
    # plt.show()

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.hist(total_counts, bins=100, density=True, cumulative=True, histtype="step")
    # ax.set_yscale("log")
    ax.set_xlim(0, np.max(total_counts))
    ax.set_ylim(0.8, 1)
    plt.show()

    cutoff = np.quantile(total_counts, 0.95)
    print(cutoff)
