#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import fish

if __name__ == "__main__":
    IN = Path(__file__).parent.parent / "data"
    OUT = Path(__file__).parent / "out"
    OUT.mkdir(exist_ok=True)

    plt.figure(figsize=(8, 8), dpi=600)

    for data in ["control", "drug"]:
        for idx in (1, 2):
            frames = fish.load_or_read(IN / data)
            print(frames.shape)

            averaged = np.average(frames, axis=(0, idx))
            print(averaged.shape)

            plt.plot(averaged, label=f"{data}, {idx}")

    plt.legend()

    plt.savefig(str(OUT / f"gradient"))
    plt.close()
