from pathlib import Path

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

import fish

blobs = fish.load_blobs(Path("out") / "find_blobs" / "D1-1__frame=1000.blobs")

bb = blobs[0]
bv = blobs[-1]
bs = [bb, bv]

MB = 10 ** 6

frames = fish.cached_read(Path.cwd().parent / "data" / f"{blobs[0].movie}.hsv")
print(frames.shape)

frame = frames[blobs[0].frame_idx]
f = fish.convert_colorspace(frame, cv.COLOR_GRAY2BGR)

for b in blobs:
    print(b)
    for k, v in sorted(b.__dict__.items()):
        msg = f"{k} {type(v)}"
        if isinstance(v, np.ndarray):
            msg += f" | {v.dtype} | {v.shape} {v.size} | {v.nbytes / MB} MB"

        print(msg)
    x = np.where(b.points_in_label)
    print(x)
    f[x] = fish.MAGENTA

    print()


fig = fish.show_image(f)

plt.savefig("hi.png")
