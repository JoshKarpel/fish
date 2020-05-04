#!/usr/bin/env python

import sys
from pathlib import Path

import htmap
from tqdm import tqdm

import fish
import fish.io

tag, target = sys.argv[1:]

map = htmap.load(tag)

export_dir = Path(target) / map.tag
export_dir.mkdir(parents=True, exist_ok=True)

for movie_name, blobs_by_frame in tqdm(map):
    fish.io.save_object(
        blobs_by_frame, export_dir / Path(movie_name).with_suffix(".blobs")
    )
