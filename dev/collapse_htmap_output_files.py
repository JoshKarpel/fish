#!/usr/bin/env python

import shutil
import sys
from pathlib import Path

import htmap
from tqdm import tqdm

tag, target = sys.argv[1:]

map = htmap.load(tag)

export_dir = Path(target) / map.tag
export_dir.mkdir(parents = True, exist_ok = True)

for d in tqdm(map.output_files):
    for path in d.iterdir():
        to = export_dir / path.name
        shutil.copy2(path, to)
