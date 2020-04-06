from typing import Mapping
import dataclasses
import csv

import numpy as np


@dataclasses.dataclass(frozen=True)
class HandCounted:
    movie: str
    times_to_counts: Mapping[int, int]
    total: int

    @property
    def times(self):
        return np.array(list(self.times_to_counts.keys()))

    @property
    def counts(self):
        return np.array(list(self.times_to_counts.values()))


def load_hand_counted_data(path):
    data = []
    with path.open(mode="r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(
                HandCounted(
                    movie=row.pop("Movie"),
                    total=int(row.pop("Total")),
                    times_to_counts={int(k): int(v) for k, v in row.items()},
                )
            )
    return data
