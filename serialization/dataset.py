from __future__ import annotations

from pathlib import Path

import numpy as np


Record = tuple[Path, int]


def enumerate_records(data: str | Path) -> list[Record]:

    data_path = Path(data)
    shard_paths = [data_path] if data_path.is_file() else sorted(data_path.rglob("shard-*.npz"))
    records: list[Record] = []
    for path in shard_paths:
        with np.load(path) as shard:
            records.extend((path, row) for row in range(len(shard["pd_pu"])))
    return records


def deterministic_split(
    records: list[Record], validation_fraction: float = 0.1, seed: int = 42
) -> tuple[list[Record], list[Record]]:

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between 0 and 1")
    if not records:
        return [], []
    order = np.random.default_rng(seed).permutation(len(records))
    validation_size = max(1, int(round(len(records) * validation_fraction)))
    validation_indices = order[:validation_size]
    train_indices = order[validation_size:]
    return [records[int(i)] for i in train_indices], [records[int(i)] for i in validation_indices]
