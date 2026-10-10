"""Documentation."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterator

import numpy as np


def temporal_splits(
    years: np.ndarray,
    cutpoints: list[int],
    *,
    min_train: int = 100,
    min_test: int = 20,
) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """Documentation."""
    valid = ~np.isnan(years)
    for T in cutpoints:
        tr = np.where(valid & (years < T))[0]
        te = np.where(valid & (years >= T))[0]
        if len(tr) < min_train or len(te) < min_test:
            continue
        yield f"T{T}", tr, te


def scaffold_kfold_splits(
    scaffolds: list[str],
    n_splits: int = 5,
    *,
    seed: int = 66,
    shuffle: bool = True,
) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """Documentation."""
    rng = np.random.RandomState(seed)
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, sc in enumerate(scaffolds):
        groups[sc].append(idx)

    folds: list[list[int]] = [[] for _ in range(n_splits)]
    for group in sorted(groups.values(), key=len, reverse=True):
        folds[int(np.argmin([len(f) for f in folds]))].extend(group)

    for i in range(n_splits):
        te = np.array(folds[i], dtype=int)
        tr = np.concatenate([folds[j] for j in range(n_splits) if j != i]).astype(int)
        if shuffle:
            rng.shuffle(tr)          
        yield f"fold{i + 1}", tr, te


def random_holdout_splits(
    n: int,
    *,
    test_size: float = 0.2,
    seeds: list[int],
) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """Documentation."""
    for seed in seeds:
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n)
        n_test = max(1, int(round(test_size * n)))
        yield f"seed{seed}", perm[n_test:].astype(int), perm[:n_test].astype(int)


def suggest_cutpoints(
    years: np.ndarray,
    *,
    min_test: int = 20,
    max_test_frac: float = 0.6,
) -> list[int]:
    """Documentation."""
    valid = years[~np.isnan(years)]
    if len(valid) == 0:
        return []
    n = len(valid)
    out = []
    for T in range(int(valid.min()) + 1, int(valid.max()) + 1):
        n_test = int((valid >= T).sum())
        if min_test <= n_test <= max_test_frac * n:
            out.append(T)
    return out
