"""Balanced minibatch traversal of the mixed local training set.

The composite loss contains a clustering term against per-class
centroids, so batches are built with a per-class quota of ``|B| / |classes|``
samples: every class present in ``D_mix`` contributes to each batch, and one
traversal consumes the whole mixed set once per local epoch.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch


def class_blocks(y: np.ndarray):
    """Sorted-order class blocks: index array plus per-class boundaries."""
    order = np.argsort(y, kind="stable")
    ys = y[order]
    uniq = np.unique(ys)
    starts = np.searchsorted(ys, uniq, side="left").tolist() + [len(y)]
    return order, uniq, starts


def balanced_index_batches(y: np.ndarray, batch_size: int, epochs: int, seed: int,
                           per_class: int = None) -> List[torch.Tensor]:
    """Batch index lists, one full traversal per epoch."""
    order, uniq, starts = class_blocks(np.asarray(y))
    pools = []
    for ci in range(len(uniq)):
        idx = order[starts[ci]:starts[ci + 1]].copy()
        pools.append(idx)
    quota = max(1, int(batch_size) // max(1, len(uniq))) if per_class is None else int(per_class)
    rng = np.random.RandomState(int(seed))
    batches: List[torch.Tensor] = []
    for _ in range(max(1, int(epochs))):
        shuffled = []
        for pool in pools:
            pool = pool.copy()
            rng.shuffle(pool)
            shuffled.append(pool)
        offsets = [0] * len(shuffled)
        while True:
            parts = []
            for ci, pool in enumerate(shuffled):
                take = min(quota, len(pool) - offsets[ci])
                if take > 0:
                    parts.append(pool[offsets[ci]:offsets[ci] + take])
                    offsets[ci] += take
            if not parts:
                break
            batch = np.concatenate(parts)
            if batch.shape[0] >= 2:      # a single sample cannot drive BatchNorm
                batches.append(torch.from_numpy(batch.astype(np.int64)))
    return batches
