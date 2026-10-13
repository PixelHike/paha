"""Random partitioning of the class order into disjoint task label spaces."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np


def make_task_splits(num_classes: int, num_tasks: int, seed: int) -> List[List[int]]:
    """Split ``range(num_classes)`` into ``num_tasks`` disjoint class groups.

    The class order is shuffled once with ``seed`` so that every method in a
    comparison sees the same task sequence. Groups are contiguous in the
    shuffled order and differ by at most one class when the division is uneven.
    """
    if num_tasks <= 0 or num_classes <= 0:
        raise ValueError("num_classes and num_tasks must be positive")
    rng = np.random.RandomState(seed)
    classes = list(range(num_classes))
    rng.shuffle(classes)
    base, rem = divmod(num_classes, num_tasks)
    splits: List[List[int]] = []
    start = 0
    for i in range(num_tasks):
        k = base + (1 if i < rem else 0)
        splits.append(sorted(classes[start:start + k]))
        start += k
    return splits


def cumulative_classes(splits: Sequence[Sequence[int]], up_to: int) -> List[int]:
    """Union of the class groups of the first ``up_to`` tasks."""
    return sorted(int(c) for s in list(splits)[:up_to] for c in s)
