"""Random streams that drive participation and local shuffling.

Every stream is derived analytically from ``(seed, task, round)``, so a repeated
run replays the same active-client sets and the same balanced-batch orders
without carrying generator objects through the loop. Client selection samples
``ceil(p * N)`` clients uniformly without replacement, so every client carries
the same selection probability in each round.
"""

from __future__ import annotations

from typing import List

import numpy as np

from ..config import Config


def sample_active_clients(cfg: Config, task: int, round_index: int) -> List[int]:
    rng = np.random.RandomState(cfg.seed * 100003 + 1000 * int(task) + int(round_index))
    n_active = cfg.n_active()
    if n_active >= cfg.n_clients:
        return list(range(cfg.n_clients))
    return sorted(int(j) for j in rng.choice(cfg.n_clients, size=n_active, replace=False))


def task_round_seed(cfg: Config, task: int, round_index: int) -> int:
    """Seed of the balanced-batch traversal for one client-round."""
    return int(cfg.seed * 7 + 100 * int(task) + int(round_index))


def partition_seed(cfg: Config, task: int) -> int:
    """Seed of the per-task client partition."""
    return int(cfg.seed + 17 * int(task))
