"""Client partitioning of the current-task training set.

Both schemes act on one task at a time: at task ``t`` the federation splits the
samples whose label lies in ``Y_t``, which is the private-data model used when
raw data of earlier tasks becomes unavailable.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch


def iid_assign(y: np.ndarray, n_clients: int, rng: np.random.RandomState) -> np.ndarray:
    """Uniform random assignment of samples to clients."""
    assign = np.empty(len(y), dtype=np.int64)
    order = rng.permutation(len(y))
    assign[order] = np.arange(len(y)) % n_clients
    return assign


def dirichlet_assign(y: np.ndarray, n_clients: int, alpha: float,
                     rng: np.random.RandomState) -> np.ndarray:
    """Label-skew assignment: per class draw client proportions ~ Dir(alpha).

    A class with no samples in a client simply contributes nothing to that
    client, which is what produces the label skew. Clients left empty by the
    draw receive one random sample so that every client can take part in
    training rounds.
    """
    n_samples = len(y)
    assign = np.zeros(n_samples, dtype=np.int64)
    classes = np.unique(y)
    proportions = np.array([rng.dirichlet([alpha] * n_clients) for _ in classes])
    for row, cls in enumerate(classes):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        counts = np.round(proportions[row] * len(idx)).astype(np.int64)
        counts[-1] = len(idx) - counts[:-1].sum()
        if counts[-1] < 0:
            counts[counts < 0] = 0
            counts[-1] = 0
        offset = 0
        for cl in range(n_clients):
            seg = idx[offset:offset + int(counts[cl])]
            assign[seg] = cl
            offset += int(counts[cl])
    present = set(np.unique(assign).tolist())
    for cl in range(n_clients):
        if cl not in present:
            assign[rng.choice(n_samples, size=1, replace=False)] = cl
    return assign


def partition_indices(y: np.ndarray, task_classes: Sequence[int], n_clients: int,
                      alpha: float, rng: np.random.RandomState) -> List[np.ndarray]:
    """Indices of the current-task samples, split across ``n_clients`` clients."""
    task_arr = np.array(sorted(int(c) for c in task_classes), dtype=np.int64)
    idx_task = np.nonzero(np.isin(y, task_arr))[0]
    if len(idx_task) == 0:
        return [np.empty(0, dtype=np.int64) for _ in range(n_clients)]
    if alpha and alpha > 0:
        assign = dirichlet_assign(y[idx_task], n_clients, alpha, rng)
        return [idx_task[np.where(assign == j)[0]] for j in range(n_clients)]
    order = rng.permutation(len(idx_task))
    return [idx_task[p] for p in np.array_split(order, n_clients)]


def build_client_data(x: torch.Tensor, y: torch.Tensor, task_classes: Sequence[int],
                      n_clients: int, alpha: float, seed: int
                      ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Per-client tensors of the current task, IID or Dirichlet partitioned."""
    rng = np.random.RandomState(int(seed))
    parts = partition_indices(y.detach().cpu().numpy(), task_classes, n_clients, alpha, rng)
    cxs, cys = [], []
    for idx in parts:
        t = torch.from_numpy(idx.astype(np.int64)).to(x.device)
        cxs.append(x[t])
        cys.append(y[t])
    return cxs, cys
