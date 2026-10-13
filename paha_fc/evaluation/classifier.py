"""Nearest-centroid read-out.

The classifier is the final encoder composed with the aligned centroid set:

    y_hat(x) = argmin_k || mu_phi(x) - p_k ||,   k in Y_{1:t},

so predictions range over every class seen so far and no task identity is
available at test time. The centroid set is fixed after the task-boundary
alignment; the encoder alone is what the federated rounds optimise.
"""

from __future__ import annotations

from typing import List, Sequence

import torch

from ..centroids import CentroidMatrix


def _pairwise_sq(mu: torch.Tensor, cents: torch.Tensor) -> torch.Tensor:
    return (mu.pow(2).sum(1, keepdim=True) - 2.0 * (mu @ cents.t())
            + cents.pow(2).sum(1).unsqueeze(0))


@torch.no_grad()
def nearest_centroid_predict(encoder, x: torch.Tensor, centroid_matrix: CentroidMatrix,
                             class_ids: Sequence[int], chunk: int = 4096) -> torch.Tensor:
    """Predicted class ids for ``x`` under the cumulative centroid set."""
    class_ids = torch.tensor(sorted(int(c) for c in class_ids), device=centroid_matrix.device)
    cents = centroid_matrix.buffer[class_ids]
    preds = []
    for i in range(0, x.shape[0], chunk):
        mu = encoder.mean_embedding(x[i:i + chunk], chunk=chunk)
        preds.append(class_ids[_pairwise_sq(mu, cents).argmin(dim=1)])
    return torch.cat(preds) if preds else torch.empty(0, dtype=torch.long, device=x.device)


@torch.no_grad()
def class_il_per_task_accuracy(encoder, x: torch.Tensor, y: torch.Tensor,
                               seen_task_splits: Sequence[Sequence[int]],
                               centroid_matrix: CentroidMatrix,
                               chunk: int = 4096) -> List[float]:
    """Per-task class-incremental accuracies, aligned with ``seen_task_splits``."""
    seen = sorted(int(c) for task in seen_task_splits for c in task)
    mask = torch.isin(y, torch.tensor(seen, device=y.device))
    xs, ys = x[mask], y[mask]
    if int(xs.shape[0]) == 0:
        return [0.0] * len(seen_task_splits)
    preds = nearest_centroid_predict(encoder, xs, centroid_matrix, seen, chunk=chunk)
    correct = preds == ys
    out: List[float] = []
    for task in seen_task_splits:
        task_t = torch.tensor([int(c) for c in task], device=y.device)
        m = torch.isin(ys, task_t)
        out.append(float(correct[m].float().mean()) * 100.0 if int(m.sum()) else 0.0)
    return out
