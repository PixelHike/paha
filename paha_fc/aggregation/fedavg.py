"""FedAvg of client model differences."""

from __future__ import annotations

from typing import Dict, Sequence

import torch

Delta = Dict[str, torch.Tensor]


def fedavg(deltas: Sequence[Delta], weights: Sequence[float]) -> Delta:
    """``d_r = sum_j alpha_{j,r} d_r^{c_j}`` with ``alpha_{j,r} = q_j / sum_i q_i``.

    The weights are the current-task sample counts of the active clients, so the
    aggregate is the size-weighted mean of the uploaded model differences.
    """
    if not deltas:
        raise ValueError("fedavg needs at least one delta")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("FedAvg weights must sum to a positive value")
    keys = list(deltas[0].keys())
    out: Delta = {}
    for k in keys:
        acc = torch.zeros_like(deltas[0][k])
        for w, d in zip(weights, deltas):
            acc = acc + d[k] * (float(w) / total)
        out[k] = acc
    return out


def zero_delta(reference: Delta) -> Delta:
    """Neutral upload for a client whose local model did not move."""
    return {k: torch.zeros_like(v) for k, v in reference.items()}


def delta_norm(delta: Delta) -> float:
    return float(sum(float(v.pow(2).sum()) for v in delta.values()) ** 0.5)
