"""Task-boundary centroid exchange.

The exchange involves all ``N`` registered clients and runs once per task, so it
is independent of the participation rate that governs the training rounds. Each
client uploads one mean embedding per new class it possesses; the server
aggregates them and checks that every new class is covered before alignment and
before the training rounds start.
"""

from __future__ import annotations

from typing import Dict, Sequence

import torch

from ..models import AutoEncoder
from .table import CentroidTable


class CentroidCoverageError(RuntimeError):
    """Some new class received no centroid, so alignment cannot start."""


@torch.no_grad()
def client_class_centroids(anchor: AutoEncoder, x: torch.Tensor, y: torch.Tensor,
                           classes: Sequence[int], chunk: int = 4096) -> CentroidTable:
    """Mean anchor embedding of this client's class-``k`` examples, per new class."""
    anchor.enc.eval()
    out: CentroidTable = {}
    for c in sorted(int(v) for v in classes):
        idx = (y == c)
        n = int(idx.sum())
        if n == 0:
            continue            # a client uploads a centroid only for classes it holds
        mu = anchor.enc.mean_embedding(x[idx], chunk=chunk)
        out[c] = mu.mean(dim=0).detach().cpu()
    return out


def aggregate_centroids(per_client: Sequence[CentroidTable],
                        classes: Sequence[int]) -> CentroidTable:
    """Unweighted mean of the client centroids of each new class."""
    out: CentroidTable = {}
    for c in sorted(int(v) for v in classes):
        votes = [cc[c] for cc in per_client if c in cc]
        if votes:
            out[c] = torch.stack(votes).mean(dim=0)
    return out


def verify_coverage(aggregated: CentroidTable, classes: Sequence[int]) -> None:
    """Require at least one supporting client for every class of ``Y_t``."""
    missing = [int(c) for c in classes if int(c) not in aggregated]
    if missing:
        raise CentroidCoverageError(
            f"centroid exchange incomplete for classes {missing}: the all-client "
            "exchange produced no supporting client, so alignment and global "
            "replay cannot start")


def coverage_report(per_client: Sequence[CentroidTable], classes: Sequence[int]) -> Dict[int, int]:
    """Number of supporting clients per new class (diagnostics)."""
    return {int(c): sum(1 for cc in per_client if int(c) in cc) for c in classes}
