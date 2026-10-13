"""Task-boundary state operations: centroid exchange and memory refresh.

Both are executed with the task anchor ``W_{t-1}`` (centroids) or the model after
the ``R`` training rounds (memory refresh), and both are separate from the
participation-controlled training rounds.
"""

from __future__ import annotations

from typing import Dict, Sequence

import torch

from ..centroids import (CentroidMatrix, aggregate_centroids, align_centroids,
                         alignment_report, client_class_centroids, verify_coverage)
from ..config import Config
from ..logging_utils import get_logger
from ..memory import LatentBank
from ..models import AutoEncoder

LOG = get_logger("training.boundary")


@torch.no_grad()
def collect_and_align_centroids(cfg: Config, anchor: AutoEncoder, cxs, cys,
                                task_classes: Sequence[int],
                                centroids: CentroidMatrix) -> Dict[str, float]:
    """All-client centroid collection, aggregation, coverage check and alignment."""
    task_classes = sorted(int(c) for c in task_classes)
    anchor.enc.eval()

    per_client = [client_class_centroids(anchor, cxs[j], cys[j], task_classes)
                  for j in range(cfg.n_clients)]
    aggregated = aggregate_centroids(per_client, task_classes)
    if cfg.verify_centroid_coverage:
        verify_coverage(aggregated, task_classes)

    before = {k: v.clone() for k, v in aggregated.items()}
    aligned = align_centroids(aggregated, centroids.as_dict(), cfg, centroids.buffer.device)
    centroids.update(aligned)

    support = [sum(1 for cc in per_client if c in cc) for c in task_classes]
    report = alignment_report(before, {k: v for k, v in aligned.items() if k in before})
    return {"classes_covered": float(len(aggregated)),
            "min_support": float(min(support)) if support else 0.0,
            "align_mean_shift": report["mean_shift"],
            "align_max_shift": report["max_shift"]}


@torch.no_grad()
def _adapt_old_codes(cfg: Config, autoencoder: AutoEncoder, anchor: AutoEncoder,
                     bank: LatentBank) -> int:
    """Old-code adaptation ``m <- mu_phi(g_init(m))``, labels retained."""
    codes = bank.codes()
    if codes is None:
        return 0
    labels = bank.labels()
    device = next(autoencoder.parameters()).device
    recon = anchor.dec(codes.to(device))
    updated = autoencoder.enc.mean_embedding(recon)
    bank.replace_all(updated.detach().cpu(), labels.tolist())
    return int(updated.shape[0])


@torch.no_grad()
def _enqueue_new_codes(cfg: Config, autoencoder: AutoEncoder, bank: LatentBank,
                       x: torch.Tensor, y: torch.Tensor, task_classes: Sequence[int],
                       generator: torch.Generator) -> int:
    """Push ``mu_phi(x)`` of a fixed number of representatives per new class."""
    added = 0
    for c in sorted(int(v) for v in task_classes):
        idx = torch.nonzero(y == c).flatten()
        n = int(idx.numel())
        if n == 0 or cfg.memory_sample_per_class <= 0:
            continue
        keep = min(int(cfg.memory_sample_per_class), n)
        if cfg.enqueue_sample == "random" and keep < n:
            perm = torch.randperm(n, generator=generator)[:keep]
            idx = idx[perm.to(idx.device)]
        else:
            idx = idx[:keep]
        mu = autoencoder.enc.mean_embedding(x[idx])
        bank.extend(mu.detach().cpu(), [c] * int(idx.numel()))
        added += int(idx.numel())
    return added


def refresh_client_memories(cfg: Config, autoencoder: AutoEncoder, anchor: AutoEncoder,
                            banks: Sequence[LatentBank], cxs, cys,
                            task_classes: Sequence[int], participants: Sequence[int],
                            seed: int) -> Dict[str, float]:
    """Refresh the latent banks of the clients that took part in this task.

    With ``refresh_participants_only`` disabled every registered client refreshes,
    which is only equivalent when the participation rate is 1.
    """
    autoencoder.eval()
    targets = (sorted(int(j) for j in participants) if cfg.refresh_participants_only
               else list(range(cfg.n_clients)))
    adapted = added = 0
    for j in targets:
        generator = torch.Generator().manual_seed(int(seed) + 31 * int(j))
        if cfg.refresh_old_codes:
            adapted += _adapt_old_codes(cfg, autoencoder, anchor, banks[j])
        added += _enqueue_new_codes(cfg, autoencoder, banks[j], cxs[j], cys[j],
                                   task_classes, generator)
    return {"banks_refreshed": float(len(targets)),
            "codes_adapted": float(adapted),
            "codes_enqueued": float(added)}
