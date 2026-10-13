"""New-class centroid alignment by gradient descent on the LJ potential.

Only the new-class positions move; the aligned centroids of earlier tasks stay
fixed, which is what keeps the classifier anchors stable across the task
sequence. The returned set is broadcast to the clients and then held fixed for
the whole task, serving clustering, global replay and nearest-centroid read-out.

The descent direction is always ``-grad U``. ``lj_trust_region`` caps the
displacement of any single centroid per iteration at a fraction of the
equilibrium spacing ``sigma``, which keeps a well-separated start from being
blown apart when two aggregated centroids coincide and ``sigma^12 / s^13``
becomes very large; with a large enough radius the update reduces to a plain
fixed-step gradient descent ``p <- p - eta_align * grad U``.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from ..config import Config
from ..logging_utils import get_logger
from .lj_potential import lj_potential_and_grad
from .table import CentroidTable

LOG = get_logger("centroids.align")


def resolve_sigma(new_p: torch.Tensor, old_p: Optional[torch.Tensor], cfg: Config) -> float:
    """Equilibrium spacing ``sigma`` used by the potential."""
    if cfg.lj_sigma_mode == "fixed":
        return float(cfg.lj_sigma)
    with torch.no_grad():  # rescaled variant: spacing relative to the centroid spread
        parts = [new_p] if old_p is None or old_p.numel() == 0 else [new_p, old_p]
        ref = torch.cat(parts, dim=0)
        mean_dist = torch.cdist(ref, ref).mean().clamp_min(1e-6).item()
    return float(cfg.lj_sigma) * (mean_dist / 4.0)


def align_centroids(new_centroids: CentroidTable, old_centroids: CentroidTable,
                    cfg: Config, device: torch.device) -> CentroidTable:
    """Optimise the new-class positions given the fixed historical centroids."""
    new_keys = sorted(new_centroids)
    if not new_keys:
        return dict(old_centroids)
    old_keys = sorted(old_centroids)

    reference = new_centroids[new_keys[0]]
    new_p = torch.stack([new_centroids[k].to(device).reshape(-1) for k in new_keys])
    if old_keys:
        old_p = torch.stack([old_centroids[k].to(device).reshape(-1) for k in old_keys])
    else:
        old_p = torch.empty((0, new_p.shape[1]), device=device)

    sigma = resolve_sigma(new_p, old_p, cfg)
    start = new_p.double()
    aligned, iterations = start.clone(), 0
    radius = float(cfg.lj_trust_region) * sigma
    for _ in range(int(cfg.lj_iters)):
        with torch.no_grad():
            _, grad = lj_potential_and_grad(aligned, old_p.double(), cfg.lj_eps, sigma,
                                            cfg.lj_delta)
            if not torch.isfinite(grad).all():
                LOG.warning("non-finite LJ gradient; keeping the pre-alignment centroids")
                aligned = start
                break
            step = float(cfg.lj_lr)
            direction = grad
            if cfg.lj_step_normalize:
                direction = grad / grad.abs().max().clamp_min(1e-30)
                step *= sigma
            if radius > 0:
                longest = direction.norm(dim=1).max().clamp_min(1e-30)
                step = min(step, radius / longest)
            candidate = aligned - step * direction
            if not torch.isfinite(candidate).all():
                LOG.warning("non-finite LJ iterate; keeping the pre-alignment centroids")
                aligned = start
                break
            aligned, iterations = candidate, iterations + 1

    out: CentroidTable = {int(k): v.detach().cpu().clone() for k, v in old_centroids.items()}
    out.update({int(k): aligned[i].to(reference.dtype).detach().cpu().clone()
                for i, k in enumerate(new_keys)})
    return out


def alignment_report(new_before: CentroidTable, new_after: CentroidTable) -> Dict[str, float]:
    """Mean and maximum displacement induced by the alignment (diagnostics)."""
    keys = sorted(set(new_before) & set(new_after))
    if not keys:
        return {"mean_shift": 0.0, "max_shift": 0.0}
    shifts = torch.stack([(new_after[k] - new_before[k]).norm() for k in keys])
    return {"mean_shift": float(shifts.mean()), "max_shift": float(shifts.max())}
