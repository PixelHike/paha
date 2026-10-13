"""Lennard-Jones potential over the cumulative centroid set.

Old-class centroids act as fixed sources; only the new-class positions carry a
gradient, which is exactly what ``lj_potential_and_grad`` returns: the potential
value over all pairs that involve a new centroid, and the derivative with respect
to the new positions.
"""

from __future__ import annotations

import torch


def lj_potential_and_grad(new_p: torch.Tensor, old_p: torch.Tensor, eps: float,
                          sigma: float, delta: float):
    """Return ``(U, grad_new)`` with ``s_ab = ||p_a - p_b|| + delta_LJ``.

    ``grad_new[i] = 24 eps sum_b (sigma^6 / s^7 - 2 sigma^12 / s^13) d_ab``, the
    per-new-centroid derivative of the pair potential, with
    ``d_ab = (p_a - p_b) / ||p_a - p_b||``.

    The arithmetic runs in float64: coincident centroids drive ``sigma^12 / s^13``
    to magnitudes that overflow single precision, and the offset ``delta_LJ`` only
    keeps the potential finite in exact arithmetic.
    """
    k_new = int(new_p.shape[0])
    work = torch.float64
    new_p = new_p.to(work)
    has_old = old_p is not None and old_p.numel() > 0
    old_p = old_p.to(work) if has_old else None
    all_p = torch.cat([new_p, old_p], dim=0) if has_old else new_p

    diff = new_p.unsqueeze(1) - all_p.unsqueeze(0)          # (K, N, m)
    dist = diff.norm(dim=2)                                  # (K, N)

    # Self pairs are excluded by pushing their separation far out, which makes
    # sigma^k / s^k underflow instead of dividing by zero.
    keep = torch.ones_like(dist, dtype=torch.bool)
    keep[torch.arange(k_new, device=new_p.device), torch.arange(k_new, device=new_p.device)] = False
    s = (dist + delta).masked_fill(~keep, 1.0e6)

    sig6 = float(sigma) ** 6
    sig12 = sig6 * sig6
    inv6 = sig6 / s.pow(6)
    psi = 4.0 * eps * (inv6 * inv6 - inv6)                   # pair potentials
    coeff = 24.0 * eps * (sig6 / s.pow(7) - 2.0 * sig12 / s.pow(13))
    unit = diff / dist.clamp_min(1e-12).unsqueeze(2)
    unit = torch.where(keep.unsqueeze(2), unit, torch.zeros_like(unit))
    grad = (coeff.unsqueeze(2) * unit).sum(dim=1)

    # Each unordered pair once: new-new pairs by their upper triangle, plus every
    # new-old pair. Old-old terms are constant w.r.t. the new positions.
    tri = torch.triu(torch.ones(k_new, k_new, dtype=torch.bool, device=new_p.device), diagonal=1)
    energy = psi[:, :k_new][tri].sum()
    if has_old:
        energy = energy + psi[:, k_new:].sum()
    return energy, grad
