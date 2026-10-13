"""Solvers for the layerwise dual of the historical projection.

For one layer the primal is ``min_dhat 1/2 ||d - dhat||^2  s.t.  V^T dhat >= 0``,
whose dual is

    max_{lam >= 0}  -1/2 lam^T (V^T V) lam - lam^T V^T d,

equivalently ``min_{lam >= 0} f(lam) = 1/2 lam^T G lam + lam^T g`` with
``G = V^T V`` (positive semidefinite) and ``g = V^T d``. Every dual variable
corresponds to one retained client direction, so the QP has at most ``N``
variables and is solved once per layer, batched across layers.

Downstream reasoning about the projection assumes the exact solution, so both
solvers report a KKT residual and stop only once it falls below ``qp_tol`` (up to
``qp_max_iters``).
"""

from __future__ import annotations

from typing import Tuple

import torch


def _max_eigenvalue(G: torch.Tensor, iters: int = 100) -> torch.Tensor:
    """Largest eigenvalue per layer by power iteration.

    The Gram matrices are nearly degenerate whenever two client histories are
    close, a regime in which batched ``eigh`` may fail to converge; the scalar
    bound only has to be an upper estimate to keep the step size safe.
    """
    n = G.shape[-1]
    v = torch.ones(G.shape[:-1], device=G.device, dtype=G.dtype) / (n ** 0.5)
    with torch.no_grad():
        for _ in range(iters):
            w = torch.bmm(G, v.unsqueeze(2)).squeeze(2)
            v = w / w.norm(dim=1, keepdim=True).clamp_min(1e-30)
        rayleigh = (v * torch.bmm(G, v.unsqueeze(2)).squeeze(2)).sum(1)
    return rayleigh.clamp_min(1e-12)


def kkt_residual(G: torch.Tensor, g: torch.Tensor, lam: torch.Tensor,
                 lmax: torch.Tensor = None, g_max: torch.Tensor = None) -> torch.Tensor:
    """Per-layer stationarity measure ``max_j |min(lam_j, grad_j)|``.

    Zero exactly at a KKT point of the non-negative orthant constraint: an index
    either sits on the boundary with a non-negative gradient, or is strictly
    positive with a vanishing gradient. Values are normalised by the problem scale
    so that ``qp_tol`` is a relative tolerance. Pass ``lmax`` and ``g_max`` to reuse
    the spectral and linear-term estimates instead of recomputing them.
    """
    grad = torch.baddbmm(g.unsqueeze(2), G, lam.unsqueeze(2)).squeeze(2)
    violation = torch.minimum(lam, grad).abs().max(dim=1).values
    lmax = _max_eigenvalue(G, iters=10) if lmax is None else lmax
    g_max = g.abs().max(dim=1).values if g_max is None else g_max
    scale = 1.0 + g_max + lmax * lam.abs().max(dim=1).values
    return violation / scale


def solve_dual_pgd(G: torch.Tensor, g: torch.Tensor, tol: float = 1e-10,
                   max_iters: int = 4000) -> Tuple[torch.Tensor, torch.Tensor]:
    """Accelerated projected gradient on the dual, batched over layers.

    Suitable when many layers share the same number of retained directions: the
    whole autoencoder is solved in a handful of batched kernels.
    """
    lmax = _max_eigenvalue(G)
    g_max = g.abs().max(dim=1).values
    step = (1.0 / (lmax * 1.05)).unsqueeze(1)
    lam = torch.zeros_like(g)
    y = lam.clone()
    t = 1.0
    for iteration in range(int(max_iters)):
        grad = torch.baddbmm(g.unsqueeze(2), G, y.unsqueeze(2)).squeeze(2)
        lam_new = (y - step * grad).clamp_min(0.0)
        t_new = 0.5 * (1.0 + (1.0 + 4.0 * t * t) ** 0.5)
        y = lam_new + ((t - 1.0) / t_new) * (lam_new - lam)
        lam, t = lam_new, t_new
        if iteration % 5 == 4 or iteration == int(max_iters) - 1:
            if float(kkt_residual(G, g, lam, lmax=lmax, g_max=g_max).max()) <= tol:
                break
    lam = torch.nan_to_num(lam, nan=0.0, posinf=0.0, neginf=0.0)
    return lam, kkt_residual(G, g, lam, lmax=lmax, g_max=g_max)


def _solve_layer_active_set(G: torch.Tensor, g: torch.Tensor, tol: float,
                            max_iters: int) -> torch.Tensor:
    """Lawson-Hanson active-set solve for one layer; exact up to ``tol``."""
    n = G.shape[0]
    lam = torch.zeros(n, device=G.device, dtype=G.dtype)
    passive = torch.zeros(n, dtype=torch.bool, device=G.device)
    for _ in range(max(int(max_iters), 4 * n)):
        grad = G @ lam + g
        want_in = (~passive) & (grad < -tol)
        if not bool(want_in.any()):
            break
        candidates = torch.nonzero(want_in).flatten()
        pick = candidates[grad[candidates].argmin()]
        passive[int(pick)] = True
        idx = torch.nonzero(passive).flatten()
        sub = G[idx][:, idx]
        try:
            sol = torch.linalg.lstsq(sub, -g[idx]).solution
        except Exception:
            sol = torch.linalg.lstsq(sub + 1e-8 * torch.eye(len(idx), device=G.device),
                                     -g[idx]).solution
        if bool((sol <= 0).any()):
            # Lawson-Hanson feasibility step: move along the segment until the
            # first coordinate hits zero, then retire that index to the active set.
            direction = torch.zeros_like(lam)
            direction[idx] = sol - lam[idx]
            free = direction > 0
            ratio = torch.where(free, -lam / direction.clamp_min(1e-30),
                                torch.full_like(lam, float("inf")))
            drop = int(ratio.argmin())
            lam = lam + ratio[drop].clamp(0.0, 1.0) * direction
            lam = lam.clamp_min(0.0)
            passive[drop] = False
        else:
            lam = torch.zeros_like(lam)
            lam[idx] = sol.clamp_min(0.0)
    return lam.clamp_min(0.0)


def solve_dual_active_set(G: torch.Tensor, g: torch.Tensor, tol: float = 1e-12,
                          max_iters: int = 200) -> Tuple[torch.Tensor, torch.Tensor]:
    """Active-set solves, one layer at a time; the reference implementation."""
    outs = [_solve_layer_active_set(G[i], g[i], tol, max_iters) for i in range(G.shape[0])]
    lam = torch.stack(outs)
    return lam, kkt_residual(G, g, lam)


def solve_dual(G: torch.Tensor, g: torch.Tensor, solver: str = "pgd",
               tol: float = 1e-10, max_iters: int = 4000) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dispatch on ``solver`` and return ``(lam, kkt_residual)``."""
    if solver == "active_set":
        return solve_dual_active_set(G, g, tol=tol, max_iters=max_iters)
    return solve_dual_pgd(G, g, tol=tol, max_iters=max_iters)
