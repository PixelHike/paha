"""Layerwise historical projection (LHP).

The aggregate is mapped to the nearest update that keeps a non-negative inner
product with every retained client direction *inside each layer*. Because the
whole-model inner product is the sum of the layer inner products, the per-layer
system implies the whole-model non-negative-alignment constraint while removing
the possibility that positive layers mask a conflicting layer.

Clients that never uploaded keep no entry and impose nothing; a client whose
history does not cover a given tensor contributes a zero column, which
corresponds to the inert constraint ``0 >= 0``.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from ..config import Config
from .cum import ClientUpdateMemory
from .dual_qp import solve_dual

Delta = Dict[str, torch.Tensor]


class LayerwiseHistoricalProjection:
    def __init__(self, cum: ClientUpdateMemory, cfg: Config):
        self.cum = cum
        self.cfg = cfg

    def _history_block(self, name: str, reference: torch.Tensor,
                       clients: List[int]) -> torch.Tensor:
        flat_ref = reference.reshape(-1)
        columns = []
        for j in clients:
            entry = self.cum.memory[j].get(name)
            columns.append(entry.reshape(-1) if entry is not None
                           else torch.zeros_like(flat_ref))
        return torch.stack(columns, dim=1)

    def project(self, d_agg: Delta) -> Tuple[Delta, Dict[str, float]]:
        """Closest aggregate satisfying ``V^[l]^T d_hat^[l] >= 0`` for all ``l``."""
        clients = self.cum.active_history_clients()
        if not self.cfg.projection or not clients:
            return ({k: v.clone() for k, v in d_agg.items()},
                    {"lhp_applied": 0.0, "lhp_J": float(len(clients)), "lhp_kkt": 0.0})

        names = list(d_agg.keys())
        gram, linear, blocks = [], [], []
        for name in names:
            V = self._history_block(name, d_agg[name], clients)
            dv = d_agg[name].reshape(-1)
            gram.append(V.t().matmul(V))
            linear.append(V.t().matmul(dv))
            blocks.append((V, dv))

        G = torch.stack(gram)
        g = torch.stack(linear)
        lam, residual = solve_dual(G, g, solver=self.cfg.qp_solver, tol=self.cfg.qp_tol,
                                   max_iters=self.cfg.qp_max_iters)

        out: Delta = {}
        for i, name in enumerate(names):
            V, dv = blocks[i]
            out[name] = (dv + V.matmul(lam[i])).view_as(d_agg[name])
        return out, {"lhp_applied": 1.0, "lhp_J": float(len(clients)),
                     "lhp_kkt": float(residual.max())}
