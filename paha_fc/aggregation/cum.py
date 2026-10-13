"""Client update memory (CUM).

The repository ``H = {m_cj}`` keeps one running estimate per client of that
client's model-update direction, across rounds and across tasks. Active clients
blend their new upload in with the participation-dependent coefficient
``rho_{j,u}``; inactive clients only get their stored vector scaled by ``gamma``,
which shrinks its norm while leaving the half-space constraint
``<m_cj, d_hat> >= 0`` untouched.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import torch

Delta = Dict[str, torch.Tensor]


class ClientUpdateMemory:
    def __init__(self, n_clients: int, rho_base: float, gamma: float,
                 device: torch.device = None):
        self.n_clients = int(n_clients)
        self.rho_base = float(rho_base)
        self.gamma = float(gamma)
        self.device = device
        self.memory: List[Dict[str, torch.Tensor]] = [{} for _ in range(self.n_clients)]
        self.counts: List[int] = [0] * self.n_clients
        self.u: int = 0                     # global training-round index across tasks

    # ------------------------------------------------------------- reads
    def mean_participation(self) -> float:
        """Federation mean ``n_bar_u`` of the participation counts."""
        return sum(self.counts) / float(self.n_clients)

    def rho(self, j: int) -> float:
        """Participation-dependent coefficient from the *pre-round* counts."""
        n_bar = self.mean_participation()
        return min(1.0, self.rho_base * (self.counts[j] + 1) / (n_bar + 1.0))

    def active_history_clients(self) -> List[int]:
        return [j for j in range(self.n_clients) if self.memory[j]]

    def has_history(self, j: int) -> bool:
        return bool(self.memory[j])

    def history_matrix(self, name: str, clients: Sequence[int]) -> Optional[torch.Tensor]:
        """Columns ``m_cj[l]`` for one parameter tensor, i.e. the history matrix ``V^[l]``."""
        cols = [self.memory[j][name].reshape(-1) for j in clients if name in self.memory[j]]
        if not cols:
            return None
        return torch.stack(cols, dim=1)

    # ------------------------------------------------------------ update
    def refresh(self, active: Iterable[int], deltas: Dict[int, Delta]) -> Dict[str, float]:
        """Repository refresh, then the participation-count update.

        ``rho`` is evaluated from the pre-round counts for every active client
        before any entry is written, so the stored directions that constrain this
        round are not modified by this round's uploads.
        """
        active = sorted(int(j) for j in active)
        coefficients = {j: self.rho(j) for j in active}
        keys = list(next(iter(deltas.values())).keys()) if deltas else []

        touched = 0
        for j in range(self.n_clients):
            entry = self.memory[j]
            if j in coefficients:
                rho = coefficients[j]
                upload = deltas[j]
                for k in keys:
                    prev = entry.get(k)
                    if prev is None:
                        entry[k] = rho * upload[k]
                    else:
                        entry[k] = (1.0 - rho) * self.gamma * prev + rho * upload[k]
                touched += 1
            elif self.gamma != 1.0:
                for k in entry:
                    entry[k] = self.gamma * entry[k]

        for j in active:
            self.counts[j] += 1
        self.u += 1
        return {"rho_mean": (sum(coefficients.values()) / len(coefficients)) if coefficients else 0.0,
                "n_clients_updated": float(touched)}

    # --------------------------------------------------------- diagnostics
    def history_sizes(self) -> List[int]:
        return [len(self.memory[j]) for j in range(self.n_clients)]

    def stored_entries(self) -> int:
        return sum(len(m) for m in self.memory)
