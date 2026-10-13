"""Server-side handling of one communication round.

The steps run in a fixed order:

1. FedAvg of the active uploads;
2. layerwise projection against the repository *as it stood before this round*;
3. participation-dependent refresh of the repository and the counts;
4. global update ``W <- W - eta_g * d_hat``.

Projecting before the refresh lets the retained histories constrain the current
round, and reserves the new information for subsequent rounds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch

from ..config import Config
from ..models import AutoEncoder, build_autoencoder, pack_state, unpack_state
from .cum import ClientUpdateMemory
from .fedavg import fedavg, zero_delta
from .lhp import LayerwiseHistoricalProjection

State = Dict[str, torch.Tensor]
Delta = Dict[str, torch.Tensor]


@dataclass
class ServerStep:
    d_agg: Delta
    d_proj: Delta
    info: Dict[str, float] = field(default_factory=dict)

    @property
    def agg_norm(self) -> float:
        return float(sum(float(v.pow(2).sum()) for v in self.d_agg.values()) ** 0.5)

    @property
    def proj_norm(self) -> float:
        return float(sum(float(v.pow(2).sum()) for v in self.d_proj.values()) ** 0.5)


class PAHAServer:
    def __init__(self, cfg: Config, init_state: Optional[State] = None,
                 autoencoder: Optional[AutoEncoder] = None):
        from ..utils import resolve_device
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        if init_state is None:
            autoencoder = autoencoder or build_autoencoder(
                cfg.latent_dim, cfg.resolve_input_size(), self.device)
            init_state = pack_state(autoencoder.enc, autoencoder.dec, self.device)
        else:
            init_state = {k: v.to(self.device) for k, v in init_state.items()}
        self.global_state: State = init_state
        self.cum = ClientUpdateMemory(cfg.n_clients, cfg.rho_base, cfg.gamma, self.device)
        self.projector = LayerwiseHistoricalProjection(self.cum, cfg)

    # ------------------------------------------------------------- helpers
    def snapshot(self) -> State:
        """Task anchor ``W_{t-1}``: a detached copy of the current global model."""
        return {k: v.detach().clone() for k, v in self.global_state.items()}

    def load_into(self, autoencoder: AutoEncoder) -> None:
        unpack_state(self.global_state, autoencoder.enc, autoencoder.dec)

    # ------------------------------------------------------------- stepping
    def round_step(self, active, deltas: Dict[int, Delta],
                   sizes: Dict[int, float]) -> ServerStep:
        """One round: aggregate, project against the stored history, refresh, update ``W``.

        ``sizes[j]`` is the current-task sample count ``q_j`` of client ``j``, which
        sets the FedAvg weight ``alpha_{j,r} = q_j / sum_i q_i``.
        """
        active = sorted(int(j) for j in active)
        weights = [float(sizes[int(j)]) for j in active]
        d_agg = fedavg([deltas[j] for j in active], weights)
        d_proj, info = self.projector.project(d_agg)

        cum_info = self.cum.refresh(active, {j: deltas[j] for j in active})
        info.update(cum_info)
        info["agg_norm"] = float(sum(float(v.pow(2).sum()) for v in d_agg.values()) ** 0.5)
        info["proj_norm"] = float(sum(float(v.pow(2).sum()) for v in d_proj.values()) ** 0.5)

        with torch.no_grad():
            for k in self.global_state:
                self.global_state[k] = self.global_state[k] - self.cfg.server_lr * d_proj[k]
        return ServerStep(d_agg=d_agg, d_proj=d_proj, info=info)

    def zero_delta(self) -> Delta:
        return zero_delta(self.global_state)

    # -------------------------------------------------------- diagnostics
    def repository_summary(self) -> Dict[str, float]:
        return {
            "repo_entries": float(self.cum.stored_entries()),
            "repo_clients": float(len(self.cum.active_history_clients())),
            "counts_min": float(min(self.cum.counts)),
            "counts_max": float(max(self.cum.counts)),
            "round_index": float(self.cum.u),
        }
