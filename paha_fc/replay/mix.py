"""The mixed local training set ``D_mix = D_cj^t + D_hat_loc + D_hat_glo``.

Alongside images and labels the set carries the cached task-anchor outputs
``mu_init(x)`` and ``g_init(mu_init(x))``; they are the targets of the two
distillation terms and are constant during a task, so they are computed once and
indexed like any other feature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch

from ..config import Config
from ..memory import LatentBank
from ..models import AutoEncoder, anchor_forward
from .batches import balanced_index_batches
from .channels import global_replay_samples, local_replay_samples


@dataclass
class MixedSet:
    x: torch.Tensor
    y: torch.Tensor
    mu_init: torch.Tensor
    rec_init: torch.Tensor
    n_real: int
    n_local: int
    n_global: int

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def batches(self, cfg: Config, seed: int) -> List[torch.Tensor]:
        return balanced_index_batches(self.y.detach().cpu().numpy(), cfg.batch_size,
                                      cfg.local_epochs, seed)

    def counts(self) -> str:
        return f"real={self.n_real} local={self.n_local} global={self.n_global}"


def build_mixed_set(cfg: Config, anchor: AutoEncoder, x: torch.Tensor, y: torch.Tensor,
                    bank: LatentBank, centroid_matrix: torch.Tensor,
                    task_classes: Sequence[int], device: torch.device,
                    real_anchor: Optional[tuple] = None,
                    generator: Optional[torch.Generator] = None) -> MixedSet:
    """Assemble real data plus the enabled replay channels with anchor targets."""
    xs: List[torch.Tensor] = [x]
    ys: List[torch.Tensor] = [y]
    mus: List[torch.Tensor] = []
    recs: List[torch.Tensor] = []
    if real_anchor is not None:
        mus.append(real_anchor[0].detach())
        recs.append(real_anchor[1].detach())

    n_local = 0
    if cfg.use_local_replay and not bank.is_empty():
        x_loc, y_loc = local_replay_samples(anchor, bank, device)
        if x_loc.numel():
            xs.append(x_loc)
            ys.append(y_loc)
            n_local = int(x_loc.shape[0])

    n_global = 0
    if cfg.use_global_replay:
        x_glo, y_glo = global_replay_samples(anchor, centroid_matrix, task_classes,
                                             cfg.global_replay_per_class, cfg.sigma_syn,
                                             device, generator=generator)
        if x_glo.numel():
            xs.append(x_glo)
            ys.append(y_glo)
            n_global = int(x_glo.shape[0])

    x_all = torch.cat(xs)
    y_all = torch.cat(ys)
    replay_x = x_all[x.shape[0]:]

    if mus and replay_x.numel():
        mu_rep, rec_rep = anchor_forward(anchor, replay_x)
        mu_all = torch.cat(mus + [mu_rep])
        rec_all = torch.cat(recs + [rec_rep])
    elif mus:
        mu_all = torch.cat(mus)
        rec_all = torch.cat(recs)
    else:
        mu_all, rec_all = anchor_forward(anchor, x_all)

    return MixedSet(x=x_all, y=y_all, mu_init=mu_all.detach(), rec_init=rec_all.detach(),
                    n_real=int(x.shape[0]), n_local=n_local, n_global=n_global)
