"""Local training of one client inside one communication round."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch

from ..centroids import CentroidMatrix
from ..config import Config
from ..memory import LatentBank
from ..models import AutoEncoder, anchor_forward
from ..replay import MixedSet, build_mixed_set
from .losses import CompositeLoss, LossTerms

State = Dict[str, torch.Tensor]
Delta = Dict[str, torch.Tensor]


def build_optimizer(cfg: Config, autoencoder: AutoEncoder) -> torch.optim.Optimizer:
    params = list(autoencoder.parameters())
    if cfg.optimizer == "adam":
        return torch.optim.Adam(params, lr=cfg.lr)
    # Heavy-ball momentum without Nesterov acceleration or dampening, so the
    # uploaded model difference stays an exact unrolled sum of the local steps.
    return torch.optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum,
                          weight_decay=cfg.weight_decay)


class ClientWorker:
    """Owns the local optimiser so that a momentum buffer may be carried over.

    ``Config.reset_momentum`` selects between a zero initial buffer ``v_{j,0} = 0``
    at every round and a buffer that persists across the rounds of a task.
    """

    def __init__(self, cfg: Config, client_id: int, autoencoder: AutoEncoder):
        self.cfg = cfg
        self.client_id = int(client_id)
        self.autoencoder = autoencoder
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._key = None

    def optimizer(self) -> torch.optim.Optimizer:
        key = (self.cfg.optimizer, self.cfg.lr, self.cfg.momentum, self.cfg.weight_decay)
        if self._optimizer is None or self._key != key or self.cfg.reset_momentum:
            self._optimizer = build_optimizer(self.cfg, self.autoencoder)
            self._key = key
        return self._optimizer

    def train(self, mixed: MixedSet, centroids: CentroidMatrix, batches: List[torch.Tensor]
              ) -> LossTerms:
        cfg = self.cfg
        objective = CompositeLoss(cfg)
        autoencoder = self.autoencoder
        optimizer = self.optimizer()
        autoencoder.train()
        totals = LossTerms()
        n_batches = 0
        for index in batches:
            index = index.to(mixed.x.device, non_blocking=True)
            xb = mixed.x[index]
            yb = mixed.y[index]
            mu_init = mixed.mu_init[index].detach()
            rec_init = mixed.rec_init[index].detach()
            optimizer.zero_grad(set_to_none=True)
            loss, terms = objective(autoencoder, xb, yb, centroids, mu_init, rec_init)
            loss.backward()
            if cfg.max_grad_norm > 0:
                # Optional numerical safeguard: it rescales the local update but
                # leaves the training objective untouched.
                torch.nn.utils.clip_grad_norm_(autoencoder.parameters(),
                                               cfg.max_grad_norm)
            optimizer.step()
            totals = LossTerms(**{k: getattr(totals, k) + getattr(terms, k)
                                  for k in LossTerms.__dataclass_fields__})
            n_batches += 1
        autoencoder.eval()
        if n_batches:
            totals = LossTerms(**{k: getattr(totals, k) / n_batches
                                  for k in LossTerms.__dataclass_fields__})
        return totals


@torch.no_grad()
def cached_anchor_targets(cfg: Config, anchor: AutoEncoder, x: torch.Tensor):
    """Task-anchor outputs for the client's real data, reused across the rounds."""
    return anchor_forward(anchor, x)


def local_update(cfg: Config, global_state: State, anchor: AutoEncoder,
                 autoencoder: AutoEncoder, worker: ClientWorker, x: torch.Tensor,
                 y: torch.Tensor, bank: LatentBank, centroids: CentroidMatrix,
                 task_classes: Sequence[int], rng_seed: int,
                 real_anchor: Optional[tuple] = None):
    """Run one local update and return ``(d, terms)`` with ``d = W_{t,r-1} - M_cj``.

    A ``None`` difference signals that the client contributed no usable batch, in
    which case the server treats the upload as a zero difference.
    """
    if int(x.shape[0]) == 0:
        return None, LossTerms()

    # 1. local model starts from the current global model
    from ..models import unpack_state
    unpack_state(global_state, autoencoder.enc, autoencoder.dec)

    # 2. mixed set: real data + enabled replay channels, with anchor targets
    mixed = build_mixed_set(cfg, anchor, x, y, bank, centroids.buffer, task_classes,
                            x.device, real_anchor=real_anchor)
    batches = mixed.batches(cfg, rng_seed)
    if not batches:
        return None, LossTerms()

    # 3. balanced minibatch traversal of D_mix for I local epochs
    terms = worker.train(mixed, centroids, batches)

    # 4. upload the model difference
    local_state = autoencoder.state(x.device)
    delta = {k: global_state[k] - local_state[k] for k in global_state}
    return delta, terms
