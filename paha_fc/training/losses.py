"""Composite encoder-decoder objective of local training.

``L(B) = L_rec(B) + lambda_enc L_enc(B) + lambda_dec L_dec(B)`` where ``L_rec``
combines the decoder likelihood, the analytic Gaussian KL term and the attraction
toward the labelled class centroid, and the two distillation terms keep the mean
embedding and the reconstruction close to the frozen task-start model. All three
terms are minibatch averages; both distillation terms read mean embeddings, so
neither injects posterior-sampling noise.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Tuple

import torch

from ..centroids import CentroidMatrix
from ..config import Config
from ..models import AutoEncoder, gaussian_kl, reparameterize


@dataclass
class LossTerms:
    total: float = 0.0
    rec: float = 0.0
    kl: float = 0.0
    cluster: float = 0.0
    enc: float = 0.0
    dec: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def _squared_error(pred: torch.Tensor, target: torch.Tensor, reduction: str) -> torch.Tensor:
    """Per-sample squared error, pixel-summed for the Gaussian likelihood."""
    per = (pred - target).flatten(1).pow(2).mean(dim=1)
    return per * float(target[0].numel()) if reduction == "sum" else per


class CompositeLoss:
    """Callable objective with per-term diagnostics for logging."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def __call__(self, autoencoder: AutoEncoder, xb: torch.Tensor, yb: torch.Tensor,
                 centroids: CentroidMatrix, mu_init: torch.Tensor,
                 rec_init: torch.Tensor) -> Tuple[torch.Tensor, LossTerms]:
        cfg = self.cfg
        mu, log_s2 = autoencoder.enc(xb)
        rec = autoencoder.dec(reparameterize(mu, log_s2))

        recon = _squared_error(rec, xb, cfg.mse_reduction)
        kl = cfg.kl_weight * gaussian_kl(mu, log_s2)

        targets = centroids.targets(yb)
        covered = centroids.covered.to(yb.device)[yb]
        cluster = (mu - targets).flatten(1).pow(2).sum(dim=1)
        cluster = cluster[covered] if bool(covered.any()) else cluster.sum() * 0.0

        l_rec = (recon + kl + cfg.lam_cluster * cluster).mean()

        # Drift of the mean embedding and of the reconstruction relative to the
        # task anchor.
        l_enc = (mu - mu_init).flatten(1).pow(2).sum(dim=1).mean()
        rec_from_mean = autoencoder.dec(mu)
        l_dec = _squared_error(rec_from_mean, rec_init, cfg.mse_reduction).mean()

        total = cfg.lam_rec * l_rec + cfg.lam_enc * l_enc + cfg.lam_dec * l_dec
        terms = LossTerms(total=float(total.detach()),
                          rec=float(recon.detach().mean()),
                          kl=float(kl.detach().mean()),
                          cluster=float(cluster.detach().mean()),
                          enc=float(l_enc.detach()),
                          dec=float(l_dec.detach()))
        return total, terms


def composite_loss(cfg: Config, autoencoder: AutoEncoder, xb: torch.Tensor, yb: torch.Tensor,
                   centroids: CentroidMatrix, mu_init: torch.Tensor,
                   rec_init: torch.Tensor) -> torch.Tensor:
    """Plain objective without diagnostics."""
    return CompositeLoss(cfg)(autoencoder, xb, yb, centroids, mu_init, rec_init)[0]
