"""Local and global replay channels.

Both channels map latent codes through the frozen task-start decoder ``g_init``,
so within a task the replay mapping does not move while the trainable decoder is
being updated.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from ..memory import LatentBank
from ..models import AutoEncoder


def _empty(x_ref: torch.Tensor, device: torch.device):
    return (torch.empty(0, device=device),
            torch.empty(0, dtype=torch.long, device=device))


@torch.no_grad()
def local_replay_samples(anchor: AutoEncoder, bank: LatentBank,
                         device: torch.device, chunk: int = 4096):
    """``x_hat = g_init(m)`` for every stored code ``m`` in ``E_cj``."""
    codes = bank.codes()
    if codes is None:
        return _empty(None, device)
    labels = bank.labels().to(device)
    imgs = torch.cat([anchor.dec(codes[i:i + chunk].to(device))
                      for i in range(0, codes.shape[0], chunk)])
    return imgs, labels


@torch.no_grad()
def global_replay_samples(anchor: AutoEncoder, centroid_matrix: torch.Tensor,
                          class_ids: Sequence[int], per_class: int, sigma: float,
                          device: torch.device,
                          generator: Optional[torch.Generator] = None):
    """``x_hat = g_init(p_{t,k} + eps)``, ``eps ~ N(0, sigma_syn^2 I)``, per current class."""
    cls = torch.tensor(sorted(int(c) for c in class_ids), dtype=torch.long, device=device)
    if cls.numel() == 0 or per_class <= 0:
        return _empty(None, device)
    p = centroid_matrix[cls]
    z = p.repeat_interleave(int(per_class), dim=0)
    z = z + torch.randn(z.shape, device=device, generator=generator) * float(sigma)
    imgs = anchor.dec(z)
    labels = cls.repeat_interleave(int(per_class))
    return imgs, labels
