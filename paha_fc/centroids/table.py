"""Cumulative centroid set ``P`` used by clustering, replay and read-out."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import torch

CentroidTable = Dict[int, torch.Tensor]


class CentroidMatrix:
    """Dense ``(num_classes, latent_dim)`` view of the centroid set, by class id.

    The dense form lets the clustering term fetch ``p_y`` for a whole minibatch
    with one gather, while ``as_dict`` exposes the sparse set the alignment step
    and the classifier work with.
    """

    def __init__(self, num_classes: int, latent_dim: int, device: torch.device):
        self.num_classes = int(num_classes)
        self.latent_dim = int(latent_dim)
        self.device = device
        self.buffer = torch.zeros(self.num_classes, self.latent_dim, device=device)
        self.covered = torch.zeros(self.num_classes, dtype=torch.bool, device=device)

    # --------------------------------------------------------------- update
    def update(self, table: CentroidTable) -> None:
        if not table:
            return
        keys = torch.tensor([int(k) for k in table], dtype=torch.long, device=self.device)
        values = torch.stack([torch.as_tensor(table[k]).to(self.device).view(-1)
                             for k in sorted(table, key=lambda k: int(k))])
        if values.shape[1] != self.latent_dim:
            raise ValueError(f"centroid dim {values.shape[1]} != {self.latent_dim}")
        self.buffer[keys] = values
        self.covered[keys] = True

    # --------------------------------------------------------------- access
    def __getitem__(self, class_id: int) -> torch.Tensor:
        return self.buffer[int(class_id)]

    def targets(self, labels: torch.Tensor) -> torch.Tensor:
        """``p_y`` for every element of a minibatch, the clustering term target."""
        return self.buffer[labels]

    def missing(self, classes: Iterable[int]) -> List[int]:
        cls = [int(c) for c in classes]
        if not cls:
            return []
        idx = torch.tensor(cls, dtype=torch.long, device=self.device)
        return [c for c, ok in zip(cls, self.covered[idx].tolist()) if not ok]

    def as_dict(self, classes: Optional[Sequence[int]] = None) -> CentroidTable:
        ids = range(self.num_classes) if classes is None else [int(c) for c in classes]
        return {int(c): self.buffer[int(c)].detach().cpu().clone() for c in ids
                if bool(self.covered[int(c)])}

    def norm_matrix(self, classes: Sequence[int]) -> torch.Tensor:
        return self.buffer[[int(c) for c in classes]]
