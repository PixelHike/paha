"""Latent exemplar bank ``E_cj``.

The bank stores mean embeddings ``m = mu_phi(x)`` together with their class
labels; raw images of past tasks are never kept. Two operations touch it at a
task boundary: the old-code adaptation through the frozen anchor decoder, and the
enqueue of new-class codes. A per-class capacity keeps the bank compact, as
required by the storage budget of the local channel.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import torch


class LatentBank:
    """Per-client FIFO bank of ``(mean embedding, class label)`` pairs."""

    def __init__(self, per_class_capacity: int = 10):
        self.capacity = int(per_class_capacity) if per_class_capacity else 0
        self._by_class: Dict[int, List[torch.Tensor]] = {}

    # ------------------------------------------------------------- reading
    def __len__(self) -> int:
        return sum(len(v) for v in self._by_class.values())

    def classes(self) -> List[int]:
        return sorted(self._by_class)

    def is_empty(self) -> bool:
        return len(self) == 0

    def codes(self) -> Optional[torch.Tensor]:
        """Stacked codes ``(n, m)`` on CPU, or ``None`` when the bank is empty."""
        flat = [c for items in self._by_class.values() for c in items]
        return torch.stack(flat) if flat else None

    def labels(self) -> Optional[torch.Tensor]:
        pairs = [(c, lab) for lab, items in self._by_class.items() for c in items]
        if not pairs:
            return None
        return torch.tensor([lab for _, lab in pairs], dtype=torch.long)

    def items(self) -> List[Tuple[torch.Tensor, int]]:
        return [(code, lab) for lab, items in self._by_class.items() for code in items]

    # ------------------------------------------------------------- writing
    def append(self, code: torch.Tensor, label: int) -> None:
        label = int(label)
        self._by_class.setdefault(label, []).append(code.detach().to("cpu").clone())
        if self.capacity:
            del self._by_class[label][:-self.capacity]

    def extend(self, codes: torch.Tensor, labels: Iterable[int]) -> None:
        labels = [int(l) for l in labels]
        if codes.shape[0] != len(labels):
            raise ValueError("codes and labels must have matching length")
        for i, lab in enumerate(labels):
            self.append(codes[i], lab)

    def replace_class(self, label: int, codes: torch.Tensor) -> None:
        """Overwrite all codes of one class, used by the old-code adaptation."""
        label = int(label)
        items = [codes[i].detach().to("cpu").clone() for i in range(codes.shape[0])]
        if self.capacity:
            items = items[-self.capacity:]
        if items:
            self._by_class[label] = items
        else:
            self._by_class.pop(label, None)

    def replace_all(self, codes: torch.Tensor, labels: Iterable[int]) -> None:
        labels = [int(l) for l in labels]
        self._by_class = {}
        self.extend(codes, labels)
