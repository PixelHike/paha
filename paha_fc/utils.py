"""Seeding and small numeric helpers."""

from __future__ import annotations

import os
import random
from typing import Dict

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed every generator a run touches."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not deterministic


def resolve_device(requested: str = "cuda") -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def enable_tf32(enabled: bool = True) -> None:
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled


def state_norm(state: Dict[str, torch.Tensor]) -> float:
    return float(sum(float(v.pow(2).sum()) for v in state.values()) ** 0.5)
