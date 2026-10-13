"""Dataset entry points shared by every run.

Loaders return uint8 images in HWC layout plus integer labels; conversion to the
network layout (CHW float in [-1, 1], matching the tanh decoder output) happens
once, in :func:`to_device_tensor`, before the tensors are cached on device.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from . import tinyimagenet


@dataclass
class DatasetBundle:
    train_x: np.ndarray          # (n, H, W, 3) uint8
    train_y: np.ndarray          # (n,) int64, 0-based class ids
    test_x: np.ndarray
    test_y: np.ndarray
    num_classes: int
    input_size: int

    def summary(self) -> str:
        return (f"{self.num_classes} classes, train {self.train_x.shape}, "
                f"test {self.test_x.shape}, {self.input_size}px")


def _resize(images: np.ndarray, size: int) -> np.ndarray:
    if images.shape[1] == size and images.shape[2] == size:
        return images
    xt = torch.from_numpy(images).permute(0, 3, 1, 2).float()
    xt = torch.nn.functional.interpolate(xt, size=(size, size), mode="bilinear",
                                         align_corners=False)
    return xt.permute(0, 2, 3, 1).round().clamp(0, 255).to(torch.uint8).numpy()


def _load_cifar100(root: str) -> DatasetBundle:
    from torchvision import datasets
    npz_dir = os.path.join(root, "cifar100_npz")
    tr_path = os.path.join(npz_dir, "cifar100_train.npz")
    te_path = os.path.join(npz_dir, "cifar100_test.npz")
    if os.path.exists(tr_path) and os.path.exists(te_path):
        tr = np.load(tr_path)
        te = np.load(te_path)
        train_x, train_y = tr["data"].reshape(-1, 3, 32, 32), tr["labels"]
        test_x, test_y = te["data"].reshape(-1, 3, 32, 32), te["labels"]
        train_x = np.transpose(train_x, (0, 2, 3, 1))
        test_x = np.transpose(test_x, (0, 2, 3, 1))
    else:
        os.makedirs(root, exist_ok=True)
        tr = datasets.CIFAR100(root=root, train=True, download=True)
        te = datasets.CIFAR100(root=root, train=False, download=True)
        train_x, train_y = tr.data, np.asarray(tr.targets)
        test_x, test_y = te.data, np.asarray(te.targets)
    return DatasetBundle(_as_uint8(train_x), _as_int(train_y), _as_uint8(test_x),
                         _as_int(test_y), 100, 32)


def _load_tinyimagenet(root: str) -> DatasetBundle:
    train_x, train_y, test_x, test_y, num_classes = tinyimagenet.load_tinyimagenet(root)
    return DatasetBundle(train_x, train_y, test_x, test_y, num_classes, 64)


def _load_stanford_dogs(root: str) -> DatasetBundle:
    from .stanforddogs import load_stanford_dogs
    train_x, train_y, test_x, test_y, num_classes = load_stanford_dogs(root)
    return DatasetBundle(train_x, train_y, test_x, test_y, num_classes,
                         int(train_x.shape[1]))


def _as_uint8(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    return x.astype(np.uint8) if x.dtype != np.uint8 else x


def _as_int(y: np.ndarray) -> np.ndarray:
    return np.asarray(y).astype(np.int64)


def load_dataset(name: str, root: str, resize: Optional[int] = None) -> DatasetBundle:
    """Load a benchmark, optionally resizing every image to ``resize``."""
    key = name.lower().replace("-", "").replace("_", "")
    if key in ("cifar100", "cifar100py"):
        bundle = _load_cifar100(root)
    elif key in ("tinyimagenet", "tiny"):
        bundle = _load_tinyimagenet(root)
    elif key in ("stanforddogs", "dogs"):
        bundle = _load_stanford_dogs(root)
    elif key == "synthetic":
        from .synthetic import make_synthetic_bundle
        bundle = make_synthetic_bundle()
    else:
        raise ValueError(f"unknown dataset {name!r}")

    size = resize if resize is not None else bundle.input_size
    if size != bundle.input_size:
        bundle.train_x = _resize(bundle.train_x, size)
        bundle.test_x = _resize(bundle.test_x, size)
        bundle.input_size = size
    return bundle


def to_device_tensor(images: np.ndarray, device: torch.device) -> torch.Tensor:
    """HWC uint8 -> CHW float32 in [-1, 1] on ``device``."""
    x = torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2)
    return x.float().div_(127.5).sub_(1.0).to(device)


def labels_to_tensor(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(_as_int(labels)).to(device)
