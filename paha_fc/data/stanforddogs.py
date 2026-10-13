"""Stanford Dogs loader.

Two layouts are supported. When the installed torchvision still ships
``datasets.StanfordDogs`` it is used directly; otherwise the public mirror
layout ``<root>/[<subset>/]Images/{train,test}/<breed>/*.jpg`` is scanned from
disk. Images are decoded to a fixed resolution so that they can be stacked, and
class ids come from the sorted union of breed names.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

SIZE = 64
_SUBDIR_CANDIDATES = ("", "stanford_dogs", "stanford-dogs-dataset", "stanford_dogs-dataset",
                      "StanfordDogs", "Stanford Dogs")


def _find_images_root(root: str) -> str:
    for sub in _SUBDIR_CANDIDATES:
        candidate = os.path.join(root, sub) if sub else root
        for name in ("Images", "images"):
            images_dir = os.path.join(candidate, name)
            if os.path.isdir(os.path.join(images_dir, "train")) and \
               os.path.isdir(os.path.join(images_dir, "test")):
                return images_dir
    raise FileNotFoundError(
        f"Stanford Dogs not found under {root}. Expected the mirror layout "
        "<root>/Images/{train,test}/<breed>/*.jpg "
        "(https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset).")


def _scan_split(images_root: str, split: str, class_to_idx: Dict[str, int],
                fill: bool) -> Tuple[List[np.ndarray], List[int]]:
    split_dir = os.path.join(images_root, split)
    xs: List[np.ndarray] = []
    ys: List[int] = []
    for breed in sorted(os.listdir(split_dir)):
        breed_dir = os.path.join(split_dir, breed)
        if not os.path.isdir(breed_dir):
            continue
        if fill and breed not in class_to_idx:
            class_to_idx[breed] = len(class_to_idx)
        if breed not in class_to_idx:
            continue
        for fn in sorted(os.listdir(breed_dir)):
            if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            with Image.open(os.path.join(breed_dir, fn)) as im:
                xs.append(np.asarray(im.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)))
            ys.append(class_to_idx[breed])
    return xs, ys


def _load_from_torchvision(root: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    from torchvision import datasets

    def _decode(dataset) -> Tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        for img, label in dataset:
            if hasattr(img, "size"):                      # PIL image
                arr = np.asarray(img.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR))
            else:                                         # already a tensor
                arr = (img.permute(1, 2, 0).mul(255).round().clamp(0, 255)
                       .byte().numpy() if img.dtype.is_floating_point
                       else img.permute(1, 2, 0).numpy())
            xs.append(arr)
            ys.append(int(label))
        return np.stack(xs), np.asarray(ys, dtype=np.int64)

    train = datasets.StanfordDogs(root=root, split="train", download=False)
    test = datasets.StanfordDogs(root=root, split="test", download=False)
    train_x, train_y = _decode(train)
    test_x, test_y = _decode(test)
    return train_x, train_y, test_x, test_y, int(max(train_y.max(), test_y.max())) + 1


def load_stanford_dogs(root: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Return ``(train_x, train_y, test_x, test_y, num_classes)`` at 64 px."""
    from torchvision import datasets
    if hasattr(datasets, "StanfordDogs"):
        try:
            return _load_from_torchvision(root)
        except Exception:
            pass  # fall back to the on-disk mirror layout

    images_root = _find_images_root(root)
    class_to_idx: Dict[str, int] = {}
    train_xs, train_ys = _scan_split(images_root, "train", class_to_idx, fill=True)
    test_xs, test_ys = _scan_split(images_root, "test", class_to_idx, fill=False)
    if not train_xs:
        raise FileNotFoundError(f"no training images found under {images_root}/train")
    return (np.stack(train_xs), np.asarray(train_ys, dtype=np.int64),
            np.stack(test_xs), np.asarray(test_ys, dtype=np.int64), len(class_to_idx))
