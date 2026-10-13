"""Tiny-ImageNet-200 loader.

The archive is read directly from ``<root>/tiny-imagenet-200`` (extracted on
demand from ``tiny-imagenet-200.zip``). Images are resized to 64x64 RGB, and the
validation split -- mapped to classes through ``val_annotations.txt`` -- serves
as the test set.
"""

from __future__ import annotations

import os
import zipfile
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

SIZE = 64


def _read_images(paths: List[str], labels: List[int], out_x: List[np.ndarray],
                 out_y: List[int]) -> None:
    for path, label in zip(paths, labels):
        with Image.open(path) as im:
            out_x.append(np.asarray(im.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)))
        out_y.append(label)


def _train_split(train_dir: str) -> Tuple[List[str], List[int], Dict[str, int]]:
    class_dirs = sorted(d for d in os.listdir(train_dir)
                        if os.path.isdir(os.path.join(train_dir, d)))
    class_to_idx = {c: i for i, c in enumerate(class_dirs)}
    paths: List[str] = []
    labels: List[int] = []
    for cls in class_dirs:
        img_dir = os.path.join(train_dir, cls, "images")
        if not os.path.isdir(img_dir):
            continue
        for fn in sorted(os.listdir(img_dir)):
            if fn.lower().endswith((".jpeg", ".jpg", ".png")):
                paths.append(os.path.join(img_dir, fn))
                labels.append(class_to_idx[cls])
    return paths, labels, class_to_idx


def _val_split(val_dir: str, class_to_idx: Dict[str, int]) -> Tuple[List[str], List[int]]:
    ann = os.path.join(val_dir, "val_annotations.txt")
    per_class = defaultdict(list)
    with open(ann, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2 or parts[1] not in class_to_idx:
                continue
            per_class[class_to_idx[parts[1]]].append(parts[0])
    img_dir = os.path.join(val_dir, "images")
    paths, labels = [], []
    for idx in sorted(per_class):
        for fn in per_class[idx]:
            paths.append(os.path.join(img_dir, fn))
            labels.append(idx)
    return paths, labels


def load_tinyimagenet(root: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Return ``(train_x, train_y, test_x, test_y, num_classes)``."""
    base = os.path.join(root, "tiny-imagenet-200")
    if not os.path.isdir(base):
        zip_path = os.path.join(root, "tiny-imagenet-200.zip")
        if not os.path.exists(zip_path):
            raise FileNotFoundError(
                "Tiny-ImageNet not found. Download tiny-imagenet-200.zip from "
                "https://cs231n.stanford.edu/tiny-imagenet-200.zip into "
                f"{root} or pre-extract it to {base}.")
        os.makedirs(root, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(root)

    train_dir = os.path.join(base, "train")
    paths, labels, class_to_idx = _train_split(train_dir)
    xs: List[np.ndarray] = []
    ys: List[int] = []
    _read_images(paths, labels, xs, ys)
    train_x = np.stack(xs) if xs else np.empty((0, SIZE, SIZE, 3), dtype=np.uint8)
    train_y = np.asarray(ys, dtype=np.int64)

    vpaths, vlabels = _val_split(os.path.join(base, "val"), class_to_idx)
    vxs: List[np.ndarray] = []
    vys: List[int] = []
    _read_images(vpaths, vlabels, vxs, vys)
    test_x = np.stack(vxs) if vxs else np.empty((0, SIZE, SIZE, 3), dtype=np.uint8)
    test_y = np.asarray(vys, dtype=np.int64)
    return train_x, train_y, test_x, test_y, len(class_to_idx)
