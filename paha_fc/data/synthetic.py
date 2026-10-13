"""Small deterministic synthetic benchmark.

Used by the unit tests and by ``--dataset synthetic`` for quick end-to-end
checks of the full pipeline on CPU. Classes are noisy colour blobs with a
per-class prototype, which is enough to exercise replay, centroid alignment and
nearest-centroid read-out without downloading any data.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .loaders import DatasetBundle

SIZE = 16


def make_synthetic_bundle(num_classes: int = 12, per_class_train: int = 24,
                          per_class_test: int = 8, size: int = SIZE,
                          seed: int = 0) -> DatasetBundle:
    rng = np.random.RandomState(seed)
    prototypes = rng.uniform(0.1, 0.9, size=(num_classes, 3))

    def _block(n: int) -> Tuple[np.ndarray, np.ndarray]:
        xs = np.empty((n * num_classes, size, size, 3), dtype=np.float32)
        ys = np.empty(n * num_classes, dtype=np.int64)
        for c in range(num_classes):
            sl = slice(c * n, (c + 1) * n)
            base = prototypes[c][None, None, None, :]
            noise = rng.normal(0.0, 0.08, size=(n, size, size, 3)).astype(np.float32)
            # A per-class spatial ramp keeps the images distinguishable.
            ramp = np.linspace(0.0, 1.0, size, dtype=np.float32)[None, :, None, None] * (c + 1) / num_classes
            xs[sl] = np.clip(base + noise + ramp, 0.0, 1.0)
            ys[sl] = c
        return (xs * 255.0).round().clip(0, 255).astype(np.uint8), ys

    train_x, train_y = _block(per_class_train)
    test_x, test_y = _block(per_class_test)
    order_t = np.random.RandomState(seed + 1).permutation(len(train_y))
    order_v = np.random.RandomState(seed + 2).permutation(len(test_y))
    return DatasetBundle(train_x[order_t], train_y[order_t], test_x[order_v],
                         test_y[order_v], num_classes, int(size))
