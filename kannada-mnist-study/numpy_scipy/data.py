"""Small, explicit NPZ loader for the NumPy/SciPy implementation."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _load_one(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if len(archive.files) != 1:
            raise ValueError(f"expected one array in {path}, found {archive.files}")
        return np.asarray(archive[archive.files[0]])


def load_split(data_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    if split not in {"train", "test"}:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    images = _load_one(data_dir / f"X_kannada_MNIST_{split}.npz")
    labels = _load_one(data_dir / f"y_kannada_MNIST_{split}.npz")
    if images.ndim != 3 or images.shape[1:] != (28, 28):
        raise ValueError(f"expected images shaped (N, 28, 28), got {images.shape}")
    if labels.shape != (images.shape[0],):
        raise ValueError(f"images/labels mismatch: {images.shape}, {labels.shape}")

    x = images.astype(np.float32).reshape(-1, 28 * 28) / 255.0
    y = labels.astype(np.int64, copy=False)
    return x, y
