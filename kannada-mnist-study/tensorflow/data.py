"""Kannada-MNIST tf.data pipeline from external NPZ files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf


def _load_one(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if len(archive.files) != 1:
            raise ValueError(f"expected one array in {path}, found {archive.files}")
        return np.asarray(archive[archive.files[0]])


def load_arrays(data_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    if split not in {"train", "test"}:
        raise ValueError(f"unknown split: {split}")
    images = _load_one(data_dir / f"X_kannada_MNIST_{split}.npz")
    labels = _load_one(data_dir / f"y_kannada_MNIST_{split}.npz")
    if images.ndim != 3 or images.shape[1:] != (28, 28):
        raise ValueError(f"expected (N, 28, 28), got {images.shape}")
    if labels.shape != (images.shape[0],):
        raise ValueError(f"images/labels mismatch: {images.shape}, {labels.shape}")
    x = images.astype(np.float32)[..., None] / 255.0
    y = labels.astype(np.int64, copy=False)
    return x, y


def make_datasets(
    data_dir: Path, *, batch_size: int, seed: int
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    x_train, y_train = load_arrays(data_dir, "train")
    x_test, y_test = load_arrays(data_dir, "test")

    train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train = train.shuffle(len(y_train), seed=seed, reshuffle_each_iteration=True)
    train = train.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test = test.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return train, test
