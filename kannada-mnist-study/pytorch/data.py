"""Kannada-MNIST Dataset/DataLoader without torchvision."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def _load_one(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if len(archive.files) != 1:
            raise ValueError(f"expected one array in {path}, found {archive.files}")
        return np.asarray(archive[archive.files[0]])


class KannadaNpzDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, data_dir: Path, split: str) -> None:
        if split not in {"train", "test"}:
            raise ValueError(f"unknown split: {split}")
        images = _load_one(data_dir / f"X_kannada_MNIST_{split}.npz")
        labels = _load_one(data_dir / f"y_kannada_MNIST_{split}.npz")
        if images.ndim != 3 or images.shape[1:] != (28, 28):
            raise ValueError(f"expected (N, 28, 28), got {images.shape}")
        if labels.shape != (images.shape[0],):
            raise ValueError(f"images/labels mismatch: {images.shape}, {labels.shape}")

        self.images = torch.from_numpy(images.astype(np.float32)[:, None]) / 255.0
        self.labels = torch.from_numpy(labels.astype(np.int64, copy=False))

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[index], self.labels[index]


def make_loaders(
    data_dir: Path,
    *,
    batch_size: int,
    seed: int,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        KannadaNpzDataset(data_dir, "train"),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        KannadaNpzDataset(data_dir, "test"),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, test_loader
