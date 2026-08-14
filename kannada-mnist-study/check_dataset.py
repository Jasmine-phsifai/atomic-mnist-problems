"""Validate an external Kannada-MNIST NPZ directory before training."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


FILES = {
    "train": ("X_kannada_MNIST_train.npz", "y_kannada_MNIST_train.npz", 60_000),
    "test": ("X_kannada_MNIST_test.npz", "y_kannada_MNIST_test.npz", 10_000),
}


@dataclass(frozen=True)
class SplitSummary:
    name: str
    samples: int
    image_dtype: str
    label_dtype: str
    pixel_min: float
    pixel_max: float
    labels: tuple[int, ...]


def load_single_array(path: Path) -> np.ndarray:
    """Load the single array stored in an official Kannada-MNIST NPZ file."""
    if not path.is_file():
        raise FileNotFoundError(f"missing required file: {path}")

    with np.load(path, allow_pickle=False) as archive:
        if len(archive.files) != 1:
            raise ValueError(
                f"expected exactly one array in {path.name}, found {archive.files}"
            )
        return np.asarray(archive[archive.files[0]])


def validate_split(
    data_dir: Path, split: str, *, enforce_official_count: bool = True
) -> SplitSummary:
    image_name, label_name, expected_count = FILES[split]
    images = load_single_array(data_dir / image_name)
    labels = load_single_array(data_dir / label_name)

    if images.ndim != 3 or images.shape[1:] != (28, 28):
        raise ValueError(f"{image_name}: expected (N, 28, 28), got {images.shape}")
    if labels.ndim != 1:
        raise ValueError(f"{label_name}: expected (N,), got {labels.shape}")
    if images.shape[0] != labels.shape[0]:
        raise ValueError(
            f"{split}: image count {images.shape[0]} != label count {labels.shape[0]}"
        )
    if enforce_official_count and images.shape[0] != expected_count:
        raise ValueError(
            f"{split}: expected official count {expected_count}, got {images.shape[0]}"
        )
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError(f"{label_name}: labels must be integers, got {labels.dtype}")

    unique_labels = tuple(int(value) for value in np.unique(labels))
    if unique_labels != tuple(range(10)):
        raise ValueError(f"{label_name}: expected labels 0..9, got {unique_labels}")

    pixel_min = float(images.min())
    pixel_max = float(images.max())
    if pixel_min < 0 or pixel_max > 255:
        raise ValueError(
            f"{image_name}: expected pixel range within 0..255, got "
            f"{pixel_min}..{pixel_max}"
        )

    return SplitSummary(
        name=split,
        samples=int(images.shape[0]),
        image_dtype=str(images.dtype),
        label_dtype=str(labels.dtype),
        pixel_min=pixel_min,
        pixel_max=pixel_max,
        labels=unique_labels,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="directory containing the six official NPZ files; defaults to "
        "KANNADA_MNIST_DATA_DIR",
    )
    parser.add_argument(
        "--skip-count-check",
        action="store_true",
        help="validate format without requiring exactly 60,000/10,000 samples",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_data_dir = args.data_dir or os.environ.get("KANNADA_MNIST_DATA_DIR")
    if not raw_data_dir:
        raise SystemExit(
            "provide --data-dir or set the KANNADA_MNIST_DATA_DIR environment variable"
        )

    data_dir = Path(raw_data_dir).expanduser().resolve()
    if not data_dir.is_dir():
        raise SystemExit(f"data directory does not exist: {data_dir}")

    print(f"data_dir: {data_dir}")
    for split in ("train", "test"):
        summary = validate_split(
            data_dir, split, enforce_official_count=not args.skip_count_check
        )
        print(
            f"{summary.name:>5}: {summary.samples:>6} samples | "
            f"images={summary.image_dtype} {summary.pixel_min:g}..{summary.pixel_max:g} | "
            f"labels={summary.label_dtype} {summary.labels}"
        )
    print("Kannada-MNIST data contract: OK")


if __name__ == "__main__":
    main()
