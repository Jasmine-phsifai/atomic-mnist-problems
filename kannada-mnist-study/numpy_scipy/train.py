"""Training-loop scaffold for the handwritten NumPy/SciPy baseline."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from data import load_split
from model import initialize_parameters


def train(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    _ = initialize_parameters(rng)
    _ = (x_train, y_train, epochs, batch_size, learning_rate)
    raise NotImplementedError(
        "Exercise N8: write shuffle -> minibatch -> forward -> backward -> SGD -> metrics"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=os.environ.get("KANNADA_MNIST_DATA_DIR"),
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.data_dir is None:
        raise SystemExit("provide --data-dir or set KANNADA_MNIST_DATA_DIR")
    x_train, y_train = load_split(Path(args.data_dir), "train")
    train(
        x_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
