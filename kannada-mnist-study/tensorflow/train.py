"""Custom TensorFlow GradientTape training-loop scaffold for Kannada-MNIST."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import tensorflow as tf

from data import make_datasets
from model import SmallMLP


def train_step(
    model: tf.keras.Model,
    images: tf.Tensor,
    labels: tf.Tensor,
    loss_fn: tf.keras.losses.Loss,
    optimizer: tf.keras.optimizers.Optimizer,
) -> tuple[tf.Tensor, tf.Tensor]:
    _ = (model, images, labels, loss_fn, optimizer)
    raise NotImplementedError(
        "Exercise T1: implement GradientTape, gradients, apply_gradients, and correct count"
    )


def evaluate(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    loss_fn: tf.keras.losses.Loss,
) -> tuple[float, float]:
    _ = (model, dataset, loss_fn)
    raise NotImplementedError(
        "Exercise T2: run with training=False and aggregate sample-weighted metrics"
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
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.data_dir is None:
        raise SystemExit("provide --data-dir or set KANNADA_MNIST_DATA_DIR")

    tf.keras.utils.set_random_seed(args.seed)
    train_data, test_data = make_datasets(
        Path(args.data_dir), batch_size=args.batch_size, seed=args.seed
    )
    model = SmallMLP()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    _ = (args.epochs, train_data, test_data, model, loss_fn, optimizer)
    raise NotImplementedError(
        "Exercise T3: call train_step/evaluate per epoch and print comparable metrics"
    )


if __name__ == "__main__":
    main()
