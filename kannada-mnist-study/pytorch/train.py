"""Custom PyTorch training-loop scaffold for Kannada-MNIST."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from data import make_loaders
from model import SmallMLP


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    _ = (model, loader, loss_fn, optimizer, device)
    raise NotImplementedError(
        "Exercise P1: implement train mode, device copies, zero_grad, backward, step, metrics"
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    _ = (model, loader, loss_fn, device)
    raise NotImplementedError(
        "Exercise P2: implement eval mode and sample-weighted loss/accuracy"
    )


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is false")
    return device


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
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.data_dir is None:
        raise SystemExit("provide --data-dir or set KANNADA_MNIST_DATA_DIR")

    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    train_loader, test_loader = make_loaders(
        Path(args.data_dir), batch_size=args.batch_size, seed=args.seed
    )
    model = SmallMLP().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    _ = (args.epochs, train_loader, test_loader, model, loss_fn, optimizer)
    raise NotImplementedError(
        "Exercise P3: call train_one_epoch/evaluate per epoch and print comparable metrics"
    )


if __name__ == "__main__":
    main()
