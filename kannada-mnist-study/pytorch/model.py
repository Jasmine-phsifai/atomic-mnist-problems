"""The same two-layer MLP used by the handwritten NumPy implementation."""

from __future__ import annotations

import torch
from torch import nn


class SmallMLP(nn.Module):
    def __init__(self, classes: int = 10) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.layers(images)
