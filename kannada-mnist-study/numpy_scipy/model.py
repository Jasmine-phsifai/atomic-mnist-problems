"""Two-layer MLP exercises implemented only with NumPy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Parameters:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray


def initialize_parameters(
    rng: np.random.Generator,
    input_dim: int = 28 * 28,
    hidden_dim: int = 128,
    classes: int = 10,
) -> Parameters:
    """Create float32 parameters with He initialization for the hidden layer."""
    w1 = rng.normal(0.0, np.sqrt(2.0 / input_dim), (input_dim, hidden_dim))
    w2 = rng.normal(0.0, np.sqrt(2.0 / hidden_dim), (hidden_dim, classes))
    return Parameters(
        w1=w1.astype(np.float32),
        b1=np.zeros(hidden_dim, dtype=np.float32),
        w2=w2.astype(np.float32),
        b2=np.zeros(classes, dtype=np.float32),
    )


def relu(x: np.ndarray) -> np.ndarray:
    raise NotImplementedError("Exercise N1: implement ReLU with NumPy")


def log_softmax(logits: np.ndarray) -> np.ndarray:
    raise NotImplementedError(
        "Exercise N2: implement numerically stable log-softmax without SciPy"
    )


def forward(
    x: np.ndarray, parameters: Parameters
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    raise NotImplementedError(
        "Exercise N3: implement affine -> ReLU -> affine and return a backward cache"
    )


def cross_entropy(logits: np.ndarray, labels: np.ndarray) -> float:
    raise NotImplementedError("Exercise N4: implement mean sparse cross-entropy")


def backward(
    labels: np.ndarray,
    parameters: Parameters,
    cache: dict[str, np.ndarray],
) -> Parameters:
    raise NotImplementedError("Exercise N5: derive and implement all four gradients")


def sgd_step(parameters: Parameters, gradients: Parameters, learning_rate: float) -> None:
    raise NotImplementedError("Exercise N6: update parameters in place with SGD")
