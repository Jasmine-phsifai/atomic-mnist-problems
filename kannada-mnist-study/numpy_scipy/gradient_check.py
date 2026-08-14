"""Use SciPy as an independent oracle for the handwritten backward pass."""

from __future__ import annotations

from scipy.optimize import check_grad


def main() -> None:
    # `check_grad(func, grad, x0)` expects one flat float64 parameter vector.
    # Keep the check tiny: for example 3 samples, 4 input features, 5 hidden
    # units, and 3 classes. Implement pack/unpack helpers here only after the
    # forward and backward exercises in model.py are complete.
    _ = check_grad
    raise NotImplementedError(
        "Exercise N7: flatten parameters and compare analytical vs numerical gradients"
    )


if __name__ == "__main__":
    main()
