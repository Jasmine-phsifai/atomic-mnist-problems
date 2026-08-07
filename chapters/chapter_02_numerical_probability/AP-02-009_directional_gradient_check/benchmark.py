"""Diagnostic benchmark for AP-02-009: A Random Direction Checks the Whole Gradient.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import numpy as np

from starter import directional_gradient_check


def main() -> None:
    """Compare correct/corrupted gradient error distributions. / 比较正确与损坏梯度的误差分布。"""
    rng = np.random.default_rng(9)
    matrix = rng.normal(size=(64, 64))
    matrix = matrix.T @ matrix / 64.0
    x = rng.normal(size=64)
    f = lambda v: float(0.5 * v @ matrix @ v)
    correct = lambda v: matrix @ v

    def corrupted(v: np.ndarray) -> np.ndarray:
        g = matrix @ v
        g[17] += 0.05
        return g

    correct_errors, bad_errors = [], []
    try:
        for _ in range(40):
            direction = rng.normal(size=64)
            correct_errors.append(directional_gradient_check(f, correct, x, direction, h=1e-5)["relative_error"])
            bad_errors.append(directional_gradient_check(f, corrupted, x, direction, h=1e-5)["relative_error"])
    except NotImplementedError as exc:
        print(f"[UNIMPLEMENTED] {exc}")
        raise SystemExit(2) from exc
    print(f"correct_median_error={float(np.median(correct_errors)):.12e}")
    print(f"correct_max_error={float(np.max(correct_errors)):.12e}")
    print(f"corrupted_median_error={float(np.median(bad_errors)):.12e}")
    print(f"corrupted_max_error={float(np.max(bad_errors)):.12e}")


if __name__ == "__main__":
    main()
