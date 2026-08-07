"""Diagnostic benchmark for AP-02-007: Derive and Implement the Softmax-Loss Gradient.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from starter import softmax_cross_entropy_gradient


def main() -> None:
    """Compare a random coordinate sample with finite differences. / 用有限差分比较随机坐标样本。"""
    rng = np.random.default_rng(7)
    logits = rng.normal(size=(8, 10))
    labels = rng.integers(0, 10, size=8, dtype=np.int64)
    try:
        analytic = softmax_cross_entropy_gradient(logits, labels)
    except NotImplementedError as exc:
        print(f"[UNIMPLEMENTED] {exc}")
        raise SystemExit(2) from exc
    errors = []
    relatives = []
    h = 1e-6
    for flat in rng.choice(logits.size, size=24, replace=False):
        index = np.unravel_index(flat, logits.shape)
        plus, minus = logits.copy(), logits.copy()
        plus[index] += h
        minus[index] -= h
        fp = np.mean(logsumexp(plus, axis=1) - plus[np.arange(8), labels])
        fm = np.mean(logsumexp(minus, axis=1) - minus[np.arange(8), labels])
        numeric = (fp - fm) / (2 * h)
        error = abs(float(analytic[index]) - float(numeric))
        errors.append(error)
        relatives.append(error / max(1e-12, abs(float(analytic[index])) + abs(float(numeric))))
    print(f"max_abs_error={max(errors):.12e}")
    print(f"max_relative_error={max(relatives):.12e}")
    print(f"shift_null_residual={float(np.max(np.abs(analytic.sum(axis=1)))):.12e}")


if __name__ == "__main__":
    main()
