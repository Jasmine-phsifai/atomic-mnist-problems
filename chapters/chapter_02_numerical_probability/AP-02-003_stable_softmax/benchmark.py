"""Diagnostic benchmark for AP-02-003: Softmax With Shift Invariance and a Named Axis.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import numpy as np
from scipy.special import softmax as scipy_softmax

from starter import stable_softmax


def main() -> None:
    """Print independent stability invariants. / 打印相互独立的稳定性不变量。"""
    rng = np.random.default_rng(3)
    logits = rng.normal(size=(1024, 10)).astype(np.float64) * 80.0 + 10_000.0
    try:
        stable = stable_softmax(logits, axis=1)
        shifted = stable_softmax(logits - 50_000.0, axis=1)
    except NotImplementedError as exc:
        print(f"[UNIMPLEMENTED] {exc}")
        raise SystemExit(2) from exc
    with np.errstate(over="ignore", invalid="ignore"):
        naive_exp = np.exp(logits)
        naive = naive_exp / naive_exp.sum(axis=1, keepdims=True)
    oracle = scipy_softmax(logits, axis=1)
    print(f"simplex_max_error={float(np.max(np.abs(stable.sum(axis=1) - 1.0))):.12e}")
    print(f"shift_invariance_max_error={float(np.max(np.abs(stable - shifted))):.12e}")
    print(f"oracle_max_error={float(np.max(np.abs(stable - oracle))):.12e}")
    print(f"naive_finite_fraction={float(np.mean(np.isfinite(naive))):.6f}")


if __name__ == "__main__":
    main()
