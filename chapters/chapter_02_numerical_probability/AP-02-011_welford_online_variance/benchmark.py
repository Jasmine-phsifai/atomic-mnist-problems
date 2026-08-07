"""Diagnostic benchmark for AP-02-011: One-Pass Variance Without Catastrophic Cancellation.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import numpy as np

from starter import online_mean_variance


def main() -> None:
    """Compare offset-sensitive and stable formulas. / 比较偏移敏感公式与稳定公式。"""
    rng = np.random.default_rng(11)
    residual = rng.normal(scale=3.0, size=200_000)
    values = 1e12 + residual
    try:
        mean, welford = online_mean_variance(values, ddof=0)
    except NotImplementedError as exc:
        print(f"[UNIMPLEMENTED] {exc}")
        raise SystemExit(2) from exc
    naive = float(np.mean(values * values) - np.mean(values) ** 2)
    oracle = float(np.var(values))
    _, shifted = online_mean_variance(residual, ddof=0)
    print(f"mean={mean:.6f}")
    print(f"oracle_variance={oracle:.12f}")
    print(f"naive_variance={naive:.12f}")
    print(f"naive_error={naive - oracle:.12e}")
    print(f"welford_error={welford - oracle:.12e}")
    print(f"offset_invariance_error={welford - shifted:.12e}")


if __name__ == "__main__":
    main()
