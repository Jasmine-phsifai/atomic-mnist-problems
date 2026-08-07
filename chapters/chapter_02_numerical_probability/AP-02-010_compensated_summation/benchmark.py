"""Diagnostic benchmark for AP-02-010: Compensated Summation Recovers Lost Increments.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import math

import numpy as np

from starter import kahan_sum


def main() -> None:
    """Print reduction errors and order sensitivity. / 打印归约误差与顺序敏感性。"""
    values = np.concatenate([np.array([1.0]), np.full(1_000_000, 1e-16)])
    oracle = math.fsum(values.tolist())
    try:
        forward = kahan_sum(values)
        reverse = kahan_sum(values[::-1].copy())
    except NotImplementedError as exc:
        print(f"[UNIMPLEMENTED] {exc}")
        raise SystemExit(2) from exc
    python_forward = sum(values.tolist())
    numpy_forward = float(np.sum(values))
    print(f"oracle_fsum={oracle:.17g}")
    print(f"python_sum_error={python_forward - oracle:.12e}")
    print(f"numpy_sum_error={numpy_forward - oracle:.12e}")
    print(f"kahan_forward_error={forward - oracle:.12e}")
    print(f"kahan_reverse_error={reverse - oracle:.12e}")
    print(f"kahan_order_sensitivity={forward - reverse:.12e}")


if __name__ == "__main__":
    main()
