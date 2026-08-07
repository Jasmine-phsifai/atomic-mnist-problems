"""Diagnostic benchmark for AP-01-008: Epoch Means Are Sample-Weighted.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import numpy as np

from starter import sample_weighted_mean


def main() -> None:
    """Print aggregation bias and implementation error separately. / 分别打印聚合偏差与实现误差。"""
    means = np.array([0.2, 0.4, 1.0], dtype=np.float32)
    sizes = np.array([64, 64, 2], dtype=np.int64)
    try:
        weighted = sample_weighted_mean(means, sizes)
    except NotImplementedError as exc:
        print(f"[UNIMPLEMENTED] {exc}")
        raise SystemExit(2) from exc
    oracle = float(np.repeat(means.astype(np.float64), sizes).mean())
    batch_mean = float(means.astype(np.float64).mean())
    print(f"sample_weighted_mean={weighted:.12f}")
    print(f"naive_batch_mean={batch_mean:.12f}")
    print(f"aggregation_bias={batch_mean - oracle:.12f}")
    print(f"implementation_error={weighted - oracle:.12e}")


if __name__ == "__main__":
    main()
