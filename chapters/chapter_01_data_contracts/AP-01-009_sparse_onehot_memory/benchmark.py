"""Diagnostic benchmark for AP-01-009: Sparse Labels and One-Hot Storage.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import numpy as np

from starter import one_hot


def main() -> None:
    """Print storage amplification. / 打印存储放大倍数。"""
    labels = np.arange(60_000, dtype=np.int64) % 10
    try:
        encoded = one_hot(labels, num_classes=10, dtype=np.float32)
    except NotImplementedError as exc:
        print(f"[UNIMPLEMENTED] {exc}")
        raise SystemExit(2) from exc
    print(f"sparse_bytes={labels.nbytes}")
    print(f"onehot_bytes={encoded.nbytes}")
    print(f"storage_amplification={encoded.nbytes / labels.nbytes:.3f}")
    print(f"row_sum_max_error={float(np.max(np.abs(encoded.sum(axis=1) - 1.0))):.3e}")
    print(f"round_trip_accuracy={float(np.mean(encoded.argmax(axis=1) == labels)):.6f}")


if __name__ == "__main__":
    main()
