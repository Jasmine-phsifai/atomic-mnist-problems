"""Diagnostic benchmark for AP-01-003: Pixel Normalization Without Integer Surprises.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import numpy as np

from starter import normalize_uint8


def main() -> None:
    """Measure rounding, levels, and aliasing separately. / 分别测量舍入、层级和别名关系。"""
    source = np.arange(256, dtype=np.uint8).reshape(1, 16, 16)
    snapshot = source.copy()
    try:
        actual = normalize_uint8(source, dtype=np.float32)
    except NotImplementedError as exc:
        print(f"[UNIMPLEMENTED] {exc}")
        raise SystemExit(2) from exc
    oracle = source.astype(np.float64) / 255.0
    print(f"max_abs_error={float(np.max(np.abs(actual.astype(np.float64) - oracle))):.12e}")
    print(f"distinct_output_levels={np.unique(actual).size}")
    print(f"shares_source_storage={np.shares_memory(source, actual)}")
    print(f"source_unchanged={np.array_equal(source, snapshot)}")
    print(f"interval_ok={bool(np.all((actual >= 0) & (actual <= 1)))}")


if __name__ == "__main__":
    main()
