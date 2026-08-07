"""Diagnostic benchmark for AP-01-002: Executable Dataset Schema.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import numpy as np

from starter import validate_split


def main() -> None:
    """Print a diagnostic vector, not one opaque score. / 打印诊断向量，而非单一不透明分数。"""
    images = np.zeros((30, 28, 28), dtype=np.uint8)
    labels = np.repeat(np.arange(10, dtype=np.int64), 3)
    try:
        report = validate_split(images, labels, expected_size=30)
    except NotImplementedError as exc:
        print(f"[UNIMPLEMENTED] {exc}")
        raise SystemExit(2) from exc
    print(f"n={report['n']}")
    print(f"shape={tuple(report['image_shape'])}")
    print(f"class_counts={np.asarray(report['class_counts']).tolist()}")
    print(f"pixel_interval=[{report['pixel_min']}, {report['pixel_max']}]")
    print("schema_violation_count=0")


if __name__ == "__main__":
    main()
