"""Diagnostic benchmark for AP-01-006: Deterministic Stratification by Integer Apportionment.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import hashlib

import numpy as np

from starter import stratified_split


def signature(indices: np.ndarray) -> str:
    """Hash exact membership/order. / 对精确成员及顺序计算哈希。"""
    return hashlib.sha256(indices.astype("<i8", copy=False).tobytes()).hexdigest()[:16]


def main() -> None:
    """Print allocation error and reproducibility evidence. / 打印分配误差与可复现证据。"""
    labels = np.repeat(np.arange(10, dtype=np.int64), np.arange(31, 41))
    try:
        train, valid = stratified_split(labels, valid_fraction=0.173, seed=2026)
    except NotImplementedError as exc:
        print(f"[UNIMPLEMENTED] {exc}")
        raise SystemExit(2) from exc
    total_counts = np.bincount(labels, minlength=10)
    valid_counts = np.bincount(labels[valid], minlength=10)
    allocation_l1 = float(np.abs(valid_counts - 0.173 * total_counts).sum())
    print(f"train_size={train.size}")
    print(f"valid_size={valid.size}")
    print(f"valid_class_counts={valid_counts.tolist()}")
    print(f"allocation_l1={allocation_l1:.6f}")
    print(f"intersection_size={np.intersect1d(train, valid).size}")
    print(f"valid_signature={signature(valid)}")


if __name__ == "__main__":
    main()
