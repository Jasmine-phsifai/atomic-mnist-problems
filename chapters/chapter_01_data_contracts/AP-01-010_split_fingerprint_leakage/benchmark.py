"""Diagnostic benchmark for AP-01-010: Content Fingerprints for Split Leakage.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import numpy as np

from starter import cross_split_duplicates


def main() -> None:
    """Print leakage diagnostics. / 打印数据泄漏诊断。"""
    rng = np.random.default_rng(11)
    train = rng.integers(0, 256, size=(1000, 28, 28), dtype=np.uint8)
    valid = rng.integers(0, 256, size=(200, 28, 28), dtype=np.uint8)
    valid[[5, 17, 199]] = train[[7, 7, 81]]
    try:
        pairs = cross_split_duplicates(train, valid)
    except NotImplementedError as exc:
        print(f"[UNIMPLEMENTED] {exc}")
        raise SystemExit(2) from exc
    affected = {j for _, j in pairs}
    print(f"duplicate_pair_count={len(pairs)}")
    print(f"affected_validation_count={len(affected)}")
    print(f"cross_split_duplicate_rate={len(affected) / valid.shape[0]:.6f}")
    print(f"pairs={pairs}")


if __name__ == "__main__":
    main()
