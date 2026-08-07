"""Diagnostic benchmark for AP-01-007: A Minibatch Iterator With an Explicit Tail Policy.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import numpy as np

from starter import iter_minibatches


def main() -> None:
    """Print coverage diagnostics for both tail policies. / 打印两种尾批策略的覆盖诊断。"""
    indices = np.arange(1009, dtype=np.int64)
    try:
        kept = list(iter_minibatches(indices, batch_size=64, shuffle=True, seed=7, drop_last=False))
        dropped = list(iter_minibatches(indices, batch_size=64, shuffle=True, seed=7, drop_last=True))
    except NotImplementedError as exc:
        print(f"[UNIMPLEMENTED] {exc}")
        raise SystemExit(2) from exc
    joined = np.concatenate(kept)
    dropped_count = indices.size - sum(batch.size for batch in dropped)
    print(f"batch_sizes_keep={[batch.size for batch in kept]}")
    print(f"coverage_ratio={np.unique(joined).size / indices.size:.6f}")
    print(f"duplicate_count={joined.size - np.unique(joined).size}")
    print(f"discarded_count_drop_last={dropped_count}")


if __name__ == "__main__":
    main()
