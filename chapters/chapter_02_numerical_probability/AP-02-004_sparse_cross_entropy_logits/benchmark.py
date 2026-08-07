"""Diagnostic benchmark for AP-02-004: Sparse Cross-Entropy Directly From Logits.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from starter import sparse_cross_entropy


def main() -> None:
    """Compare fused and probability-domain evidence. / 比较融合式与概率域证据。"""
    rng = np.random.default_rng(4)
    logits = rng.normal(size=(2048, 10)) * 300.0
    labels = rng.integers(0, 10, size=2048, dtype=np.int64)
    try:
        fused = sparse_cross_entropy(logits, labels, reduction="none")
    except NotImplementedError as exc:
        print(f"[UNIMPLEMENTED] {exc}")
        raise SystemExit(2) from exc
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        exp_logits = np.exp(logits)
        probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        naive = -np.log(probabilities[np.arange(labels.size), labels])
    oracle = logsumexp(logits, axis=1) - logits[np.arange(labels.size), labels]
    print(f"fused_nonfinite_count={int((~np.isfinite(fused)).sum())}")
    print(f"naive_nonfinite_count={int((~np.isfinite(naive)).sum())}")
    print(f"fused_oracle_max_error={float(np.max(np.abs(fused - oracle))):.12e}")
    print(f"mean_loss={float(np.mean(fused, dtype=np.float64)):.8f}")


if __name__ == "__main__":
    main()
