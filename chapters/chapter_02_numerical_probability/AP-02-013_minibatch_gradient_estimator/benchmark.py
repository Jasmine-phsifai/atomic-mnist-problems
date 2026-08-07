"""Diagnostic benchmark for AP-02-013: Measure Minibatch Gradient Unbiasedness and Variance.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from starter import sample_batch_means


def main() -> None:
    """Compare empirical and finite-population variance. / 比较经验方差与有限总体方差。
    """
    rng = np.random.default_rng(13)
    gradients = rng.normal(size=(100, 12)) @ rng.normal(size=(12, 6))
    sizes = np.array([1, 2, 5, 10, 20, 50, 100])
    empirical, theoretical = [], []
    s_cov_trace = float(np.trace(np.cov(gradients, rowvar=False, ddof=1)))
    try:
        for batch_size in sizes:
            samples = sample_batch_means(gradients, batch_size=int(batch_size), repeats=4000, seed=100 + int(batch_size))
            empirical.append(float(np.trace(np.cov(samples, rowvar=False, ddof=1))))
            theoretical.append((1 - batch_size / gradients.shape[0]) * s_cov_trace / batch_size)
    except NotImplementedError as exc:
        print(f"[UNIMPLEMENTED] {exc}")
        raise SystemExit(2) from exc
    empirical = np.asarray(empirical)
    theoretical = np.asarray(theoretical)
    output = Path("diagnostic_minibatch_variance.png")
    plt.loglog(sizes[:-1], empirical[:-1], "o-", label="empirical")
    plt.loglog(sizes[:-1], theoretical[:-1], "s--", label="finite-population theory")
    plt.xlabel("batch size")
    plt.ylabel("trace covariance of batch mean")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()
    for b, observed, expected in zip(sizes, empirical, theoretical, strict=True):
        ratio = observed / expected if expected > 0 else 1.0 if observed == 0 else np.inf
        print(f"batch_size={int(b):3d} empirical={observed:.8e} theoretical={expected:.8e} ratio={ratio:.5f}")
    print(f"plot={output}")


if __name__ == "__main__":
    main()
