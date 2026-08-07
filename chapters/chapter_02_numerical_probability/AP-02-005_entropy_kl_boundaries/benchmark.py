"""Diagnostic benchmark for AP-02-005: Entropy and KL at the Boundary of the Simplex.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import numpy as np

from starter import entropy, kl_divergence


def main() -> None:
    """Probe Gibbs nonnegativity on random simplexes. / 在随机单纯形上检验 Gibbs 非负性。
    """
    rng = np.random.default_rng(5)
    p = rng.dirichlet(np.ones(10), size=5000)
    q = rng.dirichlet(np.ones(10), size=5000)
    try:
        h = entropy(p, axis=1)
        kl = kl_divergence(p, q, axis=1)
    except NotImplementedError as exc:
        print(f"[UNIMPLEMENTED] {exc}")
        raise SystemExit(2) from exc
    print(f"entropy_interval=[{float(h.min()):.8f}, {float(h.max()):.8f}]")
    print(f"min_kl={float(kl.min()):.12e}")
    print(f"materially_negative_kl_count={int(np.sum(kl < -1e-12))}")
    support_mismatch = kl_divergence(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    print(f"support_mismatch_is_infinite={bool(np.isinf(support_mismatch))}")


if __name__ == "__main__":
    main()
