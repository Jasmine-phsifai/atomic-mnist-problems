"""Diagnostic benchmark for AP-02-002: Stable Softplus Across Extreme Inputs.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from starter import stable_softplus


def main() -> None:
    """Compare stable and naive finite behavior. / 比较稳定式与朴素式的有限性。
    """
    x = np.linspace(-1000.0, 1000.0, 2001, dtype=np.float64)
    try:
        stable = stable_softplus(x)
    except NotImplementedError as exc:
        print(f"[UNIMPLEMENTED] {exc}")
        raise SystemExit(2) from exc
    with np.errstate(over="ignore"):
        naive = np.log1p(np.exp(x))
    oracle = np.logaddexp(0.0, x)
    output = Path("diagnostic_softplus.png")
    plt.plot(x, stable, label="stable")
    plt.plot(x, np.minimum(naive, 1100.0), "--", label="naive (clipped for display)")
    plt.xlabel("x")
    plt.ylabel("softplus(x)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()
    print(f"stable_finite_count={int(np.isfinite(stable).sum())}/{stable.size}")
    print(f"naive_finite_count={int(np.isfinite(naive).sum())}/{naive.size}")
    print(f"max_abs_error={float(np.max(np.abs(stable - oracle))):.12e}")
    print(f"plot={output}")


if __name__ == "__main__":
    main()
