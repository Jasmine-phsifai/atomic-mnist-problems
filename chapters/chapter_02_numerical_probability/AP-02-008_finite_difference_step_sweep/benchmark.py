"""Diagnostic benchmark for AP-02-008: The Finite-Difference Step Has a U-Curve.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp, softmax

from starter import central_difference


def main() -> None:
    """Draw the truncation/cancellation U-curve. / 绘制截断误差与消去误差的 U 形曲线。"""
    x = np.array([0.2, -0.7, 1.1, 2.0], dtype=np.float64)
    label = 2

    def loss(v: np.ndarray) -> float:
        return float(logsumexp(v) - v[label])

    analytic = softmax(x)[0]  # Coordinate zero of p - one_hot(label). / p-one_hot 的第零坐标。
    steps = np.logspace(-1, -16, 80)
    try:
        approximations = np.array([central_difference(loss, x, index=0, h=float(h)) for h in steps])
    except NotImplementedError as exc:
        print(f"[UNIMPLEMENTED] {exc}")
        raise SystemExit(2) from exc
    errors = np.abs(approximations - analytic)
    best = int(np.argmin(errors))
    output = Path("diagnostic_step_u_curve.png")
    plt.loglog(steps, errors, marker=".")
    plt.xlabel("h")
    plt.ylabel("absolute derivative error")
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()
    print(f"empirical_best_h={steps[best]:.12e}")
    print(f"empirical_best_error={errors[best]:.12e}")
    print(f"smallest_h_error={errors[-1]:.12e}")
    print(f"plot={output}")


if __name__ == "__main__":
    main()
