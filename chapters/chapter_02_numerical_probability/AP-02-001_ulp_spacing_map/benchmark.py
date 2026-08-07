"""Diagnostic benchmark for AP-02-001: Map the Floating-Point Lattice.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from starter import local_spacing


def main() -> None:
    """Draw spacing staircases and print ratios. / 绘制间距阶梯并打印比率。"""
    magnitudes = np.logspace(-6, 4, 220)
    output = Path("diagnostic_spacing.png")
    try:
        profiles = {dtype.__name__: local_spacing(magnitudes, dtype=dtype) for dtype in (np.float16, np.float32, np.float64)}
    except NotImplementedError as exc:
        print(f"[UNIMPLEMENTED] {exc}")
        raise SystemExit(2) from exc
    for name, spacing in profiles.items():
        plt.loglog(magnitudes, spacing.astype(np.float64), label=name)
    plt.xlabel("magnitude x")
    plt.ylabel("nextafter(x, +inf) - x")
    plt.title("Forward floating-point spacing")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()
    for dtype in (np.float16, np.float32, np.float64):
        s = local_spacing(np.array([1.0, 1024.0]), dtype=dtype).astype(np.float64)
        print(f"{dtype.__name__}_spacing_at_1={s[0]:.12e}")
        print(f"{dtype.__name__}_spacing_growth_1_to_1024={s[1] / s[0]:.1f}")
    print(f"plot={output}")


if __name__ == "__main__":
    main()
