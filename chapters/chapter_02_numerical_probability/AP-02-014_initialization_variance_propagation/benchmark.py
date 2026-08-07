"""Diagnostic benchmark for AP-02-014: Initialization by Propagated Energy.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import numpy as np

from starter import initialize_weights


def main() -> None:
    """Report repeated propagated-energy distributions. / 报告重复的传播能量分布。"""
    ratios_x, ratios_h = [], []
    input_rng = np.random.default_rng(144)
    inputs = input_rng.normal(size=(2500, 192))
    input_energy = float(np.mean(inputs**2))
    try:
        for seed in range(12):
            xavier = initialize_weights(192, 192, scheme="xavier_normal", rng=np.random.default_rng(1000 + seed))
            he = initialize_weights(192, 192, scheme="he_normal", rng=np.random.default_rng(2000 + seed))
            ratios_x.append(float(np.mean((inputs @ xavier) ** 2) / input_energy))
            ratios_h.append(float(np.mean(np.maximum(inputs @ he, 0.0) ** 2) / input_energy))
    except NotImplementedError as exc:
        print(f"[UNIMPLEMENTED] {exc}")
        raise SystemExit(2) from exc
    print(f"xavier_linear_ratio_mean={np.mean(ratios_x):.8f}")
    print(f"xavier_linear_ratio_std={np.std(ratios_x, ddof=1):.8f}")
    print(f"he_relu_ratio_mean={np.mean(ratios_h):.8f}")
    print(f"he_relu_ratio_std={np.std(ratios_h, ddof=1):.8f}")


if __name__ == "__main__":
    main()
