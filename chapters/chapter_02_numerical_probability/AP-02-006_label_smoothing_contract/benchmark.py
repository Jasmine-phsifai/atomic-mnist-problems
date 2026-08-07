"""Diagnostic benchmark for AP-02-006: Label Smoothing Is a Convention, Not a Slogan.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import numpy as np
from scipy.special import softmax

from starter import smooth_targets


def main() -> None:
    """Print entropy and gradient-norm response curves. / 打印熵与梯度范数响应曲线。"""
    labels = np.array([0], dtype=np.int64)
    probabilities = softmax(np.array([[12.0, -2.0, -3.0, -4.0]]), axis=1)
    try:
        for epsilon in (0.0, 0.05, 0.1, 0.2, 0.5):
            target = smooth_targets(labels, num_classes=4, epsilon=epsilon, dtype=np.float64)
            entropy = -np.sum(np.where(target > 0, target * np.log(target), 0.0))
            gradient_norm = np.linalg.norm(probabilities - target)
            print(f"epsilon={epsilon:.2f} target_entropy={entropy:.8f} gradient_norm={gradient_norm:.8f}")
    except NotImplementedError as exc:
        print(f"[UNIMPLEMENTED] {exc}")
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
