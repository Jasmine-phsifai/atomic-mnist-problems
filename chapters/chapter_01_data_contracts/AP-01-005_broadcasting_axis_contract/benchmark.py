"""Diagnostic benchmark for AP-01-005: Broadcasting With a Named Feature Axis.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import numpy as np

from starter import standardize_features


def main() -> None:
    """Expose feature residuals and wrong-axis distance. / 展示特征残差与错误轴距离。"""
    x = np.arange(16, dtype=np.float64).reshape(4, 4)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    try:
        actual = standardize_features(x, mean, scale)
    except NotImplementedError as exc:
        print(f"[UNIMPLEMENTED] {exc}")
        raise SystemExit(2) from exc
    wrong_axis = (x - mean.reshape(4, 1)) / scale.reshape(4, 1)
    print(f"per_feature_mean={actual.mean(axis=0).tolist()}")
    print(f"per_feature_std={actual.std(axis=0).tolist()}")
    print(f"axis_confusion_l2={float(np.linalg.norm(actual - wrong_axis)):.8f}")


if __name__ == "__main__":
    main()
