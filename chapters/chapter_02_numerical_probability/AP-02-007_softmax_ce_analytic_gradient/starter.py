"""Starter for AP-02-007: Derive and Implement the Softmax-Loss Gradient.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import numpy as np


def softmax_cross_entropy_gradient(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Return the analytic gradient of mean sparse CE. / 返回稀疏交叉熵均值的解析梯度。

    English API notes:
    - Derive the row derivative before writing array code; place that derivation here.
    - Subtract row maxima before exponentiation; the shift does not alter probabilities.
    - Advanced indexing can update one true-class entry per row without one-hot storage.

    中文 API 提示：
    - 先推导单行导数，再写数组代码；请把推导写在本注释附近。
    - 指数化前减去每行最大值；该平移不改变概率。
    - 高级索引可直接修改每行真实类别位置，无需 one-hot。
    """
    # TODO: add the derivation, validate inputs, and implement the analytic gradient. / 补充推导、校验输入并实现解析梯度。
    raise NotImplementedError("AP-02-007")
