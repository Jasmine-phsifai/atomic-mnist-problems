"""Starter for AP-01-004: Flattening: Shape Is Not Ownership.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import numpy as np


def flatten_images(images: np.ndarray, *, require_independent: bool) -> np.ndarray:
    """Flatten images under an ownership policy. / 按存储所有权策略展平图像。

    English API notes:
    - ``reshape(n, -1)`` preserves logical C-order but may return a view or copy.
    - ``array.flags.c_contiguous`` describes one important layout property.
    - ``array.copy(order="C")`` requests independent C-contiguous storage.

    中文 API 提示：
    - ``reshape(n, -1)`` 保持逻辑 C 顺序，但可能返回视图或副本。
    - ``flags.c_contiguous`` 描述关键的内存布局属性。
    - ``copy(order="C")`` 可请求独立且 C 连续的存储。
    """
    # TODO: satisfy both value and ownership contracts. / 同时满足数值与所有权契约。
    raise NotImplementedError("AP-01-004")
