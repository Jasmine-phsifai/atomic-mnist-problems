"""Starter for AP-01-003: Pixel Normalization Without Integer Surprises.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import numpy as np


def normalize_uint8(
    images: np.ndarray,
    *,
    dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    """Normalize pixels into [0, 1]. / 将像素归一化到 [0, 1]。

    English API notes:
    - ``array.astype(dtype, copy=True)`` makes dtype conversion and ownership explicit.
    - NumPy true division follows ufunc casting rules; inspect the result dtype.
    - ``np.shares_memory`` is a test aid, not the implementation itself.

    中文 API 提示：
    - ``astype(dtype, copy=True)`` 可显式完成类型转换并取得独立存储。
    - NumPy 真除法遵循 ufunc 类型转换规则，必须检查结果类型。
    - ``np.shares_memory`` 用于测试存储关系，不是实现逻辑本身。
    """
    # TODO: validate, cast before division, and preserve ownership. / 校验后先转换类型，再做除法并保持独立存储。
    if images.dtype!=np.uint8:
        raise ValueError(f"expected images of dtype uint8, got {images.dtype}")
    if images.ndim not in (2,3):
        raise ValueError(f"expected images of shape (batchsize,length,width) or (length,width), got {images.shape[-3:]}")
    if dtype not in (np.float64, np.float32):
        raise ValueError(f"expected dtype to be float32 or float64, got {dtype}")
    result= (images/255).astype(dtype,copy=True)
    return result
