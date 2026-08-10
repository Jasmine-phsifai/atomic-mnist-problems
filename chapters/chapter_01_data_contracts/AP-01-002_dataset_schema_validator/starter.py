"""Starter for AP-01-002: Executable Dataset Schema.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import numpy as np


def validate_split(
    images: np.ndarray,
    labels: np.ndarray,
    *,
    expected_size: int | None = None,
) -> dict[str, object]:
    """Validate and summarize one split. / 校验并汇总一个数据划分。

    English API notes:
    - ``array.ndim``, ``array.shape``, and ``array.dtype`` are independent checks.
    - ``np.issubdtype(labels.dtype, np.integer)`` recognizes integer families.
    - ``np.bincount(labels, minlength=10)`` is useful only after range validation.

    中文 API 提示：
    - ``ndim``、``shape`` 与 ``dtype`` 是彼此独立的检查。
    - ``np.issubdtype(..., np.integer)`` 可识别整数类型族。
    - 只有先校验标签范围，才能安全使用 ``np.bincount``。
    """
    # TODO: reject instead of silently reshaping or casting. / 应拒绝错误输入，而非静默整形或转换。
    imgshape=images.shape
    lblshape=labels.shape
    if expected_size is not None and (expected_size <= 0 or expected_size.is_integer() == False):
        raise ValueError(f"expected_size must be a positive integer, got {expected_size}")
    if len(imgshape)!=images.ndim or len(lblshape)!=labels.ndim: #zombie code, but the problem explicitly requested this conservative checking.
        raise ValueError(f"expected images.ndim={len(imgshape)} and labels.ndim={len(lblshape)}, got {images.ndim} and {labels.ndim}")
    if len(imgshape)!=3 or imgshape[1:]!=(28,28):
        raise ValueError(f"expected images of shape (N,28,28), got {imgshape}[:4]")
    if len(lblshape)!=1:
        raise ValueError(f"expected labels of shape (N,), got {lblshape}[:2]")
    if imgshape[0] != lblshape[0] or (expected_size is not None and (imgshape[0]!= expected_size or lblshape[0]!= expected_size)):
        raise ValueError(f"expected images, labels, and expected_size to have the same length/patchsize or expected_size to be None, got {imgshape[0]} vs {lblshape[0]} vs {expected_size}")
    if images.dtype!=np.uint8:
        raise ValueError(f"expected images of dtype uint8, got {images.dtype}")
    if (not np.issubdtype(labels.dtype, np.integer)):
        raise ValueError(f"expected labels of numpy integer dtype, got {labels.dtype}")
    projection= {k: np.eye(10)[k] for k in range(10)}
    try:
        records = sum((projection[i] for i in labels))
    except KeyError as e:
        raise ValueError(f"unexpected label value outside 0~9, got {e.args[0]} that went out of range") # todo: How can I catch the keyerror and get the triggering unexpected value?
    ResultDict={
        "n": imgshape[0],
        "image_shape": imgshape,
        "image_dtype": images.dtype,
        "label_dtype": labels.dtype,
        "pixel_min": images.min(),
        "pixel_max": images.max(),
        "class_counts": records
    }
    return ResultDict
    raise NotImplementedError("AP-01-002")
