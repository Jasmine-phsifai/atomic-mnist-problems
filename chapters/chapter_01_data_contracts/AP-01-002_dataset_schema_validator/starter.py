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
    if expected_size is not None:  ## fix
        if not isinstance(expected_size, int):
            raise ValueError(f"expected None or standard integers, got {expected_size.__class__.__name__}")
        elif expected_size <= 0:
            raise ValueError(f"expected positive integer, got {expected_size}")
        elif expected_size == True:  ## new
            expected_size = 1
    if images.ndim!=3 or images.shape[1:]!=(28,28):
        raise ValueError(f"expected images of shape (N,28,28), got {images.shape[:4]}")
    if labels.ndim!=1:
        raise ValueError(f"expected labels of shape (N,), got {labels.shape[:2]}")
    if images.shape[0] != labels.shape[0]:
        raise ValueError(f"expected images, labels to have the same length/patchsize, got {images.shape[0]} vs {labels.shape[0]}")
    if expected_size:
        if images.shape[0] != expected_size:
            raise ValueError(f"expected images, labels to have length/patchsize {expected_size}, got images and labels' batchsize {images.shape[0]} vs expected size {expected_size}")
    if images.size == 0 or labels.size == 0:
        raise ValueError(f"expected non-empty images and labels, got images.size {images.size} and labels.size {labels.size}")
    if images.dtype!=np.uint8:
        raise ValueError(f"expected images of dtype uint8, got {images.dtype}")
    if (not np.issubdtype(labels.dtype, np.integer)):
        raise ValueError(f"expected labels of numpy integer dtype, got {labels.dtype}")
    """
    Former approach like this
    projection= {k: np.eye(10)[k] for k in range(10)}
    try:
        records = sum((projection[i] for i in labels))
    except KeyError as e:
        raise ValueError(f"unexpected label value outside 0~9, got {e.args[0]} that went out of range") # todo: How can I catch the keyerror and get the triggering unexpected value?
    ResultDict={
        "n": images.shape[0],
        "image_shape": images.shape,
        "image_dtype": images.dtype,
        "label_dtype": labels.dtype,
        "pixel_min": images.min(),
        "pixel_max": images.max(),
        "class_counts": records
    }
    return ResultDict
    raise NotImplementedError("AP-01-002")
    """

    # Second iteration approach like this
    """newRecords = np.bincount(labels, minlength=10)
    if len(newRecords) != 10:
        raise ValueError(f"unexpected integer label value outside 0~9, got {len(newRecords)-1} that went out of range")
    newResultDict = {
        "n": images.shape[0],
        "image_shape": images.shape,
        "image_dtype": images.dtype,
        "label_dtype": labels.dtype,
        "pixel_min": images.min(),
        "pixel_max": images.max(),
        "class_counts": newRecords
    }
    return newResultDict
    """
    # third iteration approach like this
    if np.any(labels < 0) or np.any(labels > 9):
        raise ValueError("unexpected integer label value outside 0~9")
    Record_3 = np.bincount(labels, minlength=10)
    newResultDict_3 = {
        "n": images.shape[0],
        "image_shape": images.shape,
        "image_dtype": images.dtype,
        "label_dtype": labels.dtype,
        "pixel_min": images.min(),
        "pixel_max": images.max(),
        "class_counts": Record_3
    }
    return newResultDict_3
