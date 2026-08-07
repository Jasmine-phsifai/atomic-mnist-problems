"""Starter for AP-01-010: Content Fingerprints for Split Leakage.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import hashlib

import numpy as np


def fingerprint_rows(array: np.ndarray) -> np.ndarray:
    """Digest each first-axis record. / 对第一轴上的每条记录计算摘要。

    English API notes:
    - ``np.ascontiguousarray(row)`` gives canonical C-order bytes for the row.
    - ``row.dtype.str`` records byte order and scalar dtype.
    - ``hashlib.sha256(payload).hexdigest()`` returns a 64-character digest.

    中文 API 提示：
    - ``np.ascontiguousarray`` 可得到行记录的规范 C 顺序字节。
    - ``dtype.str`` 同时记录字节序与标量类型。
    - ``sha256(...).hexdigest()`` 返回 64 字符摘要。
    """
    # TODO: include dtype, shape, and bytes in each digest. / 每个摘要都应包含类型、形状和字节。
    raise NotImplementedError("AP-01-010 fingerprint_rows")


def cross_split_duplicates(train: np.ndarray, valid: np.ndarray) -> list[tuple[int, int]]:
    """Return sorted cross-split duplicate pairs. / 返回排序后的跨划分重复索引对。

    English: A mapping from digest to all train indices avoids a quadratic byte comparison.
    中文：建立“摘要到全部训练索引”的映射，可避免二次方级字节比较。
    """
    # TODO: reuse fingerprint_rows and preserve repeated matches. / 复用 fingerprint_rows 并保留重复匹配。
    raise NotImplementedError("AP-01-010 cross_split_duplicates")
