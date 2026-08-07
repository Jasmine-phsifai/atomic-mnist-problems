"""Starter for AP-01-001: A Trust-Bounded NPZ Loader.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_npz_array(path: Path, *, expected_key: str) -> np.ndarray:
    """Load one trusted numerical member. / 加载一个受约束的数值成员。

    English API notes:
    - ``np.load(path, allow_pickle=False)`` returns an ``NpzFile`` for ``.npz``.
    - Use it as a context manager; inspect ``archive.files`` before indexing.
    - ``np.asarray`` does not by itself prove that an object array is safe.

    中文 API 提示：
    - ``np.load(path, allow_pickle=False)`` 对 ``.npz`` 返回 ``NpzFile``。
    - 请使用上下文管理器，并在索引前检查 ``archive.files``。
    - 仅调用 ``np.asarray`` 不能证明对象数组是安全的。
    """
    # TODO: validate suffix, exact key set, and non-object dtype. / 校验后缀、唯一键和非对象类型。
    raise NotImplementedError("AP-01-001")
