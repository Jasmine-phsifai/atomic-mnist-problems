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
    suff = path.suffixes
    if len(suff)!=1 or suff[0]!='.npz':
        raise ValueError(f"expected a single .npz artifact, got {suff}[:3]")
        with np.load(path,allow_pickle=False) as TempNPZ:
            if set(TempNPZ.files) != {expected_key}:
                raise ValueError(f"expected a single key {expected_key}, got {TempNPZ.files}[:3]")
                array = TempNPZ[expected_key]
                if not array.dtype.hasobject:
                    return array.copy()
                else:
                    raise ValueError(f"expected a non-object array, got {array.dtype}")
    raise NotImplementedError("AP-01-001")

# ---------------------------------------------------------------------------
# Reference design (advanced): judge the header before touching any payload.
# 参考设计（进阶）：在触碰 payload 之前先判头部。
#
# Why the in-body dtype check above can never fire: ``archive[key]`` runs
# ``np.lib.format.read_array``, which itself refuses object dtypes when
# ``allow_pickle=False``. Only arrays that already passed NumPy's gate ever
# reach the body, so the body check is defense in depth, not the real gate.
# 为什么函数体内的 dtype 检查永远不会触发：``archive[key]`` 内部已拒绝
# object dtype，能回到函数体的数组必然已通过 NumPy 的检查。
#
# np.load on .npz is lazy: it reads only the ZIP name table; a member is
# parsed only at ``archive[key]``. A member is raw NPY bytes laid out as
# ``magic | version | header | payload``, and the dtype string lives in the
# header, so it can be judged without reading any data.
# np.load 对 .npz 是惰性的：只读 ZIP 名字表，成员在 ``archive[key]`` 时才
# 解析。成员字节布局为 magic | 版本 | 头部 | 数据，dtype 写在头部里。
#
#     with archive.zip.open(expected_key + ".npy") as member:
#         version = np.lib.format.read_magic(member)
#         if version == (1, 0):
#             _shape, _fortran, dtype = np.lib.format.read_array_header_1_0(member)
#         elif version == (2, 0):
#             _shape, _fortran, dtype = np.lib.format.read_array_header_2_0(member)
#         else:
#             raise ValueError(f"unsupported npy format version {version}")
#     if dtype.hasobject:
#         raise ValueError(f"untrusted object dtype: {dtype!r}")
#
# ``archive.zip`` is a stdlib ``ZipFile``; ``.open(name)`` streams one
# member's decompressed bytes. Use ``dtype.hasobject`` rather than
# ``dtype != object``: a structured dtype like ``[('meta', 'O'),
# ('x', '<f4')]`` hides an object field that ``!=`` would wave through.
# ``archive.zip`` 是标准库 ``ZipFile``；用 ``hasobject`` 而非 ``!= object``，
# 因为结构化 dtype 可以把 object 字段藏起来。
# ---------------------------------------------------------------------------