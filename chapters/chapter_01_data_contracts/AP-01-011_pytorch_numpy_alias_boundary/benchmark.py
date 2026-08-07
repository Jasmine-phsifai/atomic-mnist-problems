"""Diagnostic benchmark for AP-01-011: NumPy to PyTorch: Shared View Versus Owned Copy.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import numpy as np

try:
    import torch
except ImportError as exc:  # Keep failure actionable. / 让依赖失败可操作。
    raise SystemExit("Install requirements/pytorch-cpu.txt before running AP-01-011") from exc

from starter import numpy_to_torch_pair


def main() -> None:
    """Print the mutation-transmission matrix. / 打印变更传播矩阵。"""
    array = np.arange(4, dtype=np.float32)
    try:
        shared, owned = numpy_to_torch_pair(array)
    except NotImplementedError as exc:
        print(f"[UNIMPLEMENTED] {exc}")
        raise SystemExit(2) from exc
    print(f"numpy_ptr={array.__array_interface__['data'][0]}")
    print(f"shared_ptr={shared.data_ptr()}")
    print(f"owned_ptr={owned.data_ptr()}")
    print(f"numpy_to_shared={array.__array_interface__['data'][0] == shared.data_ptr()}")
    print(f"numpy_to_owned={array.__array_interface__['data'][0] == owned.data_ptr()}")
    print(f"devices={[shared.device.type, owned.device.type]}")


if __name__ == "__main__":
    main()
