"""Diagnostic benchmark for AP-01-012: NumPy to TensorFlow: A Snapshot Boundary.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import numpy as np

try:
    import tensorflow as tf
except ImportError as exc:  # Keep failure actionable. / 让依赖失败可操作。
    raise SystemExit("Install requirements/tensorflow-cpu.txt before running AP-01-012") from exc

from starter import numpy_to_tensor_snapshot, tensor_to_numpy_snapshot


def main() -> None:
    """Print the mutation-transmission matrix. / 打印变更传播矩阵。"""
    source = np.arange(4, dtype=np.float32)
    try:
        tensor = numpy_to_tensor_snapshot(source, dtype=tf.float32)
        snapshot = tensor_to_numpy_snapshot(tensor)
    except NotImplementedError as exc:
        print(f"[UNIMPLEMENTED] {exc}")
        raise SystemExit(2) from exc
    baseline = tensor.numpy().copy()
    source[0] = 100.0
    source_reached_tensor = not np.array_equal(tensor.numpy(), baseline)
    snapshot[1] = 200.0
    snapshot_reached_tensor = not np.array_equal(tensor.numpy(), baseline)
    print(f"source_mutation_reached_tensor={source_reached_tensor}")
    print(f"snapshot_mutation_reached_tensor={snapshot_reached_tensor}")
    print(f"tensor_dtype={tensor.dtype.name}")
    print(f"tensor_shape={tuple(tensor.shape)}")


if __name__ == "__main__":
    main()
