"""Contract tests for AP-01-012: NumPy to TensorFlow: A Snapshot Boundary.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import unittest

import numpy as np

try:
    import tensorflow as tf
except ImportError as exc:  # Dependency message is deliberate. / 此依赖提示是有意设计的。
    raise SystemExit("Install requirements/tensorflow-cpu.txt before running AP-01-012") from exc

from starter import numpy_to_tensor_snapshot, tensor_to_numpy_snapshot


class TensorFlowBoundaryTests(unittest.TestCase):
    """Snapshot-isolation tests. / 快照隔离测试。"""

    def test_bidirectional_isolation_and_dtype(self) -> None:
        source = np.arange(6, dtype=np.float64).reshape(2, 3)
        expected = source.astype(np.float32)
        tensor = numpy_to_tensor_snapshot(source, dtype=tf.float32)
        self.assertIsInstance(tensor, tf.Tensor)
        self.assertNotIsInstance(tensor, tf.Variable)
        self.assertEqual(tensor.dtype, tf.float32)
        self.assertEqual(tuple(tensor.shape), (2, 3))
        source[0, 0] = 999.0
        np.testing.assert_array_equal(tensor.numpy(), expected)

        snapshot = tensor_to_numpy_snapshot(tensor)
        snapshot[0, 1] = 777.0
        np.testing.assert_array_equal(tensor.numpy(), expected)

    def test_rejects_variable(self) -> None:
        with self.assertRaises((TypeError, ValueError)):
            tensor_to_numpy_snapshot(tf.Variable([1.0, 2.0]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
