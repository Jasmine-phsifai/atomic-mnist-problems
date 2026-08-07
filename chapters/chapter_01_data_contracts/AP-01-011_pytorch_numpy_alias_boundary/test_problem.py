"""Contract tests for AP-01-011: NumPy to PyTorch: Shared View Versus Owned Copy.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import unittest

import numpy as np

try:
    import torch
except ImportError as exc:  # Dependency message is deliberate. / 此依赖提示是有意设计的。
    raise SystemExit("Install requirements/pytorch-cpu.txt before running AP-01-011") from exc

from starter import numpy_to_torch_pair


class TorchBoundaryTests(unittest.TestCase):
    """Mutation-based alias tests. / 基于变更传播的别名测试。"""

    def test_mutation_truth_table(self) -> None:
        array = np.arange(6, dtype=np.float32)
        shared, owned = numpy_to_torch_pair(array)
        self.assertEqual(shared.device.type, "cpu")
        self.assertEqual(owned.device.type, "cpu")
        self.assertEqual(shared.dtype, torch.float32)
        self.assertEqual(owned.dtype, torch.float32)
        np.testing.assert_array_equal(shared.detach().numpy(), array)
        np.testing.assert_array_equal(owned.detach().numpy(), array)

        array[0] = 101.0
        self.assertEqual(float(shared[0]), 101.0)
        self.assertNotEqual(float(owned[0]), 101.0)
        shared[1] = 202.0
        self.assertEqual(float(array[1]), 202.0)
        before = float(array[2])
        owned[2] = 303.0
        self.assertEqual(float(array[2]), before)
        self.assertNotEqual(float(shared[2]), 303.0)

    def test_rejects_ambiguous_layouts(self) -> None:
        base = np.arange(12, dtype=np.float32).reshape(3, 4)
        with self.assertRaises(ValueError):
            numpy_to_torch_pair(base[:, ::2])
        readonly = base.copy()
        readonly.flags.writeable = False
        with self.assertRaises(ValueError):
            numpy_to_torch_pair(readonly)


if __name__ == "__main__":
    unittest.main(verbosity=2)
