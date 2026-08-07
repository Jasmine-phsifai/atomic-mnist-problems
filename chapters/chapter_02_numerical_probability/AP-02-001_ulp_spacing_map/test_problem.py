"""Contract tests for AP-02-001: Map the Floating-Point Lattice.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import unittest

import numpy as np

from starter import local_spacing


class SpacingContractTests(unittest.TestCase):
    """Floating-lattice tests. / 浮点格点测试。"""

    def test_exact_forward_neighbors_for_three_dtypes(self) -> None:
        values = np.array([0.0, 1.0, 2.0, 1024.0])
        for dtype in (np.float16, np.float32, np.float64):
            with self.subTest(dtype=dtype):
                x = values.astype(dtype)
                infinity = np.array(np.inf, dtype=dtype)
                expected = np.nextafter(x, infinity) - x
                actual = local_spacing(values, dtype=dtype)
                self.assertEqual(actual.dtype, np.dtype(dtype))
                np.testing.assert_array_equal(actual, expected)
                self.assertTrue(np.all(actual > 0))

    def test_rejects_invalid_domain_and_dtype(self) -> None:
        for values, dtype in [
            (np.array([-1.0]), np.float32),
            (np.array([np.inf]), np.float32),
            (np.array([np.nan]), np.float64),
            (np.array([1.0]), np.int32),
        ]:
            with self.assertRaises(ValueError):
                local_spacing(values, dtype=dtype)


if __name__ == "__main__":
    unittest.main(verbosity=2)
