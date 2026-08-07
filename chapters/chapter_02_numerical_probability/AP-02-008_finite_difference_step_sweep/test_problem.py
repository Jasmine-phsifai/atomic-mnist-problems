"""Contract tests for AP-02-008: The Finite-Difference Step Has a U-Curve.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import unittest

import numpy as np

from starter import central_difference


class CentralDifferenceTests(unittest.TestCase):
    """Accuracy and side-effect tests. / 精度与副作用测试。"""

    def test_quadratic_coordinate_and_input_snapshot(self) -> None:
        x = np.array([1.5, -2.0, 0.25], dtype=np.float64)
        snapshot = x.copy()
        weights = np.array([2.0, 3.0, 5.0])

        def f(v: np.ndarray) -> float:
            return float(np.sum(weights * v**2))

        actual = central_difference(f, x, index=1, h=1e-5)
        expected = 2 * weights[1] * x[1]
        self.assertAlmostEqual(actual, expected, places=9)
        np.testing.assert_array_equal(x, snapshot)

    def test_rejects_invalid_contract(self) -> None:
        x = np.ones(3)
        for index, h in ((-1, 1e-5), (3, 1e-5), (0, 0.0), (0, np.inf)):
            with self.assertRaises(ValueError):
                central_difference(lambda v: float(v.sum()), x, index=index, h=h)
        with self.assertRaises(ValueError):
            central_difference(lambda v: v, x, index=0, h=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
