"""Contract tests for AP-02-011: One-Pass Variance Without Catastrophic Cancellation.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import unittest

import numpy as np

from starter import online_mean_variance


class WelfordTests(unittest.TestCase):
    """Large-offset variance tests. / 大偏移方差测试。"""

    def test_large_offset_matches_two_pass_oracle(self) -> None:
        residual = np.tile(np.array([-3.0, -1.0, 0.0, 2.0, 4.0]), 2000)
        values = 1e12 + residual
        for ddof in (0, 1):
            mean, variance = online_mean_variance(values, ddof=ddof)
            self.assertAlmostEqual(mean, float(np.mean(values)), places=3)
            self.assertAlmostEqual(variance, float(np.var(values, ddof=ddof)), places=10)

    def test_constant_variance_is_exact_zero(self) -> None:
        mean, variance = online_mean_variance(np.full(17, 3.5), ddof=1)
        self.assertEqual(mean, 3.5)
        self.assertEqual(variance, 0.0)

    def test_rejects_bad_input(self) -> None:
        for values, ddof in [
            (np.array([], dtype=np.float64), 0),
            (np.ones((2, 2), dtype=np.float64), 0),
            (np.ones(2, dtype=np.float32), 0),
            (np.array([1.0, np.nan]), 0),
            (np.ones(1), 1),
            (np.ones(2), 2),
        ]:
            with self.assertRaises(ValueError):
                online_mean_variance(values, ddof=ddof)


if __name__ == "__main__":
    unittest.main(verbosity=2)
