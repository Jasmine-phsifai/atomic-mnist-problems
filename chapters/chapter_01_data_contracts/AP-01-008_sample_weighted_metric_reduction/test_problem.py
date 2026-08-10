"""Contract tests for AP-01-008: Epoch Means Are Sample-Weighted.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import unittest

import numpy as np

from starter import sample_weighted_mean


class MetricReductionTests(unittest.TestCase):
    """Measure-selection tests. / 测度选择测试。"""

    def test_unequal_tail_matches_sample_oracle(self) -> None:
        means = np.array([0.2, 0.4, 1.0], dtype=np.float32)
        sizes = np.array([64, 64, 2], dtype=np.int64)
        actual = sample_weighted_mean(means, sizes)
        oracle = float(np.repeat(means.astype(np.float64), sizes).mean())
        self.assertEqual(type(actual), float)
        self.assertAlmostEqual(actual, oracle, places=14)
        self.assertNotAlmostEqual(actual, float(means.mean()), places=4)

    def test_equal_sizes_reduce_to_batch_mean(self) -> None:
        means = np.array([0.1, 0.3, 0.8], dtype=np.float64)
        sizes = np.array([7, 7, 7], dtype=np.int64)
        self.assertAlmostEqual(sample_weighted_mean(means, sizes), float(means.mean()), places=15)

    def test_rejects_bad_summaries(self) -> None:
        cases = [
            (np.array([1.0]), np.array([1, 2])),
            (np.array([[1.0]]), np.array([1])),
            (np.array([1.0, 2.0]), np.array([[1, 2]])),
            (np.array([np.nan]), np.array([1])),
            (np.array([np.inf]), np.array([1])),
            (np.array([1.0]), np.array([0])),
            (np.array([1.0]), np.array([-2])),
            (np.array([1.0]), np.array([1.5])),
        ]
        for means, sizes in cases:
            with self.assertRaises(ValueError):
                sample_weighted_mean(means, sizes)


if __name__ == "__main__":
    unittest.main(verbosity=2)
