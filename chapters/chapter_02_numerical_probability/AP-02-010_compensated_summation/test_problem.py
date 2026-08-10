"""Contract tests for AP-02-010: Compensated Summation Recovers Lost Increments.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from starter import kahan_sum


class CompensatedSummationTests(unittest.TestCase):
    """Accumulation-error tests. / 累积误差测试。"""

    def test_recovers_many_small_increments(self) -> None:
        values = np.concatenate([np.array([1.0]), np.full(20_000, 1e-16)])
        oracle = math.fsum(values.tolist())
        # Explicit fl(s + x) loop: builtin sum() is Neumaier-compensated since
        # Python 3.12 and would be as accurate as Kahan here. / 显式顺序累加循环。
        naive = 0.0
        for value in values.tolist():
            naive += value
        actual = kahan_sum(values)
        self.assertLessEqual(abs(actual - oracle), np.finfo(np.float64).eps)
        self.assertLess(abs(actual - oracle), abs(naive - oracle))

    def test_exact_small_sum(self) -> None:
        self.assertEqual(kahan_sum(np.array([1.0, 2.0, 3.0], dtype=np.float64)), 6.0)

    def test_rejects_bad_input(self) -> None:
        for values in [
            np.array([], dtype=np.float64),
            np.ones((2, 2), dtype=np.float64),
            np.ones(2, dtype=np.float32),
            np.array([1.0, np.inf]),
            np.array([1.0, np.nan]),
        ]:
            with self.assertRaises(ValueError):
                kahan_sum(values)


if __name__ == "__main__":
    unittest.main(verbosity=2)
