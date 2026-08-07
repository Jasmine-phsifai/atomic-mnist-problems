"""Contract tests for AP-02-012: Find the Exponential Overflow Frontier.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import unittest

import numpy as np

from starter import exp_range_audit


class ExponentRangeAuditTests(unittest.TestCase):
    """Adjacent-frontier tests. / 相邻边界测试。"""

    def test_adjacent_finite_and_overflow_inputs(self) -> None:
        for dtype in (np.float16, np.float32, np.float64):
            with self.subTest(dtype=dtype):
                report = exp_range_audit(dtype)
                lower = np.array(report["last_finite"], dtype=dtype)
                upper = np.array(report["first_overflow"], dtype=dtype)
                expected_upper = np.nextafter(lower, np.array(np.inf, dtype=dtype))
                self.assertEqual(upper, expected_upper)
                with np.errstate(over="ignore"):
                    self.assertTrue(np.isfinite(np.exp(lower)))
                    self.assertFalse(np.isfinite(np.exp(upper)))
                self.assertEqual(report["bracket_width"], float(upper - lower))
                self.assertAlmostEqual(report["log_max"], float(np.log(np.array(np.finfo(dtype).max, dtype=dtype))), places=3)

    def test_rejects_nonfloating_or_extended_type(self) -> None:
        with self.assertRaises(ValueError):
            exp_range_audit(np.int32)


if __name__ == "__main__":
    unittest.main(verbosity=2)
