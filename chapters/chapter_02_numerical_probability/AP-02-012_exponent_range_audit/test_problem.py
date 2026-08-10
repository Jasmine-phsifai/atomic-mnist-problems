"""Contract tests for AP-02-012: Find the Exponential Overflow Frontier.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import unittest
import warnings

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
                # Real float64 estimate, not a dtype-rounded log. / 实数估计值，非 dtype 舍入。
                self.assertAlmostEqual(report["log_max"], float(np.log(float(np.finfo(dtype).max))), places=3)

    def test_probe_suppresses_the_overflow_warning(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            exp_range_audit(np.float32)

    def test_rejects_nonfloating_or_extended_type(self) -> None:
        for bad in (np.int32, np.complex128, np.dtype(object)):
            with self.subTest(dtype=bad):
                with self.assertRaises(ValueError):
                    exp_range_audit(bad)


if __name__ == "__main__":
    unittest.main(verbosity=2)
