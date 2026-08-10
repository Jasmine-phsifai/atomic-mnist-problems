"""Contract tests for AP-01-003: Pixel Normalization Without Integer Surprises.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import unittest

import numpy as np

from starter import normalize_uint8


class NormalizationContractTests(unittest.TestCase):
    """Numerical and ownership tests. / 数值与存储所有权测试。"""

    def test_endpoints_levels_dtype_and_ownership(self) -> None:
        source = np.array([0, 1, 254, 255], dtype=np.uint8).reshape(1, 2, 2)
        snapshot = source.copy()
        actual = normalize_uint8(source, dtype=np.float32)
        expected = snapshot.astype(np.float32) / np.float32(255.0)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.dtype, np.float32)
        self.assertFalse(np.shares_memory(source, actual))
        np.testing.assert_array_equal(source, snapshot)

    def test_all_levels_and_float64(self) -> None:
        source = np.arange(256, dtype=np.uint8).reshape(1, 16, 16)[:, :, ::-1]
        actual = normalize_uint8(source, dtype=np.float64)
        self.assertEqual(np.unique(actual).size, 256)
        self.assertTrue(np.all((0.0 <= actual) & (actual <= 1.0)))
        np.testing.assert_allclose(actual, source.astype(np.float64) / 255.0, rtol=0.0, atol=0.0)
        actual32 = normalize_uint8(source, dtype=np.float32)
        self.assertEqual(np.unique(actual32).size, 256)

    def test_default_dtype_and_keyword_only(self) -> None:
        source = np.arange(4, dtype=np.uint8).reshape(2, 2)
        actual = normalize_uint8(source)
        self.assertEqual(actual.dtype, np.float32)
        with self.assertRaises(TypeError):
            normalize_uint8(source, np.float64)  # dtype is keyword-only. / dtype 仅接受关键字传参。

    def test_rejects_bad_dtypes(self) -> None:
        with self.assertRaises(ValueError):
            normalize_uint8(np.zeros((1, 28, 28), dtype=np.int16))
        with self.assertRaises(ValueError):
            normalize_uint8(np.zeros((1, 28, 28), dtype=np.uint8), dtype=np.int32)


if __name__ == "__main__":
    unittest.main(verbosity=2)
