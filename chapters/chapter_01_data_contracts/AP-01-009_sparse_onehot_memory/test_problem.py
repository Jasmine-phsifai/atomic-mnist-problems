"""Contract tests for AP-01-009: Sparse Labels and One-Hot Storage.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import unittest

import numpy as np

from starter import one_hot


class OneHotContractTests(unittest.TestCase):
    """Representation tests. / 表示形式测试。"""

    def test_indicator_invariants_and_dtype(self) -> None:
        labels = np.array([2, 0, 3, 2], dtype=np.int64)
        snapshot = labels.copy()
        encoded = one_hot(labels, num_classes=4, dtype=np.float32)
        self.assertEqual(encoded.shape, (4, 4))
        self.assertEqual(encoded.dtype, np.float32)
        self.assertTrue(encoded.flags.c_contiguous)
        np.testing.assert_array_equal(encoded.sum(axis=1), np.ones(4, dtype=np.float32))
        np.testing.assert_array_equal(encoded.argmax(axis=1), labels)
        np.testing.assert_array_equal(labels, snapshot)

    def test_float64_and_repeated_labels(self) -> None:
        labels = np.array([1, 1, 1], dtype=np.uint8)
        encoded = one_hot(labels, num_classes=3, dtype=np.float64)
        np.testing.assert_array_equal(encoded, np.array([[0, 1, 0]] * 3, dtype=np.float64))

    def test_rejects_invalid_domain(self) -> None:
        cases = [
            (np.array([[0, 1]]), 2, np.float32),
            (np.array([0.0, 1.0]), 2, np.float32),
            (np.array([-1, 0]), 2, np.float32),
            (np.array([0, 2]), 2, np.float32),
            (np.array([0, 1]), 0, np.float32),
            (np.array([0, 1]), 2, np.int32),
        ]
        for labels, classes, dtype in cases:
            with self.assertRaises(ValueError):
                one_hot(labels, num_classes=classes, dtype=dtype)


if __name__ == "__main__":
    unittest.main(verbosity=2)
