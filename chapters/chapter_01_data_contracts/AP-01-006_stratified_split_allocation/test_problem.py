"""Contract tests for AP-01-006: Deterministic Stratification by Integer Apportionment.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import unittest

import numpy as np

from starter import stratified_split


class StratifiedSplitTests(unittest.TestCase):
    """Allocation and partition tests. / 分配与划分测试。"""

    def test_hamilton_counts_and_partition(self) -> None:
        labels = np.repeat(np.arange(3, dtype=np.int64), [3, 5, 7])
        train, valid = stratified_split(labels, valid_fraction=0.3, seed=17)
        np.testing.assert_array_equal(np.bincount(labels[valid], minlength=3), [1, 1, 2])
        self.assertEqual(valid.size, round(0.3 * labels.size))
        self.assertEqual(np.intersect1d(train, valid).size, 0)
        np.testing.assert_array_equal(np.sort(np.concatenate([train, valid])), np.arange(labels.size))
        self.assertEqual(np.unique(train).size, train.size)
        self.assertEqual(np.unique(valid).size, valid.size)
        self.assertEqual(train.dtype, np.int64)
        self.assertEqual(valid.dtype, np.int64)

    def test_tied_remainders_favor_smaller_class_label(self) -> None:
        # Quotas 0.5, 0.5, 1.0 -> remainders tie between classes 0 and 1;
        # the single leftover seat must go to class 0. / 余数并列时座位归较小类别。
        labels = np.repeat(np.arange(3, dtype=np.int64), [1, 1, 2])
        _, valid = stratified_split(labels, valid_fraction=0.5, seed=3)
        np.testing.assert_array_equal(np.bincount(labels[valid], minlength=3), [1, 0, 1])

    def test_seed_reproducibility_without_allocation_drift(self) -> None:
        labels = np.repeat(np.arange(4, dtype=np.int64), [10, 11, 12, 13])
        a_train, a_valid = stratified_split(labels, valid_fraction=0.25, seed=5)
        b_train, b_valid = stratified_split(labels, valid_fraction=0.25, seed=5)
        c_train, c_valid = stratified_split(labels, valid_fraction=0.25, seed=6)
        np.testing.assert_array_equal(a_train, b_train)
        np.testing.assert_array_equal(a_valid, b_valid)
        self.assertFalse(np.array_equal(a_valid, c_valid))
        np.testing.assert_array_equal(
            np.bincount(labels[a_valid], minlength=4),
            np.bincount(labels[c_valid], minlength=4),
        )

    def test_rejects_invalid_domain(self) -> None:
        for labels, fraction in [
            (np.array([], dtype=np.int64), 0.2),
            (np.array([0.0, 1.0]), 0.2),
            (np.array([0, -1]), 0.2),
            (np.array([0, 1]), 0.0),
            (np.array([0, 1]), 1.0),
        ]:
            with self.assertRaises(ValueError):
                stratified_split(labels, valid_fraction=fraction, seed=1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
