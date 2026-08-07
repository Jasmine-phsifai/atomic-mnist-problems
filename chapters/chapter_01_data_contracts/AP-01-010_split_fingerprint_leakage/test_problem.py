"""Contract tests for AP-01-010: Content Fingerprints for Split Leakage.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import unittest

import numpy as np

from starter import cross_split_duplicates, fingerprint_rows


class FingerprintLeakageTests(unittest.TestCase):
    """Content-identity tests. / 内容同一性测试。"""

    def test_exact_pairs_include_copied_rows(self) -> None:
        train = np.array([[[0, 1]], [[2, 3]], [[0, 1]]], dtype=np.uint8)
        valid = np.array([[[9, 9]], [[0, 1]], [[2, 3]]], dtype=np.uint8).copy()
        self.assertFalse(np.shares_memory(train, valid))
        pairs = cross_split_duplicates(train, valid)
        self.assertEqual(pairs, [(0, 1), (1, 2), (2, 1)])

    def test_pixel_and_dtype_are_in_digest_domain(self) -> None:
        base = np.array([[[1, 2], [3, 4]]], dtype=np.uint8)
        changed = base.copy()
        changed[0, 1, 1] = 5
        wider = base.astype(np.uint16)
        h_base = fingerprint_rows(base)[0]
        self.assertEqual(len(str(h_base)), 64)
        self.assertNotEqual(h_base, fingerprint_rows(changed)[0])
        self.assertNotEqual(h_base, fingerprint_rows(wider)[0])

    def test_rejects_incompatible_rows(self) -> None:
        with self.assertRaises(ValueError):
            fingerprint_rows(np.array(3))
        with self.assertRaises(ValueError):
            cross_split_duplicates(np.zeros((2, 2)), np.zeros((2, 3)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
