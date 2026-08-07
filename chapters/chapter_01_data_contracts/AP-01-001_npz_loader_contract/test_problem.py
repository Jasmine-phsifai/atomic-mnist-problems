"""Contract tests for AP-01-001: A Trust-Bounded NPZ Loader.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from starter import load_npz_array


class LoaderContractTests(unittest.TestCase):
    """Executable loader contract. / 可执行的加载器契约。"""

    def test_safe_round_trip_and_closed_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "images.npz"
            expected = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
            np.savez_compressed(path, images=expected)
            actual = load_npz_array(path, expected_key="images")
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(int(actual[1, 2, 3]), 23)  # Archive is already closed. / 归档此时已关闭。

    def test_rejects_wrong_container_and_key_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.save(root / "images.npy", np.zeros(2, dtype=np.uint8))
            with self.assertRaises(ValueError):
                load_npz_array(root / "images.npy", expected_key="images")
            np.savez(root / "missing.npz", labels=np.zeros(2, dtype=np.uint8))
            with self.assertRaises((KeyError, ValueError)):
                load_npz_array(root / "missing.npz", expected_key="images")
            np.savez(root / "extra.npz", images=np.zeros(2), labels=np.zeros(2))
            with self.assertRaises(ValueError):
                load_npz_array(root / "extra.npz", expected_key="images")

    def test_rejects_object_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "object.npz"
            np.savez(path, images=np.array([{"not": "numeric"}], dtype=object))
            with self.assertRaises(ValueError):
                load_npz_array(path, expected_key="images")


if __name__ == "__main__":
    unittest.main(verbosity=2)
