"""Diagnostic benchmark for AP-01-001: A Trust-Bounded NPZ Loader.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np

from starter import load_npz_array


def main() -> None:
    """Print observational I/O evidence. / 打印观察性的 I/O 证据。"""
    rng = np.random.default_rng(7)
    fixture = rng.integers(0, 256, size=(6000, 28, 28), dtype=np.uint8)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "images.npz"
        np.savez_compressed(path, images=fixture)
        start = time.perf_counter()
        try:
            loaded = load_npz_array(path, expected_key="images")
        except NotImplementedError as exc:
            print(f"[UNIMPLEMENTED] {exc}")
            raise SystemExit(2) from exc
        elapsed = time.perf_counter() - start
        mib = loaded.nbytes / (1024**2)
        print(f"payload_mib={mib:.3f}")
        print(f"wall_seconds={elapsed:.6f}")
        print(f"payload_mib_per_second={mib / max(elapsed, 1e-12):.3f}")
        print(f"post_close_checksum={int(loaded.astype(np.uint64).sum())}")


if __name__ == "__main__":
    main()
