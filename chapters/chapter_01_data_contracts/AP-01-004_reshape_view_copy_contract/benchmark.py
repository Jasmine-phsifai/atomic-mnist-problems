"""Diagnostic benchmark for AP-01-004: Flattening: Shape Is Not Ownership.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import numpy as np

from starter import flatten_images


def main() -> None:
    """Print the alias/mutation truth table. / 打印别名与变更传播真值表。"""
    images = np.arange(4 * 28 * 28, dtype=np.float32).reshape(4, 28, 28)
    try:
        shared = flatten_images(images, require_independent=False)
        owned = flatten_images(images, require_independent=True)
    except NotImplementedError as exc:
        print(f"[UNIMPLEMENTED] {exc}")
        raise SystemExit(2) from exc
    original = float(images[0, 0, 0])
    shared[0, 0] = original + 100.0
    print(f"contiguous_shared_alias={np.shares_memory(images, shared)}")
    print(f"contiguous_owned_alias={np.shares_memory(images, owned)}")
    print(f"shared_mutation_reached_source={images[0, 0, 0] == original + 100.0}")
    print(f"owned_bytes={owned.nbytes}")


if __name__ == "__main__":
    main()
