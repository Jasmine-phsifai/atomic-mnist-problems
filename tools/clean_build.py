#!/usr/bin/env python3
"""Remove only generated build/output trees. / 仅删除生成的构建与输出目录。"""

from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Delete two explicit, repository-local targets. / 删除两个明确的仓库内目标。"""
    for name in ("build", "output"):
        target = (ROOT / name).resolve()
        if target.parent != ROOT.resolve():
            raise RuntimeError(f"refusing unexpected target: {target}")
        if target.exists():
            shutil.rmtree(target)
            print(f"[REMOVED] {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
