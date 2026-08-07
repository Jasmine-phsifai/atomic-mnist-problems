#!/usr/bin/env python3
"""Print the curriculum catalog. / 打印课程题目目录。"""

from __future__ import annotations

import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Render one concise line per problem. / 每题输出一行简明信息。"""
    with (ROOT / "catalog.toml").open("rb") as handle:
        problems = tomllib.load(handle)["problem"]
    for item in problems:
        print(f"{item['id']}  D{item['difficulty']}  {item['title']}")


if __name__ == "__main__":
    main()
