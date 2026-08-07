#!/usr/bin/env python3
"""Validate structure without solving exercises. / 验证脚手架结构而不求解习题。"""

from __future__ import annotations

import ast
import re
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {
    "README.md",
    "statement.tex",
    "problem.tex",
    "starter.py",
    "test_problem.py",
    "benchmark.py",
}


def fail(message: str) -> None:
    """Print one failure and stop. / 打印一条失败信息并停止。"""
    raise SystemExit(f"[FAIL] {message}")


def main() -> int:
    """Check counts, links, syntax, and teaching markers. / 检查数量、链接、语法与教学标记。"""
    catalog_path = ROOT / "catalog.toml"
    if not catalog_path.is_file():
        fail("catalog.toml is missing")

    with catalog_path.open("rb") as handle:
        problems = tomllib.load(handle).get("problem", [])

    if len(problems) != 28:
        fail(f"expected 28 catalog entries, found {len(problems)}")

    counts: dict[int, int] = {}
    seen_ids: set[str] = set()
    main_text = (ROOT / "book" / "main.tex").read_text(encoding="utf-8")

    for item in problems:
        problem_id = item["id"]
        chapter = int(item["chapter"])
        folder = ROOT / item["path"]
        counts[chapter] = counts.get(chapter, 0) + 1
        if problem_id in seen_ids:
            fail(f"duplicate id: {problem_id}")
        seen_ids.add(problem_id)

        missing = sorted(name for name in REQUIRED if not (folder / name).is_file())
        if missing:
            fail(f"{problem_id} missing: {', '.join(missing)}")

        statement = (folder / "statement.tex").read_text(encoding="utf-8")
        wrapper = (folder / "problem.tex").read_text(encoding="utf-8")
        starter = (folder / "starter.py").read_text(encoding="utf-8")

        if problem_id not in statement:
            fail(f"{problem_id} is absent from its statement")
        if "\\input{statement.tex}" not in wrapper:
            fail(f"{problem_id} wrapper does not input canonical statement.tex")
        if f"../{item['path']}/statement.tex" not in main_text:
            fail(f"{problem_id} is absent from the main book")
        if "NotImplementedError" not in starter:
            fail(f"{problem_id} starter has no explicit unfinished body")
        if not re.search(r"[\u4e00-\u9fff]", starter):
            fail(f"{problem_id} starter lacks Chinese teaching annotation")

        for py_file in folder.glob("*.py"):
            source = py_file.read_text(encoding="utf-8")
            try:
                ast.parse(source, filename=str(py_file))
            except SyntaxError as exc:
                fail(f"syntax error in {py_file.relative_to(ROOT)}: {exc}")
            if not re.search(r"[\u4e00-\u9fff]", source):
                fail(f"{py_file.relative_to(ROOT)} lacks bilingual annotation")

    if counts != {1: 12, 2: 16}:
        fail(f"expected chapter counts {{1: 12, 2: 16}}, found {counts}")

    # Parse repository tools too. / 同时解析仓库工具脚本。
    for py_file in (ROOT / "tools").glob("*.py"):
        ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))

    print("[PASS] 28 folders: chapter 1 = 12, chapter 2 = 16")
    print("[PASS] canonical LaTeX inputs, bilingual Python, and Python syntax")
    print("[NOTE] learner implementations remain intentionally unfinished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
