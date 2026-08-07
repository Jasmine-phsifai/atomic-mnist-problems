#!/usr/bin/env python3
"""Build the main and per-problem PDFs. / 构建总册与每题独立 PDF。"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build" / "latex"
OUTPUT = ROOT / "output" / "pdf"


def compile_tex(tex_file: Path, destination: Path) -> None:
    """Compile in the source directory so relative inputs stay valid. / 在源目录编译以保持相对输入有效。"""
    job_build = BUILD / destination.stem
    job_build.mkdir(parents=True, exist_ok=True)
    command = [
        "latexmk",
        "-xelatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        f"-outdir={job_build}",
        tex_file.name,
    ]
    subprocess.run(command, cwd=tex_file.parent, check=True)
    produced = job_build / f"{tex_file.stem}.pdf"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(produced, destination)


def main() -> int:
    """Build all 29 deliverable PDFs. / 构建全部 29 份交付 PDF。"""
    subprocess.run([sys.executable, str(ROOT / "tools" / "validate_scaffold.py")], check=True)
    with (ROOT / "catalog.toml").open("rb") as handle:
        problems = tomllib.load(handle)["problem"]

    compile_tex(ROOT / "book" / "main.tex", OUTPUT / "atomic-mnist-problems.pdf")
    for item in problems:
        folder = ROOT / item["path"]
        filename = f"{item['id']}_{item['slug']}.pdf"
        compile_tex(folder / "problem.tex", OUTPUT / "problems" / filename)

    print(f"[PASS] built 1 main PDF and {len(problems)} individual PDFs")
    print(f"[PASS] output: {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
