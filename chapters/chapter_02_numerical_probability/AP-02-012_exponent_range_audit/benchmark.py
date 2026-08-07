"""Diagnostic benchmark for AP-02-012: Find the Exponential Overflow Frontier.

English: Read statement.tex before editing or interpreting this file.
中文：编辑或解释本文件前，请先阅读 statement.tex。
"""

from __future__ import annotations

import numpy as np

from starter import exp_range_audit


def main() -> None:
    """Print the dtype-dependent frontier table. / 打印依赖类型的边界表。"""
    try:
        reports = [(dtype.__name__, exp_range_audit(dtype)) for dtype in (np.float16, np.float32, np.float64)]
    except NotImplementedError as exc:
        print(f"[UNIMPLEMENTED] {exc}")
        raise SystemExit(2) from exc
    print("dtype      log_max            last_finite        first_overflow      bracket_width")
    for name, report in reports:
        print(
            f"{name:<10} {report['log_max']:>17.10g} {report['last_finite']:>17.10g} "
            f"{report['first_overflow']:>19.10g} {report['bracket_width']:>18.10g}"
        )


if __name__ == "__main__":
    main()
