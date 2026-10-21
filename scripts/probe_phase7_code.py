"""Live probe for Phase 7 modules. Written before any run.

Prints one line per check so a reader can see which assertion carried
which number. Fails loudly on the first hard error.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from henri.code_verify import executor_status, solve_verified_code  # noqa: E402
from henri.semantic_backbone import SemanticBackbone  # noqa: E402


def main() -> int:
    print("=== A: executor loads by path ===")
    st = executor_status()
    print("executor_status:", st)
    if not st.get("available"):
        print("FAIL: executor not loadable")
        return 2

    print("\n=== B: solve without tests is UNVERIFIABLE, not a pass ===")
    bb = SemanticBackbone()
    r = solve_verified_code(bb, "return the sum of two numbers",
                            ["def add(a, b):\n    return a + b\n"])
    print("verdict:", r["verdict"])
    print("n_tests:", r["n_tests"], "n_executed_absent:", "executed" in r)
    assert r["verdict"] == "NO_TESTS_UNVERIFIABLE", r["verdict"]
    assert r["executed"] == [], r["executed"]

    print("\n=== C: solve WITH tests, correct candidate must pass ===")
    good = "def add(a, b):\n    return a + b\n"
    bad = "def add(a, b):\n    return a - b\n"
    tests = ["assert add(2, 3) == 5", "assert add(0, 0) == 0"]
    r2 = solve_verified_code(bb, "return the sum of two numbers",
                             [bad, good], tests=tests)
    print("verdict:", r2["verdict"])
    print("executed:", r2["executed"])
    print("winner:", r2["winner"])
    print("tie_group_size:", r2["tie_group_size"])
    assert r2["verdict"] == "EXEC_VERIFIED", r2["verdict"]

    print("\n=== D: all-wrong pool must NOT verify ===")
    r3 = solve_verified_code(bb, "sum", [bad, "def add(a, b):\n    return 0\n"],
                             tests=tests)
    print("verdict:", r3["verdict"])
    assert r3["verdict"] == "NO_CANDIDATES_PASS", r3["verdict"]

    print("\n=== E: empty pool ===")
    r4 = solve_verified_code(bb, "sum", [], tests=tests)
    print("verdict:", r4["verdict"])
    assert r4["verdict"] == "NO_CANDIDATES"

    print("\nALL CODE_VERIFY CHECKS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
