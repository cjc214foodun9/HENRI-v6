
"""Cycle U full closure: every stage suite plus one terminal gauntlet.

    python scripts/run_holarch_full_closure.py

The historical cycle runners run with --skip-gauntlet, so their stage
batteries verify their own contracts and the clean-room gauntlet runs once,
here, as the terminal gate.  Every command is a real child process; the
manifest records its exit code, wall time and output tail.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
RECEIPTS = ROOT / "receipts" / "holarch"
PYTHON = sys.executable


def _commands(include_pytest: bool = True) -> List[Tuple[str, List[str]]]:
    commands: List[Tuple[str, List[str]]] = [
        ("seed_cartridge", [PYTHON, str(ROOT / "scripts" / "build_holarch_seed_cartridge.py")]),
        ("aaii_v5_eval", [PYTHON, str(ROOT / "scripts" / "run_holarch_aaii_v5_eval.py")]),
    ]
    for cycle in ("j", "k", "m", "o", "p", "q", "r", "s", "t"):
        commands.append((f"cycle_{cycle}", [PYTHON, str(ROOT / "scripts" / f"run_holarch_{cycle}_cycle.py"),
                                            "--skip-gauntlet"]))
    commands.append(("t_cycle", [PYTHON, str(ROOT / "scripts" / "run_holarch_ts_cycle.py"),
                                  "--stages", "t1,t2,t3,t4"]))
    commands.append(("gauntlet", [PYTHON, str(ROOT / "scripts" / "run_holarch_agent_eval.py"),
                                   "--out", str(RECEIPTS / "agent_eval.json")]))
    commands.append(("u_cycle", [PYTHON, str(ROOT / "scripts" / "run_holarch_u_cycle.py")]))
    if include_pytest:
        commands.append(("pytest_holarch", [PYTHON, "-m", "pytest", str(ROOT / "tests"),
                                             "-q", "-k", "holarch"]))
    commands.append(("verify_consistency", [PYTHON, str(ROOT / "scripts" / "verify_holarch_consistency.py")]))
    return commands


def main() -> int:
    parser = argparse.ArgumentParser(description="Cycle U full closure")
    parser.add_argument("--only", default="", help="comma separated command names")
    parser.add_argument("--no-pytest", action="store_true")
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()
    wanted = {token.strip() for token in str(args.only).split(",") if token.strip()}
    commands = [row for row in _commands(include_pytest=not args.no_pytest)
                if not wanted or row[0] in wanted]
    rows: List[Dict[str, object]] = []
    started = time.perf_counter()
    for name, command in commands:
        print(f"[closure] {name} ...", flush=True)
        began = time.perf_counter()
        try:
            completed = subprocess.run(command, cwd=str(ROOT), capture_output=True,
                                       text=True, timeout=int(args.timeout))
            returncode = int(completed.returncode)
            stdout = completed.stdout[-6000:]
            stderr = completed.stderr[-4000:]
        except subprocess.TimeoutExpired as error:
            returncode = 124
            stdout = str(getattr(error, "stdout", "") or "")[-6000:]
            stderr = f"timeout after {args.timeout}s"
        seconds = float(time.perf_counter() - began)
        rows.append({
            "name": name,
            "command": [str(value) for value in command],
            "returncode": returncode,
            "wall_seconds": seconds,
            "stdout_tail": stdout,
            "stderr_tail": stderr,
        })
        print(f"[closure] {name} rc={returncode} {seconds:.1f}s", flush=True)
    payload = {
        "cycle": "U",
        "commands": rows,
        "total_seconds": float(time.perf_counter() - started),
        "passed": bool(rows and all(int(row["returncode"]) == 0 for row in rows)),
        "evidence": "OBSERVED",
    }
    path = RECEIPTS / "closure_manifest.json"
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    (path.with_name(path.name + ".sha256")).write_text(
        hashlib.sha256(path.read_bytes()).hexdigest() + "\n", encoding="utf-8")
    print(json.dumps({"passed": payload["passed"], "commands": len(rows),
                      "total_seconds": payload["total_seconds"]}, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
