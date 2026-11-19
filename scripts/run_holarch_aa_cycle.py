"""Cycle AA aggregate ledger.

    python scripts/run_holarch_aa_cycle.py

The ledger does not re-measure a stage: it reads the sealed stage receipts,
checks their SHA-256 sidecars, and records the supersession chain.  A stage is
green only when its receipt says passed and its sidecar matches.  The prior
ledgers t, u, v, w and x must stay passed; y and z stay false and are recorded
as superseded by this file.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_holarch_p_cycle as pc

RECEIPTS = ROOT / "receipts" / "holarch"
STAGES = {
    "aa1": "aa1_per_token_phase_egress.json",
    "aa2": "aa2_external_prose_and_loop_synthesis.json",
    "aa3": "aa3_external_arc_agi_audit.json",
    "aa4": "aa4_complete_cascade_and_aaii80.json",
    "aa5": "aa5_hardware_and_ledger_closure.json",
}
SECRET_PATTERNS = (
    re.compile(r"sk-or-v1-[A-Za-z0-9]{16,}"),
    re.compile(r"sk-[A-Za-z0-9]{32,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9\-\._~\+/]{24,}"),
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - a corrupt receipt is not a pass
        return {}


def _sidecar_ok(path: Path) -> bool:
    sidecar = path.with_name(path.name + ".sha256")
    return bool(
        sidecar.exists()
        and sidecar.read_text(encoding="utf-8").strip() == _sha(path)
    )


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Cycle AA ledger")
    parser.add_argument("--receipts", default=str(RECEIPTS))
    args = parser.parse_args(argv)
    directory = Path(args.receipts)
    started = time.perf_counter()
    stages: Dict[str, Any] = {}
    for name, filename in STAGES.items():
        path = directory / filename
        payload = _read(path)
        gate = payload.get("gate") or {}
        stages[name] = {
            "receipt": str(path.relative_to(ROOT)) if path.exists() else "",
            "sha256": _sha(path) if path.exists() else "",
            "sidecar_ok": _sidecar_ok(path) if path.exists() else False,
            "passed": bool(payload.get("passed", False)),
            "gate": gate,
            "failed_gates": [
                key for key, value in gate.items() if not bool(value)
            ],
            "wall_seconds": float(payload.get("wall_seconds", 0.0) or 0.0),
        }
    history = {}
    for name in ("t", "u", "v", "w", "x", "y", "z"):
        payload = _read(directory / ("%s_cycle.json" % name))
        history[name] = {
            "passed": bool(payload.get("passed", False)),
            "present": bool(payload),
        }
    docs = (ROOT / "docs" / "HOLARCH.md").read_text(encoding="utf-8")
    leaks: List[str] = []
    for path in directory.glob("*.json"):
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                leaks.append(path.name)
                break
    gates = {
        "aa_prior_cycles_passed": bool(
            all(history[name]["passed"] for name in ("t", "u", "v", "w", "x"))
        ),
        "aa_yz_historical_false": bool(
            not history["y"]["passed"] and not history["z"]["passed"]
        ),
        "aa_all_stage_sidecars_match": bool(
            all(row["sidecar_ok"] for row in stages.values())
        ),
        "aa_docs_section_42": bool("## 42." in docs),
        "aa_zero_credential_leaks": bool(not leaks),
    }
    payload = {
        "cycle": "AA",
        "supersedes": ["y_cycle.json", "z_cycle.json"],
        "stages": stages,
        "history": history,
        "closure": {
            "docs_section_42": "## 42." in docs,
            "credential_leaks": leaks,
            "prior_cycles": history,
        },
        "gate": pc._finalise(gates),
        "passed": bool(
            all(row["passed"] for row in stages.values()) and all(gates.values())
        ),
        "stage_summary": {
            name: row["passed"] for name, row in stages.items()
        },
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
    }
    written = pc._write(directory, "aa_cycle.json", payload)
    print(
        "[aa] stages=%s closure=%s passed=%s receipt=%s"
        % (
            payload["stage_summary"],
            gates,
            payload["passed"],
            written["path"],
        )
    )
    return 0 if bool(payload["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
