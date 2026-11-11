"""Cycle X3: carry-latched and ripple-insert one-cell closure (12/12).

    python scripts/run_holarch_x3_turing_closure.py

Runs the shared one-cell induction audit in this process and seals:

  * 12/12 tasks induced and length-generalised on L in [12, 32]
  * 4/4 contradictory specifications refused
  * every transition reads and writes exactly one cell and moves by one
  * zero text tokens and max_nodes_used <= 320000
  * task_relabelled == false (the declared task family is unchanged)

No held-out pair is consulted during induction.  The two new micro-
architectures are the declared ones: Q = Qctrl x R_sym x C_latch with
(sum, next_carry) = full_add(...), and the strict one-cell ripple_insert
sweep.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_holarch_p_cycle as pc
from henri.holarch.induction import TASK_TRANSFORMS, SovereignInducer, sovereign_tasks_v2
from henri.holarch.onecell_audit import audit_induction

RECEIPTS = ROOT / "receipts" / "holarch"


def _strict_one_cell(tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Every transition reads and writes one cell and moves by one."""
    rows = 0
    bad: List[str] = []
    for row in tasks:
        table = dict(row.get("table") or {})
        for (state, symbol), action in table.items():
            rows += 1
            next_state, write, move = action
            if not isinstance(state, int):
                bad.append(f"{row['name']}:state {state!r}")
            if not isinstance(symbol, str) or len(symbol) > 2:
                bad.append(f"{row['name']}:symbol {symbol!r}")
            if not isinstance(write, str) or len(write) > 2:
                bad.append(f"{row['name']}:write {write!r}")
            if int(move) not in (-1, 0, 1):
                bad.append(f"{row['name']}:move {move!r}")
            if int(move) == 0 and int(next_state) != -1:
                bad.append(f"{row['name']}:zero move without halt")
    return {"rows": int(rows), "violations": bad[:16], "strict": bool(not bad)}


def measure() -> Dict[str, Any]:
    started = time.perf_counter()
    audit = audit_induction(node_budget=320000, max_states=8)
    tasks = list(audit["tasks"])
    # audit_induction strips the table from every row, so the strict one-cell
    # check re-runs the same inducer on the same specs and audits the actual
    # transition tables: every row must read and write one cell and move by 1.
    inducer = SovereignInducer(node_budget=320000, max_states=8)
    tables = [
        {"name": str(spec.name), "table": dict(inducer.induce(spec).get("table") or {})}
        for spec in sovereign_tasks_v2()
    ]
    strict = _strict_one_cell(tables)
    max_nodes = int(max(int(row.get("nodes_evaluated", 0)) for row in tasks))
    mechanisms = sorted({str(row.get("mechanism_used", "")) for row in tasks})
    names = [str(row["name"]) for row in tasks]
    declared = [str(spec.name) for spec in sovereign_tasks_v2()]
    transforms: Dict[str, str] = {}
    for name, transform in TASK_TRANSFORMS.items():
        probe = "a#x#z" if name == "delimiter_bounded_copy" else ("|||" if (
            name.startswith("unary")) else "0110")
        try:
            value = str(transform(str(probe)))
        except Exception as error:  # noqa: BLE001 - a probe mismatch is recorded
            value = type(error).__name__
        transforms[name] = hashlib.sha256(
            json.dumps([str(probe), value], sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
    carry = next((row for row in tasks if row["name"] == "two_bit_carry_add"), {})
    copy = next((row for row in tasks if row["name"] == "delimiter_bounded_copy"), {})
    gates = {
        "x3_tasks_solved_12": bool(int(audit["solved"]) == 12),
        "x3_generalized_12": bool(int(audit["generalized"]) == 12),
        "x3_length_generalized_12": bool(int(audit["v1_length_generalized"]) == 12),
        "x3_contradictions_4": bool(int(audit["contradictions_failed_closed"]) == 4),
        "x3_carry_add_4_of_4": bool(
            carry.get("status") == "INDUCED" and int(carry.get("v1_exact", 0)) == 4
            and str(carry.get("mechanism_used")) == "carry_latch_macro"),
        "x3_delimiter_copy_4_of_4": bool(
            copy.get("status") == "INDUCED" and int(copy.get("v1_exact", 0)) == 4
            and str(copy.get("mechanism_used")) == "ripple_insert_macro"),
        "x3_task_relabelled_false": bool(names == declared),
        "x3_max_nodes_le_320000": bool(max_nodes <= 320000),
        "x3_strict_one_cell": bool(strict["strict"] and int(strict["rows"]) > 0),
        "x3_text_tokens_zero": bool(int(audit["text_tokens"]) == 0),
        "x3_two_cell_excluded": bool(audit["foveated_counted_toward_gate"] is False),
    }
    return {
        "stage": "X3",
        "tasks": [
            {
                "name": str(row["name"]),
                "status": str(row.get("status", "")),
                "mechanism": str(row.get("mechanism_used", "")),
                "nodes": int(row.get("nodes_evaluated", 0)),
                "states": int(row.get("state_count_q", 0)),
                "test_exact": int(row.get("test_exact", 0)),
                "v1_exact": int(row.get("v1_exact", 0)),
                "v1_total": int(row.get("v1_total", 0)),
                "text_tokens": int(row.get("text_tokens", 0)),
            }
            for row in tasks
        ],
        "solved": int(audit["solved"]),
        "generalized": int(audit["generalized"]),
        "v1_length_generalized": int(audit["v1_length_generalized"]),
        "v1_lengths": list(audit["v1_lengths"]),
        "contradictions": list(audit["contradictions"]),
        "contradictions_failed_closed": int(audit["contradictions_failed_closed"]),
        "max_nodes_used": int(max_nodes),
        "node_budget": int(audit["node_budget"]),
        "mechanisms": mechanisms,
        "strict_one_cell": dict(strict),
        "two_cell_excluded": bool(audit["foveated_counted_toward_gate"] is False),
        "text_tokens": int(audit["text_tokens"]),
        "task_relabelled": bool(names != declared),
        "declared_family": transforms,
        "carry_latch": {
            "present": bool(str(carry.get("mechanism_used")) == "carry_latch_macro"),
            "states": int(carry.get("state_count_q", 0)),
            "latch_bits": 1,
            "control_states": 4,
            "register_symbols": 1,
        },
        "ripple_insert": {
            "present": bool(str(copy.get("mechanism_used")) == "ripple_insert_macro"),
            "states": int(copy.get("state_count_q", 0)),
            "sweep": "strict one-cell swap-and-ripple",
        },
        "gate": pc._finalise(gates),
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
        "passed": bool(all(gates.values())),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Cycle X3 one-cell closure")
    parser.add_argument("--receipts", default=str(RECEIPTS))
    args = parser.parse_args(argv)
    directory = Path(args.receipts)
    directory.mkdir(parents=True, exist_ok=True)
    payload = measure()
    written = pc._write(directory, "x3_twelve_of_twelve_turing.json", payload)
    print("[x3] solved=%d generalized=%d v1=%d contradictions=%d max_nodes=%d strict=%s" % (
        payload["solved"], payload["generalized"], payload["v1_length_generalized"],
        payload["contradictions_failed_closed"], payload["max_nodes_used"],
        payload["strict_one_cell"]["strict"]))
    print("[x3] passed=%s receipt=%s" % (payload["passed"], written["path"]))
    return 0 if bool(payload["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())