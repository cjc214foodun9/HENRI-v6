"""DIAGNOSTIC ONLY. No gate, no bar, no verdict, no code change.

WHY THIS EXISTS
`G-SYNTH2` (4ceabd3) measured the held-out residual histogram:

    exception      105
    type_mismatch   91
    unclassified    50
    name_error      34
    sign_flip       18
    numeric_offset   5

`exception` is the largest class and currently earns ZERO repair budget. I
claimed it is "the next leverage". That was an inference from a count. This
script tests the claim by measuring WHICH exceptions occur.

The distinction is decision-relevant:
  - If 105 exceptions are 2-3 recurring types from one template family, a
    bounded repair exists and is worth pre-registering.
  - If they are scattered across a dozen types with no recurrence, no single
    bounded repair helps and the leverage claim is FALSIFIED.

This script establishes NO threshold, changes NO code, and registers NO gate.
It reports a histogram. That is all it does.
"""

import importlib.util
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DIM = 65536
RANK = 64
K_TEMPLATES = 8
MAX_VARIANTS = 2
TOP_K = K_TEMPLATES * (1 + MAX_VARIANTS)
N_TRAIN = 800
N_TASKS = 40


def main() -> int:
    t0 = time.time()
    from henri.api_server import get_api
    from henri.code_synth import CodeTemplateBank, Template
    from henri.code_verify import solve_verified_code

    bb = get_api().backbone

    p = REPO / "data" / "official_benchmarks" / "mbpp.jsonl"
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    train, held = rows[:N_TRAIN], rows[N_TRAIN:]

    bank = CodeTemplateBank(dim=DIM, rank=RANK)
    for r in train:
        bank.add(bb, Template(docstring=(r.get("text") or "").strip(),
                              source=(r.get("code") or "").strip(),
                              task_id=str(r.get("task_id"))))
    bank.finalize()
    print("bank %d  held %d" % (bank.rows, len(held)))

    kind_hist = Counter()
    exc_hist = Counter()
    msg_examples = defaultdict(list)
    exc_tasks = defaultdict(list)
    unclassified_notes = []

    tasks = held[:N_TASKS]
    for i, r in enumerate(tasks, 1):
        spec = (r.get("text") or "").strip()
        tests = list(r.get("test_list") or [])
        res = solve_verified_code(bb, spec, [], tests, bank=bank,
                                  synthesize_from_bank=True,
                                  k_templates=K_TEMPLATES,
                                  max_variants=MAX_VARIANTS,
                                  top_k=TOP_K, max_cegis_rounds=2)
        ceg = res.get("cegis") or {}
        for q in ceg.get("probes", []):
            rd = q.get("residual") or {}
            k = rd.get("kind")
            kind_hist[k] += 1
            if k == "exception":
                et = rd.get("exception_type") or "<none>"
                exc_hist[et] += 1
                tid = str(r.get("task_id"))
                if tid not in exc_tasks[et]:
                    exc_tasks[et].append(tid)
                if len(msg_examples[et]) < 3:
                    m = (rd.get("exception_msg") or "")
                    if m:
                        msg_examples[et].append(m[:110])
            if k == "unclassified" and len(unclassified_notes) < 8:
                unclassified_notes.append({
                    "task_id": str(r.get("task_id")),
                    "exception_type": rd.get("exception_type"),
                    "exception_msg": (rd.get("exception_msg") or "")[:110],
                    "actual_repr": str(rd.get("actual_repr"))[:60],
                    "expected_repr": str(rd.get("expected_repr"))[:60],
                })
        if i % 10 == 0:
            print("  %d/%d  %.0f s" % (i, len(tasks), time.time() - t0))

    print()
    print("=== residual kinds (DIAGNOSTIC, no verdict) ===")
    for k, c in kind_hist.most_common():
        print("  %-16s %d" % (k, c))

    print()
    print("=== exception subtypes, ranked ===")
    total_exc = sum(exc_hist.values())
    cum = 0
    for et, c in exc_hist.most_common():
        cum += c
        print("  %-22s %4d  %5.1f%%  cum %5.1f%%  tasks %s"
              % (et, c, 100.0 * c / max(total_exc, 1),
                 100.0 * cum / max(total_exc, 1),
                 ",".join(exc_tasks[et][:6])))
        for m in msg_examples[et]:
            print("      | %s" % m)

    print()
    print("=== concentration test ===")
    top1 = exc_hist.most_common(1)[0] if exc_hist else ("<none>", 0)
    top3 = sum(c for _, c in exc_hist.most_common(3))
    print("  distinct exception types: %d" % len(exc_hist))
    print("  top-1 %s = %d/%d (%.1f%%)"
          % (top1[0], top1[1], total_exc,
             100.0 * top1[1] / max(total_exc, 1)))
    print("  top-3 share = %d/%d (%.1f%%)"
          % (top3, total_exc, 100.0 * top3 / max(total_exc, 1)))

    print()
    print("=== unclassified samples ===")
    for u in unclassified_notes:
        print("  task %s %s %s" % (u["task_id"], u["exception_type"],
                                   u["exception_msg"]))
        print("      actual=%s expected=%s"
              % (u["actual_repr"], u["expected_repr"]))

    out = REPO / "receipts" / "diag_synth_exception_breakdown.json"
    out.write_text(json.dumps({
        "probe": "DIAGNOSTIC_synth_exception_breakdown",
        "note": ("Diagnostic only. No gate, no bar, no verdict. Establishes "
                 "no threshold and changes no code."),
        "corpus": {"train": len(train), "held": len(held),
                   "n_tasks": len(tasks)},
        "kind_histogram": dict(kind_hist),
        "exception_histogram": dict(exc_hist),
        "exception_tasks": {k: v for k, v in exc_tasks.items()},
        "exception_message_examples": {k: v for k, v in msg_examples.items()},
        "concentration": {
            "n_distinct": len(exc_hist),
            "top1": {"type": top1[0], "count": top1[1]},
            "top3_count": top3,
            "total": total_exc,
        },
        "unclassified_samples": unclassified_notes,
        "wall_clock_s": round(time.time() - t0, 2),
    }, indent=1), encoding="utf-8")
    print()
    print("wrote %s" % out.name)
    print("wall %.0f s" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
