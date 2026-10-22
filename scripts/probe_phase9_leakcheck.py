"""Adversarial check on G-SYNTH1: is 0.225 retrieval leakage or adaptation?

THE SUSPICION
`phase9_g_synth1.json` reports overlap 0 BY TASK ID. That is not the same as
overlap 0 BY CODE. MBPP contains near-duplicate problems under different
task_ids, so the 800-row bank can hold a body identical to a held-out gold
solution. If it does, a winner proves retrieval of a stored answer, not
generation, and the 0.225 would be an upper bound on a leak rather than a
capability.

MEASURES, per winning task:
  exact  gold source equals some bank source after normalisation
  ast    AST dumps are equal after stripping the function NAME only, which
         catches the same program renamed
  none   no bank row matches by either test

No verdict. This is a limit measurement on a published number, and the result
goes into the register either way.
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, ".")

REPO = Path(__file__).resolve().parents[1]
RECEIPT = REPO / "receipts" / "phase9_g_synth1.json"


def norm(src: str) -> str:
    src = re.sub(r"#.*", "", src or "")
    src = re.sub(r"\s+", "", src)
    return src.strip().strip('"')


def ast_key(src: str):
    """AST dump with the entry function name replaced by a constant.

    A bank row under a different task_id can hold the SAME program under a
    different function name. Comparing raw text would call that 'no match'
    and understate leakage, so the name is normalised away first.
    """
    try:
        tree = ast.parse(src or "")
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            node.name = "_fn"
            break
    try:
        return ast.dump(tree, annotate_fields=False)
    except Exception:                          # noqa: BLE001
        return None


def main() -> int:
    r = json.loads(RECEIPT.read_text(encoding="utf-8"))
    rows = [json.loads(l) for l in
            (REPO / "data" / "official_benchmarks" / "mbpp.jsonl")
            .read_text(encoding="utf-8").splitlines() if l.strip()]
    bank = rows[:r["params"]["n_train"]]

    bank_norm = {}
    for b in bank:
        k = norm(b.get("code") or "")
        if k:
            bank_norm.setdefault(k, str(b.get("task_id")))
    bank_ast = {}
    for b in bank:
        k = ast_key(b.get("code") or "")
        if k:
            bank_ast.setdefault(k, str(b.get("task_id")))

    by_id = {str(x.get("task_id")): x for x in rows}
    winners = [t["task_id"] for t in r["per_task"]
               if t.get("full_verdict") == "EXEC_VERIFIED"]
    print("winners %d of %d" % (len(winners), len(r["per_task"])))
    print("bank rows %d  distinct exact %d  distinct ast %d"
          % (len(bank), len(bank_norm), len(bank_ast)))

    counts = {"exact": 0, "ast": 0, "none": 0}
    detail = []
    for tid in winners:
        gold = (by_id.get(tid) or {}).get("code") or ""
        e = bank_norm.get(norm(gold))
        a = bank_ast.get(ast_key(gold)) if not e else None
        if e:
            counts["exact"] += 1
            kind, who = "exact", e
        elif a:
            counts["ast"] += 1
            kind, who = "ast", a
        else:
            counts["none"] += 1
            kind, who = "none", ""
        detail.append({"task_id": tid, "leak": kind, "matches_bank_task": who})

    for d in detail:
        print("  %-6s %-6s %s" % (d["task_id"], d["leak"], d["matches_bank_task"]))

    n = len(winners) or 1
    print()
    print("leak_exact_share  %.4f  (%d/%d)"
          % (counts["exact"] / n, counts["exact"], n))
    print("leak_ast_share    %.4f  (%d/%d)"
          % ((counts["exact"] + counts["ast"]) / n,
             counts["exact"] + counts["ast"], n))
    print("clean_share       %.4f  (%d/%d)"
          % (counts["none"] / n, counts["none"], n))

    out = REPO / "receipts" / "phase9_g_synth1_leakcheck.json"
    payload = {"probe": "phase9_g_synth1_leakcheck",
               "counts": counts, "n_winners": len(winners),
               "detail": detail,
               "note": ("exact = normalised text equality; ast = equal AST "
                        "after function-name stripping; none = no match")}
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    print("receipt %s" % out.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
