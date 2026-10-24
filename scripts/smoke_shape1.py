"""Smoke-test G-SHAPE1 counter wiring on 2 tasks BEFORE the 40-task run.

WHY THIS EXISTS
A definition is not a measurement. `shape_adapt_enabled` existing, and the
accumulation line existing in the probe source, does not prove that the real
`solve_verified_code` path returns `synthesis.origins.shape_adapted`, nor that
the counter moves. A counter initialized but never fed reads 0 in both arms,
which makes the engagement precondition vacuous and yields a verdict that
looks measured when it is not.

WHAT IT CHECKS, IN ORDER
1. The probe module loads (register in sys.modules BEFORE exec_module).
2. A direct call returns `synthesis.origins.shape_adapted` > 0 with the flag
   ON, so the key exists on this path.
3. `run_arm(enabled=True)` increments `shape_candidates`.
4. `run_arm(enabled=False)` leaves it at 0.

SMALL BANK ON PURPOSE
60 train rows, 2 tasks. This is a wiring check, not a measurement. Numbers here
are NOT comparable to the gate.
"""
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

PROBE = REPO / "scripts" / "probe_phase11_shape1.py"
spec = importlib.util.spec_from_file_location("p11s1", PROBE)
mod = importlib.util.module_from_spec(spec)
sys.modules["p11s1"] = mod          # register BEFORE exec (harness lesson)
spec.loader.exec_module(mod)

from henri.api_server import get_api                      # noqa: E402
from henri.code_synth import CodeTemplateBank, Template   # noqa: E402

N_TRAIN = 60
N_TASKS = 2


def main() -> int:
    bb = get_api().backbone
    rows = [json.loads(l)
            for l in (REPO / "data" / "official_benchmarks" / "mbpp.jsonl")
            .read_text(encoding="utf-8").splitlines() if l.strip()]
    train, held = rows[:N_TRAIN], rows[N_TRAIN:N_TRAIN + N_TASKS]

    bank = CodeTemplateBank(dim=mod.DIM, rank=mod.RANK)
    for r in train:
        bank.add(bb, Template(docstring=(r.get("text") or "").strip(),
                              source=(r.get("code") or "").strip(),
                              task_id=str(r.get("task_id"))))
    bank.finalize()
    print("bank rows %d   tasks %d" % (bank.rows, len(held)))

    # (2) direct call: does the key exist on the real path?
    import os
    os.environ[mod.ENV] = "1"
    from henri.code_verify import solve_verified_code
    r0 = held[0]
    res = solve_verified_code(bb, (r0.get("text") or "").strip(), [],
                              list(r0.get("test_list") or []), bank=bank,
                              synthesize_from_bank=True,
                              k_templates=mod.K_TEMPLATES,
                              max_variants=mod.MAX_VARIANTS,
                              top_k=mod.TOP_K,
                              max_cegis_rounds=mod.ROUNDS)
    syn = (res.get("synthesis") or {}).get("origins") or {}
    print("direct origins %s" % (syn,))
    print("KEY_PRESENT %s" % ("shape_adapted" in syn))
    # RUN1 BLIND SPOT, CLOSED HERE. The run1 smoke checked only that the
    # `origins` KEY existed. It never checked the channel the gate actually
    # reads: `winner["provenance"]`. That is why the unreachable-T2 defect
    # survived a passing smoke. Print it, and require the key to exist.
    # RUN1 BLIND SPOT, CLOSED HERE — AND THE FIRST VERSION OF THIS CHECK WAS
    # ITSELF VACUOUS. It printed `WINNER_KEY_PRESENT False` and still said
    # SMOKE PASS, because `winner` is None whenever nothing verifies: `{}`
    # has no `provenance` whether or not the field is written. So the check
    # could not fail for the reason it claimed. The verdict now decides
    # whether the check applies, and the outcome enters `ok`.
    win = res.get("winner")
    verified = res.get("verdict") == "EXEC_VERIFIED"
    win_prov = (win or {}).get("provenance") if win else None
    print("verdict %s" % (res.get("verdict"),))
    print("winner_provenance %r" % (win_prov,))
    print("WINNER_CHECK_APPLIES %s" % (verified,))
    win_ok = (not verified) or (win_prov is not None)
    print("WINNER_OK %s" % (win_ok,))
    os.environ.pop(mod.ENV, None)

    # (3)/(4) counter movement per arm
    _, _, ctr_off = mod.run_arm(bb, bank, held, enabled=False)
    _, _, ctr_on = mod.run_arm(bb, bank, held, enabled=True)
    print("arm off shape_candidates %d" % ctr_off["shape_candidates"])
    print("arm on  shape_candidates %d" % ctr_on["shape_candidates"])

    ok = ("shape_adapted" in syn
          and ctr_on["shape_candidates"] > 0
          and ctr_off["shape_candidates"] == 0
          and win_ok)
    print("SMOKE %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
