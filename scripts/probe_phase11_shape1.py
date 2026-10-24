"""Phase 11 Step 5 probe: G-SHAPE1, argument-shape adaptation ablation.

PRE-REGISTERED BEFORE THE FIRST RUN OF THIS PROBE.

WHAT THIS GATE EXISTS TO SEPARATE
Register 24 measured the `exception` bucket as 87/105 `TypeError`, with
observed messages `'int' object is not subscriptable` and `'int' object is
not iterable`. That is argument SHAPE. `adapt_arity` resizes the parameter
LIST and cannot fix a body that subscripts a scalar it was handed.

`aaf6e49` ships `adapt_shape` DEFAULT OFF. This gate measures whether the
mechanism earns being turned on.

BOTH ARMS RUN max_cegis_rounds=2, THE SAME BANK, SEED, AND TASK LIST.
Exactly one variable differs:
    HENRI_CODE_SYNTH_SHAPE_ADAPT = 0  (control)
    HENRI_CODE_SYNTH_SHAPE_ADAPT = 1  (treatment)

PRE-REGISTERED BAR, IN THE UNITS IT IS DEFINED OVER
    BAR_TASKS = 1   (one task of forty, integer count, never a float ratio)
Item 1 measured the cost of a float-ratio bar: 12/40 - 11/40 reads
0.024999999999999967 and fails a 0.025 comparison by 3.47e-17. This probe
compares TASK COUNTS from the start.

DELTA = 0 IS A REPORTABLE OUTCOME, NOT A HARNESS FAILURE.

ENGAGEMENT PRECONDITION, FAIL-CLOSED, BOTH DIRECTIONS
1. The control must emit ZERO shape candidates. Any emission proves the
   switch leaked into the control arm.
2. The treatment must emit AT LEAST ONE. Zero emissions prove the mechanism
   never fired, and every efficacy clause is then vacuous — the same shape as
   the Phase 9 defect this project already paid for.
A ONE-DIRECTION check is the defect: testing only (1) reports a silent
treatment arm as a measured negative.

CEILING, STATED BEFORE THE RUN
At most 6/40 held-out tasks carry `exception` as the LAST residual
(801, 805, 810, 817, 825, 830). Zero tasks carry it as the SOLE residual.
So the mechanism's ceiling is +6 tasks and the expected effect is small. A
result above 6 would itself be suspect and would be investigated, not
celebrated.

KNOWN LIMIT, CARRIED FROM THE UNIT LAYER
A guard makes the CALL legal. It does not make the function CORRECT.
`def f(xs): return xs[0] + 1` guarded and called with 5 returns 6, not the
caller's intent. A pass therefore requires the guarded candidate to satisfy
the REAL tests, which is what `EXEC_VERIFIED` means.

COUNTER WIRING IS VERIFIED BEFORE THE RUN
`counters["shape_candidates"]` is read from `synthesis.origins.shape_adapted`
per task and incremented in the loop. A counter that is initialized but never
incremented reads 0 in both arms and makes the gate vacuous. Callers must
smoke-test `run_arm` on one task before the full run; the probe cannot detect
this defect about itself.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

TAG = "run1"
T0 = 0.0

# Corpus knobs, identical to G-SYNTH3 so arms stay comparable across gates.
N_TRAIN = 800
N_TASKS = 40
DIM = 65536
RANK = 64
K_TEMPLATES = 8
MAX_VARIANTS = 2
TOP_K = K_TEMPLATES * (1 + MAX_VARIANTS)
ROUNDS = 2
ENV = "HENRI_CODE_SYNTH_SHAPE_ADAPT"
BAR_TASKS = 1          # one task of forty, integer, pre-registered
CEILING_TASKS = 6      # exception-as-last-residual, measured in register 24


def _arg_tag():
    global TAG
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run1")
    a, _ = ap.parse_known_args()
    TAG = a.tag


def out_path():
    return REPO / "receipts" / ("phase11_g_shape1_%s.json" % TAG)


def _ok(entry):
    """Execution outcome for one candidate row, tolerant of key name."""
    for key in ("ok", "passed", "verified"):
        if key in entry:
            return bool(entry[key])
    return False


def run_arm(bb, bank, tasks, enabled):
    """One arm over the held-out tasks. Returns (hits, per_task, counters)."""
    os.environ[ENV] = "1" if enabled else "0"
    from henri.code_verify import solve_verified_code

    hits = 0
    rows = []
    counters = {
        "shape_candidates": 0,
        "shape_candidates_ok": 0,
        "all_candidates": 0,
        "exception_probes": 0,
        "tasks_with_shape": 0,
    }

    for i, r in enumerate(tasks, 1):
        spec = (r.get("text") or "").strip()
        tests = list(r.get("test_list") or [])
        res = solve_verified_code(bb, spec, [], tests, bank=bank,
                                  synthesize_from_bank=True,
                                  k_templates=K_TEMPLATES,
                                  max_variants=MAX_VARIANTS,
                                  top_k=TOP_K, max_cegis_rounds=ROUNDS)
        ok = res.get("verdict") == "EXEC_VERIFIED"
        hits += int(ok)

        ceg = res.get("cegis") or {}
        kinds = [q["residual"]["kind"] for q in ceg.get("probes", [])]
        counters["exception_probes"] += sum(1 for k in kinds
                                            if k == "exception")

        # Candidate provenance comes from synthesis origins, not from the
        # winner only: a mechanism can emit candidates that lose and still be
        # engaged. Counting winners would understate engagement and could make
        # the treatment arm look silent when it in fact fired.
        syn = (res.get("synthesis") or {}).get("origins") or {}
        n_shape = int(syn.get("shape_adapted") or 0)
        counters["shape_candidates"] += n_shape
        counters["tasks_with_shape"] += int(n_shape > 0)
        counters["all_candidates"] += (int(syn.get("retrieved") or 0)
                                       + int(syn.get("mutated") or 0)
                                       + n_shape)

        exec_rows = res.get("executed") or []
        shape_ok = [str(e.get("mutation"))
                    for e in exec_rows
                    if e.get("shape_adapted") and _ok(e)]
        counters["shape_candidates_ok"] += len(shape_ok)

        rows.append({
            "task_id": str(r.get("task_id")),
            "ok": bool(ok),
            "verdict": res.get("verdict"),
            "winner_provenance": (res.get("winner") or {}).get("provenance"),
            "n_shape": n_shape,
            "shape_candidates_ok": shape_ok,
            "residual_kinds": kinds,
            "n_repairs": int(ceg.get("n_repairs") or 0),
        })
        if i % 10 == 0:
            print("  [%s] %d/%d  %.0f s  hits %d  shape_cand %d"
                  % ("on" if enabled else "off", i, len(tasks),
                     time.time() - T0, hits, counters["shape_candidates"]),
                  flush=True)

    os.environ.pop(ENV, None)
    return hits, rows, counters


def evaluate(n, c_hits, c_ctr, t_hits, t_ctr, c_rows, t_rows):
    """Clauses and verdict. Split out so a smoke test can exercise it."""
    c_by = {r["task_id"]: r for r in c_rows}
    t_by = {r["task_id"]: r for r in t_rows}
    gains = sorted(k for k in t_by if t_by[k]["ok"] and not c_by[k]["ok"])
    losses = sorted(k for k in t_by if c_by[k]["ok"] and not t_by[k]["ok"])
    gain_tasks = t_hits - c_hits

    # Both directions. See the module docstring: one direction is the defect.
    control_leak = c_ctr["shape_candidates"] > 0
    treatment_silent = t_ctr["shape_candidates"] == 0
    inert = bool(control_leak or treatment_silent)

    # A gain counts as attributed only when the winning candidate's provenance
    # is `shape_adapted`. A gain from another mechanism does not clear the bar.
    attributed = [k for k in gains
                  if t_by[k]["winner_provenance"] == "shape_adapted"]

    t1 = t_hits >= c_hits
    t2 = bool(gain_tasks >= BAR_TASKS and attributed)
    t3 = (len(losses) == 0)

    if inert:
        verdict = "G_SHAPE1_ABLATION_INERT"
    elif t1 and t2 and t3:
        verdict = "G_SHAPE1_PASS"
    elif t1 and t3:
        verdict = "G_SHAPE1_NO_EFFICACY"
    else:
        verdict = "G_SHAPE1_FAIL"

    clauses = [
        {"id": "T1", "pass": bool(t1), "counted": True,
         "what": "treatment does not lose",
         "value": {"control": c_hits, "treatment": t_hits}},
        {"id": "T2", "pass": bool(t2), "counted": True,
         "what": "efficacy, attributed to the mechanism",
         "value": {"gain_tasks": gain_tasks, "bar_tasks": BAR_TASKS,
                   "attributed": attributed}},
        {"id": "T3", "pass": bool(t3), "counted": True,
         "what": "zero losses", "value": {"losses": losses}},
        {"id": "T4", "pass": None, "counted": False,
         "what": "attribution detail, reported not a bar",
         "value": {"gains": gains, "attributed": attributed,
                   "unattributed": sorted(set(gains) - set(attributed))}},
        {"id": "T5", "pass": not inert, "counted": False,
         "what": "engagement precondition, both directions",
         "value": {"control_leak": control_leak,
                   "treatment_silent": treatment_silent,
                   "control_candidates": c_ctr["shape_candidates"],
                   "treatment_candidates": t_ctr["shape_candidates"]}},
    ]
    return {"verdict": verdict, "gains": gains, "losses": losses,
            "attributed": attributed, "gain_tasks": gain_tasks,
            "inert": inert, "clauses": clauses}


def main() -> int:
    global T0
    _arg_tag()
    T0 = time.time()

    from henri.api_server import get_api
    from henri.code_synth import CodeTemplateBank, Template

    print("building bank from %d train rows (shared by both arms)" % N_TRAIN,
          flush=True)
    bb = get_api().backbone
    p = REPO / "data" / "official_benchmarks" / "mbpp.jsonl"
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    train, held = rows[:N_TRAIN], rows[N_TRAIN:]
    overlap = ({str(r.get("task_id")) for r in train}
               & {str(r.get("task_id")) for r in held})

    bank = CodeTemplateBank(dim=DIM, rank=RANK)
    for r in train:
        bank.add(bb, Template(docstring=(r.get("text") or "").strip(),
                              source=(r.get("code") or "").strip(),
                              task_id=str(r.get("task_id"))))
    bank.finalize()
    tasks = held[:N_TASKS]
    print("corpus %d train %d held %d overlap %d bank %d tasks %d"
          % (len(rows), len(train), len(held), len(overlap), bank.rows,
             len(tasks)), flush=True)

    n = len(tasks)
    print("arm CONTROL (rounds=%d, %s=0)" % (ROUNDS, ENV), flush=True)
    c_hits, c_rows, c_ctr = run_arm(bb, bank, tasks, enabled=False)
    print("arm TREATMENT (rounds=%d, %s=1)" % (ROUNDS, ENV), flush=True)
    t_hits, t_rows, t_ctr = run_arm(bb, bank, tasks, enabled=True)

    ev = evaluate(n, c_hits, c_ctr, t_hits, t_ctr, c_rows, t_rows)
    verdict = ev["verdict"]
    inert = ev["inert"]

    print()
    print("control %d/%d   treatment %d/%d   gain_tasks %+d"
          % (c_hits, n, t_hits, n, ev["gain_tasks"]), flush=True)
    print("shape candidates control %d   treatment %d   verified %d"
          % (c_ctr["shape_candidates"], t_ctr["shape_candidates"],
             t_ctr["shape_candidates_ok"]), flush=True)
    print("tasks_with_shape treatment %d   exception probes c/t %d/%d"
          % (t_ctr["tasks_with_shape"], c_ctr["exception_probes"],
             t_ctr["exception_probes"]), flush=True)
    print("gains %s  losses %s  attributed %s"
          % (ev["gains"], ev["losses"], ev["attributed"]), flush=True)
    print("verdict %s   inert %s" % (verdict, inert), flush=True)

    receipt = {
        "gate": "G-SHAPE1",
        "tag": TAG,
        "verdict": verdict,
        "pre_registered": {
            "bar_tasks": BAR_TASKS,
            "bar_units": "integer task count, never a float ratio",
            "ceiling_tasks": CEILING_TASKS,
            "ceiling_basis": ("6/40 tasks carry exception as LAST residual; "
                              "0 as sole residual (register 24)"),
            "engagement": ("control must emit zero AND treatment must emit "
                           "at least one"),
            "delta_zero_is_reportable": True,
            "rounds": ROUNDS,
        },
        "n": n,
        "control": {"hits": c_hits, "counters": c_ctr},
        "treatment": {"hits": t_hits, "counters": t_ctr},
        "gain_tasks": ev["gain_tasks"],
        "bar_tasks": BAR_TASKS,
        "gains": ev["gains"],
        "losses": ev["losses"],
        "attributed_gains": ev["attributed"],
        "ablation_inert": inert,
        "clauses": ev["clauses"],
        "per_task_control": c_rows,
        "per_task_treatment": t_rows,
        "wall_clock_s": round(time.time() - T0, 1),
    }

    out = out_path()
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8", newline="\n")
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    side = out.with_suffix(out.suffix + ".sha256")
    side.write_bytes((digest + "  " + out.name + "\n").encode("ascii"))

    print("receipt %s" % out.name, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
