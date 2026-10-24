"""Phase 11 Step 4 probe: G-SYNTH3, round-matched name_error ablation.

PRE-REGISTERED BEFORE THE FIRST RUN OF THIS PROBE.

WHAT THIS GATE EXISTS TO SEPARATE
`4ceabd3` (`G-SYNTH2`) reported held-out MBPP 0.2250 -> 0.3000 with 3 gains.
Attribution showed 1 of 3 came from a name_error mechanism and 2 from Phase 10
value-residual mechanisms. But the paired control differed in TWO ways, not one:

    control   max_cegis_rounds=1
    treatment max_cegis_rounds=2

So "+3" measures "an extra CEGIS round helps". It does NOT measure the shipped
name_error branch. The receipt says so in `confound_note`. This gate removes the
confound.

BOTH ARMS RUN max_cegis_rounds=2
Only `HENRI_CEGIS_NAME_ERROR_REPAIR` differs:
    control   0  -> no name_error candidate is emitted
    treatment 1  -> shipped behavior from 4ceabd3

THE PREDICTION, RECORDED BEFORE THE RUN
Run 3's single name_error gain (task 816) is round-count-confounded, so it is
NOT evidence for this gate. The marginal effect at matched rounds is genuinely
unknown. Predicted: 0 to 1 tasks.

A ZERO RESULT IS A RESULT
If treatment equals control, the honest verdict is `G_SYNTH3_NO_MARGINAL_GAIN`:
the branch earns nothing beyond an extra round. That would partially retract the
Step 3 headline attribution, which is exactly why it must be reportable. It is
not a gate failure and not a license to tune the bar.

CLAUSES
  T1 no regression:    rate(treat) >= rate(control)
  T2 efficacy (COUNTED): delta >= 0.025 (one task of 40) AND attributed gains
  T3 zero losses:      no task solved by control is failed by treatment
  T4 attribution:      reported, not counted

ENGAGEMENT IS A PRECONDITION, NOT A CLAUSE
If the ablation never removed a candidate, the arms executed identical code and
every clause above is vacuous. The probe records `name_error_candidates` per arm
and fails closed with `G_SYNTH3_ABLATION_INERT` when the control emitted any.
"""

import argparse
import ast
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

TAG = "run1"


def _arg_tag():
    global TAG
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run1")
    a, _ = ap.parse_known_args()
    TAG = a.tag


TAG_DEFAULT_DONE = [False]

# Corpus knobs. MEASURED FROM probe_phase11_synth2.py AND FROM A REAL ENCODE.
#
# DEFECT IN MY FIRST VERSION, kept here because it cost a run: I set DIM=384
# from `backbone.dim_sem`, the sentence-embedding width. The cartridge bank
# stores WAVES, whose width is 65536. `bank.finalize()` raised
# `ValueError: expected length 384, got 65536` before any task ran, so the
# first attempt was UNMEASURED, not FAILED.
#
# Measured: `np.asarray(backbone.encode("probe").psi).shape == (65536,)`.
N_TRAIN = 800
N_TASKS = 40
DIM = 65536
RANK = 64
K_TEMPLATES = 8
MAX_VARIANTS = 2
TOP_K = K_TEMPLATES * (1 + MAX_VARIANTS)
ROUNDS = 2          # matched across both arms. This is the whole point.
ENV = "HENRI_CEGIS_NAME_ERROR_REPAIR"
BAR_DELTA = 0.025   # one task of forty, fixed before the run


def out_path():
    return REPO / "receipts" / ("phase11_g_synth3_%s.json" % TAG)


def _new_prefixes():
    return ("import_", "rebind_")


def run_arm(bb, bank, tasks, enabled):
    """One arm over the held-out tasks. Returns (hits, per_task, counters)."""
    os.environ[ENV] = "1" if enabled else "0"
    from henri.code_verify import solve_verified_code

    hits = 0
    rows = []
    counters = {
        "name_error_probes": 0,
        "name_error_candidates": 0,
        "name_error_candidates_ok": 0,
        "all_candidates": 0,
        "kind_hist": {},
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
        for k in kinds:
            counters["kind_hist"][k] = counters["kind_hist"].get(k, 0) + 1
        counters["name_error_probes"] += sum(1 for k in kinds
                                             if k == "name_error")

        mechs = [str(x.get("mechanism", "")) for x in ceg.get("repairs", [])]
        oks = [str(x.get("mechanism", ""))
               for x in ceg.get("repairs", []) if x.get("ok")]
        counters["all_candidates"] += len(mechs)
        ne_mechs = [m for m in mechs if m.startswith(_new_prefixes())]
        ne_oks = [m for m in oks if m.startswith(_new_prefixes())]
        counters["name_error_candidates"] += len(ne_mechs)
        counters["name_error_candidates_ok"] += len(ne_oks)

        rows.append({
            "task_id": str(r.get("task_id")),
            "ok": bool(ok),
            "verdict": res.get("verdict"),
            "mechanisms": mechs,
            "mechanisms_ok": oks,
            "name_error_candidates": ne_mechs,
            # MEASURED DEFECT, fixed here. This key was missing while the
            # attribution block read it, so the probe crashed with
            # `KeyError: 'name_error_candidates_ok'` AFTER both arms had run
            # and BEFORE the receipt was written. The numbers existed only in
            # the log, which is not a receipt. Third occurrence of this class
            # in this phase: report-time code is not exercised until the run
            # reaches it.
            "name_error_candidates_ok": ne_oks,
            "winner_origin": (res.get("winner") or {}).get("origin"),
            "residual_kinds": kinds,
            "n_repairs": int(ceg.get("n_repairs") or 0),
        })
        if i % 10 == 0:
            print("  [%s] %d/%d  %.0f s  hits %d  ne_cand %d"
                  % ("on" if enabled else "off", i, len(tasks),
                     time.time() - T0, hits,
                     counters["name_error_candidates"]))

    os.environ.pop(ENV, None)
    return hits, rows, counters


def main() -> int:
    global T0
    _arg_tag()
    T0 = time.time()

    from henri.api_server import get_api
    from henri.code_synth import CodeTemplateBank, Template

    print("building bank from %d train rows (shared by both arms)" % N_TRAIN)
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
             len(tasks)))

    n = len(tasks)
    print("arm CONTROL (rounds=%d, %s=0)" % (ROUNDS, ENV))
    c_hits, c_rows, c_ctr = run_arm(bb, bank, tasks, enabled=False)
    print("arm TREATMENT (rounds=%d, %s=1)" % (ROUNDS, ENV))
    t_hits, t_rows, t_ctr = run_arm(bb, bank, tasks, enabled=True)

    rate_c = c_hits / float(n) if n else 0.0
    rate_t = t_hits / float(n) if n else 0.0
    delta = rate_t - rate_c

    c_by = {r["task_id"]: r for r in c_rows}
    t_by = {r["task_id"]: r for r in t_rows}
    gains = [k for k in t_by if t_by[k]["ok"] and not c_by[k]["ok"]]
    losses = [k for k in t_by if c_by[k]["ok"] and not t_by[k]["ok"]]
    attributed = [k for k in gains if t_by[k]["name_error_candidates_ok"]]

    print()
    print("control %.4f (%d/%d)   treatment %.4f (%d/%d)   delta %+.4f"
          % (rate_c, c_hits, n, rate_t, t_hits, n, delta))
    print("gains %s" % sorted(gains))
    print("losses %s" % sorted(losses))
    print("attributed via name_error %s" % sorted(attributed))
    print("control name_error candidates %d   treatment %d"
          % (c_ctr["name_error_candidates"], t_ctr["name_error_candidates"]))

    # ---- engagement precondition ---------------------------------------
    # A control that emitted a name_error candidate proves the switch did not
    # take effect, so the two arms ran identical code and NO clause below can
    # mean anything.
    inert = c_ctr["name_error_candidates"] > 0

    # -- BAR ARITHMETIC DEFECT, MEASURED, ENCODING-ONLY REPAIR ------------
    #
    # `BAR_DELTA = 0.025` is the decimal form of one task of forty. Computing
    # `rate_t - rate_c` in binary floating point yields 0.024999999999999967,
    # which is SMALLER than 0.025 by 3.47e-17, so the bar failed on its own
    # representation: `12/40 - 11/40 != 0.025` in float.
    #
    # The bar was REGISTERED as "one task of forty" (see the module docstring
    # and the BAR_DELTA comment). It is evaluated here in TASK COUNTS, the
    # quantity it was defined over. The threshold is UNCHANGED: 0.025 either
    # way. This is an encoding correction, NOT a threshold change.
    #
    # Audit trail: the run that exposed the defect is preserved as
    # tag=run3 with verdict G_SYNTH3_BELOW_BAR. That receipt stays. This
    # receipt records the same 11 vs 12 numbers with the bar computed exactly.
    needed = max(1, int(round(BAR_DELTA * n)))      # 1 task of 40
    gain_tasks = t_hits - c_hits

    t1 = rate_t >= rate_c
    t2 = bool(gain_tasks >= needed and attributed)
    t3 = (len(losses) == 0)
    t4 = {"counted": False, "gains": sorted(gains),
          "attributed": sorted(attributed),
          "unattributed": sorted(set(gains) - set(attributed))}

    clauses = [
        {"id": "T1", "pass": bool(t1), "counted": True,
         "what": "no regression at matched rounds",
         "value": {"control": rate_c, "treatment": rate_t, "delta": delta},
         "bar": "treatment >= control"},
        {"id": "T2", "pass": bool(t2), "counted": True,
         "what": "efficacy, not engagement",
         "value": {"delta": delta, "bar": BAR_DELTA,
                   "gain_tasks": gain_tasks, "needed_tasks": needed,
                   "attributed_gains": sorted(attributed)},
         "bar": "task gain >= 1 of 40 AND >=1 attributed gain"},
        {"id": "T3", "pass": bool(t3), "counted": True,
         "what": "zero losses",
         "value": {"losses": sorted(losses)}, "bar": "losses == 0"},
        {"id": "T4", "pass": True, "counted": False,
         "what": "attribution integrity", "value": t4,
         "bar": "reported only"},
    ]

    counted = [c for c in clauses if c.get("counted", True)]
    npass = sum(1 for c in counted if c["pass"])

    if inert:
        verdict = "G_SYNTH3_ABLATION_INERT"
    elif not t1:
        verdict = "G_SYNTH3_REGRESSION"
    elif delta >= BAR_DELTA and attributed:
        verdict = "G_SYNTH3_PASS"
    elif delta >= BAR_DELTA and not attributed:
        verdict = "G_SYNTH3_PASS_CAUSE_UNATTRIBUTED"
    elif abs(delta) < 1e-12:
        verdict = "G_SYNTH3_NO_MARGINAL_GAIN"
    elif delta > 0:
        verdict = "G_SYNTH3_BELOW_BAR"
    else:
        verdict = "G_SYNTH3_NO_MARGINAL_GAIN"

    print()
    for c in clauses:
        print("  %-3s %-5s %s" % (c["id"], "PASS" if c["pass"] else "FAIL",
                                  c["what"]))
    print("verdict %s   %d/%d counted clauses" % (verdict, npass, len(counted)))

    body = {
        "probe": "phase11_g_synth3_round_matched_name_error_ablation",
        "tag": TAG,
        "prediction_recorded_before_run": (
            "Run 3's single name_error gain (task 816) is round-count "
            "confounded and does not predict this gate. Predicted marginal "
            "effect: 0 to 1 tasks. A zero result is a finding."),
        "design": {
            "rounds_both_arms": ROUNDS,
            "differing_flag": ENV,
            "control_value": "0", "treatment_value": "1",
            "why_matched": ("G-SYNTH2's control used rounds=1, so its +3 was "
                            "an extra-round effect, not a branch effect."),
        },
        "corpus": {"train": len(train), "held": len(held),
                   "n_tasks": n, "overlap": len(overlap),
                   "bank_rows": int(bank.rows)},
        "control": {"rate": rate_c, "hits": c_hits, "counters": c_ctr},
        "treatment": {"rate": rate_t, "hits": t_hits, "counters": t_ctr},
        "delta": delta,
        "gains": sorted(gains), "losses": sorted(losses),
        "attributed_gains": sorted(attributed),
        "ablation_inert": bool(inert),
        "clauses": clauses,
        "n_clauses_pass": npass,
        "n_clauses_counted": len(counted),
        "verdict": verdict,
        "wall_clock_s": round(time.time() - T0, 2),
        "per_task_control": c_rows,
        "per_task_treatment": t_rows,
    }

    from henri.receipt import write_receipt, verify_receipt
    p_out = out_path()
    ids = write_receipt(p_out, body)
    v = verify_receipt(p_out)
    print("receipt %s" % p_out.name)
    print("body_match %s file_match %s" % (v["body_match"], v["file_match"]))
    print("body_field %s" % str(ids.get("file_sha256_on_disk"))[:16])
    return 0


T0 = time.time()

if __name__ == "__main__":
    raise SystemExit(main())
