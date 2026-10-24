"""Re-evaluate G-SYNTH3's T2 bar in the units it was registered in.

NO NEW MEASUREMENT. This reads the SEALED `phase11_g_synth3_run3.json` and
recomputes one clause. It runs no task, touches no model, and writes a
separate correction receipt. `run3.json` is not edited.

WHY THIS EXISTS
`BAR_DELTA = 0.025` was registered as "one task of forty". The clause compared
binary floats:

    delta = 12/40 - 11/40 = 0.024999999999999967
    0.024999999999999967 >= 0.025  ->  False
    shortfall = 3.47e-17

So the bar failed on its own representation. In exact arithmetic
`12/40 - 11/40 = 1/40 = 0.025`, and the bar is met.

WHY A RE-ANALYSIS AND NOT A RE-RUN
Re-running would mint a PASS from the same underlying numbers, and an auditor
could not distinguish that from moving the bar after seeing the outcome. A
re-analysis from sealed data is checkable: every input number below is read out
of `run3.json`, and the arithmetic is reproduced in this file.

WHY THIS IS NOT BAR-MOVING
The threshold is unchanged at 0.025. Only the COMPARISON DOMAIN changes, from
float ratio difference to integer task count. The bar was stated in tasks. The
strict-float reading is recorded too, and `run3`'s `G_SYNTH3_BELOW_BAR` verdict
stays on record. Both readings appear in the correction receipt.

THE RESULT IS SMALL AND MUST BE REPORTED AS SMALL
+1 task of 40 is the point estimate. Both arms are deterministic on the same
bank, seed, and task list, so the +1 is reproducible rather than sampling noise.
It is still ONE task. This does not establish a general capability gain.
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

SRC = REPO / "receipts" / "phase11_g_synth3_run3.json"
OUT = REPO / "receipts" / "phase11_g_synth3_encoding_correction.json"

BAR_DELTA = 0.025


def main() -> int:
    from henri.receipt import verify_receipt, write_receipt

    # The source must verify before it is used as an input.
    src_v = verify_receipt(SRC)
    if not (src_v["body_match"] and src_v["file_match"]):
        print("source receipt FAILED verification: %s" % src_v)
        return 2

    d = json.loads(SRC.read_bytes().decode("utf-8"))
    n = int(d["corpus"]["n_tasks"])
    c_ctr = int(d["control"]["hits"])
    t_ctr = int(d["treatment"]["hits"])

    # Recompute from the per-task records, not from the stored scalars, so a
    # stored-scalar error cannot propagate.
    c_rows = {r["task_id"]: r for r in d["per_task_control"]}
    t_rows = {r["task_id"]: r for r in d["per_task_treatment"]}
    assert set(c_rows) == set(t_rows), "arm task sets differ"
    c_hits = sum(1 for r in c_rows.values() if r["ok"])
    t_hits = sum(1 for r in t_rows.values() if r["ok"])
    assert (c_hits, t_hits) == (c_ctr, t_ctr), "stored scalars disagree"

    gains = sorted(k for k in t_rows if t_rows[k]["ok"] and not c_rows[k]["ok"])
    losses = sorted(k for k in t_rows if c_rows[k]["ok"] and not t_rows[k]["ok"])
    attributed = [k for k in gains if t_rows[k].get("name_error_candidates_ok")]

    gain_tasks = t_hits - c_hits
    needed = max(1, int(round(BAR_DELTA * n)))
    delta_float = t_hits / float(n) - c_hits / float(n)

    t2_float = bool(delta_float >= BAR_DELTA and attributed)
    t2_tasks = bool(gain_tasks >= needed and attributed)
    t3 = (len(losses) == 0)

    # Engagement: the ablation must have removed candidates from the control.
    inert = int(d["control"]["counters"]["name_error_candidates"]) > 0

    if inert:
        verdict = "G_SYNTH3_ENCODING_CORRECTION_ABLATION_INERT"
    elif not t2_tasks:
        verdict = "G_SYNTH3_ENCODING_CORRECTION_STILL_BELOW_BAR"
    elif not t3:
        verdict = "G_SYNTH3_ENCODING_CORRECTION_LOSSES"
    else:
        # The name carries the boundary so it cannot be read as a clean pass.
        verdict = "G_SYNTH3_PASS_AT_TASK_COUNT_BAR"

    print("source        %s" % SRC.name)
    print("source verify body_match %s file_match %s"
          % (src_v["body_match"], src_v["file_match"]))
    print("n %d  control %d  treatment %d  gain_tasks %d  needed %d"
          % (n, c_hits, t_hits, gain_tasks, needed))
    print("delta float   %r" % delta_float)
    print("delta tasks   %d/%d" % (gain_tasks, n))
    print("shortfall     %r" % (BAR_DELTA - delta_float))
    print("T2 float      %s   T2 tasks  %s" % (t2_float, t2_tasks))
    print("gains %s  losses %s  attributed %s" % (gains, losses, attributed))
    print("verdict %s" % verdict)

    body = {
        "probe": "phase11_g_synth3_bar_encoding_correction",
        "kind": "REANALYSIS_OF_SEALED_RECEIPT",
        "runs_no_tasks": True,
        "does_not_edit_source": True,
        "source_receipt": SRC.name,
        "source_receipt_verified": {
            "body_match": bool(src_v["body_match"]),
            "file_match": bool(src_v["file_match"]),
        },
        "supersedes_nothing": (
            "run3.json keeps its verdict G_SYNTH3_BELOW_BAR. This receipt "
            "records the same numbers under the bar's registered units."),
        "defect": {
            "description": (
                "T2 compared binary-float ratio differences against 0.025. "
                "12/40 - 11/40 evaluates to 0.024999999999999967, short of "
                "0.025 by 3.47e-17, so the bar failed on representation."),
            "bar_registered_as": "one task of forty",
            "threshold_unchanged": BAR_DELTA,
            "changed": "comparison domain: float ratio -> integer task count",
        },
        "both_readings": {
            "strict_float": {"delta": delta_float, "pass": t2_float,
                             "shortfall": BAR_DELTA - delta_float},
            "task_count": {"gain_tasks": gain_tasks, "needed": needed,
                           "pass": t2_tasks},
        },
        "measured": {
            "n_tasks": n, "control_hits": c_hits, "treatment_hits": t_hits,
            "delta_tasks": gain_tasks,
            "gains": gains, "losses": losses, "attributed_gains": attributed,
            "control_name_error_candidates":
                int(d["control"]["counters"]["name_error_candidates"]),
            "treatment_name_error_candidates":
                int(d["treatment"]["counters"]["name_error_candidates"]),
        },
        "clauses": [
            {"id": "C1", "pass": bool(t2_tasks), "counted": True,
             "what": "efficacy at the task-count bar",
             "value": {"gain_tasks": gain_tasks, "needed": needed,
                       "attributed": attributed}},
            {"id": "C2", "pass": bool(t3), "counted": True,
             "what": "zero losses", "value": {"losses": losses}},
            {"id": "C3", "pass": True, "counted": False,
             "what": "strict-float reading retained for audit",
             "value": {"delta": delta_float, "pass": t2_float,
                       "shortfall": BAR_DELTA - delta_float}},
        ],
        "scope_limit": (
            "+1 task of 40. Both arms are deterministic on one bank, seed, and "
            "task list, so the gain is reproducible rather than sampling "
            "noise. It is still one task and does not establish a general "
            "capability gain. 34 name_error candidates were emitted and 1 "
            "became a verified solve."),
        "verdict": verdict,
    }

    ids = write_receipt(OUT, body)
    v = verify_receipt(OUT)
    print("receipt %s" % OUT.name)
    print("body_match %s file_match %s" % (v["body_match"], v["file_match"]))
    print("file_sha256 %s" % ids["file_sha256_on_disk"][:16])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
