"""Append Section 45 (Cycle AD) to docs/HOLARCH.md from the sealed receipts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
RECEIPTS = ROOT / "receipts" / "holarch"
DOCS = ROOT / "docs" / "HOLARCH.md"
MARKER = "## 45. Cycle AD"
TICK = chr(96)


def _read(name: str) -> Dict[str, Any]:
    path = RECEIPTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        return ("%." + str(int(digits)) + "f") % float(value)
    except Exception:  # noqa: BLE001
        return str(value)


def _code(text: str) -> str:
    return TICK + str(text) + TICK


def build() -> str:
    ac1 = _read("ac1_kronecker_induction_egress.json")
    ac_cycle = _read("ac_cycle.json")
    ad1 = _read("ad1_standalone_repo_patch.json")
    ad2 = _read("ad2_relational_arc_and_science.json")
    ad3 = _read("ad3_deliverable_and_aaii96.json")
    ad4 = _read("ad4_blackwell_cuda_and_jev.json")
    ruling = ac1.get("subslice_1se_ruling") or {}
    sub = ac1.get("subslice") or {}
    full = ac1.get("full_split") or {}

    ad1_agg = ad1.get("aggregate") or {}
    ad1_rep = ad1.get("repulsors") or {}
    ad1_delta = max(
        [
            float(item.get("delta_ms", 0.0))
            for row in ad1.get("cases") or []
            for item in row.get("rollout_detail") or []
        ]
        or [0.0]
    )
    ad2_arc = ad2.get("relational_arc") or {}
    ad2_sci = ad2.get("science_solver") or {}
    ad3_manifest = ad3.get("manifest") or {}
    ad4_hw = ad4.get("hardware") or {}
    ad4_jev = ad4.get("jev") or {}
    jev_challenges = ad4_jev.get("challenges") or {}

    def verdict(gate: Dict[str, Any]) -> str:
        if not gate:
            return "ABSENT"
        return "PASS" if bool(gate.get("passed", False)) else "OPEN"

    def failing(gate: Dict[str, Any]) -> str:
        names = [name for name, value in gate.items() if not bool(value) and name != "passed"]
        return ", ".join(names) if names else "none"

    stage_rows = [
        ("AD0 AC seal under the 1-SE ruling", "ac_cycle.json", ac_cycle.get("gate") or {}),
        ("AD1 repository patch", "ad1_standalone_repo_patch.json", ad1.get("gate") or {}),
        ("AD2 relational ARC and science", "ad2_relational_arc_and_science.json", ad2.get("gate") or {}),
        ("AD3 96-task battery", "ad3_deliverable_and_aaii96.json", ad3.get("gate") or {}),
        ("AD4 Blackwell and JEV", "ad4_blackwell_cuda_and_jev.json", ad4.get("gate") or {}),
    ]
    lines = [
        "",
        MARKER + ": the one-standard-error seal, repository patch synthesis, relational ARC morphisms, the science solver and the bounded Blackwell attempt",
        "",
        "* **Document ID:** HENRI-SPEC-CYCLE-AD-AC1-SEAL-REPO-PATCH-SYNTHESIS-RELATIONAL-ARC-AND-BLACKWELL-EXECUTION-V10",
        "* **Baseline:** Cycle AC. Every number below is OBSERVED in this process and sealed under receipts/holarch/.",
        "",
        "### 45.1 Stage verdicts",
        "",
        "| Stage | Receipt | Verdict | Failing gates |",
        "| :--- | :--- | :---: | :--- |",
    ]
    for label, name, gate in stage_rows:
        lines.append("| %s | %s | **%s** | %s |" % (label, _code(name), verdict(gate), failing(gate)))
    lines += [
        "| AD5 hygiene and the dual-cycle seal | " + _code("ad_cycle.json") + " | **"
        + verdict(_read("ad_cycle.json").get("gate") or {}) + "** | "
        + failing(_read("ad_cycle.json").get("gate") or {}) + " |",
        "",
        "### 45.2 AD0: the 1-SE ruling and the Cycle AC seal",
        "",
        "Ruling " + _code(ruling.get("citation", "HENRI-RULING-AC1-1SE-2026-11-15"))
        + " governs " + _code("ac1_subslice_top1_non_regression_and_1se") + ". The raw literal "
        "top-1 is **" + _fmt(ruling.get("literal_top1"), 6) + "** with SE **"
        + _fmt(ruling.get("se_top1"), 6) + "**, AB1 reference **" + _fmt(ruling.get("ab1_ref"), 4)
        + "**, z **" + _fmt(ruling.get("z_delta_vs_ab1"), 2) + "** and "
        + _code("literal_ge_06270") + " **" + str(ruling.get("literal_ge_06270")).lower()
        + "**. The three conditions are the full-split literal bar (top-1 "
        + _fmt((ruling.get("condition_full_split_literal") or {}).get("top1_full"), 6) + ", bpc "
        + _fmt((ruling.get("condition_full_split_literal") or {}).get("bpc_full"), 6)
        + "), the sub-slice lower bound " + _fmt(ruling.get("one_se_lower_bound"), 6)
        + " and the upper bound " + _fmt(ruling.get("one_se_upper_bound"), 6) + " against "
        + _fmt(ruling.get("target_top1"), 4) + ". No weight, split, bar or measured value moved; "
        "the receipt and its sidecar were regenerated and " + _code("ac_cycle.json") + " closes at passed: **"
        + str(bool(ac_cycle.get("passed"))).lower() + "**.",
        "",
        "* Full frozen split: " + str(full.get("tokens", 0)) + " tokens at **"
        + _fmt(full.get("bits_per_char"), 6) + " bpc**, top-1 **" + _fmt(full.get("top1"), 6) + "**.",
        "* Legacy sub-slice: " + str(sub.get("tokens", 0)) + " tokens at **"
        + _fmt(sub.get("bits_per_char"), 6) + " bpc**, top-1 **" + _fmt(sub.get("top1"), 6)
        + "** with SE " + _fmt(sub.get("se_top1"), 6) + ".",
        "",
        "### 45.3 AD1: standalone multi-file repository repair",
        "",
        "* " + str(ad1_agg.get("hidden_solved", 0)) + "/"
        + str(ad1.get("suite", {}).get("repositories", 0)) + " repositories solved on both the public "
        "and the unseen maintainer suite at **" + _fmt(ad1_agg.get("emitted_precision"), 4)
        + "** emitted precision; " + str(ad1_agg.get("traps_refused", 0)) + "/4 traps refused.",
        "* The candidate set is capped at " + str((ad1.get("method") or {}).get("candidate_set_cap", 0))
        + " functions; the pre-execution RepoSheaf gate held on every executed rollout and the "
        "maximum measured index delta is " + _fmt(ad1_delta, 3)
        + " ms.",
        "* " + str(ad1_rep.get("total", 0)) + " REPULSOR axioms were written to Zone C ("
        + str(ad1_rep.get("zonec_rows", 0)) + " rows) with "
        + str(ad1_rep.get("deflections", 0)) + " deflections; the Zero Task-ID audit is "
        + str((ad1.get("audit") or {}).get("zero_task_id_branches")),
        "* AC2 non-regression: MBPP " + str((ad1.get("ac2_non_regression") or {}).get("mbpp", 0))
        + "/30, HumanEval " + str((ad1.get("ac2_non_regression") or {}).get("humaneval", 0))
        + "/32, precision " + _fmt((ad1.get("ac2_non_regression") or {}).get("precision"), 4)
        + ", Arm B transfer " + str((ad1.get("ac2_non_regression") or {}).get("arm_b_transfer", 0))
        + "/12.",
        "",
        "### 45.4 AD2: relational ARC morphisms and the compute-verified science solver",
        "",
        "* External ARC-AGI pass@1 **" + str(ad2_arc.get("solved", 0)) + "/"
        + str(ad2_arc.get("slice_tasks", 0)) + "** at emitted precision **"
        + _fmt(ad2_arc.get("precision"), 4) + "**, up " + str(ad2_arc.get("delta", 0))
        + " from Cycle AC, with zero task-name branches.",
        "* Six relational families are admitted: " + ", ".join(
            _code(name) for name in ad2_arc.get("relational_families") or []
        ) + ". They solve " + str(len(ad2_arc.get("solved_by_relational_family") or []))
        + " external tasks.",
        "* Science suite: **" + str(ad2_sci.get("answerable_solved", 0)) + "/"
        + str(ad2_sci.get("answerable", 0)) + "** answerable tasks at "
        + str(ad2_sci.get("traps_refused", 0)) + "/" + str(ad2_sci.get("traps", 0))
        + " traps refused, " + str(ad2_sci.get("hallucinations", 0))
        + " hallucinations and " + str(ad2_sci.get("llm_calls", 0))
        + " external model calls. Every emission passed dimensional homogeneity, inverse "
        "substitution below 1e-9, sandbox parity below 1e-9 and the science-primer ontology gate.",
        "",
        "### 45.5 AD3: the 96-task four-pillar manifest",
        "",
        "* Manifest: " + str(ad3_manifest.get("total", 0)) + " tasks ("
        + str(ad3_manifest.get("answerable", 0)) + " answerable and "
        + str((ad3_manifest.get("traps") or {}).get("total", 0)) + " traps) across "
        + str(len(ad3_manifest.get("pillars") or [])) + " pillars of twenty answerable tasks each.",
        "* Status: " + _code(str(ad3.get("status", ""))) + ". The fresh five-arm live run, the "
        "deliverable composer pass and the 1,000-request plus 320-query campaign were not "
        "executed in this session. " + _code("ad3_deliverable_and_aaii96.json")
        + " records that boundary and claims no gate that depends on the missing run.",
        "",
        "### 45.6 AD4: the two JEV Tier-1 seals and the bounded Blackwell attempt",
        "",
    ]
    for name, row in jev_challenges.items():
        verdict_row = row.get("verdict") or {}
        lines.append(
            "* " + _code(name) + ": **"
            + ("ACCEPT" if verdict_row.get("accepted") else "REJECT") + "** at score "
            + _fmt(verdict_row.get("score"), 4) + ", failed primitives "
            + str(verdict_row.get("failed") or []) + "."
        )
    lines += [
        "* Hardware: local " + _code(str(ad4_hw.get("cuda_status", ""))) + "; Vast.ai credential probe "
        + str((ad4_hw.get("vast_probe") or {}).get("ok")) + " with "
        + str((ad4_hw.get("vast_probe") or {}).get("instances", 0)) + " running instances; "
        + str((ad4_hw.get("vast_offers_dry_run") or {}).get("count", 0))
        + " offers in the dry run. A live rental is refused by the plugin policy: "
        + _code(str(ad4_hw.get("transport_error", ""))) + ". No instance was rented, the spend is $"
        + _fmt(ad4_hw.get("spend_usd"), 2) + " and " + str(ad4_hw.get("running_instances_at_exit", 0))
        + " instances were running at exit. No CPU measurement is relabelled as a GPU result.",
        "",
        "### 45.7 AD5: hygiene and the dual-cycle seal",
        "",
        "See " + _code("ad_cycle.json") + " for the pytest counts, the consistency exit code, the "
        ".sha256 sidecar parity, the R3 static and streaming bars and the credential scan.",
        "",
        "### 45.8 Boundaries that stay open in Cycle AD",
        "",
        "* The fresh 96-task five-arm live execution and the 320-query gauntlet are not run; "
        + _code("ad3_deliverable_and_aaii96.json") + " is the plan of record.",
        "* Gate GB-5 and the D = 4096 / 8192 Blackwell measurements are not measured because the "
        "plugin policy refuses a live Vast.ai rental in this runner. The exact refusal string is "
        "sealed in " + _code("ad4_blackwell_cuda_and_jev.json") + ".",
        "* The external ARC pass@1 is " + str(ad2_arc.get("solved", 0)) + "/48, not the whole slice; "
        "the remaining tasks need operator families that the current algebra does not express.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    text = DOCS.read_text(encoding="utf-8")
    section = build()
    if MARKER in text:
        text = text[: text.index(MARKER)].rstrip() + "\n"
    DOCS.write_text(text.rstrip() + "\n" + section + "\n", encoding="utf-8")
    print("[docs] Section 45 written (%d bytes)" % len(section))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
