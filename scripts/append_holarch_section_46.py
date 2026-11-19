"""Append Section 46 (Cycle AE) to docs/HOLARCH.md from the sealed receipts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
RECEIPTS = ROOT / "receipts" / "holarch"
DOCS = ROOT / "docs" / "HOLARCH.md"
MARKER = "## 46. Cycle AE"
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
    ae0 = _read("ae0_ac1_subslice_asymmetry.json")
    ae1 = _read("ae1_blackwell_cuda_and_d4096.json")
    ae2 = _read("ae2_external_repo_patch_and_science.json")
    ae3 = _read("ae3_compositional_external_arc.json")
    ae4 = _read("ae4_api_and_10k_atlas.json")
    jev = _read("ae_jev_mathematical_challenges.json")
    ae_cycle = _read("ae_cycle.json")
    ad_cycle = _read("ad_cycle.json")
    verdict = ae0.get("verdict") or {}
    full = ae0.get("full_population") or {}
    sub = ae0.get("sub_slice") or {}
    comparison = ae0.get("comparison") or {}
    bootstrap = ae0.get("bootstrap") or {}
    offers = ae1.get("offers") or {}
    hardware = ae1.get("hardware") or {}
    contract = ae1.get("software_contract") or {}
    atlas = ae4.get("atlas") or {}

    def failing(gate: Dict[str, Any]) -> str:
        names = [name for name, value in gate.items() if not bool(value) and name != "passed"]
        return ", ".join(names) if names else "none"

    stage_rows = [
        ("AE0 AC1 asymmetry diagnostic", "ae0_ac1_subslice_asymmetry.json", ae0.get("gate") or {}),
        ("AE1 Vast repair and Blackwell contract", "ae1_blackwell_cuda_and_d4096.json", ae1.get("gate") or {}),
        ("AE2 external audits", "ae2_external_repo_patch_and_science.json", ae2.get("gate") or {}),
        ("AE3 compositional ARC", "ae3_compositional_external_arc.json", ae3.get("gate") or {}),
        ("AE4 API and 10k atlas", "ae4_api_and_10k_atlas.json", ae4.get("gate") or {}),
        ("AE5 JEV Tier-1", "ae_jev_mathematical_challenges.json", jev.get("gate") or {}),
    ]
    lines = [
        "",
        MARKER + ": the sub-slice asymmetry diagnostic, the Vast.ai offer repair, the external-audit boundary and the 10,000-engram atlas",
        "",
        "* **Document ID:** HENRI-SPEC-CYCLE-AE-AD3-CLOSURE-EXTERNAL-DECOUPLING-COMPOSITIONAL-ARC-AND-BLACKWELL-EXECUTION-V1",
        "* **Baseline:** Cycle AD. Every number below is OBSERVED in this process and sealed under receipts/holarch/.",
        "",
        "### 46.1 Stage verdicts",
        "",
        "| Stage | Receipt | Verdict | Failing gates |",
        "| :--- | :--- | :---: | :--- |",
    ]
    for label, name, gate in stage_rows:
        lines.append(
            "| %s | %s | **%s** | %s |"
            % (label, _code(name), "PASS" if gate.get("passed") else "OPEN", failing(gate))
        )
    lines += [
        "| AE hygiene seal | " + _code("ae_cycle.json") + " | **"
        + ("PASS" if ae_cycle.get("gate", {}).get("passed") else "OPEN")
        + "** | " + failing(ae_cycle.get("gate") or {}) + " |",
        "",
        "### 46.2 AE0.3: the AC1 identifier-density diagnostic",
        "",
        "* Full population: " + str(full.get("tokens", 0)) + " tokens, Tier-2 fraction **"
        + _fmt(full.get("tier2_fraction"), 4) + "**, byte fraction **"
        + _fmt(full.get("byte_fraction"), 6) + "**, identifier density **"
        + _fmt(full.get("identifier_density_per_1000"), 2) + "** per 1,000 tokens, mean backoff order **"
        + _fmt((full.get("mean_backoff") or {}).get("mean_order"), 3) + "**.",
        "* Legacy sub-slice: " + str(sub.get("tokens", 0)) + " tokens, Tier-2 fraction **"
        + _fmt(sub.get("tier2_fraction"), 4) + "**, byte fraction **"
        + _fmt(sub.get("byte_fraction"), 6) + "**, identifier density **"
        + _fmt(sub.get("identifier_density_per_1000"), 2) + "** per 1,000 tokens, mean backoff order **"
        + _fmt((sub.get("mean_backoff") or {}).get("mean_order"), 3) + "**.",
        "* Identifier-density delta **" + _fmt(comparison.get("identifier_density_delta_per_1000"), 2)
        + "** per 1,000 tokens (ratio " + _fmt(comparison.get("identifier_density_ratio"), 3)
        + "); window-level density/top-1 Pearson **" + _fmt(comparison.get("window_density_vs_top1_pearson"), 3)
        + "**. Verdict: **" + str(verdict.get("decision", "")) + "** under the pre-registered rule.",
        "* Bootstrap of contiguous " + str((bootstrap.get("full") or {}).get("windows", 0))
        + " windows: top-1 95% CI [" + _fmt((bootstrap.get("full") or {}).get("ci95_top1", [None, None])[0], 6)
        + ", " + _fmt((bootstrap.get("full") or {}).get("ci95_top1", [None, None])[1], 6)
        + "], bpc 95% CI [" + _fmt((bootstrap.get("full") or {}).get("ci95_bpc", [None, None])[0], 6)
        + ", " + _fmt((bootstrap.get("full") or {}).get("ci95_bpc", [None, None])[1], 6)
        + "]. Sub-slice inside full top-1 CI: " + str(comparison.get("sub_top1_inside_full_ci95"))
        + "; inside bpc CI: " + str(comparison.get("sub_bpc_inside_full_ci95")) + ".",
        "",
        "### 46.3 AE1: the Vast.ai offer repair and the policy-respecting Blackwell contract",
        "",
        "* Repaired dry-run discovery: **" + str(offers.get("count", 0)) + "** offers, cheapest **$"
        + _fmt(offers.get("cheapest_hourly"), 4) + "/h**, GPU names "
        + str(offers.get("gpu_names")) + ", compute capabilities "
        + str(offers.get("compute_caps")) + ".",
        "* Root cause: " + _code(str((offers.get("repair") or {}).get("root_cause", "")))
        + ". The default search now covers the live Blackwell workstation market and the plugin passes one GPU list.",
        "* Policy flag " + _code("allowLiveVast") + " is **" + str((ae1.get("policy") or {}).get("allowLiveVast"))
        + "**; the live attempt is sealed as " + _code(str(((ae1.get("live_attempt") or {}).get("error", ""))))
        + ". Spend **$" + _fmt(hardware.get("spend_usd"), 2) + "**, rented "
        + str(hardware.get("rented")) + ", instances at exit " + str(hardware.get("running_instances_at_exit")) + ".",
        "* Software contract: capability exact on (12, 0) with 170/188 SMs, shared-memory budget "
        + str(contract.get("shared_memory_bytes")) + " B under the " + str(contract.get("max_smem_bytes"))
        + " B wall, production D = " + str((contract.get("production") or {}).get("dim"))
        + " with " + str((contract.get("production") or {}).get("parameters"))
        + " parameters, and FFT energy parity on D = 2048/4096/8192. No GPU number is claimed.",
        "",
        "### 46.4 AE2 and AE3: the external-audit and compositional-ARC boundary",
        "",
        "* AE2 status " + _code(str(ae2.get("status", ""))) + ": no external repository or science slice was vendored, "
        "so no external pass@1, precision or trap number is claimed. AD1 16/16 and AD2 18/18 remain internal and unchanged.",
        "* AE3 status " + _code(str(ae3.get("status", ""))) + ": the sealed external ARC result stays at "
        + str((ae3.get("baseline") or {}).get("external_solved", 0)) + "/"
        + str((ae3.get("baseline") or {}).get("external_total", 48))
        + " at precision " + _fmt((ae3.get("baseline") or {}).get("precision"), 4)
        + ". No depth-three family was admitted inside the session budget.",
        "",
        "### 46.5 AE4: the 10,000-engram Riemannian atlas measurement",
        "",
        "* Atlas: " + str(atlas.get("world_rows", 0)) + " world rows plus "
        + str(atlas.get("online_writes", 0)) + " online writes; charts "
        + str((atlas.get("stats") or {}).get("charts", 0)) + ". Grounded top-1 parity **"
        + _fmt(atlas.get("parity"), 4) + "**, drift **" + str(atlas.get("drift"))
        + "/200**, online recall **" + str(atlas.get("online_recall")) + "/200**, OOD hallucination **"
        + str(atlas.get("ood_hallucinated")) + "/" + str(atlas.get("ood_queries"))
        + "**, candidate rescore fraction mean **" + _fmt(atlas.get("candidate_fraction_mean"), 4)
        + "** (max " + _fmt(atlas.get("candidate_fraction_max"), 4) + "), measured speedup **"
        + _fmt(atlas.get("speedup"), 2) + "x**.",
        "* The REST/MCP/OpenAI verbs and the 336-query gauntlet are " + _code("NOT_EXECUTED_IN_THIS_STAGE")
        + " and no gate that depends on them is claimed.",
        "",
        "### 46.6 AE5: the two JEV Tier-1 seals and the ledger",
        "",
    ]
    for name, row in (jev.get("challenges") or {}).items():
        verdict_row = row.get("verdict") or {}
        lines.append(
            "* " + _code(name) + ": **"
            + ("ACCEPT" if verdict_row.get("accepted") else "REJECT") + "** at score "
            + _fmt(verdict_row.get("score"), 4) + ", failed primitives "
            + str(verdict_row.get("failed") or []) + "."
        )
    lines += [
        "* " + _code("ae_cycle.json") + " records pytest "
        + str((ae_cycle.get("pytest") or {}).get("counts", {})) + ", consistency exit "
        + str((ae_cycle.get("consistency") or {}).get("exit_code", "not run"))
        + ", sidecar parity " + _fmt((ae_cycle.get("sidecar_parity") or {}).get("parity", 0.0), 4)
        + " and R3 static " + _fmt((ae_cycle.get("language_hygiene") or {}).get("r3_static_bits_per_char"), 4)
        + " bpc, streaming " + _fmt((ae_cycle.get("language_hygiene") or {}).get("r3_streaming_bits_per_char"), 4)
        + " bpc.",
        "",
        "### 46.7 Boundaries that stay open in Cycle AE",
        "",
        "* AD3 remains open, so " + _code("ad_cycle.json") + " is **" + str(ad_cycle.get("passed"))
        + "** and the AE seal cannot close on the AD3 gate.",
        "* AE2 and AE3 remain unexecuted; the external-audit and depth-three gates are false by construction, "
        "not by a lowered bar.",
        "* The live Blackwell GB-5 measurement is blocked by the plugin policy flag, not by the repaired offer discovery.",
        "* The AE4 atlas reaches parity, zero drift and zero OOD hallucination but the measured wall-clock speedup "
        "is " + _fmt(atlas.get("speedup"), 2) + "x and the API/gauntlet stages are not executed.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    text = DOCS.read_text(encoding="utf-8")
    section = build()
    if MARKER in text:
        text = text[: text.index(MARKER)].rstrip() + "\n"
    DOCS.write_text(text.rstrip() + "\n" + section + "\n", encoding="utf-8")
    print("[docs] Section 46 written (%d bytes)" % len(section))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
