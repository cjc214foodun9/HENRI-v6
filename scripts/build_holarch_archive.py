"""Rebuild the Cycle O..T record from the sealed JSON receipts.

The reconstruction lives outside docs/HOLARCH.md on purpose: the language
corpus reads docs/HOLARCH.md, so adding long tables there would move the R3
bits-per-character.  Every number below is read from a sealed receipt; nothing
is recomputed and nothing is remembered.

    python scripts/build_holarch_archive.py --out docs/HOLARCH_CYCLES_O_TO_T.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
RECEIPTS = ROOT / "receipts" / "holarch"

CYCLES: List[Tuple[str, str, str, List[str]]] = [
    ("O", "o", "Sovereign telepathy, SWAT prefill transfer, and the holonic API",
     ["o1_swat_telepathy.json", "o2_api_repulsor_and_linearizable_lanes.json",
      "o3_live_tree_language.json", "o4_holonic_and_gauntlet_128.json"]),
    ("P", "p", "Live-corpus language closure, beyond-context telepathy, and the plugin",
     ["p1_live_corpus_language_closure.json", "p2_beyond_context_telepathy_api.json",
      "p3_end_to_end_telepathic_vla.json", "p4_henri_holon_plugin.json"]),
    ("Q", "q", "Query-agnostic telepathy, causal Action-KV operator, disjoint splits",
     ["q1_query_agnostic_telepathy_and_fact_veto.json",
      "q2_causal_trimodal_telepathic_vla.json",
      "q3_disjoint_and_full_decile_language.json", "q4_warm_holon_and_aaii_v5.json"]),
    ("R", "r", "Unstructured telepathy, balanced causal matrix, static fact veto",
     ["r1_unstructured_telepathy_and_wave_fact_veto.json",
      "r2_balanced_causal_vla_matrix.json",
      "r3_static_and_streaming_full_decile_language.json",
      "r4_real_corpus_and_coprocessor_aaii_v5.json"]),
    ("S", "s", "Suffix purity, latent wave Turing machine, hierarchical charts",
     ["s1_language_closure.json", "s2_latent_wave_turing_machine.json",
      "s3_hierarchical_charts_and_sandboxed_browser.json",
      "s4_four_pillar_and_176_query_gauntlet.json"]),
    ("T", "t", "Hygiene and determinism, centered charts, foveated Turing, CCG and sheaf",
     ["t1_hygiene_and_determinism.json", "t2_centered_charts_mmlu.json",
      "t3_foveated_bidirectional_turing.json", "t4_sovereign_ccg_and_repo_sheaf.json"]),
]

NON_CLAIMS = {
    "O": ["The SWAT arm is measured on a CPU digital twin, not on optical hardware.",
          "The 128-query gauntlet is a symbolic self-audit, not an external benchmark."],
    "P": ["Telepathy is measured on layouts drawn from the same task family.",
          "The plugin result is a dry-run of the discovery engine, not a deployment."],
    "Q": ["The causal operator is measured on a 24-step episode budget.",
          "AAII v5.0 numbers are proxies on ingested slices, not the official report."],
    "R": ["The static fact veto is measured on the ingested corpus only.",
          "The balanced matrix holds three regimes, not the full layout space."],
    "S": ["Suffix purity is a language property of the pinned corpus.",
          "The latent wave Turing machine is a 16-cell bench, not a general computer."],
    "T": ["The foveated two-cell lookahead is diagnostic and is never counted as one-cell induction.",
          "Centered charts are measured on the MMLU slice, not on open-domain traffic."],
}


def _load(name: str) -> Dict[str, object]:
    path = RECEIPTS / name
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _digest(name: str) -> str:
    path = RECEIPTS / name
    if not path.exists():
        return "absent"
    side = path.with_name(path.name + ".sha256")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if side.exists() and side.read_text(encoding="utf-8").strip() == digest:
        return digest[:16]
    return digest[:16] + " (sidecar mismatch)"


def _rows(payload: Dict[str, object], prefix: str = "", depth: int = 0) -> List[Tuple[str, object]]:
    out: List[Tuple[str, object]] = []
    for key, value in payload.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.extend(_rows(value, name + ".", depth + 1))
        elif isinstance(value, (int, float, bool)) and not isinstance(value, bool):
            out.append((name, value))
        elif isinstance(value, bool):
            out.append((name, value))
    return out


def _table(rows: List[Tuple[str, object]], limit: int = 14) -> List[str]:
    lines = ["| Measured field | Value |", "| :--- | :--- |"]
    for name, value in rows[:limit]:
        if isinstance(value, bool):
            text = "true" if value else "false"
        elif isinstance(value, float):
            text = f"{value:.6g}"
        else:
            text = str(value)
        lines.append(f"| `{name}` | {text} |")
    return lines


def build() -> str:
    out: List[str] = []
    out.append("# Project HENRI: Cycles O to T reconstructed from sealed receipts")
    out.append("")
    out.append("This archive restores the long-form record of Cycles O, P, Q, R, S and T.")
    out.append("It is deliberately **outside** `docs/HOLARCH.md`: the language corpus reads")
    out.append("that file, so the R3 streaming bits-per-character stay at their sealed value.")
    out.append("Every table cell is read from one sealed JSON receipt, and every receipt is")
    out.append("identified by the first 16 hex digits of its SHA-256.")
    out.append("")
    for letter, prefix, title, names in CYCLES:
        out.append(f"## Cycle {letter}: {title}")
        out.append("")
        cycle = _load(f"{prefix}_cycle.json")
        stages = dict(cycle.get("stages") or {})
        out.append("| Stage | Receipt | SHA-256 (16) | Passed | Wall (s) |")
        out.append("| :--- | :--- | :--- | :---: | ---: |")
        for key, row in stages.items():
            wall = float(row.get("wall_seconds", 0.0) or 0.0)
            receipt = str(row.get("receipt", "")).replace("receipts/holarch/", "").replace("receipts\\holarch\\", "")
            out.append(f"| {key} | `{receipt}` | `{str(row.get('sha256', ''))[:16]}` | "
                       f"{'true' if row.get('passed') else 'false'} | {wall:.2f} |")
        out.append("")
        out.append(f"Cycle aggregate `passed`: **{str(bool(cycle.get('passed'))).lower()}**.")
        out.append("")
        gates: List[str] = []
        for key, row in stages.items():
            for gate, value in (row.get("gate") or {}).items():
                if gate == "passed":
                    continue
                gates.append(f"`{gate}`={'true' if value else 'false'}")
        if gates:
            out.append("Measured gates: " + ", ".join(gates) + ".")
            out.append("")
        for name in names:
            payload = _load(name)
            if not payload:
                out.append(f"### `{name}`")
                out.append("")
                out.append("Receipt absent at reconstruction time.")
                out.append("")
                continue
            out.append(f"### `{name}`")
            out.append("")
            stage = payload.get("stage", "")
            evidence = payload.get("evidence", "")
            out.append(f"Stage `{stage}`, evidence `{evidence}`, "
                       f"aggregate passed **{str(bool(payload.get('passed'))).lower()}**, "
                       f"SHA-256 `{_digest(name)}`.")
            out.append("")
            rows = [row for row in _rows(payload) if "sha256" not in row[0]]
            out.extend(_table(rows))
            out.append("")
        out.append("### Explicit non-claims")
        out.append("")
        for item in NON_CLAIMS.get(letter, []):
            out.append(f"* {item}")
        out.append("")
    out.append("## Provenance")
    out.append("")
    out.append("Each stage table above is a projection of a sealed receipt. The receipt")
    out.append("and its sidecar, not this prose, are the durable evidence.")
    out.append("")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild the Cycle O..T archive")
    parser.add_argument("--out", default=str(ROOT / "docs" / "HOLARCH_CYCLES_O_TO_T.md"))
    args = parser.parse_args()
    text = build()
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    print(f"wrote {target} ({len(text)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
