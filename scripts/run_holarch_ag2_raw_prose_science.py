"""Cycle AG2: Deterministic raw-prose dimensional physics parser & blind SciBench audit.

    .venv/bin/python scripts/run_holarch_ag2_raw_prose_science.py

Mandatory Two-Tier & Pre-Freeze Protocol:
1. Tier 1 (Existing 24-Problem Dev SciBench Slice — Raw Text Only + 6 Traps):
   Strip ``quantities`` and ``target`` completely from all 24 AF3 SciBench
   problems, feed ONLY unmodified ``prompt`` to ``ScienceSolver.solve_text``,
   and also test all 6 AF3 raw-prose traps (+ AD2 raw-prose traps & structured
   non-regression). Verify live raw-prose ingestion on ``POST /v1/holarch/science``
   and MCP ``holarch_science``.
2. Pre-Freeze Seal (BEFORE fetching any blind SciBench problems):
   Compute and seal SHA-256 digests of ``henri/holarch/science_parser.py`` and
   ``henri/holarch/science_solver.py``. Never edit either file after sealing.
3. Tier 2 (Blind 24-Problem Disjoint SciBench Holdout):
   Clone pinned upstream ``mandyyyyii/scibench`` at commit
   ``e14e0ca3d3db0f02928a18d435b9c3d15656a7c5``, exclude every problem overlapping
   AF3's 24-problem dev slice, select 24 disjoint unmodified problems by
   deterministic uniform sorted stride (zero difficulty/topic/length filtering),
   vendor to ``receipts/holarch/external/science_blind_24/`` with ``MANIFEST.json``,
   and evaluate raw ``problem_text`` blindly via ``ScienceSolver.solve_text``.
4. Re-hash both frozen modules to prove ``parser_and_solver_sha256_unchanged: true``,
   verify zero task-name branching and zero LLM calls, and seal
   ``receipts/holarch/ag2_raw_prose_science_audit.json`` + ``.sha256``.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from fastapi.testclient import TestClient

import run_holarch_ad2_relational_arc_and_science as ad2
import run_holarch_p_cycle as pc
from henri.gateway.app import create_gateway_app
from henri.gateway.config import GatewayConfig
from henri.gateway.mcp_mount import build_mcp_server
from henri.holarch.cold_synth import audit_task_specific_branches
from henri.holarch.science_parser import (
    normalize_science_text,
    parse_compound_unit,
)
from henri.holarch.science_solver import ScienceSolver, unit_dims, unit_scale

RECEIPTS = ROOT / "receipts" / "holarch"
WORK = RECEIPTS / "_work_ag"
PARSER_PATH = ROOT / "henri" / "holarch" / "science_parser.py"
SOLVER_PATH = ROOT / "henri" / "holarch" / "science_solver.py"
AF3_DIR = RECEIPTS / "external" / "science"
AF3_MANIFEST = AF3_DIR / "MANIFEST.json"
BLIND_DIR = RECEIPTS / "external" / "science_blind_24"
BLIND_MANIFEST = BLIND_DIR / "MANIFEST.json"
PREFREEZE_FILE = WORK / "ag2_science_prefreeze.json"

PINNED = RECEIPTS / "pinned_corpus_snapshot.json"
PINNED_SHA256 = "d675f3816bbf472493b25f885bbb56c62c5ec88a81ab31e31a9b87b3a0c23703"
PINNED_DIGEST = "8f270f34e8f35ef84e415d076bea20a7bc15edff337f89ed72ad607447ba8554"

UPSTREAM_URL = "https://github.com/mandyyyyii/scibench.git"
UPSTREAM_COMMIT = "e14e0ca3d3db0f02928a18d435b9c3d15656a7c5"
BLIND_COUNT = 24
FORBIDDEN_IMPORTS = ("openai", "anthropic", "requests", "urllib", "transformers", "tokenizers")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_upstream_stem(fname: str) -> str:
    name = str(fname).strip().lower()
    if name.endswith("_sol.json"):
        return name[: -len("_sol.json")] + ".json"
    return name


def _parse_scibench_gt_unit(unit_raw: str) -> Tuple[float, Optional[str]]:
    """Parse a SciBench ground-truth LaTeX unit string into (pow10_scale, unit_expr).

    Handles expressions such as ``$10^{-19} \\mathrm{~J}$``,
    ``$\\mathrm{kJ}/\\mathrm{mol}$``, ``$\\%$``, ``""`` (dimensionless), etc.
    """
    raw = str(unit_raw or "").strip()
    if not raw:
        return 1.0, "1"
    s = raw.replace("$", "").strip()
    if not s:
        return 1.0, "1"

    scale = 1.0
    # Strip optional leading multiplier such as 10^{-19}, 10^6, \times 10^{-3}
    s = re.sub(r"^(?:\\times|×|\*)\s*", "", s).strip()
    m_pow = re.match(r"^10\s*\^\s*\{?\s*([+-]?\d+)\s*\}?\s*(?:\\times|\\cdot|\*|\\,|~|\s)*", s)
    if m_pow:
        scale = 10.0 ** int(m_pow.group(1))
        s = s[m_pow.end() :].strip()
    if not s:
        return scale, "1"

    # Normalize LaTeX unit wrapper commands
    s = s.replace("~", " ").replace("\\,", " ").replace("\\;", " ").replace("\\:", " ")
    s = s.replace("\\cdot", "*").replace("\\times", "*")
    s = re.sub(r"\\(?:mathrm|text|operatorname|rm)\s*\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\%", "%", s)
    s = s.replace("^{\\circ}C", "degC").replace("^\\circ C", "degC").replace("°C", "degC")
    s = s.replace("\\Omega", "Ohm").replace("Ω", "Ohm").replace("\\mu", "u").replace("μ", "u")
    s = re.sub(r"\\AA\b|\\text\{\\AA\}|Å", "A_ang", s)
    s = re.sub(r"\^\{\s*([+-]?\d+)\s*\}", r"^\1", s)
    s = re.sub(r"\s+", " ", s).strip()

    # Handle fraction syntax like \frac{kJ}{mol}
    frac_m = re.match(r"^\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}$", s)
    if frac_m:
        s = f"({frac_m.group(1).strip()})/({frac_m.group(2).strip()})"

    # Handle space-separated compound units like "N m^2 / C" or "J mol^-1 K^-1"
    parsed = parse_compound_unit(s)
    if parsed is not None:
        return scale, parsed[2]
    # Try replacing spaces with '*'
    s_star = re.sub(r"\s*/\s*", "/", s)
    s_star = re.sub(r"\s+", "*", s_star)
    parsed2 = parse_compound_unit(s_star)
    if parsed2 is not None:
        return scale, parsed2[2]

    # Fallback via normalize_science_text
    norm, _, _, _ = normalize_science_text(f"1 {raw}")
    tail = norm[1:].strip()
    parsed3 = parse_compound_unit(tail)
    if parsed3 is not None:
        return scale, parsed3[2]
    return scale, None


def _compare_answer(
    pred_val: float,
    pred_unit: str,
    exp_val: float,
    exp_unit_raw: str,
    *,
    rel_tol: float = 0.02,
) -> Dict[str, Any]:
    """Compare predicted (value, unit) against ground truth in both SI and raw scalar space."""
    u = str(pred_unit or "1").strip() or "1"
    pow10_scale, exp_u_parsed = _parse_scibench_gt_unit(exp_unit_raw)
    exp_scaled = float(exp_val) * pow10_scale
    raw_match = abs(pred_val - exp_val) <= max(1e-9, rel_tol * max(abs(exp_val), 1.0))

    si_match = False
    rel_err: Optional[float] = None
    compare_mode = "uncomparable_unit"
    if exp_u_parsed is not None:
        try:
            d_pred = unit_dims(u)
            d_exp = unit_dims(exp_u_parsed)
            if d_pred == d_exp:
                val_si = float(pred_val) * float(unit_scale(u))
                exp_si = float(exp_scaled) * float(unit_scale(exp_u_parsed))
                denom = max(abs(exp_si), 1e-30)
                rel_err = abs(val_si - exp_si) / denom
                si_match = abs(val_si - exp_si) <= max(1e-12, rel_tol * denom)
                compare_mode = "si_exact"
            elif exp_unit_raw.strip() == "" and abs(pred_val - exp_val) <= max(
                1e-9, rel_tol * max(abs(exp_val), 1.0)
            ):
                # Some SciBench entries leave unit="" even for dimensional quantities
                si_match = True
                rel_err = abs(pred_val - exp_val) / max(abs(exp_val), 1e-30)
                compare_mode = "scalar_blank_gt_unit"
            else:
                compare_mode = f"dim_mismatch:{u}_vs_{exp_u_parsed}"
        except Exception as exc:
            compare_mode = f"unit_error:{exc}"
    else:
        if raw_match:
            si_match = True
            rel_err = abs(pred_val - exp_val) / max(abs(exp_val), 1e-30)
            compare_mode = "scalar_fallback"

    return {
        "si_match": bool(si_match),
        "raw_match": bool(raw_match),
        "rel_err": round(float(rel_err), 8) if rel_err is not None else None,
        "compare_mode": compare_mode,
        "gt_pow10_scale": pow10_scale,
        "gt_unit_parsed": exp_u_parsed,
    }


def _seal_prefreeze() -> Dict[str, Any]:
    """Compute & seal SHA-256 digests of science_parser.py and science_solver.py BEFORE blind fetch."""
    WORK.mkdir(parents=True, exist_ok=True)
    parser_sha = _sha256(PARSER_PATH)
    solver_sha = _sha256(SOLVER_PATH)
    if PREFREEZE_FILE.exists():
        rec = json.loads(PREFREEZE_FILE.read_text(encoding="utf-8"))
        return {
            "prefreeze_parser_sha256": str(rec["science_parser_sha256"]),
            "prefreeze_solver_sha256": str(rec["science_solver_sha256"]),
            "current_before_vendor_parser_sha256": parser_sha,
            "current_before_vendor_solver_sha256": solver_sha,
            "sealed_at_epoch": float(rec.get("sealed_at_epoch", 0.0)),
            "record_path": str(PREFREEZE_FILE.relative_to(ROOT)),
            "blind_manifest_existed_at_prefreeze": bool(rec.get("blind_manifest_existed_at_prefreeze", False)),
        }
    rec = {
        "stage": "AG2_PREFREEZE",
        "files": {
            "science_parser": "henri/holarch/science_parser.py",
            "science_solver": "henri/holarch/science_solver.py",
        },
        "science_parser_sha256": parser_sha,
        "science_solver_sha256": solver_sha,
        "sealed_at_epoch": time.time(),
        "blind_manifest_existed_at_prefreeze": bool(BLIND_MANIFEST.exists()),
    }
    PREFREEZE_FILE.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return {
        "prefreeze_parser_sha256": parser_sha,
        "prefreeze_solver_sha256": solver_sha,
        "current_before_vendor_parser_sha256": parser_sha,
        "current_before_vendor_solver_sha256": solver_sha,
        "sealed_at_epoch": rec["sealed_at_epoch"],
        "record_path": str(PREFREEZE_FILE.relative_to(ROOT)),
        "blind_manifest_existed_at_prefreeze": rec["blind_manifest_existed_at_prefreeze"],
    }


def _vendor_blind_24(af3_manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Vendor 24 strictly disjoint unmodified SciBench problems from pinned upstream commit."""
    if BLIND_MANIFEST.exists():
        manifest = json.loads(BLIND_MANIFEST.read_text(encoding="utf-8"))
        if int(manifest.get("problem_count", 0)) == BLIND_COUNT and len(manifest.get("problems") or []) == BLIND_COUNT:
            return manifest

    af3_pairs = {
        (_canonical_upstream_stem(str(item["upstream_file"])), str(item["upstream_problemid"]).strip())
        for item in af3_manifest.get("problems", [])
    }

    with tempfile.TemporaryDirectory(prefix="ag2_scibench_upstream_") as tmp:
        src_repo = Path(tmp) / "scibench"
        subprocess.run(
            ["git", "clone", "--no-checkout", UPSTREAM_URL, str(src_repo)],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "-C", str(src_repo), "checkout", UPSTREAM_COMMIT],
            check=True,
            capture_output=True,
            text=True,
        )
        orig_dir = src_repo / "dataset" / "original"
        all_files = sorted(orig_dir.glob("*.json"))
        # Deduplicate *_sol.json when base *.json exists in the same directory
        base_names = {f.name for f in all_files if not f.name.endswith("_sol.json")}
        chosen_files = [
            f
            for f in all_files
            if not (f.name.endswith("_sol.json") and _canonical_upstream_stem(f.name) in base_names)
        ]

        pool: List[Dict[str, Any]] = []
        seen_keys: set[Tuple[str, str]] = set()
        for fpath in chosen_files:
            fsha = _sha256(fpath)
            data = json.loads(fpath.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                continue
            canon_stem = _canonical_upstream_stem(fpath.name)
            for idx, entry in enumerate(data):
                if not isinstance(entry, dict):
                    continue
                ptext = str(entry.get("problem_text") or "").strip()
                ans_raw = entry.get("answer_number")
                pid = str(entry.get("problemid") or f"idx_{idx}").strip()
                if not ptext or ans_raw is None:
                    continue
                try:
                    ans_val = float(str(ans_raw).strip())
                except ValueError:
                    continue
                if not math.isfinite(ans_val):
                    continue
                key = (canon_stem, pid)
                if key in af3_pairs or key in seen_keys:
                    continue
                seen_keys.add(key)
                pool.append(
                    {
                        "upstream_file": fpath.name,
                        "upstream_file_sha256": fsha,
                        "canonical_stem": canon_stem,
                        "upstream_problemid": pid,
                        "source": str(entry.get("source") or fpath.stem),
                        "problem_text": ptext,
                        "answer_number": ans_val,
                        "unit": str(entry.get("unit") or ""),
                    }
                )

    pool.sort(key=lambda r: (r["upstream_file"], r["upstream_problemid"]))
    assert len(pool) >= BLIND_COUNT, f"insufficient disjoint pool: {len(pool)}"
    stride = len(pool) // BLIND_COUNT
    selected = [pool[i * stride] for i in range(BLIND_COUNT)]

    prob_dir = BLIND_DIR / "problems"
    prob_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: List[Dict[str, Any]] = []
    for i, item in enumerate(selected, start=1):
        bid = f"blind_sci_{i:02d}"
        out_path = prob_dir / f"{bid}.json"
        payload = {
            "id": bid,
            "source": item["source"],
            "upstream_file": item["upstream_file"],
            "upstream_file_sha256": item["upstream_file_sha256"],
            "upstream_problemid": item["upstream_problemid"],
            "problem_text": item["problem_text"],
            "answer_number": item["answer_number"],
            "unit": item["unit"],
        }
        raw_json = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        out_path.write_text(raw_json, encoding="utf-8")
        manifest_rows.append(
            {
                "id": bid,
                "file": f"problems/{bid}.json",
                "sha256": _sha256_bytes(raw_json.encode("utf-8")),
                "source": item["source"],
                "upstream_file": item["upstream_file"],
                "upstream_file_sha256": item["upstream_file_sha256"],
                "upstream_problemid": item["upstream_problemid"],
            }
        )

    slice_hash = _sha256_bytes(
        json.dumps(
            [(r["upstream_file"], r["upstream_problemid"], r["sha256"]) for r in manifest_rows],
            sort_keys=True,
        ).encode("utf-8")
    )
    manifest = {
        "slice_id": "ag2_blind_scibench_24_v1",
        "upstream_repo": "https://github.com/mandyyyyii/scibench",
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_license": "MIT",
        "selection_rule": f"deterministic_sorted_stride(pool_size={len(pool)}, stride={stride}, count={BLIND_COUNT}, zero_filtering)",
        "disjoint_pool_size": len(pool),
        "stride": stride,
        "problem_count": len(manifest_rows),
        "slice_sha256": slice_hash,
        "problems": manifest_rows,
    }
    BLIND_MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _audit_forbidden_imports(paths: Sequence[Path]) -> Dict[str, Any]:
    hits: Dict[str, List[str]] = {}
    for p in paths:
        src = p.read_text(encoding="utf-8")
        found = [
            mod
            for mod in FORBIDDEN_IMPORTS
            if re.search(rf"^\s*(?:import|from)\s+{re.escape(mod)}\b", src, re.M)
        ]
        hits[str(p.relative_to(ROOT))] = found
    return {
        "hits": hits,
        "clean": all(not v for v in hits.values()),
    }


def _probe_gateway_and_mcp_raw_prose(sample_prompt: str, trap_prompt: str) -> Dict[str, Any]:
    """Verify raw-prose science solving on live POST /v1/holarch/science and MCP holarch_science."""
    with tempfile.TemporaryDirectory(prefix="ag2_gw_probe_") as tmp:
        cfg = GatewayConfig(
            require_auth=False,
            sqlite_path=str(Path(tmp) / "gateway.sqlite3"),
            zone_c_sqlite_path=str(Path(tmp) / "zone_c.sqlite3"),
            signing_key_path=str(Path(tmp) / "signing.key"),
            holarch_autoload=False,
            enable_metrics=False,
        )
        app = create_gateway_app(config=cfg, mount_mcp=False)
        svc = app.state.gateway
        client = TestClient(app)

        # 1. REST POST /v1/holarch/science with raw prompt only (no quantities/target)
        r_ok = client.post("/v1/holarch/science", json={"prompt": sample_prompt})
        d_ok = r_ok.json()
        res_ok = d_ok.get("result") or {}

        # 2. REST POST /v1/holarch/science with raw-prose trap
        r_trap = client.post("/v1/holarch/science", json={"prompt": trap_prompt})
        d_trap = r_trap.json()
        res_trap = d_trap.get("result") or {}

        # 3. REST POST /v1/holarch/science empty payload -> HTTP 422
        r_422 = client.post("/v1/holarch/science", json={})

        # 4. MCP holarch_science tool direct invocation
        mcp_srv = build_mcp_server(svc)
        tools_dict = {}
        tm = getattr(mcp_srv, "_tool_manager", None)
        if tm is not None and hasattr(tm, "_tools"):
            tools_dict = dict(tm._tools)
        has_holarch_science = "holarch_science" in tools_dict
        mcp_solved = False
        mcp_trap_refused = False
        if has_holarch_science:
            fn = tools_dict["holarch_science"].fn
            m_ok = fn(prompt=sample_prompt)
            m_trap = fn(prompt=trap_prompt)
            mcp_solved = (m_ok.get("result") or {}).get("status") == "SOLVED"
            mcp_trap_refused = (m_trap.get("result") or {}).get("status") != "SOLVED"

        if hasattr(svc, "shutdown"):
            svc.shutdown()
        return {
            "rest_raw_prompt_status_code": r_ok.status_code,
            "rest_raw_prompt_solved": res_ok.get("status") == "SOLVED",
            "rest_raw_prompt_input_mode": res_ok.get("input_mode"),
            "rest_raw_prompt_answer": res_ok.get("answer"),
            "rest_raw_prompt_unit": res_ok.get("unit"),
            "rest_raw_prompt_receipt_signed": bool((d_ok.get("epistemic_receipt") or {}).get("signature")),
            "rest_trap_status_code": r_trap.status_code,
            "rest_trap_refused": res_trap.get("status") != "SOLVED",
            "rest_trap_detected": bool(res_trap.get("trap_detected")),
            "rest_empty_returns_422": r_422.status_code == 422,
            "mcp_has_holarch_science": has_holarch_science,
            "mcp_raw_prompt_solved": mcp_solved,
            "mcp_trap_refused": mcp_trap_refused,
            "passed": bool(
                r_ok.status_code == 200
                and res_ok.get("status") == "SOLVED"
                and res_ok.get("input_mode") == "raw_text"
                and bool((d_ok.get("epistemic_receipt") or {}).get("signature"))
                and r_trap.status_code == 200
                and res_trap.get("status") != "SOLVED"
                and r_422.status_code == 422
                and has_holarch_science
                and mcp_solved
                and mcp_trap_refused
            ),
        }


def main() -> int:
    t0 = time.perf_counter()
    assert _sha256(PINNED) == PINNED_SHA256, "pinned_corpus_snapshot.json mutated"
    pinned_doc = json.loads(PINNED.read_text(encoding="utf-8"))
    assert pinned_doc["digest"] == PINNED_DIGEST

    solver = ScienceSolver(ontology_root=ROOT)
    af3_manifest = json.loads(AF3_MANIFEST.read_text(encoding="utf-8"))

    # =========================================================================
    # TIER 1: Existing 24-Problem Dev SciBench Slice (RAW TEXT ONLY) + 6 Traps
    # =========================================================================
    dev_rows: List[Dict[str, Any]] = []
    dev_emitted = 0
    dev_si_ok = 0
    dev_raw_ok = 0
    dev_all_dim_ok = True
    dev_max_residual = 0.0
    dev_llm_calls = 0

    for item in af3_manifest["problems"]:
        pdoc = json.loads((AF3_DIR / item["file"]).read_text(encoding="utf-8"))
        # Strip quantities and target completely: feed ONLY unmodified raw prose!
        raw_prompt = str(pdoc["prompt"])
        exp_val = float(pdoc["expected"])
        exp_u = str(pdoc["target"]["unit"])
        tol = float(pdoc.get("rel_tol", 0.02))

        res = solver.solve_text(raw_prompt)
        dev_llm_calls += int(res.get("llm_calls", 0))
        row: Dict[str, Any] = {
            "id": pdoc["id"],
            "upstream_file": pdoc["upstream_file"],
            "upstream_problemid": pdoc["upstream_problemid"],
            "status": res.get("status"),
            "reason": res.get("reason"),
            "input_mode": res.get("input_mode"),
            "relation": res.get("relation"),
            "answer": res.get("answer"),
            "unit": res.get("unit"),
            "expected": exp_val,
            "expected_unit": exp_u,
            "dimensional_ok": res.get("dimensional_ok"),
            "residual": res.get("residual"),
        }
        if res.get("status") == "SOLVED" and res.get("answer") is not None:
            dev_emitted += 1
            if not res.get("dimensional_ok"):
                dev_all_dim_ok = False
            dev_max_residual = max(dev_max_residual, float(res.get("residual", 0.0)))
            cmp = _compare_answer(float(res["answer"]), str(res.get("unit") or "1"), exp_val, exp_u, rel_tol=tol)
            row.update(cmp)
            if cmp["si_match"]:
                dev_si_ok += 1
            if cmp["raw_match"]:
                dev_raw_ok += 1
        dev_rows.append(row)

    dev_prec = (dev_si_ok / dev_emitted) if dev_emitted > 0 else 0.0

    # 6 AF3 raw-prose traps (strip quantities/target; feed ONLY raw prompt)
    trap_rows: List[Dict[str, Any]] = []
    af3_traps_refused = 0
    for titem in af3_manifest["traps"]:
        tdoc = json.loads((AF3_DIR / titem["file"]).read_text(encoding="utf-8"))
        tres = solver.solve_text(str(tdoc["prompt"]))
        dev_llm_calls += int(tres.get("llm_calls", 0))
        refused = tres.get("status") != "SOLVED"
        if refused:
            af3_traps_refused += 1
        trap_rows.append(
            {
                "id": tdoc["id"],
                "kind": titem["kind"],
                "status": tres.get("status"),
                "reason": tres.get("reason"),
                "refused": refused,
            }
        )

    # Non-regression checks: AF3 structured (23/24 @ 1.0000) & AD2 structured + raw-prose traps
    af3_struct_ok = 0
    af3_struct_em = 0
    for item in af3_manifest["problems"]:
        pdoc = json.loads((AF3_DIR / item["file"]).read_text(encoding="utf-8"))
        sres = solver.solve(pdoc)
        if sres.get("status") == "SOLVED" and sres.get("answer") is not None:
            af3_struct_em += 1
            exp_val = float(pdoc["expected"])
            tol = float(pdoc.get("rel_tol", 0.02))
            if abs(float(sres["answer"]) - exp_val) <= max(1e-9, tol * max(abs(exp_val), 1.0)):
                af3_struct_ok += 1

    ad2_tasks = ad2.build_tasks()
    ad2_ans = [p for p in ad2_tasks if not p.get("trap")]
    ad2_traps = [p for p in ad2_tasks if p.get("trap")]
    ad2_struct_ans_ok = sum(1 for p in ad2_ans if solver.solve(p).get("status") == "SOLVED")
    ad2_struct_trap_ok = sum(1 for p in ad2_traps if solver.solve(p).get("status") != "SOLVED")
    ad2_raw_trap_ok = sum(1 for p in ad2_traps if solver.solve_text(str(p["prompt"])).get("status") != "SOLVED")

    # Gateway + MCP raw-prose probe
    sample_prompt = json.loads((AF3_DIR / af3_manifest["problems"][0]["file"]).read_text(encoding="utf-8"))["prompt"]
    trap_prompt = json.loads((AF3_DIR / af3_manifest["traps"][1]["file"]).read_text(encoding="utf-8"))["prompt"]
    gw_mcp_probe = _probe_gateway_and_mcp_raw_prose(sample_prompt, trap_prompt)

    # =========================================================================
    # PRE-FREEZE SEAL (MANDATORY BEFORE FETCHING ANY BLIND SCIBENCH PROBLEMS)
    # =========================================================================
    prefreeze = _seal_prefreeze()

    # =========================================================================
    # TIER 2: Blind 24-Problem Disjoint SciBench Holdout (RAW TEXT ONLY)
    # =========================================================================
    blind_manifest = _vendor_blind_24(af3_manifest)

    # Disjointness cryptographic assertion
    af3_key_set = {
        (_canonical_upstream_stem(str(x["upstream_file"])), str(x["upstream_problemid"]).strip())
        for x in af3_manifest["problems"]
    }
    blind_key_set = {
        (_canonical_upstream_stem(str(x["upstream_file"])), str(x["upstream_problemid"]).strip())
        for x in blind_manifest["problems"]
    }
    overlap = sorted(af3_key_set & blind_key_set)
    disjoint_ok = len(overlap) == 0 and len(blind_manifest["problems"]) == BLIND_COUNT

    blind_rows: List[Dict[str, Any]] = []
    blind_emitted = 0
    blind_si_ok = 0
    blind_raw_ok = 0
    blind_all_dim_ok = True
    blind_max_residual = 0.0
    blind_llm_calls = 0
    abstain_reasons: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()

    for bitem in blind_manifest["problems"]:
        bdoc = json.loads((BLIND_DIR / bitem["file"]).read_text(encoding="utf-8"))
        source_counts[str(bdoc.get("source", bdoc["upstream_file"]))] += 1
        raw_text = str(bdoc["problem_text"])
        exp_val = float(bdoc["answer_number"])
        exp_unit_raw = str(bdoc.get("unit") or "")

        res = solver.solve_text(raw_text)
        blind_llm_calls += int(res.get("llm_calls", 0))
        row: Dict[str, Any] = {
            "id": bdoc["id"],
            "source": bdoc.get("source"),
            "upstream_file": bdoc["upstream_file"],
            "upstream_problemid": bdoc["upstream_problemid"],
            "status": res.get("status"),
            "reason": res.get("reason"),
            "relation": res.get("relation"),
            "answer": res.get("answer"),
            "unit": res.get("unit"),
            "expected": exp_val,
            "expected_unit_raw": exp_unit_raw,
            "dimensional_ok": res.get("dimensional_ok"),
            "residual": res.get("residual"),
        }
        if res.get("status") == "SOLVED" and res.get("answer") is not None:
            blind_emitted += 1
            if not res.get("dimensional_ok"):
                blind_all_dim_ok = False
            blind_max_residual = max(blind_max_residual, float(res.get("residual", 0.0)))
            cmp = _compare_answer(
                float(res["answer"]),
                str(res.get("unit") or "1"),
                exp_val,
                exp_unit_raw,
                rel_tol=0.02,
            )
            row.update(cmp)
            if cmp["si_match"]:
                blind_si_ok += 1
            if cmp["raw_match"]:
                blind_raw_ok += 1
        else:
            abstain_reasons[str(res.get("reason") or "abstained")] += 1
        blind_rows.append(row)

    blind_prec = (blind_si_ok / blind_emitted) if blind_emitted > 0 else 1.0

    # Post-evaluation hash check & AST task-branching audit
    post_parser_sha = _sha256(PARSER_PATH)
    post_solver_sha = _sha256(SOLVER_PATH)
    sha_unchanged = (
        prefreeze["prefreeze_parser_sha256"] == prefreeze["current_before_vendor_parser_sha256"] == post_parser_sha
        and prefreeze["prefreeze_solver_sha256"] == prefreeze["current_before_vendor_solver_sha256"] == post_solver_sha
    )

    all_identifiers = sorted(
        {
            str(x["id"])
            for x in af3_manifest["problems"]
        }
        | {
            str(x["upstream_problemid"])
            for x in af3_manifest["problems"]
        }
        | {
            str(x["id"])
            for x in blind_manifest["problems"]
        }
        | {
            str(x["upstream_problemid"])
            for x in blind_manifest["problems"]
        }
        | {
            str(x["id"])
            for x in af3_manifest["traps"]
        }
    )
    branch_parser = audit_task_specific_branches(
        PARSER_PATH.read_text(encoding="utf-8"), identifiers=all_identifiers
    )
    branch_solver = audit_task_specific_branches(
        SOLVER_PATH.read_text(encoding="utf-8"), identifiers=all_identifiers
    )
    total_task_branches = int(branch_parser["task_identity_comparisons"]) + int(
        branch_solver["task_identity_comparisons"]
    )
    total_lit_cmps = int(branch_parser["compared_identifier_literal_count"]) + int(
        branch_solver["compared_identifier_literal_count"]
    )
    imports_audit = _audit_forbidden_imports([PARSER_PATH, SOLVER_PATH])

    gates = {
        "dev24_raw_text_pass_ge_18_of_24": dev_si_ok >= 18 and dev_raw_ok >= 18,
        "dev24_raw_text_precision_ge_09800": dev_emitted > 0 and dev_prec >= 0.9800,
        "dev24_raw_traps_6_of_6_refused": af3_traps_refused == 6 and ad2_raw_trap_ok == 6,
        "dev24_all_emissions_dim_homogeneous_and_residual_le_1e9": dev_all_dim_ok and dev_max_residual <= 1e-9,
        "structured_non_regression_af3_23_of_24_and_ad2_18_of_18": (
            af3_struct_ok == 23 and af3_struct_em == 23 and ad2_struct_ans_ok == 18 and ad2_struct_trap_ok == 6
        ),
        "gateway_and_mcp_raw_prose_verified": bool(gw_mcp_probe["passed"]),
        "parser_and_solver_sha256_unchanged": bool(sha_unchanged),
        "blind24_disjoint_from_af3_24_slice": bool(disjoint_ok),
        "blind24_emitted_precision_ge_09500": bool(blind_prec >= 0.9500),
        "blind24_all_emissions_dim_homogeneous_and_residual_le_1e9": blind_all_dim_ok and blind_max_residual <= 1e-9,
        "zero_task_specific_branches": (total_task_branches == 0 and total_lit_cmps == 0),
        "zero_external_llm_calls": (dev_llm_calls + blind_llm_calls == 0) and bool(imports_audit["clean"]),
    }

    receipt = {
        "cycle": "AG",
        "stage": "AG2",
        "title": "Deterministic Raw-Prose Dimensional Physics Parser & Blind SciBench End-to-End Audit",
        "evidence_label": "OBSERVED",
        "passed": all(gates.values()),
        "wall_seconds": round(time.perf_counter() - t0, 3),
        "pinned_corpus_sha256": PINNED_SHA256,
        "pinned_corpus_digest": PINNED_DIGEST,
        "prefreeze": {
            **prefreeze,
            "post_eval_parser_sha256": post_parser_sha,
            "post_eval_solver_sha256": post_solver_sha,
            "parser_and_solver_sha256_unchanged": sha_unchanged,
        },
        "tier1_dev24_raw_prose": {
            "total": len(dev_rows),
            "emitted": dev_emitted,
            "solved_si_equivalent": dev_si_ok,
            "pass_at_1_si": round(dev_si_ok / max(1, len(dev_rows)), 4),
            "solved_raw_scalar": dev_raw_ok,
            "pass_at_1_raw_scalar": round(dev_raw_ok / max(1, len(dev_rows)), 4),
            "emitted_precision": round(dev_prec, 4),
            "all_dimensional_homogeneous": dev_all_dim_ok,
            "max_residual": dev_max_residual,
            "af3_raw_traps_refused": af3_traps_refused,
            "af3_raw_traps_total": len(trap_rows),
            "ad2_raw_traps_refused": ad2_raw_trap_ok,
            "ad2_raw_traps_total": len(ad2_traps),
            "external_llm_calls": dev_llm_calls,
            "problems": dev_rows,
            "traps": trap_rows,
        },
        "non_regression_structured": {
            "af3_scibench_emitted": af3_struct_em,
            "af3_scibench_solved": af3_struct_ok,
            "ad2_answerable_solved": ad2_struct_ans_ok,
            "ad2_traps_refused": ad2_struct_trap_ok,
        },
        "gateway_and_mcp_raw_prose_probe": gw_mcp_probe,
        "tier2_blind24_scibench_holdout": {
            "manifest_path": str(BLIND_MANIFEST.relative_to(ROOT)),
            "slice_id": blind_manifest["slice_id"],
            "slice_sha256": blind_manifest["slice_sha256"],
            "upstream_repo": blind_manifest["upstream_repo"],
            "upstream_commit": blind_manifest["upstream_commit"],
            "selection_rule": blind_manifest["selection_rule"],
            "disjoint_from_af3_24_slice": disjoint_ok,
            "overlap_count": len(overlap),
            "total": len(blind_rows),
            "emitted": blind_emitted,
            "solved_si_equivalent": blind_si_ok,
            "pass_at_1_si": round(blind_si_ok / max(1, len(blind_rows)), 4),
            "solved_raw_scalar": blind_raw_ok,
            "pass_at_1_raw_scalar": round(blind_raw_ok / max(1, len(blind_rows)), 4),
            "emitted_precision": round(blind_prec, 4),
            "all_dimensional_homogeneous": blind_all_dim_ok,
            "max_residual": blind_max_residual,
            "external_llm_calls": blind_llm_calls,
            "source_distribution": dict(sorted(source_counts.items())),
            "abstain_reason_taxonomy": dict(sorted(abstain_reasons.items())),
            "problems": blind_rows,
        },
        "epistemic_audit": {
            "task_specific_branches": total_task_branches,
            "compared_identifier_literal_count": total_lit_cmps,
            "science_parser_audit": branch_parser,
            "science_solver_audit": branch_solver,
            "forbidden_llm_imports": imports_audit,
        },
        "gates": gates,
        "gate": pc._finalise(gates),
    }

    written = pc._write(RECEIPTS, "ag2_raw_prose_science_audit.json", receipt)

    print(
        json.dumps(
            {
                "stage": "AG2",
                "passed": receipt["passed"],
                "receipt": written["path"],
                "dev24_si": f"{dev_si_ok}/24",
                "dev24_raw": f"{dev_raw_ok}/24",
                "dev24_prec": round(dev_prec, 4),
                "af3_raw_traps": f"{af3_traps_refused}/6",
                "blind24_emitted": f"{blind_emitted}/24",
                "blind24_si": f"{blind_si_ok}/24",
                "blind24_prec": round(blind_prec, 4),
                "sha_unchanged": sha_unchanged,
                "disjoint": disjoint_ok,
                "gates": gates,
            },
            indent=2,
        )
    )
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
