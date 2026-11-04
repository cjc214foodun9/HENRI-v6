"""HENRI v6 command line: the eight tools the Harness plugin calls.

Evidence label: OBSERVED for every verdict this module prints.

The contract with the plugin is one JSON document on stdout:

    {ok, exitCode, summary, result}

The plugin appends --json to every call, so the JSON path is the one that
matters. exitCode 1 means a gate failed and result names the cause.

Commands
--------
    doctor      environment and evidence readiness
    ste100      ASD-STE100 linter over a file or a string
    jev         evidence integrity gate over receipts and the ontology
    holon       run the six-holon scientific graph over a goal
    gates       the four promotion gates
    ledger      run history, leaderboard, integrity
    zone-check  may this path be written under this zone
    vast        GPU executor, dry run unless explicitly allowed
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from henri.ontology import OntologyLedger, sha256_file

REPO_ROOT = Path(__file__).resolve().parents[1]
ONTOLOGY_PATH = REPO_ROOT / "data" / "holarch" / "ontology.jsonl"
LEDGER_PATH = REPO_ROOT / "data" / "holarch" / "ledger.sqlite3"
RECEIPTS = REPO_ROOT / "receipts" / "holarch"


def envelope(ok: bool, summary: str, result: Dict, code: Optional[int] = None) -> Dict:
    """The single shape every tool returns."""
    return {
        "ok": bool(ok),
        "exitCode": int(code if code is not None else (0 if ok else 1)),
        "summary": str(summary),
        "result": result,
    }


def emit(payload: Dict, as_json: bool, title: str = "") -> None:
    if as_json:
        print(json.dumps(payload, indent=2, default=str))
        return
    if title:
        print(title)
    print(f"  {'status':28s} {payload['summary']}")
    for key, value in payload["result"].items():
        if isinstance(value, (dict, list)):
            print(f"  {key:28s} {json.dumps(value, default=str)[:160]}")
        else:
            print(f"  {key:28s} {value}")


# --------------------------------------------------------------------- doctor
def cmd_doctor(args) -> int:
    checks: Dict[str, Dict] = {}

    def probe(name: str, module: str) -> None:
        try:
            imported = __import__(module)
            checks[name] = {"available": True, "version": getattr(imported, "__version__", "unknown")}
        except Exception as exc:  # pragma: no cover - environment dependent
            checks[name] = {"available": False, "error": type(exc).__name__}

    probe("numpy", "numpy")
    probe("torch", "torch")
    probe("sklearn", "sklearn")
    cuda = False
    device = "cpu"
    if checks.get("torch", {}).get("available"):
        try:
            import torch

            cuda = bool(torch.cuda.is_available())
            device = "cuda" if cuda else "cpu"
        except Exception:
            cuda = False
    receipts = sorted(RECEIPTS.glob("*.json")) if RECEIPTS.exists() else []
    sidecars = sorted(RECEIPTS.glob("*.sha256")) if RECEIPTS.exists() else []
    ontology = OntologyLedger(ONTOLOGY_PATH)
    chain = ontology.verify()
    ledger_ok = LEDGER_PATH.exists()
    checks["receipts"] = {"count": len(receipts), "sidecars": len(sidecars)}
    checks["ontology"] = chain
    checks["ledger"] = {"present": ledger_ok}
    checks["repo"] = {"root": str(REPO_ROOT), "python": platform.python_version(), "device": device}
    ok = bool(checks["numpy"]["available"]) and len(receipts) > 0 and bool(chain.get("ok"))
    summary = "READY" if ok else "DEGRADED"
    payload = envelope(ok, summary, {"checks": checks, "cuda": cuda, "device": device})
    emit(payload, args.json, "HENRI environment")
    return payload["exitCode"]


# --------------------------------------------------------------------- ste100
STE_RULES: Sequence[Dict[str, str]] = (
    {"id": "semicolon", "pattern": r";", "hint": "Split the sentence. STE100 allows one instruction per sentence."},
    {"id": "perfect_tense", "pattern": r"\b(has|have|had)\s+been\b", "hint": "Use the simple past or the simple present."},
    {"id": "passive", "pattern": r"\b(is|are|was|were|be|been|being)\s+([a-z]+ed|shown|made|done|given|taken|known|built|set)\b", "hint": "Name the actor and use the active voice."},
    {"id": "banned_word", "pattern": r"\b(utiliz|utilis|leverag|facilitat|commenc|terminat|endeavour|endeavor)\w*\b", "hint": "Use the approved word: use, start, stop, try."},
    {"id": "wordy_phrase", "pattern": r"\b(prior to|in order to|at this point in time|due to the fact that)\b", "hint": "Use before, to, now, because."},
    {"id": "contraction", "pattern": r"\b\w+'(s|t|re|ve|ll|d)\b", "hint": "Write the two words in full."},
    {"id": "latin_abbreviation", "pattern": r"\b(e\.g\.|i\.e\.|etc\.)", "hint": "Write for example, that is, and so on."},
)


def lint_text(text: str, mode: str = "procedural") -> Dict:
    violations: List[Dict] = []
    for number, line in enumerate(text.splitlines(), start=1):
        for rule in STE_RULES:
            for match in re.finditer(rule["pattern"], line, re.IGNORECASE):
                violations.append(
                    {
                        "rule": rule["id"],
                        "line": number,
                        "column": int(match.start()) + 1,
                        "text": match.group(0),
                        "hint": rule["hint"],
                    }
                )
        words = len(re.findall(r"[A-Za-z0-9'-]+", line))
        if words > 25:
            violations.append(
                {
                    "rule": "long_sentence",
                    "line": number,
                    "column": 1,
                    "text": f"{words} words",
                    "hint": "Keep a procedural sentence at or below 20 words, descriptive at or below 25.",
                }
            )
    counts: Dict[str, int] = {}
    for violation in violations:
        counts[violation["rule"]] = counts.get(violation["rule"], 0) + 1
    return {"mode": mode, "violations": violations, "counts": counts, "clean": not violations}


def cmd_ste100(args) -> int:
    if args.file:
        path = Path(args.file)
        if not path.exists():
            payload = envelope(False, f"file not found: {path}", {"file": str(path)})
            emit(payload, args.json, "STE100")
            return payload["exitCode"]
        text = path.read_text(encoding="utf-8", errors="ignore")
        source = str(path)
    else:
        text = str(args.text or "")
        source = "inline"
    report = lint_text(text, str(args.mode))
    summary = "CLEAN" if report["clean"] else f"{len(report['violations'])} violations"
    payload = envelope(report["clean"], summary, {"source": source, **report})
    emit(payload, args.json, "STE100 lint")
    return payload["exitCode"]


# ----------------------------------------------------------------------- jev
def cmd_jev(args) -> int:
    ontology = OntologyLedger(ONTOLOGY_PATH)
    chain = ontology.verify()
    rows: List[Dict] = []
    for sidecar in sorted(RECEIPTS.glob("*.sha256")) if RECEIPTS.exists() else []:
        receipt = sidecar.parent / sidecar.name[: -len(".sha256")]
        recorded = sidecar.read_text(encoding="utf-8").strip()
        if not receipt.exists():
            rows.append({"receipt": receipt.name, "ok": False, "reason": "missing file"})
            continue
        actual = sha256_file(receipt)
        rows.append({"receipt": receipt.name, "ok": actual == recorded, "digest": actual})
    matched = sum(1 for row in rows if row["ok"])
    ok = bool(rows) and matched == len(rows) and bool(chain.get("ok"))
    summary = "EVIDENCE HOLDS" if ok else "EVIDENCE BROKEN"
    payload = envelope(
        ok,
        summary,
        {"receipts": len(rows), "matched": matched, "rows": rows, "ontology": chain},
    )
    emit(payload, args.json, "JEV evidence gate")
    return payload["exitCode"]


# --------------------------------------------------------------------- holon
def cmd_holon(args) -> int:
    from henri.orchestrator import Orchestrator

    started = time.perf_counter()
    orchestrator = Orchestrator(
        root=REPO_ROOT, db_path=LEDGER_PATH, ontology_path=ONTOLOGY_PATH, timeout=int(args.budget)
    )
    report = orchestrator.run(str(args.problem), profile=str(args.profile))
    report["seconds_total"] = round(time.perf_counter() - started, 3)
    ok = report["status"] == "PROMOTE"
    payload = envelope(
        ok,
        f"{report['status']} after {len(report['holons'])} holons",
        report,
    )
    emit(payload, args.json, "six-holon run")
    return payload["exitCode"]


# --------------------------------------------------------------------- gates
def cmd_gates(args) -> int:
    from henri.holon import HolonGraph

    graph = HolonGraph(root=REPO_ROOT, profile=str(args.profile), timeout=int(args.budget), ontology_path=ONTOLOGY_PATH)
    context: Dict = {}
    measure = graph.measure(context)
    falsify = graph.falsify(context)
    chain = graph.ontology.verify()
    gates = {
        "evidence_integrity": bool(context.get("measure", {}).get("receipt_integrity", {}).get("ok")),
        "documentation_coverage": bool(context.get("measure", {}).get("documentation", {}).get("ok")),
        "runtime": int(context.get("measure", {}).get("runtime", {}).get("exit", 1)) == 0,
        "falsifier_rejects_corruption": falsify.status == "PASS",
        "ontology_chain": bool(chain.get("ok")),
    }
    ok = all(gates.values())
    payload = envelope(
        ok,
        "four gates pass" if ok else "gate failure",
        {"gates": gates, "measure": measure.as_dict(), "falsify": falsify.as_dict(), "ontology": chain},
    )
    emit(payload, args.json, "promotion gates")
    return payload["exitCode"]


# -------------------------------------------------------------------- ledger
def cmd_ledger(args) -> int:
    from henri.orchestrator import Orchestrator

    orchestrator = Orchestrator(root=REPO_ROOT, db_path=LEDGER_PATH, ontology_path=ONTOLOGY_PATH)
    action = str(args.action or "list")
    if action == "leaderboard":
        rows = orchestrator.leaderboard()
        payload = envelope(True, f"{len(rows)} plan/status groups", {"rows": rows})
    elif action == "verify":
        chain = OntologyLedger(ONTOLOGY_PATH).verify()
        payload = envelope(bool(chain.get("ok")), "chain verified" if chain.get("ok") else "chain broken", chain)
    elif action == "replay":
        rows = orchestrator.history(limit=int(args.limit))
        payload = envelope(True, f"replayed {len(rows)} runs into the ontology view", {"rows": rows})
    else:
        rows = orchestrator.history(limit=int(args.limit))
        payload = envelope(True, f"{len(rows)} runs", {"rows": rows})
    emit(payload, args.json, "ledger")
    return payload["exitCode"]


# ---------------------------------------------------------------- zone-check
def cmd_zone_check(args) -> int:
    path = Path(str(args.path)).as_posix()
    zone = int(args.zone)
    reasons: List[str] = []
    allowed = True
    if zone == 1:
        if not (RECEIPTS.exists() and any(RECEIPTS.glob("*.json"))):
            allowed = False
            reasons.append("zone 1 claim has no receipt behind it")
        else:
            reasons.append("zone 1 claim is backed by receipts/holarch")
    elif zone == 2:
        module = Path(path).stem
        hits = list((REPO_ROOT / "tests").glob("*.py")) if (REPO_ROOT / "tests").exists() else []
        named = [hit.name for hit in hits if module in hit.read_text(encoding="utf-8", errors="ignore")]
        if not named:
            allowed = False
            reasons.append(f"zone 2 module {module} has no test that names it")
        else:
            reasons.append(f"zone 2 module {module} is covered by {', '.join(named[:3])}")
    elif zone == 3:
        reasons.append("zone 3 is the api and contract zone; tests are required to pass")
    else:
        allowed = False
        reasons.append("zone must be 1, 2 or 3")
    if path.endswith(".json") and "receipts" in path:
        companion = Path(str(args.path) + ".sha256")
        if not companion.exists():
            allowed = False
            reasons.append("a receipt needs a .sha256 sidecar")
    payload = envelope(allowed, "ALLOW" if allowed else "DENY", {"path": path, "zone": zone, "reasons": reasons})
    emit(payload, args.json, "zone contract")
    return payload["exitCode"]


# ---------------------------------------------------------------------- vast
def cmd_vast(args) -> int:
    try:
        import vastai  # noqa: F401

        installed = True
    except Exception:
        installed = False
    allow = bool(args.allow_live) or os.environ.get("HENRI_ALLOW_LIVE", "").lower() in {"1", "true", "yes"}
    live_requested = bool(args.live)
    if live_requested and not (installed and allow):
        payload = envelope(
            False,
            "BLOCKED: live rental needs vastai and HENRI_ALLOW_LIVE",
            {"installed": installed, "allow_live": allow, "live": True, "rented": False},
        )
        emit(payload, args.json, "vast")
        return payload["exitCode"]
    payload = envelope(
        True,
        "dry run: nothing rented",
        {"installed": installed, "allow_live": allow, "live": live_requested, "rented": False},
    )
    emit(payload, args.json, "vast")
    return payload["exitCode"]


# ---------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m henri", description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", help="emit one JSON document")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", parents=[common], help="environment and evidence readiness")
    doctor.set_defaults(func=cmd_doctor)

    ste = sub.add_parser("ste100", parents=[common], help="lint text against ASD-STE100 rules")
    ste.add_argument("--file", default=None)
    ste.add_argument("--text", default="")
    ste.add_argument("--mode", default="procedural", choices=["procedural", "descriptive", "technical"])
    ste.set_defaults(func=cmd_ste100)

    jev = sub.add_parser("jev", parents=[common], help="evidence integrity gate")
    jev.set_defaults(func=cmd_jev)

    holon = sub.add_parser("holon", parents=[common], help="run the six-holon graph")
    holon.add_argument("--problem", required=True)
    holon.add_argument("--profile", default="quick", choices=["quick", "full"])
    holon.add_argument("--budget", type=int, default=120, help="per-command timeout in seconds")
    holon.set_defaults(func=cmd_holon)

    gates = sub.add_parser("gates", parents=[common], help="the four promotion gates")
    gates.add_argument("--profile", default="quick", choices=["quick", "full"])
    gates.add_argument("--budget", type=int, default=120)
    gates.set_defaults(func=cmd_gates)

    ledger = sub.add_parser("ledger", parents=[common], help="run history and integrity")
    ledger.add_argument("action", nargs="?", default="list", choices=["list", "leaderboard", "verify", "replay"])
    ledger.add_argument("--limit", type=int, default=10)
    ledger.set_defaults(func=cmd_ledger)

    zone = sub.add_parser("zone-check", parents=[common], help="zone write contract")
    zone.add_argument("--path", required=True)
    zone.add_argument("--zone", type=int, default=3, choices=[1, 2, 3])
    zone.set_defaults(func=cmd_zone_check)

    vast = sub.add_parser("vast", parents=[common], help="GPU executor, dry run by default")
    vast.add_argument("--live", action="store_true")
    vast.add_argument("--allow-live", action="store_true")
    vast.set_defaults(func=cmd_vast)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
