"""Closed-loop code ranking with sandboxed execution verification.

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
It ranks candidate programs by wave resonance against a specification, then
runs the top candidates against real assertions in a child process.

It is NOT a code generator. It selects among candidates you supply. Absent
assertions it CANNOT verify, and it says so (`NO_TESTS_UNVERIFIABLE`)
instead of reporting a pass it never observed.

THE IMPORT TRAP, MEASURED
-------------------------
`scripts/` has no `__init__.py`, so `from scripts.mbpp_wave_eval import
run_candidate` raises `ModuleNotFoundError` at CALL time. A module-level
import would have failed loudly; a lazy one fails silently until first use.
Load by path with importlib, and cache the module.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]

_MODULE = None
_LOAD_ERROR: str | None = None


def load_mbpp_executor():
    """Load `scripts/mbpp_wave_eval.py` by path, cached.

    Returns the module, or None when the staged executor is unavailable.
    A missing executor is reported as an infrastructure limit, never
    silently converted into a failed candidate.
    """
    global _MODULE, _LOAD_ERROR
    if _MODULE is not None:
        return _MODULE
    if _LOAD_ERROR is not None:
        return None
    p = REPO / "scripts" / "mbpp_wave_eval.py"
    if not p.exists():
        _LOAD_ERROR = f"executor missing: {p}"
        return None
    try:
        spec = importlib.util.spec_from_file_location("henri_mbpp_eval", p)
        if spec is None or spec.loader is None:
            _LOAD_ERROR = "spec_from_file_location returned no loader"
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _MODULE = mod
        return mod
    except Exception as exc:                       # noqa: BLE001
        _LOAD_ERROR = f"{type(exc).__name__}: {exc}"
        return None


def executor_status() -> dict:
    mod = load_mbpp_executor()
    if mod is None:
        return {"available": False, "error": _LOAD_ERROR}
    return {
        "available": True,
        "module": str(Path(mod.__file__).name),
        "has_run_candidate": callable(getattr(mod, "run_candidate", None)),
        "exec_timeout": int(getattr(mod, "EXEC_TIMEOUT", 0)),
    }


def rank_candidates(backbone, spec: str, candidates: Sequence[str]) -> dict:
    """Score candidates by wave resonance against the spec.

    Ties are counted, not hidden. Phase 6 measured a tie plateau on MBPP
    (`mean tie-group 1.6333`), so a reader must see the group size to know
    whether `argmax` is ranking or arbitrating.
    """
    psi_spec = backbone.encode(spec).psi
    rows = []
    for i, cand in enumerate(candidates):
        psi_c = backbone.encode(cand).psi
        s = float(np.real(np.vdot(psi_spec, psi_c)))
        rows.append((s, i, cand))
    rows.sort(key=lambda t: (-t[0], t[1]))
    scores = np.array([r[0] for r in rows], dtype=float)
    mx = float(scores.max()) if scores.size else 0.0
    tie_group = int(np.isclose(scores, mx, atol=1e-12).sum()) if scores.size else 0
    return {
        "ranked": rows,
        "scores": scores,
        "top1_index": rows[0][1] if rows else None,
        "top1_score": mx,
        "tie_group_size": tie_group,
        "n": len(rows),
    }


def solve_verified_code(backbone, spec: str, candidates: Sequence[str],
                        tests: Sequence[str] = (),
                        top_k: int = 3,
                        timeout: int | None = None) -> dict:
    """Rank, then execute-verify the top candidates.

    Verdicts:
        EXEC_VERIFIED              a top-k candidate passed every assertion
        NO_CANDIDATES_PASS         every executed candidate failed
        NO_TESTS_UNVERIFIABLE      no assertions supplied; nothing was run
        NO_CANDIDATES              the pool was empty
        EXECUTOR_UNAVAILABLE       the staged executor could not load

    `NO_TESTS_UNVERIFIABLE` is the honesty gate. Without assertions there is
    no verification, and reporting a pass would be a fabricated outcome.
    """
    cands = [c for c in candidates if isinstance(c, str) and c.strip()]
    if not cands:
        return {"verdict": "NO_CANDIDATES", "n_candidates": 0,
                "n_tests": len(tests), "executed": [],
                "note": "empty candidate pool; nothing ranked or run"}

    rk = rank_candidates(backbone, spec, cands)
    base = {
        "spec_sha256": _sha(spec),
        "n_candidates": int(rk["n"]),
        "n_tests": len(tests),
        "wave_top1_index": rk["top1_index"],
        "wave_top1_score": round(float(rk["top1_score"]), 9),
        "wave_top1_text": rk["ranked"][0][2] if rk["ranked"] else None,
        "tie_group_size": rk["tie_group_size"],
        "ranking_well_posed": bool(rk["tie_group_size"] <= 1),
    }

    if not tests:
        return {**base, "verdict": "NO_TESTS_UNVERIFIABLE", "executed": [],
                "note": ("no assertions supplied, so no execution occurred. "
                         "This is not a pass and not a failure of the "
                         "candidates; it is an unverifiable request.")}

    mod = load_mbpp_executor()
    if mod is None:
        return {**base, "verdict": "EXECUTOR_UNAVAILABLE", "executed": [],
                "executor_error": _LOAD_ERROR,
                "note": "infrastructure limit, not a candidate verdict"}

    run = getattr(mod, "run_candidate", None)
    if not callable(run):
        return {**base, "verdict": "EXECUTOR_UNAVAILABLE", "executed": [],
                "executor_error": "run_candidate not callable"}

    tmo = int(timeout if timeout is not None else getattr(mod, "EXEC_TIMEOUT", 6))
    executed = []
    for pos, (score, idx, code) in enumerate(rk["ranked"][:max(1, int(top_k))]):
        res = run(code, list(tests), timeout=tmo)
        executed.append({
            "rank": pos + 1,
            "candidate_index": int(idx),
            "wave_score": round(float(score), 9),
            "ok": bool(res.get("ok", False)),
            "reason": str(res.get("reason", "unknown")),
        })

    passing = [e for e in executed if e["ok"]]
    verdict = "EXEC_VERIFIED" if passing else "NO_CANDIDATES_PASS"
    winner = None
    if passing:
        best = passing[0]
        winner = {"candidate_index": best["candidate_index"],
                  "wave_score": best["wave_score"]}
    return {**base, "verdict": verdict, "executed": executed,
            "n_executed": len(executed), "winner": winner,
            "exec_verified_top1": bool(passing and passing[0]["rank"] == 1),
            "note": ("execution-verified against supplied assertions"
                     if passing else
                     "every executed candidate failed its assertions")}


def _sha(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
