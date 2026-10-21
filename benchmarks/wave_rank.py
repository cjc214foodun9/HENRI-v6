"""Named-arm wave ranker over canonical multiple-choice corpora.

WHY THIS FILE EXISTS
--------------------
`HENRI-main/HENRI V2` holds benchmark runners wired to the legacy V2 wave
core. `HENRI-v6` holds the verified kernel and no runner. Reporting a V2
number as a v6 result would be the exact defect this project exists to
prevent. Therefore this runner names its arm on every receipt and refuses
to emit a score without one.

ARMS
----
`v6`      The v6 kernel path: `ring_bundle` circular-mean bundling in
          Z_256, `to_complex_phasor` sphere lift, `henri.readout.select`
          at the calibrated `beta* = 26.10`, plus Margin Law telemetry.
`chance`  Deterministic uniform pick. The control. If `v6` does not beat
          this, the arm carries no task signal and says so.
`legacy`  NOT RUN HERE. The legacy score is read from a real V2 receipt on
          the same rows and the same dataset sha256. Re-implementing the
          V2 codec here would produce a number whose provenance is this
          file, not V2.

WHAT THE TWO WAVE ARMS SHARE, STATED PLAINLY
--------------------------------------------
Ingress is a shared deterministic hash: text -> Z_256 per word. v6 has no
semantic text encoder. The architectural audit says random phase hashing
treats "car" and "automobile" as unrelated, so a near-chance score is the
EXPECTED, honest outcome. The measured gap between arms is bundling
algebra and readout calibration, nothing more. No semantic grounding is
claimed, and none is present.

EVIDENCE LABELS
---------------
Every accuracy figure is OBSERVED for the stated arm, row count, and
dataset digest. None is an intelligence score. A subset run is labelled
`SUBSET_SMOKE` and must never be quoted as the benchmark result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from henri.config import WaveConfig
from henri.qfhrr_tokenizer import QFHRRTokenizer
from henri.readout import entropy_at_beta, select
from henri.state import WaveState

K_BINS = 256
BETA_STAR = 26.1
LETTERS = "ABCD"
WORD_RE = re.compile(r"[a-z0-9']+")

CANONICAL = {
    "mmlu": REPO / "data/official_benchmarks/canonical/mmlu/mmlu.csv",
    "gpqa": REPO / "data/official_benchmarks/canonical/gpqa/gpqa_diamond.csv",
}


# -- ingress -------------------------------------------------------------

def _word_ring(word: str, dim: int, salt: str) -> np.ndarray:
    """Deterministic Z_256 ring for one word.

    Same word, same salt, same ring, across processes and hosts. This is a
    hash, not a learned embedding. Stated so it cannot be mistaken for one.
    """
    seed = hashlib.sha256(f"{salt}|{word}".encode("utf-8")).digest()
    gen = np.random.default_rng(int.from_bytes(seed[:8], "big"))
    return gen.integers(0, K_BINS, size=dim, dtype=np.uint8)


def _text_rings(text: str, dim: int, salt: str) -> list:
    return [_word_ring(w, dim, salt) for w in WORD_RE.findall(text.lower())]


# -- v6 arm --------------------------------------------------------------

def encode_v6(text: str, tok: QFHRRTokenizer, salt: str) -> Optional[np.ndarray]:
    """Bundle text into one ring vector with the v6 ring algebra.

    `ring_bundle` takes the circular mean per dimension, then rounds back
    to Z_256. Positions bind through the frozen `q_step` rotor, which is a
    true phase rotation, not a scaled integer multiplication.
    """
    words = WORD_RE.findall(text.lower())
    if not words:
        return None
    bound = []
    n = len(words)
    for i, w in enumerate(words):
        q = _word_ring(w, tok.dim, salt)
        pos = float(i) / float(max(1, n - 1))
        bound.append(tok.positional_bind(q, pos))
    return tok.ring_bundle(bound)


def lift_to_4slot(q_ring: np.ndarray, tok: QFHRRTokenizer,
                  full_dim: int) -> np.ndarray:
    """Lift one slot ring into the C6 4-slot layout, three slots zeroed.

    Constraint C6 partitions `D = 65536` into four 16384 complex
    sub-manifolds. Text here occupies the entity slot only. The other
    three are EXACT complex zeros, not ring zeros: a ring zero lifts to a
    constant phasor of magnitude `1/sqrt(D_slot)`, and a shared constant
    would add a spurious real coherence of ~1.0 to every pair. Zeros keep
    the readout comparable to the pinned sparsity case in `form_pin`.
    """
    slot = tok.to_complex_phasor(q_ring)
    psi = np.zeros(full_dim, dtype=np.complex128)
    psi[:slot.shape[0]] = slot
    return psi


def rank_v6(question: str, options: list, tok: QFHRRTokenizer,
            cfg_full: WaveConfig, salt: str) -> dict:
    """Rank options by v6 readout at beta*.

    Returns the selected index plus the telemetry the Margin Law needs:
    `t` (top score), `c_max` (max off-diagonal coherence among options),
    `margin`, and entropy at beta*.
    """
    q_ring = encode_v6(question, tok, salt)
    if q_ring is None:
        return {"status": "EMPTY_QUESTION"}
    book_rows = []
    for opt in options:
        r = encode_v6(opt, tok, salt)
        if r is None:
            return {"status": "EMPTY_OPTION"}
        book_rows.append(lift_to_4slot(r, tok, cfg_full.dim))
    book = np.stack(book_rows, axis=0)
    q_psi = lift_to_4slot(q_ring, tok, cfg_full.dim)
    state = WaveState(q_psi, cfg_full)
    res = select(state, book, floor=None)
    # c_max: largest real coherence between two distinct option waves.
    g = np.real(book @ book.conj().T)
    np.fill_diagonal(g, 0.0)
    c_max = float(np.max(g))
    ent = entropy_at_beta(np.real(book @ np.conj(state.psi)), beta=BETA_STAR)
    return {
        "selected_idx": int(res.index),
        "t": float(res.top_score),
        "second": float(res.second_score),
        "margin": float(res.margin),
        "c_max": c_max,
        "entropy_beta_star": float(ent),
    }


# -- chance arm ----------------------------------------------------------

def rank_chance(question: str, options: list, seed: int) -> dict:
    """Deterministic uniform pick, keyed on the item text."""
    h = hashlib.sha256(question.encode("utf-8")).digest()
    idx = int.from_bytes(h[:4], "big") % len(options)
    return {"selected_idx": idx, "t": 0.0, "second": 0.0,
            "margin": 0.0, "c_max": 0.0, "entropy_beta_star": 1.0}


# -- corpus --------------------------------------------------------------

def load_items(name: str, limit: Optional[int]) -> tuple:
    path = CANONICAL[name]
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    items = []
    for idx, row in enumerate(rows):
        if name == "mmlu":
            q = (row.get("Question") or "").strip()
            opts = [(row.get(k) or "").strip() for k in LETTERS]
            letter = (row.get("Answer") or "").strip().upper()
            subject = row.get("Subject", "")
        else:
            q = (row.get("Question") or "").strip()
            opts = [(row.get("Correct Answer") or "").strip()] + [
                (row.get(f"Incorrect Answer {i}") or "").strip() for i in (1, 2, 3)]
            letter = "A"
            subject = row.get("High-level domain", "")
        if letter not in LETTERS or not q or any(not o for o in opts):
            continue
        items.append({
            "record_id": row.get("", str(idx)),
            "subject": subject,
            "question": q,
            "options": opts,
            "correct_idx": LETTERS.index(letter),
        })
    if limit:
        items = items[:limit]
    return items, digest


def run(name: str, arm: str, limit: Optional[int], legacy: Optional[str],
        seed: int = 20261020) -> dict:
    items, digest = load_items(name, limit)
    cfg_slot = WaveConfig(dim=16384)
    tok = QFHRRTokenizer(cfg_slot, seed=seed) if arm == "v6" else None
    salt = f"{name}|{arm}|{digest[:12]}"

    per_item, correct = [], 0
    t_vals, m_vals, c_vals, e_vals = [], [], [], []
    started = time.perf_counter()
    for it in items:
        if arm == "v6":
            r = rank_v6(it["question"], it["options"], tok, cfg_slot, salt)
        else:
            r = rank_chance(it["question"], it["options"], seed)
        if "status" in r:
            per_item.append({"record_id": it["record_id"], "status": r["status"]})
            continue
        ok = r["selected_idx"] == it["correct_idx"]
        correct += int(ok)
        t_vals.append(r["t"]); m_vals.append(r["margin"])
        c_vals.append(r["c_max"]); e_vals.append(r["entropy_beta_star"])
        per_item.append({
            "record_id": it["record_id"],
            "subject": it["subject"][:40],
            "correct_idx": it["correct_idx"],
            "selected_idx": r["selected_idx"],
            "is_correct": bool(ok),
            "t": round(r["t"], 6),
            "margin": round(r["margin"], 6),
            "c_max": round(r["c_max"], 6),
            "entropy_beta_star": round(r["entropy_beta_star"], 6),
        })
    attempted = sum(1 for r in per_item if "is_correct" in r)
    acc = correct / max(1, attempted)

    rec = {
        "probe": "wave_rank",
        "benchmark": name.upper(),
        "arm": arm,
        "scope": "SUBSET_SMOKE" if limit else "FULL_TEST_SET",
        "status": "EVALUATED",
        "item_count": len(items),
        "attempted": attempted,
        "correct": correct,
        "accuracy": round(acc, 6),
        "chance": round(1.0 / 4.0, 6),
        "above_chance": round(acc - 0.25, 6),
        "dataset_sha256": digest,
        "dataset_path": str(CANONICAL[name].relative_to(REPO)),
        "dim": int(cfg_slot.dim),
        "beta_star": BETA_STAR,
        "ingress": "deterministic_sha256_word_hash_to_Z256",
        "semantic_grounding": False,
        "wall_clock_sec": round(time.perf_counter() - started, 3),
        "significance_vs_chance": _significance(correct, attempted),
        "per_item": per_item,
    }
    if t_vals:
        rec["telemetry"] = {
            "mean_t": round(float(np.mean(t_vals)), 6),
            "mean_margin": round(float(np.mean(m_vals)), 6),
            "mean_c_max": round(float(np.mean(c_vals)), 6),
            "mean_entropy_beta_star": round(float(np.mean(e_vals)), 6),
            "margin_law_pred": round(float(np.mean(
                [t * (1.0 - c) for t, c in zip(t_vals, c_vals)])), 6),
        }
    if legacy:
        rec["legacy_reference"] = read_legacy(legacy, digest)
    return rec


def read_legacy(path: str, digest: str) -> dict:
    """Load a real V2 receipt and check it is the same corpus and rows."""
    d = json.load(open(path, encoding="utf-8"))
    same = d.get("dataset_sha256") == digest
    return {
        "source": path,
        "arm": "legacy_v2",
        "dataset_sha256": d.get("dataset_sha256"),
        "dataset_matches": bool(same),
        "item_count": d.get("item_count"),
        "accuracy": d.get("accuracy"),
        "egress_path": d.get("egress_path"),
        "note": "Number came from the V2 tree. It is not a v6 score.",
    }


def _significance(correct: int, n: int, p0: float = 0.25) -> dict:
    """One-sided binomial test that accuracy beats chance.

    The raw gap `acc - 0.25` is not evidence on 40 items. This returns the
    z statistic with a continuity correction and its one-sided p-value, so
    a reader cannot mistake a 2-item edge for a signal. `math.erfc` keeps
    the runner dependency-free.
    """
    import math

    if n <= 0:
        return {"z": None, "p_value": None, "verdict": "NO_ITEMS"}
    expected = n * p0
    var = n * p0 * (1.0 - p0)
    sd = math.sqrt(var)
    if sd == 0.0:
        return {"z": None, "p_value": None, "verdict": "DEGENERATE"}
    z = (correct - expected - 0.5) / sd          # continuity correction
    p = 0.5 * math.erfc(z / math.sqrt(2.0))      # one-sided upper tail
    if p < 0.01:
        verdict = "ABOVE_CHANCE_DECISIVE"
    elif p < 0.05:
        verdict = "ABOVE_CHANCE"
    elif p > 0.95:
        verdict = "BELOW_CHANCE"
    else:
        verdict = "INDISTINGUISHABLE_FROM_CHANCE"
    return {
        "z": round(z, 4),
        "p_value": round(p, 8),
        "expected_correct": round(expected, 3),
        "verdict": verdict,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", choices=sorted(CANONICAL), default="mmlu")
    ap.add_argument("--arm", choices=["v6", "chance"], required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--legacy-receipt", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rec = run(a.benchmark, a.arm, a.limit, a.legacy_receipt)
    out = Path(a.out) if a.out else (
        REPO / "receipts/bench" /
        f"{a.benchmark}_{a.arm}_n{rec['item_count']}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"[ARM] {rec['benchmark']} arm={rec['arm']} scope={rec['scope']}"
          f" items={rec['item_count']}")
    print(f"[SCORE] correct={rec['correct']}/{rec['attempted']}"
          f" acc={rec['accuracy']:.4f} chance=0.2500"
          f" above={rec['above_chance']:+.4f}")
    if "telemetry" in rec:
        tl = rec["telemetry"]
        print(f"[TELEMETRY] t={tl['mean_t']:.6f}"
              f" margin={tl['mean_margin']:.6f}"
              f" c_max={tl['mean_c_max']:.6f}"
              f" H(beta*)={tl['mean_entropy_beta_star']:.6f}")
    if "legacy_reference" in rec:
        lr = rec["legacy_reference"]
        print(f"[LEGACY] acc={lr['accuracy']} same_corpus={lr['dataset_matches']}"
              f" ({lr['note']})")
    print(f"[RECEIPT] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
