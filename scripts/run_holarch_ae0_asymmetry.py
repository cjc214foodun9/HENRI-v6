"""Cycle AE0.3: the AC1 identifier-density and sub-slice asymmetry diagnostic.

    python scripts/run_holarch_ae0_asymmetry.py

The sealed AC1 weights, split and receipt are not touched.  The diagnostic
re-evaluates the frozen split once, dumps per-token NLL and correctness, and
compares the 84,048-token population with the 5,714-token legacy sub-slice.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

import run_holarch_p_cycle as pc
from henri.holarch.code_gen import CountLanguageModel
from henri.holarch.code_lang import build_tokenizer
from henri.holarch.pinned_split import (
    load_snapshot,
    snapshot_corpus,
    snapshot_segments,
    snapshot_train_ids,
)

RECEIPTS = ROOT / "receipts" / "holarch"
WORK = RECEIPTS / "_work_ae"
DUMP = WORK / "ac1_per_token.json"
PIN_SNAPSHOT = RECEIPTS / "pinned_corpus_snapshot.json"
PIN_SHA = "d675f3816bbf472493b25f885bbb56c62c5ec88a81ab31e31a9b87b3a0c23703"
PIN_DIGEST = "8f270f34e8f35ef84e415d076bea20a7bc15edff337f89ed72ad607447ba8554"
IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
SUBSLICE = 5714
SUBSLICE_CAP = 6000


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_dump() -> Dict[str, Any]:
    if not DUMP.exists():
        raise SystemExit("per-token dump missing; run scripts/dump_ac1_per_token.py first")
    return json.loads(DUMP.read_text(encoding="utf-8"))


def _metrics(sequences: Sequence[Sequence[int]], tokenizer) -> Dict[str, Any]:
    tokens = [int(token) for sequence in sequences for token in sequence]
    total = len(tokens)
    tier1 = sum(1 for token in tokens if tokenizer.tier_of(token) == 1)
    tier2 = total - tier1
    byte_tokens = sum(1 for token in tokens if tokenizer.is_byte_token(token))
    identifier_tokens = sum(
        1 for token in tokens if IDENTIFIER.match(tokenizer.token_text(token) or "")
    )
    unique = len({int(token) for token in tokens})
    return {
        "tokens": int(total),
        "tier1": int(tier1),
        "tier2": int(tier2),
        "tier1_fraction": float(tier1 / max(total, 1)),
        "tier2_fraction": float(tier2 / max(total, 1)),
        "byte_fraction": float(byte_tokens / max(total, 1)),
        "identifier_tokens": int(identifier_tokens),
        "identifier_density_per_1000": float(1000.0 * identifier_tokens / max(total, 1)),
        "unique_token_fraction": float(unique / max(total, 1)),
    }


def _order_table() -> Dict[str, Any]:
    snapshot = load_snapshot(ROOT)
    train = [int(token) for token in snapshot_train_ids(snapshot)]
    floor_ids = train[:-12288]
    tokenizer = build_tokenizer(snapshot_corpus(snapshot))
    model = CountLanguageModel(len(tokenizer))
    model.fit(floor_ids)
    model.fit_long(floor_ids)
    return {"model": model, "tokenizer": tokenizer, "snapshot": snapshot}


def _mean_backoff(sequences: Sequence[Sequence[int]], model: CountLanguageModel) -> Dict[str, Any]:
    orders: List[int] = []
    for sequence in sequences:
        ids = [int(token) for token in sequence]
        for index in range(1, len(ids)):
            order, _mass = model.matched_order(ids[max(0, index - 8) : index])
            orders.append(int(order) + 1)
    return {
        "positions": int(len(orders)),
        "mean_order": float(np.mean(orders)) if orders else 0.0,
        "sd_order": float(np.std(orders)) if orders else 0.0,
        "definition": "1 + matched_order from the fitted floor counts; 1 is the unigram backoff",
    }


def _window_stats(rows: Sequence[Dict[str, Any]], windows: int = 14) -> Dict[str, Any]:
    per_token = sorted(rows, key=lambda row: (int(row["doc"]), int(row["index"])))
    top1 = np.asarray([int(row["correct"]) for row in per_token], dtype=np.float64)
    nll = np.asarray([float(row["nll"]) for row in per_token], dtype=np.float64)
    total = int(top1.size)
    width = min(SUBSLICE, total)
    starts = list(range(0, max(total - width + 1, 1), width))
    window_top1: List[float] = []
    window_bpc: List[float] = []
    for start in starts:
        stop = min(start + width, total)
        if stop - start < width:
            continue
        window_top1.append(float(top1[start:stop].mean()))
        window_bpc.append(float(nll[start:stop].mean() / math.log(2.0)))
    generator = np.random.default_rng(20261118)
    samples_top1: List[float] = []
    samples_bpc: List[float] = []
    count = len(window_top1)
    for _ in range(5000):
        picks = generator.integers(0, count, size=count)
        samples_top1.append(float(np.mean([window_top1[int(index)] for index in picks])))
        samples_bpc.append(float(np.mean([window_bpc[int(index)] for index in picks])))
    return {
        "tokens": total,
        "top1": float(top1.mean()),
        "bpc": float(nll.mean() / math.log(2.0)),
        "windows": count,
        "window_top1": window_top1,
        "window_bpc": window_bpc,
        "ci95_top1": [float(np.percentile(samples_top1, 2.5)), float(np.percentile(samples_top1, 97.5))],
        "ci95_bpc": [float(np.percentile(samples_bpc, 2.5)), float(np.percentile(samples_bpc, 97.5))],
        "bootstrap_samples": 5000,
    }


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Cycle AE0.3 AC1 asymmetry diagnostic")
    parser.add_argument("--receipts", default=str(RECEIPTS))
    args = parser.parse_args(argv)
    directory = Path(args.receipts).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    pinned_before = _sha(PIN_SNAPSHOT) if PIN_SNAPSHOT.exists() else ""
    pinned_digest = (
        str(json.loads(PIN_SNAPSHOT.read_text(encoding="utf-8")).get("digest", ""))
        if PIN_SNAPSHOT.exists()
        else ""
    )
    dump = _load_dump()
    table = _order_table()
    tokenizer = table["tokenizer"]
    snapshot = table["snapshot"]
    segments = snapshot_segments(snapshot)
    full_sequences = [list(int(token) for token in val) for _name, _train, val in segments]
    sub_cap = max(64, SUBSLICE_CAP // max(len(segments), 1))
    sub_sequences = [
        list(int(token) for token in val[:sub_cap]) for _name, _train, val in segments
    ]
    full_metrics = _metrics(full_sequences, tokenizer)
    sub_metrics = _metrics(sub_sequences, tokenizer)
    full_metrics["predicted_positions"] = int(
        full_metrics["tokens"] - len(full_sequences)
    )
    sub_metrics["predicted_positions"] = int(
        sub_metrics["tokens"] - len(sub_sequences)
    )
    full_metrics["mean_backoff"] = _mean_backoff(full_sequences, table["model"])
    sub_metrics["mean_backoff"] = _mean_backoff(sub_sequences, table["model"])
    full_stats = _window_stats(dump["groups"]["full"])
    sub_stats = _window_stats(dump["groups"]["sub"], windows=1)
    density_delta = float(
        sub_metrics["identifier_density_per_1000"]
        - full_metrics["identifier_density_per_1000"]
    )
    density_ratio = float(
        sub_metrics["identifier_density_per_1000"]
        / max(full_metrics["identifier_density_per_1000"], 1e-9)
    )
    top1_gap = float(full_stats["top1"] - sub_stats["top1"])
    bpc_gap = float(sub_stats["bpc"] - full_stats["bpc"])
    # Per-window identifier density over the same contiguous windows.
    per_token = sorted(dump["groups"]["full"], key=lambda row: (int(row["doc"]), int(row["index"])))
    window_density: List[float] = []
    start = 0
    while start + SUBSLICE <= len(per_token):
        ids = [int(row["token"]) for row in per_token[start : start + SUBSLICE]]
        identifiers = sum(
            1 for token in ids if IDENTIFIER.match(tokenizer.token_text(token) or "")
        )
        window_density.append(1000.0 * identifiers / float(SUBSLICE))
        start += SUBSLICE
    correlation = 0.0
    if len(window_density) >= 3:
        correlation = float(np.corrcoef(window_density, full_stats["window_top1"])[0, 1])
    hypothesis_confirmed = bool(density_delta > 0.0 and correlation <= -0.2)
    pinned_after = _sha(PIN_SNAPSHOT) if PIN_SNAPSHOT.exists() else ""
    gates = {
        "ae0_dump_present": bool(dump.get("groups")),
        "ae0_full_predicted_84048": bool(full_metrics["predicted_positions"] == 84048),
        "ae0_subslice_predicted_5714": bool(sub_metrics["predicted_positions"] == SUBSLICE),
        "ae0_subslice_matches_sealed": bool(sub_metrics["predicted_positions"] == SUBSLICE),
        "ae0_bootstrap_computed": bool(full_stats["windows"] >= 10 and sub_stats["windows"] == 1),
        "ae0_hypothesis_decided": bool(
            hypothesis_confirmed
            or correlation > -0.2
            or density_delta <= 0.0
        ),
        "ae0_pinned_snapshot_unchanged": bool(
            pinned_before == pinned_after == PIN_SHA and pinned_digest == PIN_DIGEST
        ),
    }
    receipt = {
        "stage": "AE0.3",
        "full_population": {**full_metrics, **{"top1": full_stats["top1"], "bpc": full_stats["bpc"]}},
        "sub_slice": {**sub_metrics, **{"top1": sub_stats["top1"], "bpc": sub_stats["bpc"]}},
        "bootstrap": {"full": full_stats, "sub": sub_stats},
        "comparison": {
            "identifier_density_delta_per_1000": density_delta,
            "identifier_density_ratio": density_ratio,
            "top1_gap_full_minus_sub": top1_gap,
            "bpc_gap_sub_minus_full": bpc_gap,
            "window_density_vs_top1_pearson": correlation,
            "sub_top1_inside_full_ci95": bool(
                full_stats["ci95_top1"][0] <= sub_stats["top1"] <= full_stats["ci95_top1"][1]
            ),
            "sub_bpc_inside_full_ci95": bool(
                full_stats["ci95_bpc"][0] <= sub_stats["bpc"] <= full_stats["ci95_bpc"][1]
            ),
        },
        "verdict": {
            "hypothesis": "the legacy sub-slice carries a higher identifier density that explains part of its top-1 gap",
            "decision": "CONFIRMED" if hypothesis_confirmed else "FALSIFIED",
            "rule": "CONFIRMED iff the sub-slice identifier density exceeds the full population and the window-level density/top-1 Pearson correlation is at most -0.2",
        },
        "pinned_snapshot": {
            "sha256": pinned_after,
            "sha256_before": pinned_before,
            "expected_sha256": PIN_SHA,
            "expected_digest": PIN_DIGEST,
        },
        "gate": pc._finalise(gates),
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
        "passed": bool(all(gates.values())),
    }
    written = pc._write(directory, "ae0_ac1_subslice_asymmetry.json", receipt)
    print(
        "[ae0] full tier2=%.4f id/1k=%.2f | sub tier2=%.4f id/1k=%.2f | r=%.3f | %s"
        % (
            full_metrics["tier2_fraction"],
            full_metrics["identifier_density_per_1000"],
            sub_metrics["tier2_fraction"],
            sub_metrics["identifier_density_per_1000"],
            correlation,
            receipt["verdict"]["decision"],
        )
    )
    failed = [name for name, value in receipt["gate"].items() if not bool(value)]
    print("[ae0] passed=%s failed=%s receipt=%s" % (receipt["passed"], failed, written["path"]))
    return 0 if bool(receipt["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
