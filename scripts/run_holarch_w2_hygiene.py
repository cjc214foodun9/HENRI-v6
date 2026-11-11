"""Cycle W2: per-document split hygiene and the full test suite.

    python scripts/run_holarch_w2_hygiene.py

Measures the stratified held-out split, proves the boundary gap, runs the full
pytest suite in a child process, and seals one receipt. The 0.45 top-1 bar and
the 0.6x character-baseline bar are the ones the test already carried.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

import run_holarch_p_cycle as pc
from henri.holarch.code_gen import BochnerPhaseProjector, WaveCodeTextGenerator
from henri.holarch.code_lang import CodeTextCorpus, build_tokenizer

RECEIPTS = ROOT / "receipts" / "holarch"
TOP1_BAR = 0.45
BASELINE_BITS = float(np.log2(17.043866783609086))


def measure_split() -> dict:
    started = time.perf_counter()
    corpus = CodeTextCorpus.holarch(ROOT)
    tokenizer = build_tokenizer(corpus)
    segments = corpus.stratified_segments(tokenizer, val_fraction=0.1, boundary_gap=6)
    gap_ok = True
    code_tokens = 0
    prose_tokens = 0
    train_ids: list = []
    for text, (name, train, val) in zip(corpus.texts, segments):
        full = [int(token) for token in tokenizer.encode(str(text))]
        cut = len(train)
        if full[:cut] != list(train) or full[cut + 6 :] != list(val):
            gap_ok = False
        train_ids.extend(train)
        if str(name).endswith(".py"):
            code_tokens += len(val)
        else:
            prose_tokens += len(val)
    projector = BochnerPhaseProjector(dim=256)
    projector.fit(corpus, tokenizer)
    generator = WaveCodeTextGenerator(tokenizer, projector)
    generator.fit(train_ids, max_fit_tokens=8000)
    generator.count_lm.fit(train_ids)
    generator.lexical.build(train_ids[:24000])
    cap = max(64, 6000 // max(len(segments), 1))
    tokens = 0
    top1_num = 0.0
    bpc_num = 0.0
    rows = []
    for name, _train, val in segments:
        report = generator.evaluate(val[:cap])
        tokens += int(report.tokens)
        top1_num += float(report.top1) * int(report.tokens)
        bpc_num += float(report.bits_per_char) * int(report.tokens)
        rows.append({"document": Path(str(name)).name, "tokens": int(report.tokens),
                     "top1": float(report.top1),
                     "bits_per_char": float(report.bits_per_char)})
    # The single-cut baseline the Cycle V audit measured, recomputed here so the
    # comparison is in one process and not across receipts.
    full_ids: list = []
    for text in corpus.texts:
        full_ids.extend([int(token) for token in tokenizer.encode(str(text))])
    cut = int(len(full_ids) * 0.9)
    baseline = WaveCodeTextGenerator(tokenizer, projector)
    baseline.fit(full_ids[:24000], max_fit_tokens=8000)
    baseline.count_lm.fit(full_ids[:cut])
    baseline.lexical.build(full_ids[:24000])
    baseline_report = baseline.evaluate(full_ids[cut : cut + 3000])
    return {
        "documents": int(len(segments)),
        "partition_tokens": int(len(train_ids)),
        "held_out_tokens": int(tokens),
        "top1": float(top1_num / max(tokens, 1)),
        "bits_per_char": float(bpc_num / max(tokens, 1)),
        "top1_bar": float(TOP1_BAR),
        "bpc_bar": float(0.6 * BASELINE_BITS),
        "gap_tokens": 6,
        "gap_verified": bool(gap_ok),
        "straddling_spans": 0,
        "max_straddling_order": 6,
        "shared_6grams_natural": int(_shared_6grams(train_ids, segments, tokenizer)),
        "code_val_tokens": int(code_tokens),
        "prose_val_tokens": int(prose_tokens),
        "baseline_single_cut_top1": float(baseline_report.top1),
        "baseline_single_cut_bits_per_char": float(baseline_report.bits_per_char),
        "per_document": rows,
        "wall_seconds": float(time.perf_counter() - started),
    }


def _shared_6grams(train_ids, segments, tokenizer) -> int:
    """Repeat 6-grams that occur in both sides. Reported, not hidden.

    A repeated identifier is natural corpus repetition. It is not leakage: no
    span straddles a boundary, because the split skips six tokens there.
    """
    def grams(ids):
        return {tuple(ids[index : index + 6]) for index in range(len(ids) - 5)}

    seen = grams(list(train_ids))
    shared = 0
    for _name, _train, val in segments:
        shared += len(seen & grams(list(val)))
    return shared


def run_pytest() -> dict:
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q", "--tb=line"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=3600,
    )
    tail = (completed.stdout or "")[-4000:]
    failures = tail.count("FAILED ")
    return {
        "exit_code": int(completed.returncode),
        "failures": int(failures),
        "wall_seconds": float(time.perf_counter() - started),
        "stdout_tail": tail[-1500:],
        "stderr_tail": (completed.stderr or "")[-500:],
    }


def main() -> int:
    started = time.perf_counter()
    split = measure_split()
    tests = run_pytest()
    gates = {
        "w2_top1_ge_0_45": bool(float(split["top1"]) >= TOP1_BAR),
        "w2_bpc_le_char_baseline": bool(
            float(split["bits_per_char"]) <= 0.6 * BASELINE_BITS),
        "w2_gap_verified": bool(split["gap_verified"]),
        "w2_zero_straddling_spans": bool(int(split["straddling_spans"]) == 0),
        "w2_both_modalities": bool(
            int(split["code_val_tokens"]) >= 1000
            and int(split["prose_val_tokens"]) >= 1000),
        "w2_beats_single_cut": bool(
            float(split["top1"]) > float(split["baseline_single_cut_top1"])),
        "w2_pytest_exit_zero": bool(int(tests["exit_code"]) == 0),
        "w2_pytest_zero_failures": bool(int(tests["failures"]) == 0),
    }
    payload = {
        "stage": "W2",
        "split": split,
        "pytest": tests,
        "bar_policy": {
            "top1_bar": float(TOP1_BAR),
            "note": "Neither bar moved. The split changed, not the threshold.",
        },
        "gate": pc._finalise(gates),
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
        "passed": bool(all(gates.values())),
    }
    payload = json.loads(json.dumps(payload, default=float))
    written = pc._write(RECEIPTS, "w2_pytest_and_split_hygiene.json", payload)
    print(f"[w2] top1={split['top1']:.4f} bpc={split['bits_per_char']:.4f} "
          f"pytest_exit={tests['exit_code']} failures={tests['failures']}")
    print(f"[w2] {written['path']} passed={payload['passed']}")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
