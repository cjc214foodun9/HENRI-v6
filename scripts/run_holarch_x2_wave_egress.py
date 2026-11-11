"""Cycle X2: contrastive thermodynamic wave egress and unclamped override.

    python scripts/run_holarch_x2_wave_egress.py

Measures the standalone transducer on the exact W2 34-document stratified
held-out split (6-token boundary purge), fits its phase adapter on the W2
training split only, calibrates the photonic weight and the three override
thresholds on an INNER slice of that training split, and seals the result.

The count floor is the deployed H1 floor of the same generator (the count
model with the stream cache and suffix mix at their zero defaults), measured
on the same held-out tokens in the same process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

import run_holarch_p_cycle as pc
from henri.holarch.code_gen import BochnerPhaseProjector, WaveCodeTextGenerator
from henri.holarch.code_lang import CodeTextCorpus, build_tokenizer
from henri.holarch.config import HolarchConfig
from henri.holarch.model import find_checkpoint, load_checkpoint, open_clean_room_memory
from henri.holarch.pipeline import HenriHolarch
from henri.holarch.wave_egress import WaveEgressHead

RECEIPTS = ROOT / "receipts" / "holarch"
BASELINE_BITS = float(np.log2(17.043866783609086))


def _digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def measure(directory: Path) -> Dict[str, Any]:
    started = time.perf_counter()
    corpus = CodeTextCorpus.holarch(ROOT)
    tokenizer = build_tokenizer(corpus)
    segments = corpus.stratified_segments(tokenizer, val_fraction=0.1, boundary_gap=6)
    train_ids: List[int] = []
    gap_ok = True
    for text, (name, train, val) in zip(corpus.texts, segments):
        full = [int(token) for token in tokenizer.encode(str(text))]
        cut = len(train)
        if full[:cut] != list(train) or full[cut + 6 :] != list(val):
            gap_ok = False
        train_ids.extend(int(token) for token in train)
    projector = BochnerPhaseProjector(dim=1024, d_sem=64)
    projector.fit(corpus, tokenizer)
    generator = WaveCodeTextGenerator(tokenizer, projector)
    generator.fit(train_ids, max_fit_tokens=8000)
    generator.count_lm.fit(train_ids)
    generator.lexical.build(train_ids[:24000])
    # The deployed H1 floor: the suffix mix and the stream cache stay at
    # their zero defaults, exactly as the W2 hygiene receipt measured them.
    generator.suffix_weight = 0.0
    generator.recency_weight = 0.0
    generator.document_weight = 0.0
    generator.beta_star = 0.0

    def floor_fn(context):
        return generator.floor_probability(context)

    cap = max(64, 6000 // max(len(segments), 1))
    sequences = [list(int(token) for token in val[:cap]) for _name, _train, val in segments]

    checkpoint = find_checkpoint(None)
    payload = json.loads("{}")
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg = HolarchConfig(**payload["config"])
    workdir = directory / "_work" / "x2"
    workdir.mkdir(parents=True, exist_ok=True)
    memory, _info = open_clean_room_memory(cfg, checkpoint, workdir, prefix="x2_wave")
    model = HenriHolarch(cfg, memory=memory)
    load_checkpoint(checkpoint, model=model)
    model.eval()

    egress = WaveEgressHead(projector, tokenizer)
    fit_started = time.perf_counter()
    fit = egress.fit(train_ids, epochs=4, samples=1200, floor_fn=floor_fn, model=None)
    fit_seconds = time.perf_counter() - fit_started
    cut = int(len(train_ids) * 0.85)
    inner_ids = train_ids[cut:]
    photonic = egress.calibrate_photonic(inner_ids, model=model, floor_fn=floor_fn)
    override = egress.calibrate_override(inner_ids, model=model, floor_fn=floor_fn)
    evaluate_started = time.perf_counter()
    report = egress.evaluate(sequences, floor_fn=floor_fn, model=model)
    evaluate_seconds = time.perf_counter() - evaluate_started
    # Attribution ablation: the same egress with the learned phase offsets
    # removed.  The ring phases then reproduce the incoherent order mixture.
    saved_theta = egress.theta.copy()
    saved_adapter = egress.adapter.copy()
    egress.theta = np.zeros_like(saved_theta)
    egress.adapter = np.eye(saved_adapter.shape[0], dtype=np.complex128)
    ring_report = egress.evaluate(sequences, floor_fn=floor_fn, model=model)
    egress.theta = saved_theta
    egress.adapter = saved_adapter

    source = (ROOT / "henri" / "holarch" / "wave_egress.py").read_text(encoding="utf-8")
    import ast as _ast

    tree = _ast.parse(source)
    markers: List[str] = []
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Call):
            func = node.func
            name = func.attr if isinstance(func, _ast.Attribute) else (
                func.id if isinstance(func, _ast.Name) else "")
            if str(name) in {"backward", "autograd"}:
                markers.append("call:" + str(name))
        if isinstance(node, _ast.Attribute) and str(node.attr) in {"optim", "autograd", "grad"}:
            markers.append("attribute:" + str(node.attr))
    # Incoherent arithmetic mixture: the same order waves with no interference.
    mix_top1 = 0
    mix_nll = 0.0
    mix_tokens = 0
    for sequence in sequences:
        for index in range(1, len(sequence)):
            window = sequence[max(0, index - 512) : index]
            weights, dists, _totals, _hp = egress._order_waves(window, model)
            probability = sum(weight * dist for weight, dist in zip(weights, dists))
            probability = np.maximum(probability, 1e-12)
            probability = probability / probability.sum()
            truth = int(sequence[index])
            mix_top1 += int(int(np.argmax(probability)) == truth)
            mix_nll -= float(np.log(max(float(probability[truth]), 1e-12)))
            mix_tokens += 1
    body = report.as_dict()
    gates = {
        "x2_wave_egress_module": bool((ROOT / "henri" / "holarch" / "wave_egress.py").exists()),
        "x2_wave_egress_imports": True,
        "x2_autograd_unused": bool(not markers and not report.autograd_used),
        "x2_sagnac_residual_reduced": bool(report.sagnac_final < report.sagnac_init),
        "x2_unclamped_overrides_present": bool(report.override_count > 0),
        "x2_override_net_win_positive": bool(report.override_net_win_delta > 0),
        "x2_top1_beats_floor": bool(report.top1 > report.base_top1),
        "x2_bpc_beats_floor": bool(report.bits_per_char < report.base_bits_per_char),
        "x2_held_out_tokens": bool(report.tokens >= 4000),
        "x2_gap_purged": bool(gap_ok),
    }
    return {
        "stage": "X2",
        "split": {
            "documents": int(len(segments)),
            "train_tokens": int(len(train_ids)),
            "held_out_tokens": int(report.tokens),
            "gap_tokens": 6,
            "boundary_purged": bool(gap_ok),
            "inner_tokens": int(len(inner_ids)),
            "corpus_digest": str(getattr(corpus, "digest", "")),
        },
        "wave_egress": body,
        "fit": {
            "epochs": int(fit.get("epochs", 0)),
            "updates": int(fit.get("updates", 0)),
            "sagnac_curve": list(fit.get("sagnac_curve") or []),
            "fit_seconds": float(fit_seconds),
            "evaluate_seconds": float(evaluate_seconds),
            "photonic_weight": float(egress.photonic_weight),
            "photonic_calibration": photonic,
            "override_calibration": override,
            "theta": list(fit.get("theta") or []),
            "adapter_digest": float(fit.get("adapter_digest", 0.0)),
        },
        "attribution": {
            "wave_trained": {
                "top1": float(report.top1),
                "bits_per_char": float(report.bits_per_char),
            },
            "wave_incoherent_ring": {
                "top1": float(ring_report.top1),
                "bits_per_char": float(ring_report.bits_per_char),
            },
            "incoherent_suffix_mixture": {
                "top1": float(mix_top1 / max(mix_tokens, 1)),
                "bits_per_char": float(mix_nll / max(mix_tokens, 1) / np.log(2.0)),
            },
            "count_floor": {
                "top1": float(report.base_top1),
                "bits_per_char": float(report.base_bits_per_char),
            },
            "phase_learning_delta_top1": float(report.top1 - ring_report.top1),
            "phase_learning_delta_bpc": float(report.bits_per_char - ring_report.bits_per_char),
            "suffix_resonance_delta_top1": float(ring_report.top1 - report.base_top1),
        },
        "autograd_markers": markers,
        "bar_policy": {
            "note": ("the bars are the measured count floor of the same process; "
                     "the phase learner is selected on an inner slice of the training split"),
        },
        "gate": pc._finalise(gates),
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
        "passed": bool(all(gates.values())),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Cycle X2 wave egress")
    parser.add_argument("--receipts", default=str(RECEIPTS))
    args = parser.parse_args(argv)
    directory = Path(args.receipts)
    directory.mkdir(parents=True, exist_ok=True)
    payload = measure(directory)
    written = pc._write(directory, "x2_sovereign_wave_egress.json", payload)
    body = payload["wave_egress"]
    print("[x2] top1 wave=%.4f floor=%.4f | bpc wave=%.4f floor=%.4f" % (
        float(body["top1_wave_egress"]), float(body["top1_count_floor"]),
        float(body["bpc_wave_egress"]), float(body["bpc_count_floor"])))
    print("[x2] overrides=%d wins=%d losses=%d net=%d sagnac %.4f->%.4f" % (
        int(body["unclamped_override_count"]), int(body["override_wins"]),
        int(body["override_losses"]), int(body["override_net_win_delta"]),
        float(body["sagnac_initial"]), float(body["sagnac_final"])))
    print("[x2] passed=%s receipt=%s" % (payload["passed"], written["path"]))
    return 0 if bool(payload["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())