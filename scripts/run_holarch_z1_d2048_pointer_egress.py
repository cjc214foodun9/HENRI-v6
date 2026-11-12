"""Cycle Z1: D = 2048 pointer-resonant photonic egress and the band rotor.

    python scripts/run_holarch_z1_d2048_pointer_egress.py

The held-out stream is the PINNED split (receipts/holarch/pinned_lang_split.json):
package sources only, docs/HOLARCH.md excluded, so appending Section 41 cannot
move it.  The measurement mounts a constructed D = 2048 four-plate Kerr stack,
a closed-form subword cartridge, one contrastive imprint round, an
entropy-gated long-range pointer and a gauge-broken frequency rotor.
"""

from __future__ import annotations

import argparse
import ast as _ast
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import torch

import run_holarch_p_cycle as pc
from henri.holarch.code_gen import BochnerPhaseProjector, WaveCodeTextGenerator
from henri.holarch.code_lang import CodeTextCorpus, build_tokenizer
from henri.holarch.config import HolarchConfig
from henri.holarch.pinned_split import PinnedSplit, build_pinned_split, verify_pinned
from henri.holarch.pointer_egress import PointerEgress, core_model
from henri.holarch.wave_egress import WaveEgressHead
from henri.holarch.zone_b import PhotonicCore, open_subword_cartridge

RECEIPTS = ROOT / "receipts" / "holarch"
PIN_FILE = "pinned_lang_split.json"
DIM = 2048
BANDS = 32


def _mark(label: str, started: float) -> None:
    print("[z1] %-36s %8.1f s" % (label, time.perf_counter() - started), flush=True)


def _load_or_build_pinned() -> Tuple[PinnedSplit, Dict[str, Any]]:
    path = RECEIPTS / PIN_FILE
    split = build_pinned_split(ROOT)
    if path.exists():
        recorded = json.loads(path.read_text(encoding="utf-8"))
        verification = verify_pinned(ROOT, recorded)
    else:
        written = pc._write(RECEIPTS, PIN_FILE, split.payload())
        recorded = split.payload()
        verification = verify_pinned(ROOT, recorded)
        verification["written"] = written["path"]
    return split, verification


def _copy_recall(
    egress,
    sequences: Sequence[Sequence[int]],
    *,
    low: int = 7,
    high: int = 24,
    local: int = 6,
) -> Dict[str, float]:
    """Top-1 on positions whose truth recurs at offset 7..24 and not in the last 6."""
    hits = 0
    total = 0
    for sequence in sequences:
        sequence = [int(token) for token in sequence]
        for index in range(1, len(sequence)):
            window = sequence[max(0, index - high) : index]
            if len(window) < high:
                continue
            target = int(sequence[index])
            # The pointer unbinds window offsets low..high, so the recall probe
            # must score exactly those offsets (indices 7..23 of a 24 window).
            long_range = window[int(low) : int(high)]
            local_window = window[-local:]
            if target not in long_range or target in local_window:
                continue
            probability = egress.distribution(window, None)
            hits += int(int(np.argmax(probability)) == target)
            total += 1
    return {
        "hits": int(hits),
        "positions": int(total),
        "recall": float(hits / max(total, 1)),
    }


def measure(directory: Path) -> Dict[str, Any]:
    started = time.perf_counter()
    split, pinned = _load_or_build_pinned()
    _mark("pinned split", started)

    paths = [ROOT / record["name"] for record in split.records]
    corpus = CodeTextCorpus.from_paths(paths)
    tokenizer = build_tokenizer(corpus)
    projector = BochnerPhaseProjector(dim=DIM, d_sem=64)
    projector.fit(corpus, tokenizer)
    train_ids = list(split.train_ids)
    generator = WaveCodeTextGenerator(tokenizer, projector)
    generator.fit(train_ids, max_fit_tokens=8000)
    generator.count_lm.fit(train_ids)
    generator.lexical.build(train_ids[:24000])
    generator.suffix_weight = 0.0
    generator.recency_weight = 0.0
    generator.document_weight = 0.0
    generator.beta_star = 0.0

    def floor_fn(context):
        return generator.floor_probability(context)

    cap = max(64, 6000 // max(len(split.segments), 1))
    sequences = [
        list(int(token) for token in val[:cap]) for _name, _train, val in split.segments
    ]
    _mark("corpus + projector d2048", started)

    core = PhotonicCore(HolarchConfig(dim=DIM, photonic_layers=4))
    fit_ids = train_ids[: int(len(train_ids) * 0.85)]
    inner_ids = train_ids[int(len(train_ids) * 0.85) :]
    rng = np.random.default_rng(20261111)
    context_waves = [
        projector.sequence_wave(
            train_ids[index : index + 24], tokenizer
        )
        for index in rng.choice(len(train_ids) - 40, 700, replace=False)
    ]
    cartridge = open_subword_cartridge(core, ridge=0.1)
    cartridge_before = cartridge.fit(context_waves[:600])
    _mark("cartridge fit", started)

    imprint_residuals: List[float] = []
    with torch.no_grad():
        for wave in context_waves[:64]:
            psi = torch.from_numpy(np.asarray(wave, dtype=np.complex64)).reshape(1, -1)
            imprint_residuals.append(float(core.imprint(psi, psi, rate=0.1)))
    cartridge = open_subword_cartridge(core, ridge=0.1)
    cartridge_report = cartridge.fit(context_waves[:600])
    _mark("contrastive imprint", started)

    egress = PointerEgress(
        projector,
        tokenizer,
        core=core,
        cartridge=cartridge,
        freq_bands=BANDS,
        quadrature=True,
    )
    egress.set_cartridge(cartridge, report=cartridge_report)
    model = core_model(core)
    # qr_step = 0 keeps the seeded mixing frame: with U = I every phase is
    # gauge, so the egress would have no interference to learn on.
    fit = egress.fit(
        train_ids, epochs=4, samples=1200, floor_fn=floor_fn, model=None, qr_step=0.0
    )
    _mark("egress fit", started)
    egress.photonic_weight = 0.0
    rotor = egress.fit_rotor(
        fit_ids, epochs=8, samples=240, floor_fn=floor_fn, step=0.10
    )
    _mark("rotor torque", started)
    fusion = egress.calibrate_pointer_fusion(inner_ids, samples=120)
    _mark("pointer fusion calibration", started)
    override = egress.calibrate_override(inner_ids, model=model, floor_fn=floor_fn)
    _mark("override calibration", started)

    report = egress.evaluate(sequences, floor_fn=floor_fn, model=None)
    _mark("held-out evaluate", started)
    pointer_recall = _copy_recall(egress, sequences)

    saved_high = egress.w_high
    egress.w_high = 0.0
    without_pointer = egress.evaluate(sequences, floor_fn=floor_fn, model=None)
    _mark("w_high = 0 ablation", started)
    egress.w_high = saved_high
    saved_rotor = egress.rotor.copy()
    egress.rotor = egress.rotor0.copy()
    without_rotor = egress.evaluate(sequences, floor_fn=floor_fn, model=None)
    _mark("rotor reset ablation", started)
    egress.rotor = saved_rotor

    # The six-mode X2 suffix-only baseline on the SAME pinned split.
    baseline = WaveEgressHead(projector, tokenizer, quadrature=False)
    baseline_fit = baseline.fit(train_ids, epochs=4, samples=1200, floor_fn=floor_fn, model=None)
    baseline.calibrate_photonic(inner_ids, model=model, floor_fn=floor_fn,
                                grid=(0.0, 0.05, 0.1), samples=120)
    baseline.calibrate_override(inner_ids, model=model, floor_fn=floor_fn)
    baseline_report = baseline.evaluate(sequences, floor_fn=floor_fn, model=None)
    baseline_recall = _copy_recall(baseline, sequences)
    _mark("x2 baseline on pinned split", started)

    source = (ROOT / "henri" / "holarch" / "pointer_egress.py").read_text(encoding="utf-8")
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

    body = report.as_dict()
    inner_ce_with = float(egress.fusion_report.get("selected", {}).get("nll", 0.0))
    gates = {
        "z1_pinned_split_invariant": bool(pinned["invariant"]),
        "z1_pinned_excludes_docs": bool(
            all("HOLARCH.md" not in record["name"] for record in split.records)
        ),
        "z1_port_cosine_ge_080": bool(
            float(cartridge_report.get("mean_aligned_cosine", 0.0)) >= 0.80
        ),
        "z1_imprint_residual_reduced": bool(
            imprint_residuals
            and float(np.mean(imprint_residuals[-8:])) < float(np.mean(imprint_residuals[:8]))
        ),
        "z1_w_high_positive": bool(float(egress.w_high) > 0.0),
        "z1_pointer_improves_inner_ce": bool(
            float(body["bpc_wave_egress"]) < float(without_pointer.bits_per_char)
        ),
        "z1_rotor_displacement_gt_001": bool(float(egress.rotor_displacement()) > 0.01),
        "z1_rotor_ce_decrease": bool(
            float(rotor.get("inner_nll_gain", 0.0)) > 0.0
            and float(body["bpc_wave_egress"]) <= float(without_rotor.bits_per_char)
        ),
        "z1_top1_beats_x2": bool(float(report.top1) > float(baseline_report.top1)),
        "z1_bpc_beats_x2": bool(
            float(report.bits_per_char) < float(baseline_report.bits_per_char)
        ),
        "z1_copy_recall_gain_015": bool(
            float(pointer_recall["recall"]) - float(baseline_recall["recall"]) >= 0.15
        ),
        "z1_autograd_unused": bool(not markers),
        "z1_held_out_tokens": bool(int(report.tokens) >= 4000),
    }
    return {
        "stage": "Z1",
        "config": {
            "dim": int(DIM),
            "d_slot": int(DIM // 4),
            "freq_bands": int(BANDS),
            "photonic_layers": 4,
            "pointer_offsets": [int(egress.pointer_low), int(egress.pointer_high)],
        },
        "pinned_split": {
            "digest": split.digest,
            "files": int(len(split.records)),
            "train_tokens": int(split.train_tokens),
            "val_tokens": int(split.val_tokens),
            "documents": int(len(split.segments)),
            "verification": pinned,
            "live_corpus_digest": split.live_corpus_digest,
        },
        "photonic_cartridge": {
            "before_imprint": cartridge_before,
            "after_imprint": cartridge_report,
            "imprint_residuals": [float(value) for value in imprint_residuals],
        },
        "wave_egress": body,
        "x2_baseline": {
            "top1": float(baseline_report.top1),
            "bits_per_char": float(baseline_report.bits_per_char),
            "net_overrides": int(baseline_report.override_net_win_delta),
            "copy_recall": baseline_recall,
            "photonic_weight": float(baseline.photonic_weight),
            "fit_updates": int(baseline_fit.get("updates", 0)),
        },
        "pointer": {
            "w_low": float(egress.w_low),
            "w_high": float(egress.w_high),
            "tau_h": float(egress.tau_h),
            "fusion": egress.fusion_report,
            "inner_ce_with_pointer": inner_ce_with,
            "copy_recall": pointer_recall,
            "ablation_without_pointer": {
                "top1": float(without_pointer.top1),
                "bits_per_char": float(without_pointer.bits_per_char),
                "net_overrides": int(without_pointer.override_net_win_delta),
            },
            "contexts": int(egress.pointer_contexts),
        },
        "rotor": {
            **rotor,
            "ablation_without_rotor": {
                "top1": float(without_rotor.top1),
                "bits_per_char": float(without_rotor.bits_per_char),
            },
            "final_displacement": float(egress.rotor_displacement()),
        },
        "attribution": {
            "pointer_delta_top1": float(report.top1 - without_pointer.top1),
            "pointer_delta_bpc": float(report.bits_per_char - without_pointer.bits_per_char),
            "rotor_delta_top1": float(report.top1 - without_rotor.top1),
            "rotor_delta_bpc": float(report.bits_per_char - without_rotor.bits_per_char),
            "baseline_delta_top1": float(report.top1 - baseline_report.top1),
            "baseline_delta_bpc": float(report.bits_per_char - baseline_report.bits_per_char),
        },
        "override_calibration": override,
        "autograd_markers": markers,
        "gate": pc._finalise(gates),
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
        "passed": bool(all(gates.values())),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Cycle Z1 D=2048 pointer egress")
    parser.add_argument("--receipts", default=str(RECEIPTS))
    args = parser.parse_args(argv)
    directory = Path(args.receipts)
    directory.mkdir(parents=True, exist_ok=True)
    payload = measure(directory)
    written = pc._write(directory, "z1_d2048_pointer_wave_egress.json", payload)
    body = payload["wave_egress"]
    print("[z1] top1=%.4f bpc=%.4f vs x2 top1=%.4f bpc=%.4f" % (
        float(body["top1_wave_egress"]), float(body["bpc_wave_egress"]),
        float(payload["x2_baseline"]["top1"]), float(payload["x2_baseline"]["bits_per_char"])))
    print("[z1] w_high=%.3f rotor=%.4f rad copy=%.3f vs %.3f" % (
        float(payload["pointer"]["w_high"]), float(payload["rotor"]["final_displacement"]),
        float(payload["pointer"]["copy_recall"]["recall"]),
        float(payload["x2_baseline"]["copy_recall"]["recall"])))
    failed = [name for name, value in payload["gate"].items() if not bool(value)]
    print("[z1] passed=%s failed=%s receipt=%s" % (payload["passed"], failed, written["path"]))
    return 0 if bool(payload["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
