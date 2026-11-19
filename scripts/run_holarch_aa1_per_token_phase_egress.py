"""Cycle AA1: per-token phase-conjugate Born egress and induction pointer.

    python scripts/run_holarch_aa1_per_token_phase_egress.py

The held-out stream is the FROZEN pinned corpus snapshot
(receipts/holarch/pinned_corpus_snapshot.json), whose digest is the Cycle Z
pin 8f270f34e8f35ef84e415d076bea20a7bc15edff337f89ed72ad607447ba8554.  The
snapshot freezes the exact bytes of the 38 package files, so a code edit in
this cycle cannot move the evaluation.  docs/HOLARCH.md is excluded.

Mounted organs:

* the 6/7-gram continuation floor, calibrated on the inner slice;
* the six window-local suffix resonances;
* the token-indexed phase table Phi, trained by the exact Born cross-term
  phase gradient with a best-so-far inner line search;
* the holographic key-successor induction pointer over a 256-token causal
  window, excluding the local span already read by the suffix bank;
* the aligned D = 2048 post-photonic port.
"""

from __future__ import annotations

import ast as _ast
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import torch

import run_holarch_p_cycle as pc
from henri.holarch.code_gen import BochnerPhaseProjector, CountLanguageModel
from henri.holarch.code_lang import build_tokenizer
from henri.holarch.config import HolarchConfig
from henri.holarch.pinned_split import (
    SNAPSHOT_NAME,
    load_snapshot,
    snapshot_corpus,
    snapshot_segments,
    snapshot_train_ids,
)
from henri.holarch.wave_egress import TokenPhaseEgress
from henri.holarch.zone_b import PhotonicCore, open_subword_cartridge

RECEIPTS = ROOT / "receipts" / "holarch"
PIN = "8f270f34e8f35ef84e415d076bea20a7bc15edff337f89ed72ad607447ba8554"
Z1_BPC = 2.894812290769193
Z1_TOP1 = 0.6006300315015751
Z1_COPY = 0.684981684981685
DIM = 2048


def _mark(label: str, started: float) -> None:
    print("[aa1] %-40s %8.1f s" % (label, time.perf_counter() - started), flush=True)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy_recall(
    egress, sequences: Sequence[Sequence[int]], *, low: int = 7, high: int = 24, local: int = 6
) -> Dict[str, Any]:
    """Top-1 on the 1,092 positions whose truth recurs at offset 7..23.

    Two protocols are recorded.  `strict` passes only the 24-token window to
    the model, which is the only context Z1's pointer could read.  `natural`
    passes the model's own causal window (up to 512 tokens), which is the
    deployed protocol and the only one in which a 256-token induction head or
    a long-range copy resonance can act at all.  The gate is scored on the
    natural protocol; the strict number is reported beside it so the
    comparison with Z1's 0.6850 stays visible.
    """
    out: Dict[str, Any] = {}
    for name, ctx in (("natural", int(getattr(egress, "window", 512))), ("strict", int(high))):
        hits = 0
        total = 0
        for sequence in sequences:
            sequence = [int(token) for token in sequence]
            for index in range(1, len(sequence)):
                window = sequence[max(0, index - high) : index]
                if len(window) < high:
                    continue
                target = int(sequence[index])
                long_range = window[int(low) : int(high)]
                local_window = window[-local:]
                if target not in long_range or target in local_window:
                    continue
                probability = egress.distribution(sequence[max(0, index - ctx) : index], None)
                hits += int(int(np.argmax(probability)) == target)
                total += 1
        out[name] = {
            "hits": int(hits),
            "positions": int(total),
            "recall": float(hits / max(total, 1)),
        }
    out["recall"] = float(out["natural"]["recall"])
    out["hits"] = int(out["natural"]["hits"])
    out["positions"] = int(out["natural"]["positions"])
    return out


def measure(directory: Path) -> Dict[str, Any]:
    started = time.perf_counter()
    snapshot_path = directory / SNAPSHOT_NAME
    sha_before = _sha(snapshot_path)
    pin_before = _sha(directory / "pinned_lang_split.json")
    snapshot = load_snapshot(ROOT)
    corpus = snapshot_corpus(snapshot)
    tokenizer = build_tokenizer(corpus)
    segments = snapshot_segments(snapshot)
    train_ids = snapshot_train_ids(snapshot)
    _mark("frozen snapshot", started)

    projector = BochnerPhaseProjector(dim=DIM, d_sem=64)
    projector.fit(corpus, tokenizer)
    _mark("projector d2048", started)

    inner_cut = int(len(train_ids) * 0.85)
    fit_ids, inner_ids = train_ids[:inner_cut], train_ids[inner_cut:]
    floor_counts = CountLanguageModel(int(len(tokenizer)))
    floor_counts.fit(train_ids)

    def floor_fn(context):
        return floor_counts.distribution(context)

    cap = max(64, 6000 // max(len(segments), 1))
    sequences = [list(int(token) for token in val[:cap]) for _name, _train, val in segments]
    _mark("held-out sequences", started)

    core = PhotonicCore(HolarchConfig(dim=DIM, photonic_layers=4))
    rng = np.random.default_rng(20261111)
    context_waves = [
        projector.sequence_wave(train_ids[index : index + 24], tokenizer)
        for index in rng.choice(len(train_ids) - 40, 700, replace=False)
    ]
    cartridge = open_subword_cartridge(core, ridge=0.1)
    cartridge_before = cartridge.fit(context_waves[:600])
    imprint: List[float] = []
    with torch.no_grad():
        for wave in context_waves[:64]:
            psi = torch.from_numpy(np.asarray(wave, dtype=np.complex64)).reshape(1, -1)
            imprint.append(float(core.imprint(psi, psi, rate=0.1)))
    cartridge = open_subword_cartridge(core, ridge=0.1)
    cartridge_report = cartridge.fit(context_waves[:600])
    _mark("photonic cartridge + imprint", started)

    egress = TokenPhaseEgress(
        projector,
        tokenizer,
        core=core,
        cartridge=cartridge,
        induction_window=256,
        induction_gamma=0.8,
        induction_low=7,
        induction_tau_match=0.5,
    )
    egress.set_cartridge(cartridge, report=cartridge_report)
    floor = egress.fit_floor(train_ids, inner_ids)
    _mark("6/7-gram floor calibration", started)
    mix = egress.fit(train_ids, epochs=4, samples=1200, floor_fn=floor_fn, model=None)
    _mark("mode weights", started)
    phase = egress.fit_phase_table(train_ids, epochs=26, samples=2000, inner_samples=400)
    _mark("per-token phase table", started)
    channels = egress.calibrate_channels(inner_ids, samples=1200)
    _mark("entropy-gated channels", started)
    override = egress.calibrate_override(inner_ids, model=None, floor_fn=floor_fn)
    _mark("override thresholds", started)

    report = egress.evaluate(sequences, floor_fn=floor_fn, model=None)
    _mark("held-out evaluation", started)
    recall = _copy_recall(egress, sequences)

    saved_phase = egress.phase_table.copy()
    saved_ind = float(egress.w_induction)
    saved_port = float(egress.w_photonic)
    saved_copy = float(egress.w_copy)
    egress.phase_table = egress.phase_table0.copy()
    without_phase = egress.evaluate(sequences, floor_fn=floor_fn, model=None)
    _mark("ablation: phases reset", started)
    egress.phase_table = saved_phase.copy()
    egress.w_induction = 0.0
    egress.w_photonic = 0.0
    egress.w_copy = 0.0
    without_channels = egress.evaluate(sequences, floor_fn=floor_fn, model=None)
    _mark("ablation: channels off", started)
    egress.phase_table = egress.phase_table0.copy()
    floor_only = egress.evaluate(sequences, floor_fn=floor_fn, model=None)
    _mark("ablation: floor only", started)
    egress.phase_table = saved_phase.copy()
    egress.w_induction = saved_ind
    egress.w_photonic = saved_port
    egress.w_copy = saved_copy

    source = (ROOT / "henri" / "holarch" / "wave_egress.py").read_text(encoding="utf-8")
    markers: List[str] = []
    for node in _ast.walk(_ast.parse(source)):
        if isinstance(node, _ast.Call):
            func = node.func
            name = func.attr if isinstance(func, _ast.Attribute) else (
                func.id if isinstance(func, _ast.Name) else "")
            if str(name) in {"backward", "autograd"}:
                markers.append("call:" + str(name))
        if isinstance(node, _ast.Attribute) and str(node.attr) in {"optim", "autograd", "grad"}:
            markers.append("attribute:" + str(node.attr))

    sha_after = _sha(snapshot_path)
    pin_after = _sha(directory / "pinned_lang_split.json")
    body = report.as_dict()
    selected = channels.get("selected", {})
    gates = {
        "aa1_pinned_digest_equals_8f270f34": bool(
            str(snapshot.get("_digest_recomputed", "")) == PIN
        ),
        "aa1_pinned_snapshot_unchanged": bool(sha_before == sha_after and pin_before == pin_after),
        "aa1_zero_autograd": bool(not markers),
        "aa1_channel_weight_positive": bool(
            float(selected.get("w_induction", 0.0)) > 0.0
            or float(selected.get("w_photonic", 0.0)) > 0.0
            or float(selected.get("w_copy", 0.0)) > 0.0
        ),
        "aa1_phase_rms_gt_001": bool(
            float(phase.get("rms_displacement_rad", 0.0)) > 0.01
            and bool(phase.get("accepted_by_line_search", False))
        ),
        "aa1_bpc_beats_z1": bool(float(body["bpc_wave_egress"]) < Z1_BPC),
        "aa1_top1_beats_z1": bool(float(body["top1_wave_egress"]) > Z1_TOP1),
        "aa1_copy_recall_ge_072": bool(float(recall["recall"]) >= 0.72),
        "aa1_copy_positions_1092": bool(int(recall["positions"]) == 1092),
        "aa1_held_out_tokens": bool(int(body["tokens"]) >= 4000),
    }
    return {
        "stage": "AA1",
        "config": {
            "dim": DIM,
            "freq_bands": 32,
            "photonic_layers": 4,
            "phase_modes": int(egress.phase_modes),
            "induction_window": int(egress.induction_window),
            "induction_gamma": float(egress.induction_gamma),
            "induction_low": int(egress.induction_low),
            "induction_tau_match": float(egress.induction_tau_match),
            "orders": [int(value) for value in egress.orders],
            "coupling": str(egress.coupling),
        },
        "pinned_split": {
            "snapshot": snapshot.get("_path"),
            "snapshot_sha256": sha_before,
            "snapshot_sha256_after": sha_after,
            "pinned_lang_split_sha256": pin_before,
            "pinned_lang_split_sha256_after": pin_after,
            "digest_recorded": str(snapshot.get("digest", "")),
            "digest_frozen_recomputed": str(snapshot.get("_digest_recomputed", "")),
            "files": int(len(snapshot.get("files") or [])),
            "train_tokens": int(len(train_ids)),
            "val_tokens": int(sum(len(segment[2]) for segment in segments)),
            "documents": int(len(segments)),
            "docs_excluded": True,
            "live_tree_digest_diagnostic": "recorded separately; code edits move the live tree, not the snapshot",
        },
        "photonic_cartridge": {
            "before_imprint": cartridge_before,
            "after_imprint": cartridge_report,
            "imprint_residuals": [float(value) for value in imprint],
        },
        "floor": floor,
        "modal_mix": mix,
        "phase_table": phase,
        "channels": channels,
        "channel_weights": {
            "w_photonic": float(getattr(egress, "w_photonic", 0.0)),
            "w_induction": float(getattr(egress, "w_induction", 0.0)),
            "w_copy": float(getattr(egress, "w_copy", 0.0)),
            "coupling": str(getattr(egress, "coupling", "")),
            "tau_gate_H": float(getattr(egress, "tau_gate_h", 0.0)),
        },
        "override_calibration": override,
        "wave_egress": body,
        "copy_recall": recall,
        "z1_baseline": {
            "bits_per_char": Z1_BPC,
            "top1": Z1_TOP1,
            "copy_recall": Z1_COPY,
            "source": "receipts/holarch/z1_stage_pointer_egress.json (x2_baseline)",
        },
        "ablations": {
            "without_phase_table": {
                "top1": float(without_phase.top1),
                "bits_per_char": float(without_phase.bits_per_char),
            },
            "without_channels": {
                "top1": float(without_channels.top1),
                "bits_per_char": float(without_channels.bits_per_char),
            },
            "floor_only": {
                "top1": float(floor_only.top1),
                "bits_per_char": float(floor_only.bits_per_char),
            },
        },
        "attribution": {
            "phase_delta_bpc": float(body["bpc_wave_egress"] - without_phase.bits_per_char),
            "phase_delta_top1": float(body["top1_wave_egress"] - without_phase.top1),
            "channel_delta_bpc": float(body["bpc_wave_egress"] - without_channels.bits_per_char),
            "channel_delta_top1": float(body["top1_wave_egress"] - without_channels.top1),
            "floor_delta_bpc": float(body["bpc_wave_egress"] - floor_only.bits_per_char),
            "floor_delta_top1": float(body["top1_wave_egress"] - floor_only.top1),
            "delta_vs_z1_bpc": float(body["bpc_wave_egress"] - Z1_BPC),
            "delta_vs_z1_top1": float(body["top1_wave_egress"] - Z1_TOP1),
            "delta_vs_z1_copy": float(recall["recall"] - Z1_COPY),
        },
        "autograd_markers": markers,
        "gate": pc._finalise(gates),
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
        "passed": bool(all(gates.values())),
    }


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Cycle AA1 per-token phase egress")
    parser.add_argument("--receipts", default=str(RECEIPTS))
    args = parser.parse_args(argv)
    directory = Path(args.receipts)
    directory.mkdir(parents=True, exist_ok=True)
    payload = measure(directory)
    written = pc._write(directory, "aa1_per_token_phase_egress.json", payload)
    body = payload["wave_egress"]
    print(
        "[aa1] top1=%.4f bpc=%.4f (z1 %.4f / %.4f) copy=%.4f (z1 %.4f)"
        % (
            float(body["top1_wave_egress"]),
            float(body["bpc_wave_egress"]),
            Z1_TOP1,
            Z1_BPC,
            float(payload["copy_recall"]["recall"]),
            Z1_COPY,
        )
    )
    print(
        "[aa1] phase_rms=%.4f rad w_ind=%.3f w_port=%.3f"
        % (
            float(payload["phase_table"]["rms_displacement_rad"]),
            float(payload["channels"]["selected"].get("w_induction", 0.0)),
            float(payload["channels"]["selected"].get("w_photonic", 0.0)),
        )
    )
    failed = [name for name, value in payload["gate"].items() if not bool(value)]
    print("[aa1] passed=%s failed=%s receipt=%s" % (payload["passed"], failed, written["path"]))
    return 0 if bool(payload["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
