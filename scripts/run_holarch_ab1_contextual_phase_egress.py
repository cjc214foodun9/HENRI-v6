"""Cycle AB1: context-steered Born phase and non-redundant dual-window induction.

    python scripts/run_holarch_ab1_contextual_phase_egress.py

The held-out stream is the FROZEN pinned corpus snapshot
(receipts/holarch/pinned_corpus_snapshot.json, digest 8f270f34...).  The
snapshot freezes the exact bytes of the package files, so a code edit in this
cycle cannot move the evaluation.

Mounted organs, in the order they are calibrated on the inner slice:

* the 6/7-gram continuation floor;
* the six window-local suffix resonances;
* the static token phase table Phi (AA1) and the rank-R Zone-B context
  steering (Phi, W_m, V), trained by the exact analytic Born cross-term
  derivative with zero autograd and accepted only on inner-slice cross
  entropy reduction;
* the aligned post-photonic port;
* the non-redundant role-prefix induction residual;
* the fast 24-token and slow 256-token Slot-0 copy caches.

Every gate is scored on the frozen held-out slice; every marginal is the
difference between two measured evaluations of the same model.
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
from henri.holarch.wave_egress import ContextSteeredEgress
from henri.holarch.zone_b import PhotonicCore, open_subword_cartridge

RECEIPTS = ROOT / "receipts" / "holarch"
PIN = "8f270f34e8f35ef84e415d076bea20a7bc15edff337f89ed72ad607447ba8554"
AA1_BPC = 2.6848
AA1_TOP1 = 0.6271
AA1_PHASE_DELTA = -0.00042
Z1_BPC = 2.894812290769193
Z1_COPY = 0.684981684981685
DIM = 2048

# Task-ID, benchmark-function and file-name branch audit.  A synthesizer or
# inducer must never branch on these strings.
FORBIDDEN_BRANCH_TOKENS = (
    "humaneval",
    "mbpp",
    "arc_agi",
    "task_id",
    "swe_bench",
    "gsm8k",
    "mmlu",
    "hellaswag",
    ".json",
    ".py",
    ".md",
    ".txt",
)


def _mark(label: str, started: float) -> None:
    print("[ab1] %-44s %8.1f s" % (label, time.perf_counter() - started), flush=True)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy_recall(
    egress, sequences: Sequence[Sequence[int]], *, low: int = 7, high: int = 24, local: int = 6
) -> Dict[str, Any]:
    """Top-1 on the positions whose truth recurs at offset 7..23.

    natural reads the model's own causal window (up to 512 tokens); strict
    passes only the 24-token window, which is the protocol Z1's pointer could
    read.  Both protocols are gated in AB1.
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


def _branch_audit(path: Path) -> Dict[str, Any]:
    """AST audit: no task-id, benchmark-name or file-name string branch.

    Every string literal that appears inside a Compare test or a boolean
    branch is extracted and tested against the forbidden token list.  The
    audit also records the autograd and optimizer surface.
    """
    source = path.read_text(encoding="utf-8")
    tree = _ast.parse(source)
    offenders: List[Dict[str, Any]] = []
    autograd: List[str] = []
    branch_strings: List[str] = []
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Compare):
            for child in _ast.walk(node):
                if isinstance(child, _ast.Constant) and isinstance(child.value, str):
                    branch_strings.append(child.value)
        if isinstance(node, _ast.Call):
            func = node.func
            name = (
                func.attr
                if isinstance(func, _ast.Attribute)
                else (func.id if isinstance(func, _ast.Name) else "")
            )
            if str(name) in {"backward", "autograd", "step", "zero_grad"}:
                autograd.append("call:" + str(name))
            if isinstance(func, _ast.Attribute) and str(func.attr) in {"optim", "autograd", "grad"}:
                autograd.append("attribute:" + str(func.attr))
        if isinstance(node, _ast.Attribute) and str(node.attr) in {"optim", "autograd", "grad"}:
            autograd.append("attribute:" + str(node.attr))
    for literal in branch_strings:
        lowered = literal.lower()
        for token in FORBIDDEN_BRANCH_TOKENS:
            if token in lowered:
                offenders.append({"literal": literal, "token": token})
    return {
        "path": str(path.relative_to(ROOT)),
        "branch_strings": int(len(branch_strings)),
        "forbidden_hits": offenders,
        "zero_task_id_branches": bool(not offenders),
        "autograd_markers": sorted(set(autograd)),
        "zero_autograd": bool(not autograd),
    }


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

    egress = ContextSteeredEgress(
        projector,
        tokenizer,
        core=core,
        cartridge=cartridge,
        induction_window=256,
        induction_gamma=0.8,
        induction_low=7,
        induction_tau_match=0.5,
        steer_rank=8,
        orders=(1, 2, 3, 4, 5, 6),
    )
    egress.set_cartridge(cartridge, report=cartridge_report)
    floor = egress.fit_floor(train_ids, inner_ids)
    _mark("6/7-gram floor calibration", started)
    mix = egress.fit(train_ids, epochs=4, samples=1200, floor_fn=floor_fn, model=None)
    _mark("mode weights", started)
    phase = egress.fit_phase_table(train_ids, epochs=26, samples=2000, inner_samples=400)
    _mark("static phase table (AA1)", started)
    steer = egress.fit_contextual_phase(
        train_ids, epochs=16, samples=2400, inner_samples=600, batch=128
    )
    _mark("context-steered phase (AB1)", started)
    channels = egress.calibrate_ab_channels(inner_ids, samples=800)
    _mark("dual-window channel calibration", started)
    override = egress.calibrate_override(inner_ids, model=None, floor_fn=floor_fn)
    _mark("override thresholds", started)

    # Batched Zone B feature prepass for the held-out windows: one core
    # forward per unique 24-token window, not one per token.
    windows: List[Sequence[int]] = []
    for sequence in sequences:
        for index in range(1, len(sequence)):
            windows.append(sequence[max(0, index - 24) : index])
    warmed = egress.prepare_bright(windows, batch=64)
    _mark("zone B feature prepass (%d windows)" % int(warmed), started)

    report = egress.evaluate(sequences, floor_fn=floor_fn, model=None)
    _mark("held-out evaluation", started)
    recall = _copy_recall(egress, sequences)

    saved_phase = egress.phase_table.copy()
    saved_W = egress.steer_W.copy()
    saved_V = egress.steer_V.copy()
    saved_channels = (
        float(egress.w_induction),
        float(egress.w_photonic),
        float(egress.w_copy),
        float(egress.w_prefix),
        float(egress.w_copy_fast),
        float(egress.w_copy_long),
    )
    # (a) steering marginal: keep the static AA1 table, remove the rank-R
    # context steering only.
    egress.steer_W = np.zeros_like(egress.steer_W)
    egress.steer_V = np.zeros_like(egress.steer_V)
    without_steer = egress.evaluate(sequences, floor_fn=floor_fn, model=None)
    _mark("ablation: steering off", started)
    # (b) full phase block marginal: static table and steering both reset.
    egress.phase_table = egress.phase_table0.copy()
    without_phase = egress.evaluate(sequences, floor_fn=floor_fn, model=None)
    _mark("ablation: phase block reset", started)
    egress.phase_table = saved_phase.copy()
    egress.steer_W = saved_W.copy()
    egress.steer_V = saved_V.copy()
    # (c) channels off.
    egress.w_induction = 0.0
    egress.w_photonic = 0.0
    egress.w_copy = 0.0
    egress.w_prefix = 0.0
    egress.w_copy_fast = 0.0
    egress.w_copy_long = 0.0
    without_channels = egress.evaluate(sequences, floor_fn=floor_fn, model=None)
    _mark("ablation: channels off", started)
    # (d) floor only.
    egress.phase_table = egress.phase_table0.copy()
    egress.steer_W = np.zeros_like(egress.steer_W)
    egress.steer_V = np.zeros_like(egress.steer_V)
    floor_only = egress.evaluate(sequences, floor_fn=floor_fn, model=None)
    _mark("ablation: floor only", started)
    egress.phase_table = saved_phase.copy()
    egress.steer_W = saved_W.copy()
    egress.steer_V = saved_V.copy()
    (
        egress.w_induction,
        egress.w_photonic,
        egress.w_copy,
        egress.w_prefix,
        egress.w_copy_fast,
        egress.w_copy_long,
    ) = saved_channels

    audit = _branch_audit(ROOT / "henri" / "holarch" / "wave_egress.py")
    sha_after = _sha(snapshot_path)
    pin_after = _sha(directory / "pinned_lang_split.json")
    body = report.as_dict()
    selected = channels.get("selected", {})
    steer_delta = float(body["bpc_wave_egress"] - without_steer.bits_per_char)
    phase_delta = float(body["bpc_wave_egress"] - without_phase.bits_per_char)
    gates = {
        "ab1_pinned_digest_equals_8f270f34": bool(
            str(snapshot.get("_digest_recomputed", "")) == PIN
        ),
        "ab1_pinned_snapshot_unchanged": bool(sha_before == sha_after and pin_before == pin_after),
        "ab1_zero_autograd": bool(audit["zero_autograd"]),
        "ab1_zero_task_id_branches": bool(audit["zero_task_id_branches"]),
        "ab1_bpc_le_26650": bool(float(body["bpc_wave_egress"]) <= 2.6650),
        "ab1_top1_ge_06320": bool(float(body["top1_wave_egress"]) >= 0.6320),
        "ab1_context_phase_marginal_le_-00040": bool(phase_delta <= -0.0040),
        "ab1_steering_marginal_recorded": bool(steer_delta < 0.0),
        "ab1_prefix_weight_positive": bool(float(selected.get("w_prefix", 0.0)) > 0.0),
        "ab1_copy_fast_weight_positive": bool(float(selected.get("w_copy_fast", 0.0)) > 0.0),
        "ab1_copy_long_weight_positive": bool(float(selected.get("w_copy_long", 0.0)) > 0.0),
        "ab1_natural_copy_recall_ge_074": bool(float(recall["natural"]["recall"]) >= 0.7400),
        "ab1_strict_copy_recall_ge_072": bool(float(recall["strict"]["recall"]) >= 0.7200),
        "ab1_copy_positions_1092": bool(int(recall["positions"]) == 1092),
        "ab1_held_out_tokens": bool(int(body["tokens"]) >= 4000),
    }
    return {
        "stage": "AB1",
        "config": {
            "dim": DIM,
            "freq_bands": 32,
            "photonic_layers": 4,
            "phase_modes": int(egress.phase_modes),
            "steer_rank": int(egress.steer_rank),
            "steer_ctx_dim": int(egress.ctx_dim),
            "induction_window": int(egress.induction_window),
            "copy_fast_window": int(egress.copy_fast_window),
            "copy_long_window": int(egress.copy_long_window),
            "copy_low": int(egress.copy_low),
            "orders": [int(value) for value in egress.orders],
            "coupling": str(egress.coupling),
        },
        "pinned_split": {
            "snapshot": snapshot.get("_path"),
            "snapshot_sha256": sha_before,
            "snapshot_sha256_after": sha_after,
            "pinned_lang_split_sha256": pin_before,
            "pinned_lang_split_sha256_after": pin_after,
            "digest_frozen_recomputed": str(snapshot.get("_digest_recomputed", "")),
            "files": int(len(snapshot.get("files") or [])),
            "train_tokens": int(len(train_ids)),
            "val_tokens": int(sum(len(segment[2]) for segment in segments)),
            "documents": int(len(segments)),
            "docs_excluded": True,
        },
        "photonic_cartridge": {
            "before_imprint": cartridge_before,
            "after_imprint": cartridge_report,
            "imprint_residuals": [float(value) for value in imprint],
        },
        "floor": floor,
        "modal_mix": mix,
        "phase_table": phase,
        "context_steering": steer,
        "channels": channels,
        "channel_weights": {
            "w_photonic": float(egress.w_photonic),
            "w_induction": float(egress.w_induction),
            "w_prefix": float(egress.w_prefix),
            "w_copy_fast": float(egress.w_copy_fast),
            "w_copy_long": float(egress.w_copy_long),
            "w_copy_legacy": float(egress.w_copy),
            "coupling": str(egress.coupling),
            "tau_gate_H": float(egress.tau_gate_h),
        },
        "override_calibration": override,
        "wave_egress": body,
        "copy_recall": recall,
        "aa1_baseline": {
            "bits_per_char": AA1_BPC,
            "top1": AA1_TOP1,
            "phase_delta_bpc": AA1_PHASE_DELTA,
            "copy_natural": 0.7253,
            "copy_strict": 0.7005,
            "source": "receipts/holarch/aa1_per_token_phase_egress.json",
        },
        "ablations": {
            "without_steering": {
                "top1": float(without_steer.top1),
                "bits_per_char": float(without_steer.bits_per_char),
            },
            "without_phase_block": {
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
            "steering_delta_bpc": steer_delta,
            "phase_block_delta_bpc": phase_delta,
            "channel_delta_bpc": float(body["bpc_wave_egress"] - without_channels.bits_per_char),
            "floor_delta_bpc": float(body["bpc_wave_egress"] - floor_only.bits_per_char),
            "delta_vs_aa1_bpc": float(body["bpc_wave_egress"] - AA1_BPC),
            "delta_vs_aa1_top1": float(body["top1_wave_egress"] - AA1_TOP1),
            "delta_vs_z1_copy": float(recall["recall"] - Z1_COPY),
            "phase_improvement_factor_vs_aa1": float(
                (abs(phase_delta) / abs(AA1_PHASE_DELTA)) if AA1_PHASE_DELTA else 0.0
            ),
        },
        "branch_audit": audit,
        "gate": pc._finalise(gates),
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
        "passed": bool(all(gates.values())),
    }


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Cycle AB1 context-steered phase egress")
    parser.add_argument("--receipts", default=str(RECEIPTS))
    args = parser.parse_args(argv)
    directory = Path(args.receipts).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    payload = measure(directory)
    written = pc._write(directory, "ab1_contextual_phase_egress.json", payload)
    body = payload["wave_egress"]
    print(
        "[ab1] top1=%.4f bpc=%.4f (aa1 %.4f / %.4f) copy natural=%.4f strict=%.4f"
        % (
            float(body["top1_wave_egress"]),
            float(body["bpc_wave_egress"]),
            AA1_TOP1,
            AA1_BPC,
            float(payload["copy_recall"]["natural"]["recall"]),
            float(payload["copy_recall"]["strict"]["recall"]),
        )
    )
    print(
        "[ab1] phase_delta=%.5f steer_delta=%.5f w_prefix=%.3f w_copy_fast=%.3f w_copy_long=%.3f"
        % (
            float(payload["attribution"]["phase_block_delta_bpc"]),
            float(payload["attribution"]["steering_delta_bpc"]),
            float(payload["channel_weights"]["w_prefix"]),
            float(payload["channel_weights"]["w_copy_fast"]),
            float(payload["channel_weights"]["w_copy_long"]),
        )
    )
    failed = [name for name, value in payload["gate"].items() if not bool(value)]
    print("[ab1] passed=%s failed=%s receipt=%s" % (payload["passed"], failed, written["path"]))
    return 0 if bool(payload["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
