"""Cycle Y1: photonic-coupled wave egress and Born-rule phase breaking.

    python scripts/run_holarch_y1_photonic_wave_egress.py

Measures the same W2 34-document stratified held-out split as X2 with three
Y1 organs mounted:

  1. complex quadrature resonances (both quadratures on every mode),
  2. deterministic incommensurate per-mode / per-slot seed phases,
  3. the closed-form non-destructive subword optical cartridge on Zone B.

The count floor, the split, the inner-slice calibration protocol and the
override thresholds are the X2 ones.  Nothing is relaxed; the receipt
records the X2 numbers side by side with the Y1 numbers.
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
X2_TOP1 = 0.6110
X2_BPC = 2.8043
X2_NET = 243



def capacity_probe(
    d_slots=(64, 128, 256, 512, 1024),
    *,
    vocab: int = 64,
    window: int = 24,
    train: int = 900,
    test: int = 300,
    seed: int = 20261111,
    ridge: float = 1e-2,
    plates: int = 4,
    kerr: float = 0.25,
) -> Dict[str, Any]:
    """Minimum waveguide channel count for optical long-range retrieval.

    The task is the one a suffix bank structurally cannot do: recover the
    OLDEST token of a 24-token window from the post-photonic field.  A
    6-order count model sees only the last six tokens, so its top-1 on this
    target is chance by construction; the optical field is

        psi = sum_i chi_{w_i} * p_i

    superposed on four-slot unit codevectors with per-position phases, run
    through a four-plate phase-only stack with the angular-spectrum chirp and
    a Kerr self-phase, and read by a CLOSED-FORM ridge probe.  The matched
    filter on the pre-photonic wave is the ceiling.  The reported boundary is
    the smallest D whose probe top-1 reaches 0.99.
    """
    rng = np.random.default_rng(int(seed))
    span = 4 * max(d_slots)
    chi_all = rng.standard_normal((vocab, span)) + 1j * rng.standard_normal((vocab, span))
    pos_all = rng.standard_normal((window, span)) + 1j * rng.standard_normal((window, span))
    masks = [rng.uniform(-np.pi, np.pi, span) for _ in range(int(plates))]
    chirp = rng.uniform(-np.pi, np.pi, span)
    n = int(train) + int(test)
    stream = rng.integers(0, vocab, size=n + window + 1)
    rows: List[Dict[str, Any]] = []
    for d_slot in d_slots:
        dim = 4 * int(d_slot)
        chi = chi_all[:, :dim].copy()
        pos = pos_all[:, :dim].copy()
        chi /= np.linalg.norm(chi, axis=1, keepdims=True)
        pos /= np.linalg.norm(pos, axis=1, keepdims=True)
        stack = [mask[:dim] for mask in masks]
        phase_chirp = chirp[:dim]

        def optic(psi: np.ndarray) -> np.ndarray:
            field = psi.copy()
            for mask in stack:
                field = field * np.exp(1j * mask)
                field = np.fft.ifft(np.fft.fft(field) * np.exp(1j * phase_chirp))
                field = field * np.exp(1j * kerr * np.abs(field) ** 2)
            return field / max(float(np.linalg.norm(field)), 1e-12)

        feats = np.zeros((n, dim), dtype=np.complex128)
        targets = np.zeros(n, dtype=np.int64)
        for index in range(n):
            window_tokens = stream[index : index + window]
            targets[index] = int(window_tokens[0])
            psi = np.sum(chi[window_tokens] * pos, axis=0)
            feats[index] = optic(psi / max(float(np.linalg.norm(psi)), 1e-12))
        x_train, x_test = feats[:train], feats[train:]
        y_train, y_test = targets[:train], targets[train:]
        one_hot = np.zeros((train, vocab), dtype=np.complex128)
        one_hot[np.arange(train), y_train] = 1.0
        gram = x_train.conj() @ x_train.T
        lam = ridge * max(float(np.trace(gram).real) / train, 1e-12)
        alpha = np.linalg.solve(gram + lam * np.eye(train, dtype=np.complex128), one_hot)
        weights = x_train.conj().T @ alpha
        predicted = np.argmax(np.real(x_test @ weights), axis=1)
        probe_top1 = float(np.mean(predicted == y_test))
        ideal = 0
        for index in range(train, n):
            window_tokens = stream[index : index + window]
            psi = np.sum(chi[window_tokens] * pos, axis=0)
            value = int(np.argmax(np.abs(chi.conj() @ (psi * np.conj(pos[0])))))
            ideal += int(value == int(window_tokens[0]))
        counts: Dict[Any, np.ndarray] = {}
        for index in range(6, train):
            key = tuple(int(x) for x in stream[index - 6 : index])
            counts.setdefault(key, np.zeros(vocab))[int(stream[index])] += 1
        hits = 0
        for index in range(train, n):
            key = tuple(int(x) for x in stream[index - 6 : index])
            row = counts.get(key)
            guess = int(np.argmax(row)) if row is not None else -1
            hits += int(guess == int(stream[index]))
        rows.append(
            {
                "d_slot": int(d_slot),
                "dim": int(dim),
                "probe_top1": float(probe_top1),
                "ideal_matched_filter_top1": float(ideal / test),
                "six_order_suffix_top1": float(hits / max(n - train, 1)),
                "chance_top1": float(1.0 / vocab),
            }
        )
    boundary = next(
        (row["dim"] for row in rows if float(row["probe_top1"]) >= 0.99), 0
    )
    return {
        "task": "recover the oldest token of a 24-token window",
        "rows": rows,
        "capacity_boundary_dim": int(boundary),
        "deployed_dim": 1024,
        "deployed_probe_top1": float(
            next((row["probe_top1"] for row in rows if int(row["dim"]) == 1024), 0.0)
        ),
        "suffix_blind": bool(all(float(row["six_order_suffix_top1"]) <= 0.05 for row in rows)),
        "plates": int(plates),
        "kerr": float(kerr),
        "note": (
            "closed-form ridge probe on the post-photonic field; the suffix bank is "
            "structurally blind to the target, so any positive value is optical"
        ),
    }

def _inner_cross_entropy(egress, ids: List[int], *, model, photonic: bool, context: int = 24,
                         samples: int = 320) -> Dict[str, float]:
    """Mean NLL / top-1 of the egress on an inner slice, photonic on or off."""
    sequence = [int(token) for token in ids]
    indices = list(range(context, len(sequence) - 1))
    stride = max(1, len(indices) // max(int(samples), 1))
    indices = indices[::stride][: int(samples)]
    nll = 0.0
    top1 = 0
    for index in indices:
        window = sequence[max(0, index - context) : index]
        probability = egress.distribution(window, model if photonic else None)
        truth = int(sequence[index])
        nll -= float(np.log(max(float(probability[truth]), 1e-12)))
        top1 += int(int(np.argmax(probability)) == truth)
    tokens = max(len(indices), 1)
    return {
        "tokens": int(tokens),
        "nll": float(nll / tokens),
        "bits_per_char": float(nll / tokens / math.log(2.0)),
        "top1": float(top1 / tokens),
    }


def _mark(label: str, started: float) -> None:
    print("[y1] %-34s %8.1f s" % (label, time.perf_counter() - started), flush=True)


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
    generator.suffix_weight = 0.0
    generator.recency_weight = 0.0
    generator.document_weight = 0.0
    generator.beta_star = 0.0

    def floor_fn(context):
        return generator.floor_probability(context)

    cap = max(64, 6000 // max(len(segments), 1))
    sequences = [list(int(token) for token in val[:cap]) for _name, _train, val in segments]

    checkpoint = find_checkpoint(None)
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg = HolarchConfig(**payload["config"])
    workdir = directory / "_work" / "y1"
    workdir.mkdir(parents=True, exist_ok=True)
    memory, _info = open_clean_room_memory(cfg, checkpoint, workdir, prefix="y1_wave")
    model = HenriHolarch(cfg, memory=memory)
    load_checkpoint(checkpoint, model=model)
    model.eval()

    cut = int(len(train_ids) * 0.85)
    inner_ids = train_ids[cut:]
    fit_ids = train_ids[:cut]

    # ---- the X2 organ, unchanged, on the same split (the side-by-side bar).
    x2 = WaveEgressHead(projector, tokenizer, quadrature=False)
    fit_started = time.perf_counter()
    x2_fit = x2.fit(train_ids, epochs=4, samples=1200, floor_fn=floor_fn, model=None)
    x2_photonic = x2.calibrate_photonic(inner_ids, model=model, floor_fn=floor_fn)
    x2_override = x2.calibrate_override(inner_ids, model=model, floor_fn=floor_fn)
    _mark("x2 fit+calibrate", fit_started)
    x2_eval = time.perf_counter()
    x2_report = x2.evaluate(sequences, floor_fn=floor_fn, model=model)
    x2_seconds = float(time.perf_counter() - fit_started)
    _mark("x2 evaluate", x2_eval)
    print("[y1] x2 top1=%.4f bpc=%.4f net=%d" % (
        x2_report.top1, x2_report.bits_per_char, x2_report.override_net_win_delta), flush=True)

    # ---- the Y1 organ: quadrature on, cartridge fitted, phase torque learned.
    egress = WaveEgressHead(projector, tokenizer, quadrature=True)
    y_started = time.perf_counter()
    fit = egress.fit(train_ids, epochs=4, samples=1200, floor_fn=floor_fn, model=None)
    _mark("y1 fit", y_started)
    cartridge = egress.fit_cartridge(fit_ids, model, samples=600, context=24, ridge=1e-1)
    _mark("y1 cartridge", y_started)
    photonic = egress.calibrate_photonic(
        inner_ids, model=model, floor_fn=floor_fn,
        grid=(0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30), samples=200,
    )
    _mark("y1 photonic calibrate 1", y_started)
    phase = egress.fit_quadrature(
        fit_ids, model=model, epochs=3, samples=480, floor_fn=floor_fn,
        eta=0.02, eta_negative=0.02, clip=0.05,
    )
    _mark("y1 phase torque", y_started)
    # Re-select the photonic weight after the phase torque moved the mixture,
    # still on the inner slice only.
    photonic_after = egress.calibrate_photonic(
        inner_ids, model=model, floor_fn=floor_fn,
        grid=(0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30), samples=200,
    )
    _mark("y1 photonic calibrate 2", y_started)
    suffix_only = _inner_cross_entropy(egress, inner_ids, model=model, photonic=False)
    with_photonic = _inner_cross_entropy(egress, inner_ids, model=model, photonic=True)
    _mark("y1 inner ce ablation", y_started)
    override = egress.calibrate_override(inner_ids, model=model, floor_fn=floor_fn)
    _mark("y1 override calibrate", y_started)
    report = egress.evaluate(sequences, floor_fn=floor_fn, model=model)
    _mark("y1 held-out evaluate", y_started)

    # Attribution: the same learned phi with the quadrature support removed.
    saved_phase = egress.phase.copy()
    egress.phase = egress.phase0.copy()
    seed_report = egress.evaluate(sequences, floor_fn=floor_fn, model=model)
    egress.phase = saved_phase
    _mark("y1 seed-phase evaluate", y_started)
    saved_quadrature = egress.quadrature
    egress.quadrature = False
    real_report = egress.evaluate(sequences, floor_fn=floor_fn, model=model)
    egress.quadrature = saved_quadrature
    y_seconds = float(time.perf_counter() - y_started)
    _mark("y1 no-quadrature evaluate", y_started)

    source = (ROOT / "henri" / "holarch" / "wave_egress.py").read_text(encoding="utf-8")
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

    probe_started = time.perf_counter()
    capacity = capacity_probe()
    _mark("y1 capacity probe", probe_started)

    body = report.as_dict()
    phase_displacement = egress.phase_displacement()
    inner_bits = float(with_photonic["bits_per_char"])
    suffix_bits = float(suffix_only["bits_per_char"])
    inner_nll = float(with_photonic["nll"])
    suffix_nll = float(suffix_only["nll"])
    phase_attr = dict(egress.phase_attribution)
    gates = {
        "y1_module_present": bool((ROOT / "henri" / "holarch" / "wave_egress.py").exists()),
        "y1_autograd_unused": bool(not markers and not report.autograd_used),
        "y1_quadrature_enabled": bool(egress.quadrature),
        "y1_photonic_weight_positive": bool(float(egress.photonic_weight) > 0.0),
        "y1_photonic_improves_inner_ce": bool(inner_nll < suffix_nll),
        "y1_phase_displacement": bool(float(phase_displacement) >= 0.05),
        "y1_phase_attribution_nonzero": bool(
            abs(float(phase_attr.get("inner_nll_gain", 0.0))) > 0.0
            and int(phase_attr.get("updates", 0)) > 0
        ),
        "y1_held_out_top1": bool(float(report.top1) >= 0.6150),
        "y1_bits_per_char_beat_x2": bool(float(report.bits_per_char) < float(X2_BPC)),
        "y1_net_overrides": bool(int(report.override_net_win_delta) > 250),
        "y1_top1_beats_floor": bool(float(report.top1) > float(report.base_top1)),
        "y1_bpc_beats_floor": bool(float(report.bits_per_char) < float(report.base_bits_per_char)),
        "y1_gap_purged": bool(gap_ok),
        "y1_held_out_tokens": bool(int(report.tokens) >= 4000),
        "y1_cartridge_positive_cosine": bool(
            float(cartridge.get("mean_aligned_cosine", 0.0))
            > float(cartridge.get("mean_raw_cosine", 1.0))
        ),
        "y1_capacity_probe_ran": bool(len(capacity.get("rows") or []) == 5),
        "y1_capacity_probe_monotone": bool(
            all(
                capacity["rows"][index]["probe_top1"]
                <= capacity["rows"][index + 1]["probe_top1"] + 1e-9
                for index in range(len(capacity["rows"]) - 1)
            )
        ),
        "y1_capacity_suffix_blind": bool(capacity.get("suffix_blind", False)),
        "y1_capacity_deployed_retrieval": bool(
            float(capacity.get("deployed_probe_top1", 0.0)) > 0.5
        ),
    }
    return {
        "stage": "Y1",
        "split": {
            "documents": int(len(segments)),
            "train_tokens": int(len(train_ids)),
            "inner_tokens": int(len(inner_ids)),
            "held_out_tokens": int(report.tokens),
            "gap_tokens": 6,
            "boundary_purged": bool(gap_ok),
            "corpus_digest": str(getattr(corpus, "digest", "")),
        },
        "wave_egress": body,
        "x2_side_by_side": {
            "top1": float(x2_report.top1),
            "bits_per_char": float(x2_report.bits_per_char),
            "net_overrides": int(x2_report.override_net_win_delta),
            "photonic_weight": float(x2.photonic_weight),
            "receipt_constants": {"top1": float(X2_TOP1), "bpc": float(X2_BPC),
                                  "net": int(X2_NET)},
            "seconds": x2_seconds,
        },
        "photonic_cartridge": dict(cartridge),
        "capacity_probe": capacity,
        "photonic_calibration": photonic,
        "photonic_calibration_after_phase": photonic_after,
        "inner_ce_ablation": {
            "suffix_only_6mode": suffix_only,
            "with_photonic": with_photonic,
            "delta_bits_per_char": float(inner_bits - suffix_bits),
            "delta_nll": float(inner_nll - suffix_nll),
            "selected_weight": float(egress.photonic_weight),
        },
        "phase": {**phase_attr, "phase": [[float(v) for v in row] for row in egress.phase],
                  "phase0": [[float(v) for v in row] for row in egress.phase0],
                  "displacement": float(phase_displacement)},
        "attribution": {
            "wave_quadrature": {"top1": float(report.top1),
                                "bits_per_char": float(report.bits_per_char)},
            "wave_seed_phase": {"top1": float(seed_report.top1),
                                "bits_per_char": float(seed_report.bits_per_char)},
            "wave_no_quadrature": {"top1": float(real_report.top1),
                                   "bits_per_char": float(real_report.bits_per_char)},
            "quadrature_delta_top1": float(report.top1 - real_report.top1),
            "quadrature_delta_bpc": float(report.bits_per_char - real_report.bits_per_char),
            "phase_learning_delta_top1": float(report.top1 - seed_report.top1),
            "phase_learning_delta_bpc": float(report.bits_per_char - seed_report.bits_per_char),
        },
        "fit": {
            "epochs": int(fit.get("epochs", 0)),
            "updates": int(fit.get("updates", 0)),
            "theta": list(fit.get("theta") or []),
            "mode_gain": list(fit.get("mode_gain") or []),
            "seconds": y_seconds,
            "override_calibration": override,
            "photonic_weight": float(egress.photonic_weight),
        },
        "autograd_markers": markers,
        "bar_policy": {
            "note": ("X2's sealed numbers are the bar (top1 0.6150 target, bpc < 2.8043, "
                     "net > 250); the photonic weight and the override thresholds are "
                     "selected on an inner slice of the training split only, and the "
                     "cartridge is fitted on the training split only."),
        },
        "gate": pc._finalise(gates),
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
        "passed": bool(all(gates.values())),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Cycle Y1 photonic wave egress")
    parser.add_argument("--receipts", default=str(RECEIPTS))
    args = parser.parse_args(argv)
    directory = Path(args.receipts)
    directory.mkdir(parents=True, exist_ok=True)
    payload = measure(directory)
    written = pc._write(directory, "y1_photonic_wave_egress.json", payload)
    body = payload["wave_egress"]
    print("[y1] top1 wave=%.4f floor=%.4f | bpc wave=%.4f floor=%.4f" % (
        float(body["top1_wave_egress"]), float(body["top1_count_floor"]),
        float(body["bpc_wave_egress"]), float(body["bpc_count_floor"])))
    print("[y1] overrides=%d wins=%d losses=%d net=%d | photonic=%.3f | phase=%.4f rad" % (
        int(body["unclamped_override_count"]), int(body["override_wins"]),
        int(body["override_losses"]), int(body["override_net_win_delta"]),
        float(payload["fit"]["photonic_weight"]), float(payload["phase"]["displacement"])))
    print("[y1] inner ce suffix=%.6f photonic=%.6f" % (
        float(payload["inner_ce_ablation"]["suffix_only_6mode"]["nll"]),
        float(payload["inner_ce_ablation"]["with_photonic"]["nll"])))
    failed = [name for name, value in payload["gate"].items() if not bool(value)]
    print("[y1] passed=%s failed=%s receipt=%s" % (
        payload["passed"], failed, written["path"]))
    return 0 if bool(payload["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
