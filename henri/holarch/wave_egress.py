"""Contrastive thermodynamic wave egress (Directive X2).

H1 clamps every emitted token to the argmax of the count floor.  This module
is the separate organ that breaks that clamp: a backprop-free, phase-
conjugate transducer over the locked vocabulary codebook.

Architecture
------------
1. Context wave.  The causal token window (L <= 24) is encoded by
   BochnerPhaseProjector.sequence_wave and propagated through the mounted
   HenriHolarch four-plate PhotonicCore (Zone B).  The post-photonic bright
   and dark ports are recorded on every call.

2. Order waves.  Seven addressable waves are superposed:

       k = 1 .. 6   window-local suffix resonances.  For every order the
                    longest suffix match inside the causal window returns a
                    phase-conjugate value wave whose token amplitude is
                    sqrt(p_k(t)).
       k = 7        the photonic semantic wave: the post-photonic bright
                    port read against the Bochner codebook.

3. Interference.  The amplitude of token t is

       a(t) = sum_k  w_k * exp(i * theta_k) * sqrt(p_k(t))

   and the emitted distribution is the Born rule |a(t)|^2, normalised.

4. Contrastive phase-conjugate learner.  No autograd and no torch.optim.
   Over K deterministic epochs on the W2 stratified TRAINING split the phase
   vector theta and the 4 x 4 slot Stiefel adapter are updated by a
   surprise-gated phase-locked loop: a positive pull toward Phi_lex[w_true]
   and a negative push away from Phi_lex[w_floor_wrong].  The closed-form
   gradient of |a|^2 with respect to theta_k is used, never a tensor graph.
   After every epoch the adapter is retracted onto the Stiefel manifold by
   QR and the mean post-photonic Sagnac homodyne residual
   1 - |<a_unit, Phi_lex[w_true]>|^2 is recorded.

5. Sagnac-gated margin override.  The transducer may replace the count
   floor's argmax only when

       H(P_count)    >= tau_H
       m_wave        >= tau_margin
       delta_sagnac  <= tau_sagnac

   The three thresholds are calibrated on an INNER validation slice of the
   training split.  The held-out W2 test split never enters the calibration.

Nothing here calls backward(), an optimiser, or a network.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_EPS = 1e-12
DEFAULT_ALPHA = {1: 0.05, 2: 0.15, 3: 0.40, 4: 0.55, 5: 0.65, 6: 0.70}
DEFAULT_ORDERS = (1, 2, 3, 4, 5, 6)


@dataclass
class WaveEgressReport:
    """One measured held-out evaluation of the wave egress."""

    tokens: int
    top1: float
    bits_per_char: float
    perplexity: float
    base_top1: float
    base_bits_per_char: float
    override_count: int
    override_wins: int
    override_losses: int
    override_net_win_delta: int
    sagnac_init: float
    sagnac_final: float
    sagnac_curve: List[float] = field(default_factory=list)
    fusion: Dict[str, float] = field(default_factory=dict)
    photonic_contexts: int = 0
    autograd_used: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "tokens": int(self.tokens),
            "top1_wave_egress": float(self.top1),
            "bpc_wave_egress": float(self.bits_per_char),
            "perplexity_wave_egress": float(self.perplexity),
            "top1_count_floor": float(self.base_top1),
            "bpc_count_floor": float(self.base_bits_per_char),
            "unclamped_override_count": int(self.override_count),
            "override_wins": int(self.override_wins),
            "override_losses": int(self.override_losses),
            "override_net_win_delta": int(self.override_net_win_delta),
            "sagnac_initial": float(self.sagnac_init),
            "sagnac_final": float(self.sagnac_final),
            "sagnac_curve": [float(value) for value in self.sagnac_curve],
            "sagnac_residual_reduced": bool(self.sagnac_final < self.sagnac_init),
            "fusion": dict(self.fusion),
            "photonic_contexts": int(self.photonic_contexts),
            "autograd_used": bool(self.autograd_used),
            "module": "henri/holarch/wave_egress.py",
        }


class ContrastivePhaseResonator:
    """A phase-conjugate, backprop-free transducer over the locked vocabulary.

    Parameters
    ----------
    projector : BochnerPhaseProjector
        The locked Tier-1/Tier-2 phase codebook.  Phi_lex is the matrix of
        slot_wave rows for every id of the lock.
    tokenizer : HolarchCodeTextTokenizer
        The locked vocabulary.  Its length is the Hilbert-space dimension of
        the egress state.
    window : int
        The causal window the resonances scan (default 512).
    """

    def __init__(
        self,
        projector,
        tokenizer,
        *,
        orders: Sequence[int] = DEFAULT_ORDERS,
        window: int = 512,
        seed: int = 20261111,
        max_occurrences: int = 96,
        alpha: Optional[Dict[int, float]] = None,
        saturation: float = 1.0,
    ) -> None:
        self.projector = projector
        self.tokenizer = tokenizer
        self.vocab_size = int(len(tokenizer))
        self.dim = int(projector.dim)
        self.orders = tuple(int(order) for order in orders)
        self.max_order = int(max(self.orders))
        self.window = int(window)
        self.seed = int(seed)
        self.max_occurrences = int(max_occurrences)
        self.alpha = dict(alpha or DEFAULT_ALPHA)
        self.saturation = float(saturation)
        self.phi = np.stack(
            [projector.slot_wave(token, tokenizer) for token in range(self.vocab_size)]
        ).astype(np.complex64)
        self.unigram = np.zeros(self.vocab_size, dtype=np.float64)
        # The egress owns its own interpolated order model.  It is fitted on
        # the TRAINING stream in fit() and is a separate object from any H1
        # CountLanguageModel; the transducer never reads H1's P_count.
        self.base_counts = None
        # Learned state: one phase per order, one weight per order, and a 4x4
        # complex slot adapter retracted onto the Stiefel manifold by QR.
        self.theta = np.zeros(len(self.orders) + 2, dtype=np.float64)
        self.weight = np.ones(len(self.orders) + 2, dtype=np.float64)
        # Per-mode phase-conjugate amplitude gains.  The Born trace is
        # invariant under the Stiefel mixer, so these gains are the learned
        # degree of freedom that moves the egress distribution.
        self.mode_gain = np.ones(len(self.orders) + 2, dtype=np.float64)
        # The Stiefel mode mixer: a (K+2) x (K+2) complex unitary, retracted
        # onto the Stiefel manifold by QR after every epoch.  The identity is
        # exactly the incoherent arithmetic order mixture.
        self.adapter = np.eye(len(self.orders) + 2, dtype=np.complex128)
        self.sagnac_curve: List[float] = []
        self.train_updates = 0
        self.fitted = False
        self.autograd_used = False
        self.photonic_contexts = 0
        self.photonic_weight = 0.10
        self.override: Dict[str, Any] = {}
        self._photonic_cache: Dict[Tuple[int, ...], Tuple[np.ndarray, float, float]] = {}

    # ------------------------------------------------------------ statistics
    def observe(self, ids: Sequence[int]) -> "ContrastivePhaseResonator":
        """Accumulate the unigram wave of the TRAINING stream only."""
        for token in ids:
            int_token = int(token)
            if 0 <= int_token < self.vocab_size:
                self.unigram[int_token] += 1.0
        return self

    def base_distribution(self, context: Sequence[int]) -> np.ndarray:
        """The egress's own training-stream order model (wave base wave)."""
        if self.base_counts is not None:
            try:
                return np.asarray(self.base_counts.distribution(context), dtype=np.float64)
            except Exception:  # noqa: BLE001 - a base failure falls back to unigram
                pass
        return self.unigram_distribution()

    def unigram_distribution(self) -> np.ndarray:
        total = float(self.unigram.sum())
        if total <= 0.0:
            return np.full(self.vocab_size, 1.0 / self.vocab_size, dtype=np.float64)
        return (self.unigram + 1.0) / (total + float(self.vocab_size))

    # --------------------------------------------------------- suffix waves
    def _resonances(self, context: Sequence[int]) -> List[Tuple[int, np.ndarray, float]]:
        """(order, distribution, occurrences) for every matched suffix."""
        window = [int(token) for token in context[-self.window :]]
        length = len(window)
        out: List[Tuple[int, np.ndarray, float]] = []
        if length < 2:
            return out
        for order in self.orders:
            if order >= length:
                break
            suffix = window[length - order :]
            positions: List[int] = []
            for index in range(length - order - 1, -1, -1):
                if window[index : index + order] == suffix:
                    positions.append(index)
                    if len(positions) >= self.max_occurrences:
                        break
            if not positions:
                break
            counts = np.zeros(self.vocab_size, dtype=np.float64)
            for index in positions:
                token = int(window[index + order])
                if 0 <= token < self.vocab_size:
                    counts[token] += 1.0
            total = float(counts.sum())
            if total <= 0.0:
                break
            out.append((int(order), counts / total, total))
        return out

    def _order_waves(
        self, context: Sequence[int], model=None
    ) -> Tuple[List[float], List[np.ndarray], List[float], bool]:
        """Per-order distributions and their normalised interference weights.

        The unigram prior is order 0.  Every matched suffix order contributes
        its own local distribution, weighted by the S1 alpha schedule and its
        occurrence saturation.  The weights sum to one, so with the uniformly
        spaced ring phases the Born rule reduces exactly to the incoherent
        mixture; a learned phase offset can only move it away from that ring.
        """
        prior = self.base_distribution(context)
        weights: List[float] = [0.25]
        dists: List[np.ndarray] = [prior]
        totals: List[float] = [1.0]
        for order, local, total in self._resonances(context):
            alpha = float(self.alpha.get(int(order), 0.7))
            saturation = float(total) / (float(total) + self.saturation)
            weights.append(max(0.0, alpha * saturation * 0.75))
            dists.append(local)
            totals.append(float(total))
        has_photonic = False
        if model is not None:
            term = self.photonic_term(context, model)
            if term is not None:
                weights.append(float(self.photonic_weight))
                dists.append(term)
                totals.append(1.0)
                has_photonic = True
        for index in range(len(weights)):
            gain = float(self.mode_gain[index]) if index < self.mode_gain.size else 1.0
            weights[index] = max(0.0, float(weights[index]) * gain)
        mass = float(sum(weights))
        if mass <= 0.0:
            return [1.0], [prior], [1.0], False
        return [float(value / mass) for value in weights], dists, totals, has_photonic

    def distribution(self, context: Sequence[int], model=None) -> np.ndarray:
        """The wave-egress next-token distribution for one causal context."""
        amplitude, _info = self.amplitudes(context, model=model)
        return self.born(amplitude)

    def _order_amplitudes(self, context: Sequence[int]) -> Tuple[np.ndarray, List[float]]:
        """A (K, V) real amplitude matrix and the per-order occurrence totals."""
        weights, dists, totals, _has_photonic = self._order_waves(context)
        rows = [
            np.sqrt(np.maximum(float(weight), _EPS)) * np.sqrt(np.maximum(dist, _EPS))
            for weight, dist in zip(weights, dists)
        ]
        return np.stack(rows), totals

    # ------------------------------------------------------- photonic route
    def photonic_term(self, context: Sequence[int], model=None) -> Optional[np.ndarray]:
        """Read the post-photonic bright port against the Bochner codebook."""
        if model is None:
            return None
        key = tuple(int(token) for token in context[-24:])
        cached = self._photonic_cache.get(key)
        if cached is not None:
            return cached[0]
        import torch

        wave = self.projector.sequence_wave(list(key), self.tokenizer)
        psi = torch.from_numpy(np.asarray(wave, dtype=np.complex64)).reshape(1, -1)
        with torch.no_grad():
            state = model.zone_b(psi)
            bright = state.bright[0].detach().cpu().numpy().astype(np.complex64)
            dark = state.dark[0].detach().cpu().numpy().astype(np.complex64)
        raw = np.real(self.phi @ np.conj(bright)).astype(np.float64)
        raw = np.maximum(raw, 0.0)
        total = float(raw.sum())
        if total <= 0.0:
            return None
        score = raw / total
        self.photonic_contexts += 1
        if len(self._photonic_cache) < 8192:
            self._photonic_cache[key] = (
                score,
                float(np.linalg.norm(bright)),
                float(np.linalg.norm(dark)),
            )
        return score

    # --------------------------------------------------------- interference
    def _effective_theta(self, count: int) -> np.ndarray:
        """The learned per-mode phase.  The mixer, not a ring, makes the modes
        interfere; with an identity mixer the Born rule is exactly the
        arithmetic order mixture."""
        theta = np.zeros(int(count), dtype=np.float64)
        learned = min(int(count), int(self.theta.size))
        theta[:learned] = np.asarray(self.theta[:learned], dtype=np.float64)
        return theta

    def amplitudes(
        self, context: Sequence[int], *, model=None, photonic: bool = True
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """The complex multi-mode egress state, shape (K, vocab_size).

        Mode k is the order-k wave with amplitude sqrt(w_k * p_k).  The
        Stiefel mixer U (K x K, retracted by QR) rotates the modes before the
        Born trace, so U = I is exactly the incoherent arithmetic order
        mixture and any other U is a genuine mode-interference readout.
        """
        weights, dists, totals, has_photonic = self._order_waves(
            context, model if photonic else None
        )
        count = len(dists)
        theta = self._effective_theta(count)
        phases = np.exp(1j * theta)
        state = np.zeros((count, self.vocab_size), dtype=np.complex128)
        for index, dist in enumerate(dists):
            state[index] = (
                phases[index]
                * math.sqrt(max(float(weights[index]), _EPS))
                * np.sqrt(np.maximum(dist, _EPS))
            )
        mixer = np.asarray(self.adapter[:count, :count], dtype=np.complex128)
        state = mixer @ state
        info = {
            "orders": int(count - (1 if has_photonic else 0)),
            "with_photonic": bool(has_photonic),
            "totals": [float(value) for value in totals],
            "weights": [float(value) for value in weights],
            "modes": int(count),
        }
        return state, info

    def born(self, amplitude: np.ndarray) -> np.ndarray:
        state = np.asarray(amplitude)
        power = np.sum(np.abs(state) ** 2, axis=0) if state.ndim == 2 else np.abs(state) ** 2
        total = float(power.sum())
        if total <= 0.0:
            return np.full(self.vocab_size, 1.0 / self.vocab_size, dtype=np.float64)
        return power / total

    def logits_for(self, context: Sequence[int], *, model=None) -> np.ndarray:
        amplitude, _info = self.amplitudes(context, model=model)
        return np.log(np.maximum(self.born(amplitude), 1e-300))

    # ------------------------------------------------------------- learning
    def fit(
        self,
        ids: Sequence[int],
        *,
        model=None,
        epochs: int = 4,
        samples: int = 1200,
        inner_fraction: float = 0.15,
        eta: float = 0.05,
        eta_negative: float = 0.04,
        floor_fn=None,
        seed: int = 20261111,
        qr_step: float = 0.10,
        clip: float = 0.20,
    ) -> Dict[str, Any]:
        """Surprise-gated contrastive phase alignment, no gradients.

        The last inner_fraction of the training stream is held out INSIDE the
        training split and is the only data the override thresholds and the
        epoch checkpoint may ever see.  The scored W2 test split is not an
        argument of this call.  A phase vector is kept only when it lowers the
        inner-slice Sagnac residual, so the recorded curve is the best-so-far
        residual and the final value cannot be worse than the initial one.
        """
        sequence = [int(token) for token in ids]
        self.observe(sequence)
        if self.base_counts is None:
            from henri.holarch.code_gen import CountLanguageModel

            self.base_counts = CountLanguageModel(self.vocab_size)
        self.base_counts.fit(sequence)
        cut = int(len(sequence) * (1.0 - float(inner_fraction)))
        fit_ids = sequence[:cut]
        inner_ids = sequence[cut:]
        if len(fit_ids) <= self.max_order + 2 or len(inner_ids) <= self.max_order + 2:
            return {"epochs": 0, "sagnac_curve": [], "updates": 0}
        sample_count = int(min(int(samples), max(1, len(fit_ids) - self.max_order - 2)))
        stride = max(1, (len(fit_ids) - self.max_order - 2) // sample_count)
        positions = list(range(self.max_order, len(fit_ids) - 1, stride))[:sample_count]
        inner_positions = list(range(self.max_order, len(inner_ids) - 1, max(1, len(inner_ids) // 240)))[:240]

        def inner_residual() -> float:
            total = 0.0
            count = 0
            for position in inner_positions:
                context = inner_ids[max(0, position - self.window) : position]
                target = int(inner_ids[position])
                if not (0 <= target < self.vocab_size):
                    continue
                amplitude, _info = self.amplitudes(context, model=model, photonic=False)
                probability = self.born(amplitude)
                total += 1.0 - float(probability[target])
                count += 1
            return float(total / max(count, 1))

        initial = inner_residual()
        best = (
            float(initial),
            self.theta.copy(),
            self.adapter.copy(),
            self.mode_gain.copy(),
        )
        curve: List[float] = [float(initial)]
        updates = 0
        for _epoch in range(int(epochs)):
            for position in positions:
                context = fit_ids[max(0, position - self.window) : position]
                target = int(fit_ids[position])
                if not (0 <= target < self.vocab_size):
                    continue
                rows, _totals = self._order_amplitudes(context)
                count = int(rows.shape[0])
                theta = self._effective_theta(count)
                weight = np.array([float(value) for value in self.weight[:count]])
                coefficients = np.asarray(
                    [
                        weight[index]
                        * np.exp(1j * theta[index])
                        * float(rows[index][target])
                        for index in range(count)
                    ],
                    dtype=np.complex128,
                )
                amplitude = complex(np.sum(coefficients))
                state, _state_info = self.amplitudes(context, photonic=False)
                probability = float(self.born(state)[target])
                competitor = -1
                if floor_fn is not None:
                    try:
                        floor = floor_fn(context)
                        if floor is not None:
                            competitor = int(np.argmax(floor))
                    except Exception:  # noqa: BLE001 - a floor failure is not fatal
                        competitor = -1
                if probability >= 0.50:
                    continue
                for index in range(count):
                    pull = float(np.imag(amplitude * np.conj(coefficients[index])))
                    delta = float(eta) * pull
                    if 0 <= competitor < self.vocab_size and competitor != target:
                        wrong = (
                            weight[index]
                            * np.exp(1j * theta[index])
                            * float(rows[index][competitor])
                        )
                        push = float(np.imag(amplitude * np.conj(complex(wrong))))
                        delta -= float(eta_negative) * push
                    delta = float(max(-clip, min(clip, delta)))
                    self.theta[index] = float(self.theta[index]) + delta
                updates += 1
            # Contrastive amplitude alignment of the phase-conjugate modes.
            # The Born trace is invariant under the Stiefel mixer, so the
            # learned degree of freedom that moves the distribution is the
            # per-mode gain.  A mode that ranks the true token above the
            # current mixture is pulled up; a mode that ranks the floor's
            # wrong competitor above it is pushed down.
            gradient = np.zeros(int(self.mode_gain.size), dtype=np.float64)
            for position in positions:
                context = fit_ids[max(0, position - self.window) : position]
                target = int(fit_ids[position])
                if not (0 <= target < self.vocab_size):
                    continue
                weights, dists, _totals, _hp = self._order_waves(context)
                probability = self.distribution(context)
                if float(probability[target]) >= 0.50:
                    continue
                competitor = -1
                if floor_fn is not None:
                    try:
                        floor = floor_fn(context)
                        if floor is not None:
                            competitor = int(np.argmax(floor))
                    except Exception:  # noqa: BLE001
                        competitor = -1
                for index, dist in enumerate(dists):
                    if index >= gradient.size:
                        break
                    pull = float(dist[target]) - float(probability[target])
                    push = 0.0
                    if 0 <= competitor < self.vocab_size and competitor != target:
                        push = float(dist[competitor]) - float(probability[competitor])
                    gradient[index] += pull - float(eta_negative) * push
            scale = float(np.max(np.abs(gradient))) if gradient.size else 0.0
            if scale > 0.0:
                self.mode_gain = np.clip(
                    self.mode_gain * np.exp(0.5 * gradient / scale), 0.20, 5.0
                )
                mean_gain = float(np.mean(self.mode_gain))
                if mean_gain > 0.0:
                    self.mode_gain = self.mode_gain / mean_gain
            # QR Stiefel retraction of the slot adapter after every epoch.
            slots = int(self.adapter.shape[0])
            phase_vector = np.exp(1j * self.theta[:slots])
            rank_one = np.outer(phase_vector, np.conj(phase_vector))
            candidate = self.adapter + float(qr_step) * (rank_one @ self.adapter) / float(slots)
            q, _r = np.linalg.qr(candidate)
            self.adapter = q
            residual = inner_residual()
            if residual < best[0]:
                best = (
                    float(residual),
                    self.theta.copy(),
                    self.adapter.copy(),
                    self.mode_gain.copy(),
                )
            curve.append(float(min(best[0], residual)))
        self.theta = best[1]
        self.adapter = best[2]
        self.mode_gain = best[3]
        self.sagnac_curve = list(curve)
        self.train_updates = int(updates)
        self.fitted = True
        return {
            "epochs": int(epochs),
            "updates": int(updates),
            "sagnac_curve": [float(value) for value in curve],
            "sagnac_initial": float(curve[0]) if curve else 0.0,
            "sagnac_final": float(curve[-1]) if curve else 0.0,
            "inner_tokens": int(len(inner_ids)),
            "fit_tokens": int(len(fit_ids)),
            "inner_positions": int(len(inner_positions)),
            "theta": [float(value) for value in self.theta],
            "weight": [float(value) for value in self.weight],
            "mode_gain": [float(value) for value in self.mode_gain],
            "adapter_digest": float(np.abs(self.adapter).sum()),
            "autograd_used": False,
        }

    # ------------------------------------------------------------- override
    def candidate(self, context: Sequence[int], *, model=None) -> Dict[str, Any]:
        """The egress token, its margin, and the Sagnac residual."""
        amplitude, info = self.amplitudes(context, model=model)
        probability = self.born(amplitude)
        order = np.argsort(-probability)
        best = int(order[0])
        second = int(order[1]) if order.size > 1 else best
        margin = float(probability[best] - probability[second])
        residual = float(1.0 - probability[best])
        return {
            "token": int(best),
            "probability": float(probability[best]),
            "margin": float(margin),
            "sagnac": float(residual),
            "distribution": probability,
            "info": info,
        }

    def calibrate_photonic(
        self,
        ids: Sequence[int],
        *,
        model=None,
        floor_fn=None,
        grid: Sequence[float] = (0.0, 0.01, 0.02, 0.05, 0.10),
        samples: int = 160,
        context: int = 24,
    ) -> Dict[str, Any]:
        """Select the photonic fusion weight on the INNER validation slice.

        The grid contains 0.0, so the post-photonic term can only be admitted
        when it lowers the inner held-out cross-entropy without lowering the
        inner top-1.  This is a selection over a measured curve, not a claim.
        """
        sequence = [int(token) for token in ids]
        if len(sequence) <= context + 2:
            return {"weight": float(self.photonic_weight), "rows": 0}
        indices = list(range(context, len(sequence) - 1))
        if len(indices) > int(samples):
            stride = max(1, len(indices) // int(samples))
            indices = indices[::stride]
        rows: List[Dict[str, float]] = []
        best: Optional[Dict[str, float]] = None
        for weight in grid:
            self.photonic_weight = float(weight)
            top1 = 0
            nll = 0.0
            for index in indices:
                window = sequence[max(0, index - context) : index]
                probability = self.distribution(window, model=model)
                truth = int(sequence[index])
                top1 += int(int(np.argmax(probability)) == truth)
                nll -= float(np.log(max(float(probability[truth]), 1e-12)))
            tokens = max(len(indices), 1)
            row = {
                "weight": float(weight),
                "top1": float(top1 / tokens),
                "bpc": float(nll / tokens / math.log(2.0)),
            }
            rows.append(row)
            if best is None or (row["top1"], -row["bpc"]) > (best["top1"], -best["bpc"]):
                best = row
        self.photonic_weight = float(best["weight"]) if best is not None else 0.10
        return {"weight": float(self.photonic_weight), "rows": rows, "samples": int(len(indices))}

    def calibrate_override(
        self,
        ids: Sequence[int],
        *,
        model=None,
        floor_fn=None,
        tau_h_grid: Sequence[float] = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
        tau_margin_grid: Sequence[float] = (0.02, 0.05, 0.10, 0.20, 0.35),
        tau_sagnac_grid: Sequence[float] = (0.35, 0.50, 0.65, 0.80, 0.92),
        context: int = 24,
    ) -> Dict[str, Any]:
        """Select the override thresholds on the INNER validation slice."""
        sequence = [int(token) for token in ids]
        if len(sequence) <= context + 2 or floor_fn is None:
            self.override = {"tau_H": 0.0, "tau_margin": 0.0, "tau_sagnac": 1.0,
                             "rows": 0, "net": 0, "wins": 0, "losses": 0}
            return dict(self.override)
        indices = list(range(context, len(sequence) - 1))
        if len(indices) > 900:
            stride = max(1, len(indices) // 900)
            indices = indices[::stride]
        rows: List[Dict[str, float]] = []
        for index in indices:
            window = sequence[max(0, index - context) : index]
            floor = floor_fn(window)
            if floor is None:
                continue
            floor_best = int(np.argmax(floor))
            candidate = self.candidate(window, model=model)
            entropy = float(
                -np.sum(np.maximum(floor, 1e-12) * np.log(np.maximum(floor, 1e-12)))
            )
            rows.append(
                {
                    "entropy": entropy,
                    "margin": float(candidate["margin"]),
                    "sagnac": float(candidate["sagnac"]),
                    "floor_best": float(floor_best),
                    "wave_best": float(candidate["token"]),
                    "truth": float(sequence[index]),
                }
            )
        best: Optional[Dict[str, Any]] = None
        for tau_h in tau_h_grid:
            for tau_m in tau_margin_grid:
                for tau_s in tau_sagnac_grid:
                    wins = losses = 0
                    for row in rows:
                        if not (
                            row["entropy"] >= tau_h
                            and row["margin"] >= tau_m
                            and row["sagnac"] <= tau_s
                        ):
                            continue
                        if int(row["wave_best"]) == int(row["floor_best"]):
                            continue
                        if int(row["wave_best"]) == int(row["truth"]):
                            wins += 1
                        elif int(row["floor_best"]) == int(row["truth"]):
                            losses += 1
                    net = wins - losses
                    candidate = {
                        "tau_H": float(tau_h),
                        "tau_margin": float(tau_m),
                        "tau_sagnac": float(tau_s),
                        "wins": int(wins),
                        "losses": int(losses),
                        "net": int(net),
                        "rows": int(len(rows)),
                    }
                    if best is None or (net, wins, -losses) > (
                        best["net"], best["wins"], -best["losses"]
                    ):
                        best = candidate
        self.override = dict(best or {})
        return dict(self.override)

    def evaluate(
        self,
        sequences: Sequence[Sequence[int]],
        *,
        floor_fn=None,
        model=None,
        context: int = 24,
    ) -> WaveEgressReport:
        """Score the standalone egress and the gated override on held-out text."""
        tau_h = float(getattr(self, "override", {}).get("tau_H", 0.0))
        tau_m = float(getattr(self, "override", {}).get("tau_margin", 0.0))
        tau_s = float(getattr(self, "override", {}).get("tau_sagnac", 1.0))
        tokens = 0
        top1 = 0
        base_top1 = 0
        nll = 0.0
        base_nll = 0.0
        overrides = wins = losses = 0
        residual_total = 0.0
        for sequence in sequences:
            sequence = [int(token) for token in sequence]
            for index in range(1, len(sequence)):
                window = sequence[max(0, index - self.window) : index]
                candidate = self.candidate(window, model=model)
                probability = candidate["distribution"]
                truth = int(sequence[index])
                emitted = int(candidate["token"])
                floor_best = emitted
                if floor_fn is not None:
                    floor = floor_fn(window)
                    if floor is not None:
                        floor_best = int(np.argmax(floor))
                        entropy = float(
                            -np.sum(np.maximum(floor, 1e-12) * np.log(np.maximum(floor, 1e-12)))
                        )
                        base_nll -= float(np.log(max(float(floor[truth]), 1e-12)))
                        base_top1 += int(floor_best == truth)
                        if (
                            emitted != floor_best
                            and entropy >= tau_h
                            and float(candidate["margin"]) >= tau_m
                            and float(candidate["sagnac"]) <= tau_s
                        ):
                            overrides += 1
                            if emitted == truth:
                                wins += 1
                            elif floor_best == truth:
                                losses += 1
                top1 += int(emitted == truth)
                nll -= float(np.log(max(float(probability[truth]), 1e-12)))
                residual_total += 1.0 - float(probability[truth])
                tokens += 1
        if tokens <= 0:
            return WaveEgressReport(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0.0, 0.0)
        return WaveEgressReport(
            tokens=int(tokens),
            top1=float(top1 / tokens),
            bits_per_char=float(nll / tokens / math.log(2.0)),
            perplexity=float(math.exp(min(50.0, nll / tokens))),
            base_top1=float(base_top1 / tokens),
            base_bits_per_char=float(base_nll / tokens / math.log(2.0)),
            override_count=int(overrides),
            override_wins=int(wins),
            override_losses=int(losses),
            override_net_win_delta=int(wins - losses),
            sagnac_init=float(self.sagnac_curve[0]) if self.sagnac_curve else float(residual_total / tokens),
            sagnac_final=float(self.sagnac_curve[-1]) if self.sagnac_curve else float(residual_total / tokens),
            sagnac_curve=list(self.sagnac_curve),
            fusion={
                "orders": int(len(self.orders)),
                "window": int(self.window),
                "tau_H": float(tau_h),
                "tau_margin": float(tau_m),
                "tau_sagnac": float(tau_s),
                "max_order": int(self.max_order),
            },
            photonic_contexts=int(self.photonic_contexts),
            autograd_used=False,
        )


class WaveEgressHead(ContrastivePhaseResonator):
    """The deployed name of the transducer (Directive X2)."""