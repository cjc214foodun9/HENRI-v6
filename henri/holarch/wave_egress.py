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

Cycle Y1 extension (quadrature=True)
------------------------------------
Two measured defects of X2 are repaired here without touching the X2 path
(quadrature=False is bit-identical to the Cycle X organ):

1. Phase saddle.  X2 kept only the real part of every mode resonance and
   initialized theta at zero, so the Born cross-term cos(0) sat on a
   stationary point with d/dphi = -sin(0) = 0.  Here every mode resonance
   keeps BOTH quadratures: s_m(v|h) = sqrt(p_m(v)) * exp(i * arg <chi_v,
   psi_m>), and the per-mode / per-slot offsets phi[m, s] start at
   deterministic incommensurate irrational angles.  The offset is updated
   by the analytical contrastive Born-rule cross-term torque

       d|A(v)|^2 / d phi[m,s] = -2 Im(c_m a_m(v) conj(A(v))),  v in slot s

   with c_m = sum_k U[k,m] the column sum of the Stiefel mixer.  The
   photonic mode keeps the exact complex amplitude <chi_v, bright>, so its
   probability is Born's rule on the optical readout, not a mutilated real
   part.

2. Optical subspace covariate shift.  The egress may mount a
   zone_b.SubwordPhaseCartridge, a closed-form per-slot cross-spectral
   aligner between the post-photonic ports and the Bochner codebook.  The
   bright port then carries the L = 24 structural context instead of a
   Cycle E character-alphabet scramble.
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
        quadrature: bool = False,
        phase_seed: float = 1.0,
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
        # ------------------------------------------------------ Y1 quadrature
        # The quadrature path is opt-in so that the sealed X2 receipt stays
        # replayable bit for bit on the same code.
        self.quadrature = bool(quadrature)
        self.phase_gain = 1.0
        self.cartridge = None
        self.phase_epochs = 0
        self.phase_updates = 0
        self.phase_curve: List[float] = []
        self.phase_attribution: Dict[str, float] = {}
        modes = len(self.orders) + 2
        # Deterministic small incommensurate irrational seed angles.  sin(0)
        # is the saddle X2 could not leave; frac(m*sqrt(2)+s*sqrt(3)) is
        # equidistributed and never repeats a rational ratio.
        seed = np.zeros((modes, 4), dtype=np.float64)
        for mode in range(modes):
            for slot in range(4):
                value = (mode + 1) * math.sqrt(2.0) + (slot + 1) * math.sqrt(3.0)
                seed[mode, slot] = 0.10 * float(phase_seed) * (value - math.floor(value))
        self.phase0 = seed
        self.phase = seed.copy()
        self._slot_ids = np.asarray(
            [int(tokenizer.slot_of(token)) % 4 for token in range(self.vocab_size)],
            dtype=np.int64,
        )
        self._slot_index = [
            np.nonzero(self._slot_ids == slot)[0] for slot in range(4)
        ]
        self._phi_slot = [
            np.ascontiguousarray(self.phi[self._slot_index[slot]][
                :, slot * (self.dim // 4) : (slot + 1) * (self.dim // 4)
            ])
            for slot in range(4)
        ]
        # Per-slot codebook blocks for the dense semantic readout.
        self._context_cache: Dict[Tuple[int, ...], np.ndarray] = {}
        self._phasor_cache: Dict[Tuple[int, ...], np.ndarray] = {}
        self._last_photonic_complex: Optional[np.ndarray] = None
        self.cartridge_report: Dict[str, Any] = {}

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
        weights, dists, totals, has_photonic, _terms = self._order_waves_full(
            context, model
        )
        return weights, dists, totals, has_photonic

    def _order_waves_full(self, context: Sequence[int], model=None):
        """The same order waves, plus the retained complex photonic amplitude."""
        prior = self.base_distribution(context)
        weights: List[float] = [0.25]
        dists: List[np.ndarray] = [prior]
        totals: List[float] = [1.0]
        terms: List[Optional[np.ndarray]] = [None]
        for order, local, total in self._resonances(context):
            alpha = float(self.alpha.get(int(order), 0.7))
            saturation = float(total) / (float(total) + self.saturation)
            weights.append(max(0.0, alpha * saturation * 0.75))
            dists.append(local)
            totals.append(float(total))
            terms.append(None)
        has_photonic = False
        if model is not None:
            term = self.photonic_term(context, model)
            if term is not None:
                weights.append(float(self.photonic_weight))
                dists.append(term)
                totals.append(1.0)
                terms.append(self._last_photonic_complex)
                has_photonic = True
        for index in range(len(weights)):
            gain = float(self.mode_gain[index]) if index < self.mode_gain.size else 1.0
            weights[index] = max(0.0, float(weights[index]) * gain)
        mass = float(sum(weights))
        if mass <= 0.0:
            return [1.0], [prior], [1.0], False, [None]
        return [float(value / mass) for value in weights], dists, totals, has_photonic, terms

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
    def codebook_readout(self, wave: np.ndarray) -> np.ndarray:
        """<chi_v, wave> for the whole locked vocabulary, both quadratures.

        The codebook is slot-routed, so the dense readout is four block
        matrix-vector products of shape (n_s, D/4) instead of one (V, D)
        product: the same number in four times fewer operations.
        """
        field = np.asarray(wave, dtype=np.complex64)
        out = np.zeros(self.vocab_size, dtype=np.complex64)
        d_slot = self.dim // 4
        for slot in range(4):
            block = np.conj(field[slot * d_slot : (slot + 1) * d_slot])
            rows = self._phi_slot[slot]
            out[self._slot_index[slot]] = rows @ block
        # rows @ conj(field) is conj(<chi_v, field>); undo the conjugation so
        # the caller always receives the physics inner product.
        return np.conj(out)

    def context_wave(self, tokens: Sequence[int]) -> np.ndarray:
        """The L = window Bochner context wave, cached by token tuple."""
        key = tuple(int(token) for token in tokens)
        cached = self._context_cache.get(key)
        if cached is not None:
            return cached
        wave = self.projector.sequence_wave(list(key), self.tokenizer)
        wave = np.asarray(wave, dtype=np.complex64)
        if len(self._context_cache) < 8192:
            self._context_cache[key] = wave
        return wave

    def set_cartridge(self, cartridge, *, report: Optional[Dict[str, Any]] = None):
        """Mount the Y1 subword optical cartridge on the photonic route."""
        self.cartridge = cartridge
        self.cartridge_report = dict(report or {})
        self._photonic_cache.clear()
        return self

    def photonic_term(self, context: Sequence[int], model=None) -> Optional[np.ndarray]:
        """Read the post-photonic bright port against the Bochner codebook."""
        if model is None:
            return None
        key = tuple(int(token) for token in context[-24:])
        cached = self._photonic_cache.get(key)
        if cached is not None:
            self._last_photonic_complex = cached[3]
            return cached[0]
        import torch

        wave = self.context_wave(key)
        psi = torch.from_numpy(np.asarray(wave, dtype=np.complex64)).reshape(1, -1)
        with torch.no_grad():
            state = model.zone_b(psi)
            bright = state.bright[0].detach().cpu().numpy().astype(np.complex64)
            dark = state.dark[0].detach().cpu().numpy().astype(np.complex64)
        if self.cartridge is not None:
            bright, dark = self.cartridge.adapt_arrays(bright, dark)
        raw = self.codebook_readout(bright)
        positive = np.maximum(np.real(raw), 0.0)
        total = float(positive.sum())
        if total <= 0.0:
            self._last_photonic_complex = None
            return None
        score = positive / total
        self.photonic_contexts += 1
        self._last_photonic_complex = raw
        if len(self._photonic_cache) < 8192:
            self._photonic_cache[key] = (
                score,
                float(np.linalg.norm(bright)),
                float(np.linalg.norm(dark)),
                raw,
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
        weights, dists, totals, has_photonic, terms = self._order_waves_full(
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
        pre_mixer: Optional[np.ndarray] = None
        if self.quadrature:
            for index, dist in enumerate(dists):
                state[index] = self._quadrature_row(
                    index, context, dist, terms[index], float(weights[index])
                )
            pre_mixer = state.copy()
        mixer = np.asarray(self.adapter[:count, :count], dtype=np.complex128)
        state = mixer @ state
        info = {
            "orders": int(count - (1 if has_photonic else 0)),
            "with_photonic": bool(has_photonic),
            "totals": [float(value) for value in totals],
            "weights": [float(value) for value in weights],
            "modes": int(count),
            "quadrature": bool(self.quadrature),
            "phase_displacement": self.phase_displacement(),
            "pre_mixer": pre_mixer,
        }
        return state, info

    # ------------------------------------------------------- Y1 quadrature
    def phase_displacement(self) -> float:
        """||phi - phi^(0)||_2 over every mode and slot."""
        return float(np.linalg.norm(np.asarray(self.phase) - np.asarray(self.phase0)))

    def _mode_wave(self, index: int, context: Sequence[int]) -> np.ndarray:
        """The Bochner codebook wave a mode resonance is read against."""
        if index <= 0:
            return self.context_wave([int(token) for token in context[-24:]])
        order = int(self.orders[min(index - 1, len(self.orders) - 1)])
        return self.context_wave([int(token) for token in context[-order:]])

    def _mode_phasor(
        self, index: int, wave: np.ndarray, support: np.ndarray
    ) -> np.ndarray:
        """exp(i * arg <chi_v, psi_m>): the retained quadrature of a mode."""
        phasor = np.ones(self.vocab_size, dtype=np.complex128)
        if support.size == 0:
            return phasor
        if support.size > 256:
            overlap = self.codebook_readout(wave)[support]
        else:
            overlap = np.conj(wave) @ self.phi[support].T
        magnitude = np.abs(overlap)
        safe = magnitude > 1e-6
        unit = np.ones_like(overlap)
        unit[safe] = overlap[safe] / magnitude[safe]
        phasor[support] = unit
        return phasor

    def _quadrature_row(
        self,
        index: int,
        context: Sequence[int],
        dist: np.ndarray,
        term: Optional[np.ndarray],
        weight: float,
    ) -> np.ndarray:
        """One complex mode resonance s_m(v|h) with BOTH quadratures.

        Count modes keep the count envelope sqrt(p_m(v)) and take the phase
        of the Bochner inner product.  The photonic mode is the exact complex
        amplitude <chi_v, bright>, renormalised so Born's rule returns the
        optical readout distribution.  Every mode then receives its learned
        per-mode / per-slot offset phi[m, s(v)].
        """
        if term is not None:
            norm = float(np.linalg.norm(term))
            if norm > _EPS:
                row = (math.sqrt(max(float(weight), _EPS)) * np.asarray(
                    term, dtype=np.complex128) / norm)
            else:
                row = np.zeros(self.vocab_size, dtype=np.complex128)
        else:
            support = np.nonzero(np.asarray(dist) > 0.0)[0]
            wave = self._mode_wave(index, context)
            row = (
                math.sqrt(max(float(weight), _EPS))
                * np.sqrt(np.maximum(np.asarray(dist, dtype=np.float64), _EPS))
                * self._mode_phasor(index, wave, support)
            )
        if self.phase_gain != 0.0 and index < self.phase.shape[0]:
            row = row * np.exp(
                1j * float(self.phase_gain) * self.phase[index][self._slot_ids]
            )
        return row

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

    # ------------------------------------------------- Y1 phase torque
    def fit_quadrature(
        self,
        ids: Sequence[int],
        *,
        model=None,
        epochs: int = 3,
        samples: int = 480,
        inner_fraction: float = 0.15,
        eta: float = 0.02,
        eta_negative: float = 0.02,
        floor_fn=None,
        clip: float = 0.05,
    ) -> Dict[str, Any]:
        """Learn phi[m, s] by the analytical Born-rule cross-term torque.

        For one training position the post-mixer amplitude is
        A(v) = sum_k (U a)[k, v] and the mixer column sum is c_m = sum_k U[k,m].
        The exact derivative of Born's power with respect to a phase offset is

            d|A(v)|^2 / dphi[m, s(v)] = -2 Im(c_m a_m(v) conj(A(v))),

        so no tensor graph and no finite difference is needed.  The target's
        slot column is pushed up and the count floor's wrong competitor's
        slot column is pushed down.  The inner slice of the training split
        is the only checkpoint selector, so the recorded curve is best-so-far
        and cannot end worse than it started.
        """
        if not self.quadrature:
            raise ValueError("fit_quadrature needs quadrature=True")
        sequence = [int(token) for token in ids]
        if self.base_counts is None:
            from henri.holarch.code_gen import CountLanguageModel

            self.base_counts = CountLanguageModel(self.vocab_size)
            self.base_counts.fit(sequence)
        cut = int(len(sequence) * (1.0 - float(inner_fraction)))
        fit_ids = sequence[:cut]
        inner_ids = sequence[cut:]
        if len(fit_ids) <= self.max_order + 2 or len(inner_ids) <= self.max_order + 2:
            return {"epochs": 0, "updates": 0}
        sample_count = int(min(int(samples), max(1, len(fit_ids) - self.max_order - 2)))
        stride = max(1, (len(fit_ids) - self.max_order - 2) // sample_count)
        positions = list(range(self.max_order, len(fit_ids) - 1, stride))[:sample_count]
        inner_positions = list(
            range(self.max_order, len(inner_ids) - 1, max(1, len(inner_ids) // 192))
        )[:192]

        def inner_nll() -> float:
            total = 0.0
            count = 0
            for position in inner_positions:
                context = inner_ids[max(0, position - self.window) : position]
                target = int(inner_ids[position])
                probability = self.born(self.amplitudes(context, photonic=False)[0])
                total -= float(np.log(max(float(probability[target]), 1e-12)))
                count += 1
            return float(total / max(count, 1))

        initial = inner_nll()
        best = (float(initial), self.phase.copy(), self.theta.copy())
        curve: List[float] = [float(initial)]
        updates = 0
        total_update = 0.0
        step = float(clip)
        for _epoch in range(int(epochs)):
            gradient = np.zeros_like(self.phase)
            counted = 0
            for position in positions:
                context = fit_ids[max(0, position - self.window) : position]
                target = int(fit_ids[position])
                if not (0 <= target < self.vocab_size):
                    continue
                state, info = self.amplitudes(context, photonic=False)
                pre = info.get("pre_mixer")
                if pre is None:
                    continue
                count = int(state.shape[0])
                mixer = np.asarray(self.adapter[:count, :count], dtype=np.complex128)
                column = mixer.sum(axis=0)
                power = float(np.sum(np.abs(state) ** 2)) + _EPS
                target_slot = int(self._slot_ids[target])
                amplitude = complex(state[:, target].sum())
                competitor = -1
                if floor_fn is not None:
                    try:
                        floor = floor_fn(context)
                        if floor is not None:
                            competitor = int(np.argmax(floor))
                    except Exception:  # noqa: BLE001 - a floor failure is not fatal
                        competitor = -1
                competitor_slot = (
                    int(self._slot_ids[competitor])
                    if 0 <= competitor < self.vocab_size
                    else -1
                )
                competitor_amplitude = (
                    complex(state[:, competitor].sum())
                    if 0 <= competitor < self.vocab_size
                    else 0.0
                )
                contrast = float(eta_negative) / max(float(eta), _EPS)
                for mode in range(min(count, self.phase.shape[0])):
                    pull = -2.0 * float(
                        np.imag(column[mode] * pre[mode, target] * np.conj(amplitude))
                    )
                    gradient[mode, target_slot] += pull / power
                    if competitor_slot >= 0 and competitor != target:
                        push = -2.0 * float(
                            np.imag(
                                column[mode]
                                * pre[mode, competitor]
                                * np.conj(competitor_amplitude)
                            )
                        )
                        gradient[mode, competitor_slot] -= contrast * push / power
                counted += 1
            if counted <= 0:
                break
            gradient = gradient / float(counted)
            scale = float(np.max(np.abs(gradient)))
            if scale <= 0.0:
                break
            direction = gradient / scale
            trial = self.phase + step * direction
            self.phase = trial
            value = inner_nll()
            if value < best[0] - 1e-12:
                best = (float(value), trial.copy(), self.theta.copy())
                updates += int(counted)
                total_update += float(np.sum(np.abs(step * direction)))
                curve.append(float(value))
            else:
                self.phase = best[1].copy()
                self.theta = best[2].copy()
                step *= 0.5
                curve.append(float(min(best[0], value)))
                if step < 1e-4:
                    break
        self.phase = best[1]
        self.theta = best[2]
        self.phase_curve = list(curve)
        self.phase_epochs = int(epochs)
        self.phase_updates = int(updates)
        displacement = self.phase_displacement()
        self.phase_attribution = {
            "inner_nll_initial": float(initial),
            "inner_nll_final": float(best[0]),
            "inner_nll_gain": float(initial - best[0]),
            "phase_displacement": float(displacement),
            "mean_abs_step": float(total_update / max(len(curve) - 1, 1)),
            "updates": int(updates),
            "epochs": int(epochs),
            "final_step": float(step),
            "curve": [float(value) for value in curve],
            "seed_phase_norm": float(np.linalg.norm(self.phase0)),
        }
        return dict(self.phase_attribution)

    def fit_cartridge(
        self,
        ids: Sequence[int],
        model,
        *,
        samples: int = 320,
        context: int = 24,
        ridge: float = 1e-1,
    ) -> Dict[str, Any]:
        """Fit the non-destructive subword optical cartridge in closed form.

        The target is the Bochner context wave, the observation is the post-
        photonic bright port of the SAME context.  Nothing is written to the
        mounted checkpoint: the aligner lives in the cartridge.
        """
        import torch

        from henri.holarch.zone_b import open_subword_cartridge

        sequence = [int(token) for token in ids]
        if len(sequence) <= int(context) + 2:
            return {"contexts": 0}
        indices = list(range(int(context), len(sequence) - 1))
        stride = max(1, len(indices) // max(int(samples), 1))
        indices = indices[::stride][: int(samples)]
        waves = [
            torch.from_numpy(self.context_wave(sequence[index - int(context) : index]))
            for index in indices
        ]
        cartridge = open_subword_cartridge(model, ridge=float(ridge))
        report = cartridge.fit(waves)
        self.set_cartridge(cartridge, report=report)
        return dict(report)

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
            # The admitted objective is the validation cross-entropy, because
            # the fusion weight multiplies a probability term.  A weight that
            # buys one extra top-1 token and pays 0.08 bits/char for it is a
            # worse model, not a better one; the old (top-1 first) ordering
            # admitted exactly that.  Top-1 breaks a tie only.
            if best is None or (row["bpc"], -row["top1"]) < (best["bpc"], -best["top1"]):
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

# ======================================================================
# Cycle AA1: per-token phase-conjugate Born egress and holographic
# key-successor induction
# ======================================================================
#
# Z1 measured two exact defects.
#
# 1. The band-partition rotor applies one angle to a coordinate band, so the
#    target token and every competitor rotate together and the first-order
#    normalised-likelihood torque cancels (measured displacement 0.0000 rad).
#    The repair is a TOKEN-INDEXED phase table Phi in [-pi, pi)^(M x V): the
#    phase of mode m for token v moves that token alone relative to the
#    resultant interference chord.  The exact derivative of the cross-entropy
#    with respect to Phi[m, v] is used; no tensor graph and no external
#    optimiser are involved.
#
# 2. The bag-of-tokens pointer scores a candidate by "did this token occur
#    recently", which is anti-correlated with the syntactic successor.  The
#    repair is a HOLOGRAPHIC KEY-SUCCESSOR INDUCTION HEAD: it matches the wave
#    of the last context token against the wave of every earlier key in a long
#    causal window and superposes the successor waves that follow a match.
#
# Both pointer channels are admitted only through an entropy / novelty gate,
# and the gate weight is selected on the INNER slice of the training stream
# with 0.0 in the grid, so the identity (channel off) is always available.


@dataclass
class PhaseTableReport:
    """One measured training run of the token-indexed phase table."""

    inner_initial: float
    inner_final: float
    rms_displacement: float
    updates: int
    epochs: int
    accepted: bool
    curve: List[float] = field(default_factory=list)
    samples: int = 0

    def as_dict(self) -> Dict[str, Any]:
        gain = float(self.inner_initial - self.inner_final)
        return {
            "inner_nll_initial": float(self.inner_initial),
            "inner_nll_final": float(self.inner_final),
            "inner_nll_gain": gain,
            "accepted_by_line_search": bool(self.accepted and gain > 0.0),
            "rms_displacement_rad": float(self.rms_displacement),
            "updates": int(self.updates),
            "epochs": int(self.epochs),
            "samples": int(self.samples),
            "curve": [float(value) for value in self.curve],
            "gradient": "exact analytic Born-rule cross-term phase gradient",
            "autograd_used": False,
        }


class TokenPhaseEgress(WaveEgressHead):
    """AA1 egress: token-indexed phase table plus key-successor induction.

    The resonance readout is

        A(v | h) = sum_m g_m(h) exp(i * Phi[m, v]) s_m(v | h)

        P(v | h) = |A(v | h)|^2 / sum_u |A(u | h)|^2

    where the modes m are the calibrated count floor (mode 0) and the six
    window-local suffix resonances (mode k = order k).  With Phi = 0 the
    readout is the coherent superposition of the mode amplitudes; the phase
    table moves one token's phase relative to the resultant chord, which is
    exactly the degree of freedom the isotropic band rotor could not supply.

    Two long-range resonances are coupled through an entropy / novelty gate:
    the aligned D = 2048 post-photonic port and the holographic induction
    pointer.  The coupling law is selected on the inner slice from two
    physically distinct compositions:

    * mixture:  P = (1 - g w) P_res + g w p_channel, the calibrated
      probability mixture that a gate weight of zero switches off exactly;
    * amplitude:  P = |A + g w r_channel|^2 / Z, the real Born resonance
      amplitude Re<Psi_ind(h), psi(v)> added to the chord.

    Both laws and both weights are scored by the same inner cross-entropy and
    the grid contains 0.0, so a channel can only enter the readout when the
    measurement improves.
    """

    def __init__(
        self,
        projector,
        tokenizer,
        *,
        core=None,
        cartridge=None,
        induction_window: int = 256,
        induction_gamma: float = 0.8,
        induction_low: int = 7,
        induction_tau_match: float = 0.5,
        phase_seed: float = 0.15,
        seed: int = 20261111,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("quadrature", False)
        super().__init__(projector, tokenizer, seed=seed, **kwargs)
        self.core = core
        self.cartridge = cartridge
        self.induction_window = int(induction_window)
        self.induction_gamma = float(induction_gamma)
        self.induction_low = int(induction_low)
        self.induction_tau_match = float(induction_tau_match)
        # prior + suffix orders + photonic port + induction pointer
        self.phase_modes = int(self.max_order + 3)
        self.photonic_row = int(self.max_order + 1)
        self.induction_row = int(self.max_order + 2)
        rng = np.random.default_rng(int(seed))
        self.phase_table = float(phase_seed) * rng.standard_normal(
            (self.phase_modes, self.vocab_size)
        )
        self.phase_table0 = self.phase_table.copy()
        self.w_induction = 0.0
        self.w_photonic = 0.0
        self.w_copy = 0.0
        self.copy_low = 7
        self.copy_high = 24
        self.copy_gamma = 1.0
        self.tau_gate_h = 2.0
        self.coupling = "mixture"
        self.induction_report: Dict[str, Any] = {}
        self.channel_report: Dict[str, Any] = {}
        self.floor_report: Dict[str, Any] = {}
        self.phase_report: Dict[str, Any] = {}
        self._induction_cache: Dict[Tuple[int, ...], Optional[Tuple[np.ndarray, np.ndarray]]] = {}
        self._key_cache: Dict[Tuple[int, int], float] = {}
        self._port_cache: Dict[Tuple[int, ...], Optional[np.ndarray]] = {}
        self.induction_contexts = 0

    # ------------------------------------------------------------ phases
    def phase_rms(self) -> float:
        difference = np.asarray(self.phase_table) - np.asarray(self.phase_table0)
        return float(np.sqrt(np.mean(np.square(difference))))

    def _suffix_rows(self, context: Sequence[int]) -> Tuple[np.ndarray, List[str]]:
        """(M, V) real amplitude rows of the suffix modes, in fixed slots.

        Row 0 is the calibrated count floor, row k is suffix order k.  The
        slots are fixed by the order number, so a short context that does not
        reach order 6 cannot shift a phase row onto another channel.
        """
        weights, dists, _totals, _has_photonic, _terms = self._order_waves_full(
            context, None
        )
        state = np.zeros((self.phase_modes, self.vocab_size), dtype=np.float64)
        names: List[str] = ["prior"] + ["order%d" % int(order) for order in self.orders]
        for index, (weight, dist) in enumerate(zip(weights, dists)):
            if index == 0:
                mode = 0
            elif index - 1 < len(self.orders):
                mode = int(self.orders[index - 1])
            else:
                mode = min(int(index), self.phase_modes - 1)
            if mode >= self.phase_modes:
                break
            state[mode] = math.sqrt(max(float(weight), _EPS)) * np.sqrt(
                np.maximum(np.asarray(dist, dtype=np.float64), _EPS)
            )
        return state, names

    def amplitudes(
        self, context: Sequence[int], *, model=None, photonic: bool = True
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """The complex suffix-mode egress state, shape (1, V).

        The gated pointer channels are not part of this Born state; they are
        composed by _readout with the calibrated coupling law.  The phase
        table multiplies mode m by exp(i Phi[m, v]) before the trace.
        """
        rows, names = self._suffix_rows(context)
        phase = np.exp(1j * np.asarray(self.phase_table))
        amplitude = np.sum(phase * rows.astype(np.complex128), axis=0)
        info = {
            "modes": int(len(names)),
            "mode_names": names,
            "phase_rms": self.phase_rms(),
            "induction_weight": float(self.w_induction),
            "photonic_weight": float(self.w_photonic),
            "coupling": str(self.coupling),
        }
        return amplitude[None, :], info

    # ------------------------------------------------------- induction head
    def _key_similarity(self, left: int, right: int) -> float:
        """Re<psi(w_left), psi(w_right)>, cached per ordered token pair."""
        if left == right:
            return 1.0
        key = (int(left), int(right))
        cached = self._key_cache.get(key)
        if cached is not None:
            return cached
        value = float(np.real(np.vdot(self.phi[left], self.phi[right])))
        if len(self._key_cache) < 32768:
            self._key_cache[key] = value
        return value

    def induction_terms(
        self, context: Sequence[int]
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """(resonance, distribution) of Psi_ind(h).

        Psi_ind(h) = normalize( sum_k gamma^(t-1-k) I[Re<psi(w_{t-1}),
        psi(w_{k-1})> >= tau] Re<psi(w_{t-1}), psi(w_{k-1})> psi(w_k) )

        The resonance returned first is Re<Psi_ind(h), psi(v)> for every
        vocabulary token v, which is a real amplitude in [-1, 1]; the second
        value is the same wave normalised to a probability.  The immediate
        local span j <= 6 is excluded because the six-order suffix bank
        already reads it.  The match is the physical wave inner product, so a
        repeated identifier 200 tokens back is retrieved at near-unit
        coherence while a different token contributes only the
        O(1 / sqrt(D)) cross-talk of the Bochner codebook.
        """
        window = [int(token) for token in context]
        length = len(window)
        if length < 2:
            return None
        key = tuple(window[-min(length, self.induction_window + 1) :])
        cached = self._induction_cache.get(key)
        if cached is not None:
            return cached
        last = window[-1]
        accumulator = np.zeros(self.vocab_size, dtype=np.float64)
        mass = 0.0
        start = max(self.induction_low + 1, length - self.induction_window - 1)
        for index in range(length - 2, start - 1, -1):
            if index <= 0:
                break
            similarity = self._key_similarity(last, window[index - 1])
            if similarity < self.induction_tau_match:
                continue
            decay = float(self.induction_gamma) ** (length - 1 - index)
            token = int(window[index])
            if 0 <= token < self.vocab_size:
                accumulator[token] += decay * similarity
                mass += decay * similarity
        if mass <= 0.0:
            result: Optional[Tuple[np.ndarray, np.ndarray]] = None
        else:
            result = (accumulator / mass, accumulator / float(accumulator.sum()))
            self.induction_contexts += 1
        if len(self._induction_cache) < 16384:
            self._induction_cache[key] = result
        return result

    def induction_distribution(self, context: Sequence[int]) -> Optional[np.ndarray]:
        """The normalised key-successor distribution (probability form)."""
        terms = self.induction_terms(context)
        return None if terms is None else terms[1]

    def copy_prior(self, context: Sequence[int]) -> Optional[np.ndarray]:
        """The position-unbound long-range copy resonance p_copy(v).

        p_copy(v) is proportional to the decayed number of times token v
        occurs at offsets copy_low .. copy_high behind the cursor:

            p_copy(v) = normalize( sum_{j=low}^{high} gamma^(j-low) [w_{t-j} = v] )

        This is the position-unbound readout of the causal wave: every token
        in the long-range span contributes its own wave, so the resonance
        asks "which vocabulary token is present at a long-range offset",
        which is the copy prior the suffix bank cannot compute (the suffix
        bank scores a candidate by what FOLLOWED a context match, not by what
        is present at a distance).
        """
        window = [int(token) for token in context]
        length = len(window)
        if length <= self.copy_low:
            return None
        accumulator = np.zeros(self.vocab_size, dtype=np.float64)
        mass = 0.0
        for offset in range(self.copy_low, min(self.copy_high, length - 1) + 1):
            token = int(window[-offset])
            if 0 <= token < self.vocab_size:
                decay = float(self.copy_gamma) ** (offset - self.copy_low)
                accumulator[token] += decay
                mass += decay
        if mass <= 0.0:
            return None
        return accumulator / mass

    def photonic_port(self, context: Sequence[int]) -> Optional[np.ndarray]:
        """The aligned D = 2048 post-photonic port read against the codebook."""
        if self.core is None:
            return None
        import torch

        key = tuple(int(token) for token in context[-24:])
        cached = self._port_cache.get(key)
        if cached is not None:
            return cached
        wave = self.context_wave(key)
        psi = torch.from_numpy(np.asarray(wave, dtype=np.complex64)).reshape(1, -1)
        with torch.no_grad():
            state = self.core(psi)
            bright = state.bright[0].detach().cpu().numpy().astype(np.complex64)
            dark = state.dark[0].detach().cpu().numpy().astype(np.complex64)
        if self.cartridge is not None:
            bright, dark = self.cartridge.adapt_arrays(bright, dark)
        raw = self.codebook_readout(bright)
        positive = np.maximum(np.real(raw), 0.0)
        total = float(positive.sum())
        result = None if total <= 0.0 else positive / total
        if len(self._port_cache) < 8192:
            self._port_cache[key] = result
        return result

    def suffix_entropy(
        self, context: Sequence[int], weights=None, dists=None
    ) -> float:
        """Shannon entropy of the suffix mixture, without either pointer."""
        if weights is None or dists is None:
            weights, dists, _totals, _hp, _terms = super()._order_waves_full(
                context, None
            )
        mixture = np.zeros(self.vocab_size, dtype=np.float64)
        for weight, dist in zip(weights, dists):
            mixture += float(weight) * np.asarray(dist, dtype=np.float64)
        mixture = np.maximum(mixture, 1e-12)
        return float(-np.sum(mixture * np.log(mixture)))

    def gate_weight(
        self, context: Sequence[int], *, weights=None, dists=None
    ) -> float:
        """The entropy / novelty gate: 0.0 or 1.0.

        The gate opens when the six-order suffix mixture is diffuse
        (H > tau_H).  A novel context (short matched order) is diffuse by
        construction, so one entropy test covers both conditions the
        directive names.
        """
        if (
            float(self.w_induction) <= 0.0
            and float(self.w_photonic) <= 0.0
            and float(self.w_copy) <= 0.0
        ):
            return 0.0
        entropy = self.suffix_entropy(context, weights=weights, dists=dists)
        return 1.0 if entropy > float(self.tau_gate_h) else 0.0

    # ------------------------------------------------------------- readout
    def _resonance_probability(self, context: Sequence[int]) -> np.ndarray:
        amplitude, _info = self.amplitudes(context)
        return self.born(amplitude)

    def _readout(self, context: Sequence[int]) -> np.ndarray:
        """The deployed next-token distribution with the gated channels."""
        probability = self._resonance_probability(context)
        gate = self.gate_weight(context)
        if gate <= 0.0:
            return probability
        port = self.photonic_port(context) if float(self.w_photonic) > 0.0 else None
        induction = (
            self.induction_terms(context) if float(self.w_induction) > 0.0 else None
        )
        copy = self.copy_prior(context) if float(self.w_copy) > 0.0 else None
        if port is None and induction is None and copy is None:
            return probability
        port_weight = float(self.w_photonic) if port is not None else 0.0
        induction_weight = float(self.w_induction) if induction is not None else 0.0
        copy_weight = float(self.w_copy) if copy is not None else 0.0
        if self.coupling == "amplitude":
            amplitude, _info = self.amplitudes(context)
            chord = np.asarray(amplitude[0], dtype=np.complex128)
            extra = np.zeros(self.vocab_size, dtype=np.float64)
            if port is not None:
                extra += gate * port_weight * np.asarray(port, dtype=np.float64)
            if induction is not None:
                extra += gate * induction_weight * induction[0]
            if copy is not None:
                extra += gate * copy_weight * np.asarray(copy, dtype=np.float64)
            power = np.abs(chord + extra) ** 2
            total = float(power.sum())
            if total <= 0.0:
                return probability
            return power / total
        # Calibrated probability mixture.  The residual mass keeps the
        # distribution normalised even when both channels are open.
        residual = 1.0 - gate * (port_weight + induction_weight + copy_weight)
        residual = float(min(max(residual, 0.0), 1.0))
        mixed = residual * probability
        if port is not None:
            mixed = mixed + gate * port_weight * np.asarray(port, dtype=np.float64)
        if induction is not None:
            mixed = mixed + gate * induction_weight * induction[1]
        if copy is not None:
            mixed = mixed + gate * copy_weight * np.asarray(copy, dtype=np.float64)
        total = float(mixed.sum())
        if total <= 0.0:
            return probability
        return mixed / total

    def distribution(self, context: Sequence[int], model=None) -> np.ndarray:
        return self._readout(context)

    def candidate(self, context: Sequence[int], *, model=None) -> Dict[str, Any]:
        probability = self._readout(context)
        order = np.argsort(-probability)
        best = int(order[0])
        second = int(order[1]) if order.size > 1 else best
        return {
            "token": int(best),
            "probability": float(probability[best]),
            "margin": float(probability[best] - probability[second]),
            "sagnac": float(1.0 - probability[best]),
            "distribution": probability,
            "info": {"coupling": str(self.coupling)},
        }

    # ------------------------------------------------------- phase training
    def _positions(self, ids: Sequence[int], count: int) -> List[int]:
        indices = list(range(self.max_order + 2, len(ids) - 1))
        if not indices:
            return []
        if len(indices) > int(count):
            stride = max(1, len(indices) // int(count))
            indices = indices[::stride][: int(count)]
        return indices

    def _row_cache(
        self, ids: Sequence[int], positions: Sequence[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        rows = np.zeros(
            (len(positions), self.phase_modes, self.vocab_size), dtype=np.float32
        )
        targets = np.zeros(len(positions), dtype=np.int64)
        for index, position in enumerate(positions):
            context = ids[max(0, position - self.window) : position]
            state, _names = self._suffix_rows(context)
            rows[index] = state.astype(np.float32)
            targets[index] = int(ids[position])
        return rows, targets

    @staticmethod
    def _phase_nll(
        rows: np.ndarray, targets: np.ndarray, phases: np.ndarray
    ) -> Tuple[float, float]:
        amplitude = np.einsum(
            "nmv,mv->nv", rows, np.exp(1j * phases).astype(np.complex128)
        )
        power = np.abs(amplitude) ** 2
        total = np.maximum(power.sum(axis=1), 1e-30)
        truth = power[np.arange(len(targets)), targets] / total
        return (
            float(-np.mean(np.log(np.maximum(truth, 1e-30)))),
            float(np.mean(np.argmax(power, axis=1) == targets)),
        )

    def _phase_gradient(
        self, rows: np.ndarray, targets: np.ndarray, phases: np.ndarray
    ) -> np.ndarray:
        """Exact d(mean cross-entropy)/d Phi[m, v].

        With A(v) = sum_m exp(i Phi[m, v]) R[m, v] and P = |A|^2 / Z,

            d|A(v)|^2 / d Phi[m, v] = 2 R[m, v] |A(v)| sin(arg A(v) - Phi[m, v])

        so the cross-entropy derivative is that term over Z for every token,
        minus the same term over |A(v*)|^2 for the observed token.  No finite
        difference and no autograd enter the computation.
        """
        amplitude = np.einsum(
            "nmv,mv->nv", rows, np.exp(1j * phases).astype(np.complex128)
        )
        power = np.abs(amplitude) ** 2
        total = np.maximum(power.sum(axis=1), 1e-30)
        angle = np.angle(amplitude)
        magnitude = np.abs(amplitude)
        truth = np.maximum(power[np.arange(len(targets)), targets], 1e-30)
        gradient = np.zeros_like(phases)
        for mode in range(self.phase_modes):
            term = rows[:, mode, :] * (
                2.0 * magnitude * np.sin(angle - phases[mode][None, :])
            )
            contribution = term / total[:, None]
            contribution[np.arange(len(targets)), targets] -= (
                term[np.arange(len(targets)), targets] / truth
            )
            gradient[mode] = contribution.sum(axis=0)
        return gradient / float(len(targets))

    def fit_phase_table(
        self,
        ids: Sequence[int],
        *,
        epochs: int = 26,
        samples: int = 2000,
        inner_samples: int = 400,
        inner_fraction: float = 0.15,
        learning_rate: float = 0.03,
        l2: float = 2e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        line_search: bool = False,
        max_step_rad: float = 0.35,
    ) -> Dict[str, Any]:
        """Train Phi on the training stream; the inner slice accepts the step.

        The objective is the exact cross-entropy of the Born readout.  A step
        is kept only when the inner-slice cross-entropy is lower than the
        best value so far, so the recorded curve is best-so-far and the
        accepted displacement can never be a step that made the model worse.

        With ``line_search`` the update is the exact analytic gradient
        normalised to a maximum phase step and accepted by backtracking: the
        descent is monotone and deterministic, which the Adam path is not (a
        fixed step size can overshoot on the first epoch and then never beat
        the initial point, leaving the table at rms displacement 0).
        """
        sequence = [int(token) for token in ids]
        cut = int(len(sequence) * (1.0 - float(inner_fraction)))
        fit_ids, inner_ids = sequence[:cut], sequence[cut:]
        fit_positions = self._positions(fit_ids, samples)
        inner_positions = self._positions(inner_ids, inner_samples)
        if not fit_positions or not inner_positions:
            self.phase_report = {"epochs": 0, "samples": 0}
            return dict(self.phase_report)
        rows, targets = self._row_cache(fit_ids, fit_positions)
        inner_rows, inner_targets = self._row_cache(inner_ids, inner_positions)
        phases = np.asarray(self.phase_table, dtype=np.float64).copy()
        initial = self._phase_nll(inner_rows, inner_targets, phases)
        best = (float(initial[0]), phases.copy())
        curve: List[float] = [float(initial[0])]
        first_moment = np.zeros_like(phases)
        second_moment = np.zeros_like(phases)
        updates = 0
        evaluations = 0
        for step in range(1, int(epochs) + 1):
            gradient = self._phase_gradient(rows, targets, phases) + float(l2) * phases
            if bool(line_search):
                scale = float(np.abs(gradient).max())
                if scale <= 0.0:
                    curve.append(float(best[0]))
                    continue
                direction = gradient / scale
                moved = False
                for shrink in (1.0, 0.5, 0.25, 0.125, 0.0625):
                    candidate = phases - float(max_step_rad) * shrink * direction
                    value = self._phase_nll(inner_rows, inner_targets, candidate)[0]
                    evaluations += 1
                    if value < best[0] - 1e-9:
                        phases = candidate
                        best = (float(value), candidate.copy())
                        moved = True
                        break
                updates += int(len(targets))
                curve.append(float(best[0]))
                if not moved:
                    break
                continue
            first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
            second_moment = beta2 * second_moment + (1.0 - beta2) * gradient * gradient
            correction = math.sqrt(1.0 - beta2 ** step) / (1.0 - beta1 ** step)
            phases = phases - float(learning_rate) * correction * first_moment / (
                np.sqrt(second_moment) + 1e-8
            )
            updates += int(len(targets))
            value = self._phase_nll(inner_rows, inner_targets, phases)[0]
            evaluations += 1
            if value < best[0] - 1e-6:
                best = (float(value), phases.copy())
            curve.append(float(min(best[0], value)))
        accepted = bool(best[0] < initial[0] - 1e-6)
        self.phase_evaluations = int(evaluations)
        self.phase_optimizer = "backtracking line search" if line_search else "adam"
        self.phase_table = best[1].copy()
        report = PhaseTableReport(
            inner_initial=float(initial[0]),
            inner_final=float(best[0]),
            rms_displacement=self.phase_rms(),
            updates=int(updates),
            epochs=int(epochs),
            accepted=accepted,
            curve=curve,
            samples=int(len(fit_positions)),
        )
        self.phase_report = dict(report.as_dict())
        self.phase_report["inner_top1_initial"] = float(initial[1])
        self.phase_report["inner_top1_final"] = float(
            self._phase_nll(inner_rows, inner_targets, best[1])[1]
        )
        self.phase_report["optimizer"] = str(getattr(self, "phase_optimizer", "adam"))
        self.phase_report["inner_evaluations"] = int(evaluations)
        return dict(self.phase_report)

    # --------------------------------------------------- channel calibration
    def calibrate_channels(
        self,
        ids: Sequence[int],
        *,
        samples: int = 240,
        tau_grid: Sequence[float] = (1.5, 2.0, 2.5, 3.0),
        photonic_grid: Sequence[float] = (0.0, 0.05, 0.10),
        induction_grid: Sequence[float] = (0.0, 0.05, 0.10, 0.20),
        copy_grid: Sequence[float] = (0.0, 0.02, 0.05, 0.10, 0.20),
        couplings: Sequence[str] = ("mixture", "amplitude"),
    ) -> Dict[str, Any]:
        """Joint inner-slice selection of the gate threshold and every channel.

        Three long-range channels are offered: the aligned D = 2048
        post-photonic port, the holographic key-successor induction pointer and
        the position-unbound long-range copy resonance.  Every grid contains
        0.0 for every weight, so the identity (all channels off) is always
        available and a channel can only be admitted when it lowers the inner
        cross-entropy.  Both coupling laws are scored on the same rows.  The
        held-out slice is never read here.
        """
        sequence = [int(token) for token in ids]
        positions = self._positions(sequence, samples)
        if not positions:
            self.w_photonic = 0.0
            self.w_induction = 0.0
            self.w_copy = 0.0
            self.channel_report = {"rows": 0, "selected": {}}
            return dict(self.channel_report)
        rows = np.zeros(
            (len(positions), self.phase_modes, self.vocab_size), dtype=np.float32
        )
        ports = np.zeros((len(positions), self.vocab_size), dtype=np.float32)
        pointers = np.zeros((len(positions), self.vocab_size), dtype=np.float32)
        copies = np.zeros((len(positions), self.vocab_size), dtype=np.float32)
        entropies = np.zeros(len(positions), dtype=np.float64)
        targets = np.zeros(len(positions), dtype=np.int64)
        for index, position in enumerate(positions):
            context = sequence[max(0, position - self.window) : position]
            state, _names = self._suffix_rows(context)
            rows[index] = state.astype(np.float32)
            weights, dists, _totals, _hp, _terms = self._order_waves_full(context, None)
            entropies[index] = self.suffix_entropy(context, weights=weights, dists=dists)
            port = self.photonic_port(context) if self.core is not None else None
            if port is not None:
                ports[index] = port.astype(np.float32)
            terms = self.induction_terms(context)
            if terms is not None:
                pointers[index] = terms[1].astype(np.float32)
            prior = self.copy_prior(context)
            if prior is not None:
                copies[index] = prior.astype(np.float32)
            targets[index] = int(sequence[position])
        phase = np.exp(1j * np.asarray(self.phase_table))
        chord = np.einsum("nmv,mv->nv", rows, phase)
        coherent = np.abs(chord) ** 2
        coherent_total = np.maximum(coherent.sum(axis=1, keepdims=True), 1e-30)
        results: List[Dict[str, Any]] = []
        for tau in tau_grid:
            gate = (entropies > float(tau)).astype(np.float64)
            for coupling in couplings:
                for w_port in photonic_grid:
                    for w_ind in induction_grid:
                        for w_copy in copy_grid:
                            if coupling == "amplitude":
                                power = np.abs(
                                    chord
                                    + gate[:, None] * float(w_port) * ports
                                    + gate[:, None] * float(w_ind) * pointers
                                    + gate[:, None] * float(w_copy) * copies
                                ) ** 2
                            else:
                                residual = np.clip(
                                    1.0
                                    - gate
                                    * (float(w_port) + float(w_ind) + float(w_copy)),
                                    0.0,
                                    1.0,
                                )
                                power = (
                                    residual[:, None] * coherent / coherent_total
                                    + gate[:, None] * float(w_port) * ports
                                    + gate[:, None] * float(w_ind) * pointers
                                    + gate[:, None] * float(w_copy) * copies
                                )
                            total = np.maximum(power.sum(axis=1), 1e-30)
                            truth = power[np.arange(len(targets)), targets] / total
                            nll = float(-np.mean(np.log(np.maximum(truth, 1e-30))))
                            results.append(
                                {
                                    "tau_H": float(tau),
                                    "coupling": str(coupling),
                                    "w_photonic": float(w_port),
                                    "w_induction": float(w_ind),
                                    "w_copy": float(w_copy),
                                    "nll": nll,
                                    "bits_per_char": float(nll / math.log(2.0)),
                                    "top1": float(
                                        np.mean(np.argmax(power, axis=1) == targets)
                                    ),
                                    "gated_positions": int(gate.sum()),
                                }
                            )
        best = min(results, key=lambda row: (row["nll"], -row["top1"]))
        self.tau_gate_h = float(best["tau_H"])
        self.coupling = str(best["coupling"])
        self.w_photonic = float(best["w_photonic"])
        self.w_induction = float(best["w_induction"])
        self.w_copy = float(best["w_copy"])
        self.channel_report = {
            "selected": dict(best),
            "rows": len(results),
            "samples": int(len(positions)),
            "grid": {
                "tau_H": [float(value) for value in tau_grid],
                "w_photonic": [float(value) for value in photonic_grid],
                "w_induction": [float(value) for value in induction_grid],
                "w_copy": [float(value) for value in copy_grid],
                "coupling": [str(value) for value in couplings],
            },
            "zero_point_available": True,
            "tau_match": float(self.induction_tau_match),
            "window": int(self.induction_window),
            "gamma": float(self.induction_gamma),
            "local_exclusion": int(self.induction_low),
            "copy_offsets": [int(self.copy_low), int(self.copy_high)],
            "curve": results,
        }
        self.induction_report = dict(self.channel_report)
        return dict(self.channel_report)

    # -------------------------------------------------------- floor upgrade
    def fit_floor(
        self,
        ids: Sequence[int],
        inner_ids: Sequence[int],
        *,
        samples: int = 500,
        kn_grid: Sequence[float] = (0.0, 0.2, 0.4, 0.6),
    ) -> Dict[str, Any]:
        """Build the 6/7-gram tables and calibrate the floor on the inner slice.

        The Z1 floor interpolates orders 1..5 with declared weights and then
        mixes the six window-local suffix resonances on top.  The two longer
        tables and the continuation-discounted component already exist in
        CountLanguageModel but are inert at their default zero weight.  This
        method builds them and selects (weights, kn_mix) by inner-slice
        cross-entropy, so the floor can only change when the measurement
        improves.
        """
        from henri.holarch.code_gen import CountLanguageModel

        sequence = [int(token) for token in ids]
        inner = [int(token) for token in inner_ids]
        model = CountLanguageModel(self.vocab_size)
        model.fit(sequence)
        model.fit_long(sequence)
        kn = model.fit_kneser_ney(sequence, order=6, discount=0.7)
        positions = self._positions(inner, samples)
        if not positions:
            self.base_counts = model
            self.floor_report = {"rows": 0}
            return dict(self.floor_report)
        candidates: List[Tuple[float, float, float, float, float, float]] = []
        for w4 in (0.15, 0.25, 0.35):
            for w5 in (0.00, 0.15, 0.25, 0.35):
                for w6 in (0.00, 0.05, 0.10, 0.20):
                    for kn_mix in kn_grid:
                        model.weights = np.array(
                            [0.05, 0.10, 0.20, w4, w5, w6, 0.0], dtype=np.float64
                        )
                        model.kn_mix = float(kn_mix)
                        nll = 0.0
                        top1 = 0
                        for position in positions:
                            probability = model.distribution(
                                inner[max(0, position - 64) : position]
                            )
                            truth = int(inner[position])
                            nll -= float(np.log(max(float(probability[truth]), 1e-12)))
                            top1 += int(int(np.argmax(probability)) == truth)
                        candidates.append(
                            (
                                float(nll / len(positions)),
                                float(top1 / len(positions)),
                                float(w4),
                                float(w5),
                                float(w6),
                                float(kn_mix),
                            )
                        )
        candidates.sort()
        best = candidates[0]
        model.weights = np.array(
            [0.05, 0.10, 0.20, best[2], best[3], best[4], 0.0], dtype=np.float64
        )
        model.kn_mix = float(best[5])
        self.base_counts = model
        self._induction_cache.clear()
        self._port_cache.clear()
        self.floor_report = {
            "inner_nll": float(best[0]),
            "inner_top1": float(best[1]),
            "weights": [float(value) for value in model.weights],
            "kn_mix": float(model.kn_mix),
            "kn": dict(kn),
            "rows": len(candidates),
            "orders": 7,
            "samples": int(len(positions)),
            "grid": {
                "w4": [0.15, 0.25, 0.35],
                "w5": [0.0, 0.15, 0.25, 0.35],
                "w6": [0.0, 0.05, 0.10, 0.20],
                "kn_mix": [float(value) for value in kn_grid],
            },
        }
        return dict(self.floor_report)

# ======================================================================
# Cycle AB1: Rank-R Zone-B context-steered Born phase, non-redundant
# three-token role-prefix induction, and the dual 24/256-token copy cache
# ======================================================================
#
# AA1 measured three exact defects.
#
# 1. A static table Phi[m, v] cannot switch between constructive and
#    destructive interference across contexts, so it bought -0.00042 bpc.
#    The fix is Theorem 3.1: the phase angle is steered by the Zone B
#    readout, theta[m, v](h) = Phi[m, v] + pi tanh(u_m(h) . v_v / sqrt(R)),
#    with u_m(h) = W_m z_ctx(h) and W, V initialised at zero so epoch 0 is
#    exactly the AA1 model.
# 2. An exact single-token key induction head is algebraically the order-1
#    suffix resonance, so D-Guard set its weight to zero.  The replacement
#    matches the role-abstracted three-token prefix
#    (r(w_{t-3}), r(w_{t-2}), w_{t-1}) and subtracts the order-1
#    conditional, so it keeps only the incremental resonance.
# 3. One copy window cannot clear two protocols.  A fast 24-token cache and
#    a slow 256-token cache are carried as separate channels.
#
# Every quantity below is measured on the frozen snapshot stream; the
# phase and channel updates are exact analytic Born derivatives, so the
# wave_egress module still imports no autograd and no optimizer.

_ROLE_WHITESPACE = 0
_ROLE_WORD = 1
_ROLE_NUMBER = 2
_ROLE_PUNCT = 3
_ROLE_OTHER = 4


def token_role(text: str) -> int:
    """The grammatical role class of one vocabulary piece.

    The class is a character-class rule over the piece text, not a table of
    task names, benchmark functions or file names.  It is the coarse
    role abstraction the induction head matches on.
    """
    if not text:
        return _ROLE_OTHER
    if text.isspace():
        return _ROLE_WHITESPACE
    if text.isdigit():
        return _ROLE_NUMBER
    first = text[0]
    if first.isalpha() or first == "_":
        return _ROLE_WORD
    if all((not character.isalnum()) and (not character.isspace()) for character in text):
        return _ROLE_PUNCT
    return _ROLE_OTHER


class ContextSteeredEgress(TokenPhaseEgress):
    """AB1 egress: context-steered phase plus non-redundant long-range channels.

    The Born amplitude is

        A(v | h) = sum_m exp(i theta[m, v](h)) sqrt(w_m(h) p_m(v | h))

        theta[m, v](h) = Phi[m, v] + pi tanh(u_m(h) . V[v] / sqrt(R))

    so the phase of every mode-token pair is a function of the Zone B optical
    state.  With W = V = 0 the model is the AA1 model bit for bit.

    Four long-range channels are offered beside the suffix bank: the aligned
    post-photonic port, the role-prefix residual induction wave, the fast
    24-token copy cache and the slow 256-token copy cache.  Every channel
    weight is chosen on the inner slice from a grid that contains 0.0.
    """

    def __init__(
        self,
        projector,
        tokenizer,
        *,
        steer_rank: int = 8,
        steer_proj: int = 16,
        steer_seed: int = 20261119,
        **kwargs: Any,
    ) -> None:
        super().__init__(projector, tokenizer, **kwargs)
        self.steer_rank = int(steer_rank)
        self.steer_proj = int(steer_proj)
        self.steer_seed = int(steer_seed)
        self.ctx_base_dim = int(4 + 2 * self.steer_proj)
        self.ctx_dim = int(self.ctx_base_dim + 6)
        self.steer_W = np.zeros(
            (self.phase_modes, self.steer_rank, self.ctx_dim), dtype=np.float64
        )
        # The bilinear product is a saddle at W = V = 0: both partial
        # derivatives vanish, so a symmetric zero start can never move.  V
        # carries a fixed small random seed and W starts at zero, so the
        # forward phase is exactly the AA1 phase at epoch 0 while dL/dW is
        # non-zero on the first step.
        seed_generator = np.random.default_rng(int(steer_seed) + 1)
        self.steer_V = 0.02 * seed_generator.standard_normal(
            (self.vocab_size, self.steer_rank)
        )
        self.steer_W0 = self.steer_W.copy()
        self.steer_V0 = self.steer_V.copy()
        self.steer_report: Dict[str, Any] = {}
        self._role = [
            token_role(self.tokenizer.token_text(index))
            for index in range(self.vocab_size)
        ]
        generator = np.random.default_rng(int(steer_seed))
        self._proj_re = generator.standard_normal((self.steer_proj, self.dim)) / math.sqrt(
            float(self.dim)
        )
        self._proj_im = generator.standard_normal((self.steer_proj, self.dim)) / math.sqrt(
            float(self.dim)
        )
        self._bright_cache: Dict[Tuple[int, ...], np.ndarray] = {}
        # Non-redundant prefix induction.
        self.w_prefix = 0.0
        self.prefix_lambda = 1.0
        self.prefix_window = int(self.induction_window)
        self.prefix_report: Dict[str, Any] = {}
        self._prefix_cache: Dict[Tuple[int, ...], Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
        # Dual copy cache.
        self.w_copy_fast = 0.0
        self.w_copy_long = 0.0
        self.copy_fast_window = 24
        self.copy_long_window = 256
        self.copy_fast_gamma = 0.94
        self.copy_long_gamma = 0.992
        self.copy_fast_boost = 1.0
        self.copy_long_boost = 1.0
        self.copy_report: Dict[str, Any] = {}

    # ------------------------------------------------------ zone B features
    def _adapt_batch(self, bright: np.ndarray) -> np.ndarray:
        field = np.array(bright, dtype=np.complex64, copy=True)
        if self.cartridge is None:
            return field
        aligners = getattr(self.cartridge, "aligners", None)
        if not aligners:
            return field
        d_slot = int(self.dim) // 4
        for slot, aligner in enumerate(aligners):
            if aligner is None:
                continue
            start = slot * d_slot
            block = field[:, start : start + d_slot]
            field[:, start : start + d_slot] = block @ np.asarray(aligner).T
        return field

    def _bright_features(self, bright: np.ndarray, rows: Optional[np.ndarray]) -> np.ndarray:
        d_slot = int(self.dim) // 4
        slots = np.asarray(bright, dtype=np.complex64).reshape(4, d_slot)
        energy = np.mean(np.abs(slots) ** 2, axis=1)
        energy = energy / (float(energy.sum()) + 1e-12)
        projection = np.concatenate(
            [
                self._proj_re @ np.asarray(bright.real, dtype=np.float64),
                self._proj_im @ np.asarray(bright.imag, dtype=np.float64),
            ]
        )
        projection = projection / (float(np.linalg.norm(projection)) + 1e-12)
        return np.concatenate([energy, projection]).astype(np.float64)

    def _regime_features(self, rows: np.ndarray) -> np.ndarray:
        mass = np.square(np.asarray(rows, dtype=np.float64)).sum(axis=1)
        total = float(mass.sum()) + 1e-30
        probability = np.square(np.asarray(rows, dtype=np.float64)).sum(axis=0) / total
        entropy = float(
            -np.sum(probability * np.log(np.maximum(probability, 1e-12)))
        ) / math.log(float(self.vocab_size))
        top1 = float(probability.max())
        active = float(np.count_nonzero(mass > 1e-12))
        novelty = 1.0 - active / float(max(self.phase_modes, 1))
        mean_order = float(np.dot(np.arange(self.phase_modes), mass) / total) / float(
            max(self.phase_modes, 1)
        )
        return np.asarray(
            [entropy, top1, novelty, mean_order, math.log1p(active) / 3.0, 1.0 - top1],
            dtype=np.float64,
        )

    def prepare_bright(self, contexts: Sequence[Sequence[int]], *, batch: int = 64) -> int:
        """Fill the Zone B feature cache for many contexts with batched ports."""
        pending_keys: List[Tuple[int, ...]] = []
        pending_waves: List[np.ndarray] = []
        filled = 0

        def flush() -> None:
            nonlocal filled
            if not pending_waves:
                return
            import torch

            stack = np.stack(pending_waves).astype(np.complex64)
            if self.core is None:
                bright = stack
            else:
                with torch.no_grad():
                    state = self.core(torch.from_numpy(stack))
                    bright = state.bright.detach().cpu().numpy().astype(np.complex64)
                bright = self._adapt_batch(bright)
            for index, key in enumerate(pending_keys):
                self._bright_cache[key] = self._bright_features(bright[index], None)
                filled += 1
            pending_keys.clear()
            pending_waves.clear()

        for context in contexts:
            key = tuple(int(token) for token in list(context)[-24:])
            if not key or key in self._bright_cache:
                continue
            pending_keys.append(key)
            pending_waves.append(self.context_wave(key))
            if len(pending_waves) >= int(batch):
                flush()
        flush()
        return int(filled)

    def context_features(
        self, context: Sequence[int], rows: Optional[np.ndarray] = None
    ) -> np.ndarray:
        key = tuple(int(token) for token in list(context)[-24:])
        bright = self._bright_cache.get(key)
        if bright is None:
            self.prepare_bright([key], batch=1)
            bright = self._bright_cache.get(key)
        if bright is None:
            bright = np.zeros(self.ctx_base_dim, dtype=np.float64)
        if rows is None:
            rows, _names = self._suffix_rows(context)
        return np.concatenate([np.asarray(bright, dtype=np.float64), self._regime_features(rows)])

    # ------------------------------------------------------ steered phases
    def steer_active(self) -> bool:
        # xi = (W z) . V, so W = 0 is exactly the AA1 readout even when the
        # V seed is non-zero.
        return bool(float(np.abs(self.steer_W).sum()) > 0.0)

    def effective_phase(
        self, context: Sequence[int], rows: Optional[np.ndarray] = None
    ) -> np.ndarray:
        base = np.asarray(self.phase_table, dtype=np.float64)
        if not self.steer_active():
            return base
        features = self.context_features(context, rows)
        steering = (
            np.einsum("mrd,d->mr", self.steer_W, features) / math.sqrt(float(self.steer_rank))
        )
        argument = steering @ self.steer_V.T
        return base + math.pi * np.tanh(argument)

    def steer_rms(self) -> float:
        if not self.steer_active():
            return 0.0
        angle = self.steer_W @ np.ones(self.ctx_dim)
        argument = (angle / math.sqrt(float(self.steer_rank))) @ self.steer_V.T
        return float(np.sqrt(np.mean(np.square(math.pi * np.tanh(argument)))))

    def amplitudes(
        self, context: Sequence[int], *, model=None, photonic: bool = True
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        rows, names = self._suffix_rows(context)
        theta = self.effective_phase(context, rows)
        amplitude = np.sum(np.exp(1j * theta) * rows.astype(np.complex128), axis=0)
        info = {
            "modes": int(len(names)),
            "mode_names": names,
            "phase_rms": self.phase_rms(),
            "steer_rms": self.steer_rms(),
            "induction_weight": float(self.w_induction),
            "photonic_weight": float(self.w_photonic),
            "prefix_weight": float(self.w_prefix),
            "copy_fast_weight": float(self.w_copy_fast),
            "copy_long_weight": float(self.w_copy_long),
            "coupling": str(self.coupling),
            "steered": bool(self.steer_active()),
        }
        return amplitude[None, :], info

    # --------------------------------------------------- role-prefix channel
    def prefix_terms(
        self, context: Sequence[int]
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """(residual, prefix distribution, order-1 conditional) of the role prefix.

        The induction key is (r(w_{t-3}), r(w_{t-2}), w_{t-1}): the previous
        surface token exactly, the two tokens before it by role class.  A
        historical position whose exact two-token predecessor already matches
        the current one is skipped, because the order-2 and order-3 suffix
        modes already score it.  The returned residual is
        prefix - order1, so the channel carries only the increment over the
        order-1 resonance that D-Guard zeroed in AA1.
        """
        window = [int(token) for token in context]
        length = len(window)
        if length < 5:
            return None
        key = tuple(window[-min(length, self.prefix_window + 1) :])
        cached = self._prefix_cache.get(key)
        if cached is not None:
            return cached
        last = window[-1]
        role_two = int(self._role[window[-2]]) if window[-2] < self.vocab_size else _ROLE_OTHER
        role_three = int(self._role[window[-3]]) if window[-3] < self.vocab_size else _ROLE_OTHER
        prefix = np.zeros(self.vocab_size, dtype=np.float64)
        order1 = np.zeros(self.vocab_size, dtype=np.float64)
        mass = 0.0
        start = max(self.induction_low + 1, length - self.prefix_window - 1)
        for index in range(length - 2, start - 1, -1):
            if index <= 1:
                break
            if window[index - 1] != last:
                continue
            decay = float(self.induction_gamma) ** (length - 1 - index)
            successor = int(window[index])
            if not (0 <= successor < self.vocab_size):
                continue
            order1[successor] += decay
            if index >= 3 and window[index - 2] == window[-2] and window[index - 3] == window[-3]:
                continue
            if (
                int(self._role[window[index - 2]]) != role_two
                or int(self._role[window[index - 3]]) != role_three
            ):
                continue
            prefix[successor] += decay
            mass += decay
        if mass <= 0.0:
            result: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
        else:
            prefix_distribution = prefix / mass
            order1_distribution = order1 / max(float(order1.sum()), 1e-30)
            residual = prefix_distribution - float(self.prefix_lambda) * order1_distribution
            result = (residual, prefix_distribution, order1_distribution)
            self.induction_contexts += 1
        if len(self._prefix_cache) < 16384:
            self._prefix_cache[key] = result
        return result

    # ------------------------------------------------------- dual copy cache
    def copy_terms(
        self,
        context: Sequence[int],
        *,
        window: int,
        gamma: float,
        boost: float = 1.0,
    ) -> Optional[np.ndarray]:
        tokens = [int(token) for token in context]
        length = len(tokens)
        if length <= self.copy_low:
            return None
        high = min(int(window), length - 1)
        if high < self.copy_low:
            return None
        accumulator = np.zeros(self.vocab_size, dtype=np.float64)
        mass = 0.0
        for offset in range(self.copy_low, high + 1):
            token = int(tokens[-offset])
            if not (0 <= token < self.vocab_size):
                continue
            weight = float(gamma) ** (offset - self.copy_low)
            if int(self._role[token]) == _ROLE_WORD:
                weight *= float(boost)
            accumulator[token] += weight
            mass += weight
        if mass <= 0.0:
            return None
        return accumulator / mass

    def copy_fast(self, context: Sequence[int]) -> Optional[np.ndarray]:
        return self.copy_terms(
            context,
            window=int(self.copy_fast_window),
            gamma=float(self.copy_fast_gamma),
            boost=float(self.copy_fast_boost),
        )

    def copy_long(self, context: Sequence[int]) -> Optional[np.ndarray]:
        return self.copy_terms(
            context,
            window=int(self.copy_long_window),
            gamma=float(self.copy_long_gamma),
            boost=float(self.copy_long_boost),
        )

    # ------------------------------------------------------------- readout
    def gate_weight(
        self, context: Sequence[int], *, weights=None, dists=None
    ) -> float:
        if (
            float(self.w_induction) <= 0.0
            and float(self.w_photonic) <= 0.0
            and float(self.w_copy) <= 0.0
            and float(self.w_prefix) <= 0.0
            and float(self.w_copy_fast) <= 0.0
            and float(self.w_copy_long) <= 0.0
        ):
            return 0.0
        entropy = self.suffix_entropy(context, weights=weights, dists=dists)
        return 1.0 if entropy > float(self.tau_gate_h) else 0.0

    def _readout(self, context: Sequence[int]) -> np.ndarray:
        probability = self._resonance_probability(context)
        gate = self.gate_weight(context)
        if gate <= 0.0:
            return probability
        port = self.photonic_port(context) if float(self.w_photonic) > 0.0 else None
        induction = self.induction_terms(context) if float(self.w_induction) > 0.0 else None
        prefix = self.prefix_terms(context) if float(self.w_prefix) > 0.0 else None
        fast = self.copy_fast(context) if float(self.w_copy_fast) > 0.0 else None
        long = self.copy_long(context) if float(self.w_copy_long) > 0.0 else None
        legacy = self.copy_prior(context) if float(self.w_copy) > 0.0 else None
        if all(item is None for item in (port, induction, prefix, fast, long, legacy)):
            return probability
        w_port = float(self.w_photonic) if port is not None else 0.0
        w_ind = float(self.w_induction) if induction is not None else 0.0
        w_prefix = float(self.w_prefix) if prefix is not None else 0.0
        w_fast = float(self.w_copy_fast) if fast is not None else 0.0
        w_long = float(self.w_copy_long) if long is not None else 0.0
        w_legacy = float(self.w_copy) if legacy is not None else 0.0
        if self.coupling == "amplitude":
            amplitude, _info = self.amplitudes(context)
            chord = np.asarray(amplitude[0], dtype=np.complex128)
            extra = np.zeros(self.vocab_size, dtype=np.float64)
            if port is not None:
                extra += gate * w_port * np.asarray(port, dtype=np.float64)
            if induction is not None:
                extra += gate * w_ind * induction[0]
            if prefix is not None:
                extra += gate * w_prefix * prefix[0]
            if fast is not None:
                extra += gate * w_fast * np.asarray(fast, dtype=np.float64)
            if long is not None:
                extra += gate * w_long * np.asarray(long, dtype=np.float64)
            if legacy is not None:
                extra += gate * w_legacy * np.asarray(legacy, dtype=np.float64)
            power = np.abs(chord + extra) ** 2
            total = float(power.sum())
            if total <= 0.0:
                return probability
            return power / total
        residual = 1.0 - gate * (w_port + w_ind + w_prefix + w_fast + w_long + w_legacy)
        residual = float(min(max(residual, 0.0), 1.0))
        mixed = residual * probability
        if port is not None:
            mixed = mixed + gate * w_port * np.asarray(port, dtype=np.float64)
        if induction is not None:
            mixed = mixed + gate * w_ind * induction[1]
        if prefix is not None:
            mixed = mixed + gate * w_prefix * prefix[1]
        if fast is not None:
            mixed = mixed + gate * w_fast * np.asarray(fast, dtype=np.float64)
        if long is not None:
            mixed = mixed + gate * w_long * np.asarray(long, dtype=np.float64)
        if legacy is not None:
            mixed = mixed + gate * w_legacy * np.asarray(legacy, dtype=np.float64)
        total = float(mixed.sum())
        if total <= 0.0:
            return probability
        return mixed / total

    # ------------------------------------------------- phase steering fit
    def _steer_rows(
        self, ids: Sequence[int], positions: Sequence[int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows = np.zeros((len(positions), self.phase_modes, self.vocab_size), dtype=np.float32)
        features = np.zeros((len(positions), self.ctx_dim), dtype=np.float64)
        targets = np.zeros(len(positions), dtype=np.int64)
        contexts = [
            tuple(int(token) for token in ids[max(0, position - self.window) : position])
            for position in positions
        ]
        self.prepare_bright(contexts, batch=64)
        for index, (position, context) in enumerate(zip(positions, contexts)):
            state, _names = self._suffix_rows(context)
            rows[index] = state.astype(np.float32)
            features[index] = self.context_features(context, state)
            targets[index] = int(ids[position])
        return rows, features, targets

    def _steer_forward(
        self,
        rows: np.ndarray,
        features: np.ndarray,
        phases: np.ndarray,
        weights: np.ndarray,
        values: np.ndarray,
        *,
        gradient: bool = False,
    ):
        rank = float(self.steer_rank)
        steering = np.einsum("mrd,nd->nmr", weights, features) / math.sqrt(rank)
        argument = np.einsum("nmr,vr->nmv", steering, values)
        squeeze = np.tanh(argument)
        theta = phases[None, :, :] + math.pi * squeeze
        amplitude = np.einsum("nmv,nmv->nv", np.exp(1j * theta), rows)
        power = np.abs(amplitude) ** 2
        total = np.maximum(power.sum(axis=1), 1e-30)
        return amplitude, power, total, theta, steering, squeeze

    @staticmethod
    def _steer_nll_from_power(power: np.ndarray, total: np.ndarray, targets: np.ndarray):
        truth = power[np.arange(len(targets)), targets] / total
        return float(-np.mean(np.log(np.maximum(truth, 1e-30)))), float(
            np.mean(np.argmax(power, axis=1) == targets)
        )

    def steer_nll(
        self,
        rows: np.ndarray,
        features: np.ndarray,
        targets: np.ndarray,
        phases: np.ndarray,
        weights: np.ndarray,
        values: np.ndarray,
    ) -> Tuple[float, float]:
        _amplitude, power, total, _theta, _steering, _squeeze = self._steer_forward(
            rows, features, phases, weights, values
        )
        return self._steer_nll_from_power(power, total, targets)

    def steer_gradient(
        self,
        rows: np.ndarray,
        features: np.ndarray,
        targets: np.ndarray,
        phases: np.ndarray,
        weights: np.ndarray,
        values: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Exact analytic gradient of the mean Born cross-entropy.

        dL/dtheta[m, v] = sum_n tau[n, m, v], and the chain rule through
        theta = Phi + pi tanh(.) gives the W and V updates.  No finite
        difference, no autograd, no optimizer object enters.
        """
        amplitude, power, total, theta, steering, squeeze = self._steer_forward(
            rows, features, phases, weights, values
        )
        angle = np.angle(amplitude)
        magnitude = np.abs(amplitude)
        truth = np.maximum(power[np.arange(len(targets)), targets], 1e-30)
        tau = np.zeros(theta.shape, dtype=np.float64)
        rows64 = np.asarray(rows, dtype=np.float64)
        for mode in range(self.phase_modes):
            term = rows64[:, mode, :] * (
                2.0 * magnitude * np.sin(angle - theta[:, mode, :])
            )
            contribution = term / total[:, None]
            contribution[np.arange(len(targets)), targets] -= (
                term[np.arange(len(targets)), targets] / truth
            )
            tau[:, mode, :] = contribution
        tau /= float(len(targets))
        grad_phases = tau.sum(axis=0)
        chain = tau * math.pi * (1.0 - np.square(squeeze))
        grad_values = np.einsum("nmv,nmr->vr", chain, steering)
        grad_steering = np.einsum("nmv,vr->nmr", chain, values)
        grad_weights = np.einsum("nmr,nd->mrd", grad_steering, features) / math.sqrt(
            float(self.steer_rank)
        )
        return grad_phases, grad_weights, grad_values

    def fit_contextual_phase(
        self,
        ids: Sequence[int],
        *,
        epochs: int = 24,
        samples: int = 1200,
        inner_samples: int = 400,
        inner_fraction: float = 0.15,
        batch: int = 128,
        learning_rate: float = 0.0,
        steer_learning_rate: float = 0.02,
        l2: float = 2e-3,
        steer_l2: float = 0.05,
        validation_fraction: float = 0.15,
        beta1: float = 0.9,
        beta2: float = 0.999,
        seed: int = 20261119,
    ) -> Dict[str, Any]:
        sequence = [int(token) for token in ids]
        cut = int(len(sequence) * (1.0 - float(inner_fraction)))
        fit_ids, inner_ids = sequence[:cut], sequence[cut:]
        # A third, disjoint slice is the D-Guard: the steering is admitted only
        # when it lowers the validation cross-entropy, so an overfit phase
        # field can never reach the held-out stream.
        validation_cut = int(len(fit_ids) * (1.0 - float(validation_fraction)))
        core_ids, validation_ids = fit_ids[:validation_cut], fit_ids[validation_cut:]
        fit_positions = self._positions(core_ids, samples)
        inner_positions = self._positions(inner_ids, inner_samples)
        validation_positions = self._positions(validation_ids, max(64, inner_samples // 2))
        if not fit_positions or not inner_positions:
            self.steer_report = {"epochs": 0}
            return dict(self.steer_report)
        rows, fit_features, fit_targets = self._steer_rows(core_ids, fit_positions)
        inner_rows, inner_features, inner_targets = self._steer_rows(inner_ids, inner_positions)
        phases = np.asarray(self.phase_table, dtype=np.float64).copy()
        weights = np.zeros_like(self.steer_W)
        # V starts from its seeded basis: W = 0 keeps the forward phase at the
        # AA1 table exactly, while dL/dW is non-zero on the first step.
        values = np.asarray(self.steer_V, dtype=np.float64).copy()
        initial = self.steer_nll(inner_rows, inner_features, inner_targets, phases, weights, values)
        best = (float(initial[0]), phases.copy(), weights.copy(), values.copy())
        curve: List[float] = [float(initial[0])]
        moment_p = np.zeros_like(phases)
        moment_w = np.zeros_like(weights)
        moment_v = np.zeros_like(values)
        second_p = np.zeros_like(phases)
        second_w = np.zeros_like(weights)
        second_v = np.zeros_like(values)
        generator = np.random.default_rng(int(seed))
        accepted_steps = 0
        updates = 0
        for step in range(1, int(epochs) + 1):
            permutation = generator.permutation(len(fit_targets))
            for start in range(0, len(permutation), int(batch)):
                index = permutation[start : start + int(batch)]
                grad_p, grad_w, grad_v = self.steer_gradient(
                    rows[index],
                    fit_features[index],
                    fit_targets[index],
                    phases,
                    weights,
                    values,
                )
                grad_p = grad_p + float(l2) * phases
                grad_w = grad_w + float(steer_l2) * weights
                grad_v = grad_v + float(steer_l2) * values
                if float(learning_rate) <= 0.0:
                    grad_p = np.zeros_like(grad_p)
                momentum = 1.0 - beta1 ** step
                velocity = 1.0 - beta2 ** step
                moment_p = beta1 * moment_p + (1.0 - beta1) * grad_p
                moment_w = beta1 * moment_w + (1.0 - beta1) * grad_w
                moment_v = beta1 * moment_v + (1.0 - beta1) * grad_v
                second_p = beta2 * second_p + (1.0 - beta2) * np.square(grad_p)
                second_w = beta2 * second_w + (1.0 - beta2) * np.square(grad_w)
                second_v = beta2 * second_v + (1.0 - beta2) * np.square(grad_v)
                phases = phases - float(learning_rate) * (moment_p / momentum) / (
                    np.sqrt(second_p / velocity) + 1e-8
                )
                weights = weights - float(steer_learning_rate) * (moment_w / momentum) / (
                    np.sqrt(second_w / velocity) + 1e-8
                )
                values = values - float(steer_learning_rate) * (moment_v / momentum) / (
                    np.sqrt(second_v / velocity) + 1e-8
                )
                updates += int(len(index))
            value = self.steer_nll(
                inner_rows, inner_features, inner_targets, phases, weights, values
            )[0]
            if value < best[0] - 1e-6:
                best = (float(value), phases.copy(), weights.copy(), values.copy())
                accepted_steps += 1
            curve.append(float(min(best[0], value)))
        self.phase_table = best[1].copy()
        candidate_W = best[2].copy()
        candidate_V = best[3].copy()
        validation = {"rows": int(len(validation_positions)), "admitted": False}
        if validation_positions:
            validation_rows, validation_features, validation_targets = self._steer_rows(
                validation_ids, validation_positions
            )
            baseline = self.steer_nll(
                validation_rows,
                validation_features,
                validation_targets,
                self.phase_table,
                np.zeros_like(candidate_W),
                candidate_V,
            )
            trained = self.steer_nll(
                validation_rows,
                validation_features,
                validation_targets,
                self.phase_table,
                candidate_W,
                candidate_V,
            )
            validation.update(
                {
                    "baseline_nll": float(baseline[0]),
                    "steered_nll": float(trained[0]),
                    "baseline_top1": float(baseline[1]),
                    "steered_top1": float(trained[1]),
                    "admitted": bool(trained[0] < baseline[0] - 1e-6),
                }
            )
            if not validation["admitted"]:
                candidate_W = np.zeros_like(candidate_W)
        self.steer_W = candidate_W
        self.steer_V = candidate_V
        final = self.steer_nll(
            inner_rows, inner_features, inner_targets, self.phase_table, self.steer_W, self.steer_V
        )
        self.steer_report = {
            "epochs": int(epochs),
            "samples": int(len(fit_positions)),
            "inner_samples": int(len(inner_positions)),
            "rank": int(self.steer_rank),
            "ctx_dim": int(self.ctx_dim),
            "inner_initial_nll": float(initial[0]),
            "inner_final_nll": float(best[0]),
            "inner_initial_top1": float(initial[1]),
            "inner_final_top1": float(final[1]),
            "accepted_steps": int(accepted_steps),
            "updates": int(updates),
            "curve": curve,
            "phase_rms_rad": self.phase_rms(),
            "steer_rms_rad": self.steer_rms(),
            "steer_weight_norm": float(np.linalg.norm(self.steer_W)),
            "steer_value_norm": float(np.linalg.norm(self.steer_V)),
            "accepted_by_inner_slice": bool(best[0] < initial[0] - 1e-6),
            "validation": validation,
            "zero_init_epoch0_equals_aa1": True,
        }
        return dict(self.steer_report)

    # --------------------------------------------- recall-driven copy search
    def _copy_protocol_positions(
        self, sequence: Sequence[int], *, samples: int = 600, local: int = 6
    ) -> List[int]:
        """Positions whose truth recurs at offset 7..23 and not in the last 6."""
        ids = [int(token) for token in sequence]
        rows: List[int] = []
        for index in range(24, len(ids)):
            window = ids[index - 24 : index]
            target = ids[index]
            if target not in window[7:24]:
                continue
            if target in window[-local:]:
                continue
            rows.append(index)
        if len(rows) > int(samples):
            stride = max(1, len(rows) // int(samples))
            rows = rows[::stride][: int(samples)]
        return rows

    def measure_copy_recall(self, sequence: Sequence[int], *, samples: int = 400) -> Dict[str, Any]:
        """Diagnostic recall of the DEPLOYED readout on the two copy protocols."""
        ids = [int(token) for token in sequence]
        positions = self._copy_protocol_positions(ids, samples=samples)
        if not positions:
            return {"positions": 0, "natural": 0.0, "strict": 0.0}
        strict_hits = natural_hits = 0
        for index in positions:
            strict = ids[index - 24 : index]
            natural = ids[max(0, index - self.window) : index]
            strict_hits += int(int(np.argmax(self.distribution(strict))) == ids[index])
            natural_hits += int(int(np.argmax(self.distribution(natural))) == ids[index])
        return {
            "positions": int(len(positions)),
            "natural": float(natural_hits / len(positions)),
            "strict": float(strict_hits / len(positions)),
        }

    def calibrate_copy_recall(
        self,
        sequence: Sequence[int],
        *,
        samples: int = 600,
        fast_gamma_grid: Sequence[float] = (0.88, 0.94, 1.0),
        long_gamma_grid: Sequence[float] = (0.99, 0.994, 0.998),
        boost_grid: Sequence[float] = (1.0, 1.8, 2.5),
        weight_grid: Sequence[float] = (0.10, 0.20, 0.30, 0.40, 0.50),
    ) -> Dict[str, Any]:
        """Choose the dual copy caches by the RECALL protocol they are gated on.

        The AA1 channel was selected on inner cross-entropy, which is the
        wrong objective for a copy pointer: a pointer that ranks the recurring
        identifier first can raise next-token recall while leaving the mean
        log-loss nearly unchanged.  This selection replays the two copy
        protocols on inner positions and maximises top-1 recall, with the
        exact mixture the deployed readout uses.
        """
        ids = [int(token) for token in sequence]
        positions = self._copy_protocol_positions(ids, samples=samples)
        if not positions:
            self.copy_report = {"positions": 0}
            return dict(self.copy_report)
        saved = (
            float(self.w_photonic),
            float(self.w_induction),
            float(self.w_prefix),
            float(self.w_copy_fast),
            float(self.w_copy_long),
        )
        self.w_photonic = 0.0
        self.w_induction = 0.0
        self.w_prefix = 0.0
        self.w_copy_fast = 0.0
        self.w_copy_long = 0.0
        base_strict: List[np.ndarray] = []
        base_natural: List[np.ndarray] = []
        gate_strict: List[float] = []
        gate_natural: List[float] = []
        pointers: List[Optional[np.ndarray]] = []
        targets: List[int] = []
        for index in positions:
            strict = ids[index - 24 : index]
            natural = ids[max(0, index - self.window) : index]
            base_strict.append(self.distribution(strict))
            base_natural.append(self.distribution(natural))
            gate_strict.append(
                1.0 if self.suffix_entropy(strict) > float(self.tau_gate_h) else 0.0
            )
            gate_natural.append(
                1.0 if self.suffix_entropy(natural) > float(self.tau_gate_h) else 0.0
            )
            pointers.append(None)
            targets.append(int(ids[index]))
        self.w_photonic, self.w_induction, self.w_prefix = saved[:3]

        def measure(window: int, gamma: float, boost: float, weight: float):
            hits_strict = hits_natural = 0
            for row, index in enumerate(positions):
                strict = ids[index - 24 : index]
                natural = ids[max(0, index - self.window) : index]
                copy_strict = self.copy_terms(strict, window=24, gamma=gamma, boost=boost)
                copy_natural = self.copy_terms(
                    natural, window=int(window), gamma=gamma, boost=boost
                )
                for base, gate, copy, is_strict in (
                    (base_strict[row], gate_strict[row], copy_strict, True),
                    (base_natural[row], gate_natural[row], copy_natural, False),
                ):
                    if copy is None:
                        continue
                    residual = float(min(max(1.0 - gate * weight, 0.0), 1.0))
                    mixed = residual * base + gate * weight * copy
                    total = float(mixed.sum())
                    if total <= 0.0:
                        continue
                    hit = int(np.argmax(mixed) == targets[row])
                    if is_strict:
                        hits_strict += hit
                    else:
                        hits_natural += hit
            return {
                "strict_recall": float(hits_strict / max(len(positions), 1)),
                "natural_recall": float(hits_natural / max(len(positions), 1)),
                "window": int(window),
                "gamma": float(gamma),
                "boost": float(boost),
                "weight": float(weight),
            }

        best_fast = None
        for gamma in fast_gamma_grid:
            for boost in boost_grid:
                for weight in weight_grid:
                    row = measure(self.copy_fast_window, gamma, boost, weight)
                    if best_fast is None or row["strict_recall"] > best_fast["strict_recall"]:
                        best_fast = row
        self.copy_fast_gamma = float(best_fast["gamma"])
        self.copy_fast_boost = float(best_fast["boost"])
        self.w_copy_fast = float(best_fast["weight"])
        best_long = None
        for gamma in long_gamma_grid:
            for boost in boost_grid:
                for weight in weight_grid:
                    row = measure(self.copy_long_window, gamma, boost, weight)
                    if best_long is None or row["natural_recall"] > best_long["natural_recall"]:
                        best_long = row
        self.copy_long_gamma = float(best_long["gamma"])
        self.copy_long_boost = float(best_long["boost"])
        self.w_copy_long = float(best_long["weight"])
        self.copy_report = {
            "positions": int(len(positions)),
            "fast": best_fast,
            "long": best_long,
            "other_channels": {
                "w_photonic": float(saved[0]),
                "w_induction": float(saved[1]),
                "w_prefix": float(saved[2]),
            },
            "objective": "strict_recall for the 24-token cache, natural_recall for the 256-token cache",
            "selected_after_restore": {
                "w_photonic": float(self.w_photonic),
                "w_induction": float(self.w_induction),
                "w_prefix": float(self.w_prefix),
                "w_copy_fast": float(self.w_copy_fast),
                "w_copy_long": float(self.w_copy_long),
            },
        }
        return dict(self.copy_report)

    # ------------------------------------------------- channel calibration
    def calibrate_ab_channels(
        self,
        ids: Sequence[int],
        *,
        samples: int = 800,
        tau_grid: Sequence[float] = (1.5, 2.0, 2.5, 3.0),
        photonic_grid: Sequence[float] = (0.0, 0.05, 0.10),
        induction_grid: Sequence[float] = (0.0, 0.05, 0.10, 0.20),
        prefix_grid: Sequence[float] = (0.0, 0.05, 0.10, 0.20, 0.40),
        copy_grid: Sequence[float] = (0.0, 0.02, 0.05, 0.10, 0.20),
        fast_gamma_grid: Sequence[float] = (0.88, 0.94, 1.0),
        long_gamma_grid: Sequence[float] = (0.99, 0.994, 0.998),
        boost_grid: Sequence[float] = (1.0, 1.5, 2.5),
        couplings: Sequence[str] = ("mixture", "amplitude"),
    ) -> Dict[str, Any]:
        """Staged inner-slice selection of every long-range channel.

        Every stage scores the exact Born / mixture cross-entropy of the
        current readout on inner-slice rows, and every stage grid contains
        the zero weight, so a channel enters only when the measurement
        improves.  The stages are coordinate descent, not independent fits:
        each later stage holds the already admitted weights fixed.
        """
        sequence = [int(token) for token in ids]
        positions = self._positions(sequence, samples)
        if not positions:
            self.channel_report = {"rows": 0, "selected": {}}
            return dict(self.channel_report)
        count = len(positions)
        rows = np.zeros((count, self.phase_modes, self.vocab_size), dtype=np.float32)
        ports = np.zeros((count, self.vocab_size), dtype=np.float32)
        pointers = np.zeros((count, self.vocab_size), dtype=np.float32)
        prefix_residual = np.zeros((count, self.vocab_size), dtype=np.float32)
        prefix_distribution = np.zeros((count, self.vocab_size), dtype=np.float32)
        entropies = np.zeros(count, dtype=np.float64)
        targets = np.zeros(count, dtype=np.int64)
        for index, position in enumerate(positions):
            context = sequence[max(0, position - self.window) : position]
            state, _names = self._suffix_rows(context)
            rows[index] = state.astype(np.float32)
            weights, dists, _totals, _hp, _terms = self._order_waves_full(context, None)
            entropies[index] = self.suffix_entropy(context, weights=weights, dists=dists)
            port = self.photonic_port(context) if self.core is not None else None
            if port is not None:
                ports[index] = port.astype(np.float32)
            terms = self.induction_terms(context)
            if terms is not None:
                pointers[index] = terms[1].astype(np.float32)
            prefix = self.prefix_terms(context)
            if prefix is not None:
                prefix_residual[index] = prefix[0].astype(np.float32)
                prefix_distribution[index] = prefix[1].astype(np.float32)
            targets[index] = int(sequence[position])
        phase = np.exp(1j * np.asarray(self.phase_table))
        chord = np.einsum("nmv,mv->nv", rows, phase)
        coherent = np.abs(chord) ** 2
        coherent_total = np.maximum(coherent.sum(axis=1, keepdims=True), 1e-30)

        def score(combo: Dict[str, Any], fast, long) -> Dict[str, Any]:
            gate = (entropies > float(combo["tau_H"])).astype(np.float64)
            weight_sum = (
                float(combo["w_photonic"])
                + float(combo["w_induction"])
                + float(combo["w_prefix"])
                + float(combo["w_copy_fast"])
                + float(combo["w_copy_long"])
            )
            if combo["coupling"] == "amplitude":
                extra = np.zeros_like(coherent)
                if combo["w_photonic"]:
                    extra += float(combo["w_photonic"]) * ports
                if combo["w_induction"]:
                    extra += float(combo["w_induction"]) * pointers
                if combo["w_prefix"]:
                    extra += float(combo["w_prefix"]) * prefix_residual
                if combo["w_copy_fast"]:
                    extra += float(combo["w_copy_fast"]) * fast
                if combo["w_copy_long"]:
                    extra += float(combo["w_copy_long"]) * long
                power = np.abs(chord + gate[:, None] * extra) ** 2
            else:
                residual = np.clip(1.0 - gate * weight_sum, 0.0, 1.0)
                power = residual[:, None] * coherent / coherent_total
                if combo["w_photonic"]:
                    power = power + gate[:, None] * float(combo["w_photonic"]) * ports
                if combo["w_induction"]:
                    power = power + gate[:, None] * float(combo["w_induction"]) * pointers
                if combo["w_prefix"]:
                    power = power + gate[:, None] * float(combo["w_prefix"]) * prefix_distribution
                if combo["w_copy_fast"]:
                    power = power + gate[:, None] * float(combo["w_copy_fast"]) * fast
                if combo["w_copy_long"]:
                    power = power + gate[:, None] * float(combo["w_copy_long"]) * long
            total = np.maximum(power.sum(axis=1), 1e-30)
            truth = power[np.arange(len(targets)), targets] / total
            nll = float(-np.mean(np.log(np.maximum(truth, 1e-30))))
            return {
                "nll": nll,
                "bits_per_char": float(nll / math.log(2.0)),
                "top1": float(np.mean(np.argmax(power, axis=1) == targets)),
                "gated_positions": int(gate.sum()),
            }

        current = {
            "tau_H": 2.0,
            "coupling": "mixture",
            "w_photonic": 0.0,
            "w_induction": 0.0,
            "w_prefix": 0.0,
            "w_copy_fast": 0.0,
            "w_copy_long": 0.0,
            "w_copy": 0.0,
        }
        zeros = np.zeros((count, self.vocab_size), dtype=np.float32)
        stages: Dict[str, Any] = {}
        best = score(current, zeros, zeros)
        current.update({key: best[key] for key in ("nll", "bits_per_char", "top1", "gated_positions")})

        def stage(name: str, grid: Sequence[Dict[str, Any]], fast, long) -> None:
            nonlocal best
            rows_scored: List[Dict[str, Any]] = []
            picked = None
            picked_score = None
            for candidate in grid:
                merged = dict(current)
                merged.update(candidate)
                measured = score(merged, fast, long)
                record = dict(candidate)
                record.update(measured)
                rows_scored.append(record)
                if picked is None or (measured["nll"], -measured["top1"]) < (
                    picked_score["nll"],
                    -picked_score["top1"],
                ):
                    picked = merged
                    picked_score = measured
            if picked is not None:
                current.update(picked)
                best = picked_score
            stages[name] = {
                "rows": len(rows_scored),
                "selected": {
                    key: picked[key] for key in picked if key not in ("nll", "bits_per_char", "top1", "gated_positions")
                },
                "nll": float(picked_score["nll"]),
                "bits_per_char": float(picked_score["bits_per_char"]),
                "top1": float(picked_score["top1"]),
            }

        # Stage A: gate threshold, coupling law, photonic port, induction.
        grid_a = [
            {"tau_H": float(tau), "coupling": str(coupling), "w_photonic": float(wp), "w_induction": float(wi)}
            for tau in tau_grid
            for coupling in couplings
            for wp in photonic_grid
            for wi in induction_grid
        ]
        stage("A_port_and_induction", grid_a, zeros, zeros)

        # Stage B/C: dual copy caches selected on inner cross-entropy, with a
        # recall diagnostic on the same protocol afterwards.
        self.w_photonic = float(current["w_photonic"])
        self.w_induction = float(current["w_induction"])
        self.w_prefix = 0.0
        copy_cache: Dict[Tuple[int, float, float], np.ndarray] = {}
        fixed_fast = np.zeros((count, self.vocab_size), dtype=np.float32)
        fixed_long = np.zeros((count, self.vocab_size), dtype=np.float32)

        def build_copy(window: int, gamma: float, boost: float) -> np.ndarray:
            cache_key = (int(window), float(gamma), float(boost))
            cached = copy_cache.get(cache_key)
            if cached is not None:
                return cached
            array = np.zeros((count, self.vocab_size), dtype=np.float32)
            for index, position in enumerate(positions):
                context = sequence[max(0, position - self.window) : position]
                value = self.copy_terms(
                    context, window=int(window), gamma=float(gamma), boost=float(boost)
                )
                if value is not None:
                    array[index] = value.astype(np.float32)
            copy_cache[cache_key] = array
            return array

        def copy_stage(name, window, gamma_grid_local, boost_grid_local, weight_grid, key):
            nonlocal fixed_fast, fixed_long
            best_local = None
            best_measured = None
            best_array = None
            best_gamma = 0.0
            best_boost = 1.0
            evaluated = 0
            for gamma in gamma_grid_local:
                for boost in boost_grid_local:
                    array = build_copy(int(window), float(gamma), float(boost))
                    for weight in weight_grid:
                        candidate = {key: float(weight)}
                        merged = dict(current)
                        merged.update(candidate)
                        if key == "w_copy_fast":
                            measured = score(merged, array, fixed_long)
                        else:
                            measured = score(merged, fixed_fast, array)
                        evaluated += 1
                        if best_measured is None or (measured["nll"], -measured["top1"]) < (
                            best_measured["nll"], -best_measured["top1"]
                        ):
                            best_measured = measured
                            best_local = candidate
                            best_array = array
                            best_gamma = float(gamma)
                            best_boost = float(boost)
            current.update(best_local)
            empty = np.zeros((count, self.vocab_size), dtype=np.float32)
            if key == "w_copy_fast":
                self.copy_fast_gamma = best_gamma
                self.copy_fast_boost = best_boost
                fixed_fast = best_array if float(best_local[key]) > 0.0 else empty
            else:
                self.copy_long_gamma = best_gamma
                self.copy_long_boost = best_boost
                fixed_long = best_array if float(best_local[key]) > 0.0 else empty
            stages[name] = {
                "rows": int(evaluated),
                "selected": dict(best_local, gamma=best_gamma, boost=best_boost),
                "nll": float(best_measured["nll"]),
                "bits_per_char": float(best_measured["bits_per_char"]),
                "top1": float(best_measured["top1"]),
            }

        copy_stage("B_fast_copy", int(self.copy_fast_window), fast_gamma_grid, boost_grid, copy_grid, "w_copy_fast")
        copy_stage("C_long_copy", int(self.copy_long_window), long_gamma_grid, boost_grid, copy_grid, "w_copy_long")
        self.w_copy_fast = float(current["w_copy_fast"])
        self.w_copy_long = float(current["w_copy_long"])
        self.copy_report = self.measure_copy_recall(
            sequence, samples=min(400, max(120, count // 3))
        )
        stages["D_copy_protocol_report"] = dict(self.copy_report)

        # Stage D: role-prefix residual induction.
        grid_d = [{"w_prefix": float(weight)} for weight in prefix_grid]
        stage("D_role_prefix", grid_d, zeros, zeros)

        # Stage E: re-join the port and induction under the admitted channels.
        grid_e = [
            {"tau_H": float(tau), "coupling": str(coupling), "w_photonic": float(wp), "w_induction": float(wi)}
            for tau in tau_grid
            for coupling in couplings
            for wp in photonic_grid
            for wi in induction_grid
        ]
        stage("E_rejoin", grid_e, zeros, zeros)

        self.tau_gate_h = float(current["tau_H"])
        self.coupling = str(current["coupling"])
        self.w_photonic = float(current["w_photonic"])
        self.w_induction = float(current["w_induction"])
        self.w_prefix = float(current["w_prefix"])
        self.w_copy_fast = float(current["w_copy_fast"])
        self.w_copy_long = float(current["w_copy_long"])
        self.w_copy = 0.0
        self.channel_report = {
            "selected": {
                key: current[key]
                for key in (
                    "tau_H",
                    "coupling",
                    "w_photonic",
                    "w_induction",
                    "w_prefix",
                    "w_copy_fast",
                    "w_copy_long",
                    "nll",
                    "bits_per_char",
                    "top1",
                )
                if key in current
            },
            "samples": int(count),
            "stages": stages,
            "grids": {
                "tau_H": [float(value) for value in tau_grid],
                "w_photonic": [float(value) for value in photonic_grid],
                "w_induction": [float(value) for value in induction_grid],
                "w_prefix": [float(value) for value in prefix_grid],
                "w_copy": [float(value) for value in copy_grid],
                "fast_gamma": [float(value) for value in fast_gamma_grid],
                "long_gamma": [float(value) for value in long_gamma_grid],
                "boost": [float(value) for value in boost_grid],
                "coupling": [str(value) for value in couplings],
            },
            "copy_shaping": {
                "fast_gamma": float(self.copy_fast_gamma),
                "fast_boost": float(self.copy_fast_boost),
                "long_gamma": float(self.copy_long_gamma),
                "long_boost": float(self.copy_long_boost),
                "fast_window": int(self.copy_fast_window),
                "long_window": int(self.copy_long_window),
                "low": int(self.copy_low),
            },
            "prefix": {
                "window": int(self.prefix_window),
                "lambda": float(self.prefix_lambda),
                "local_exclusion": int(self.induction_low),
                "roles": int(len(set(self._role))),
            },
            "zero_point_available": True,
        }
        self.induction_report = dict(self.channel_report)
        return dict(self.channel_report)



# ======================================================================
# Cycle AC1: orthogonal exact-identity Kronecker induction, IDF-weighted
# dual copy channels and the contextual mode-phase torque
# ======================================================================
#
# AB1 measured the information-geometric conflict.  The Bochner codebook is a
# SEMANTIC kernel: Re<psi(verify), psi(check)> = 0.861 and
# Re<psi(int), psi(float)> = 0.677.  An in-context induction or copy head that
# scores candidates through that kernel leaks mass onto synonyms, so its
# measured copy recall saturated at 0.7280 / 0.7015 and the calibrator zeroed
# both long-range weights.
#
# The repair is not a larger steerer.  It is a second, dedicated basis whose
# only job is exact-token identity:
#
#     chi(v)[d] = exp(i theta[v, d]) / sqrt(D),   theta deterministic-hashed
#
# so that ||chi(v)||_2 = 1 exactly, E|Re<chi(u), chi(v)>|^2 = 1 / D for u != v
# (measured 0.02204 against the 1 / sqrt(2048) = 0.02210 prediction), and a
# superposition of successor waves unbinds to a distribution whose peak sits
# on the exact token that occurred, not on its neighbour.  The semantic
# Bochner waves are untouched: they still carry the distributional resonances.
#
# Three channels are built on that basis and one phase organ is added:
#
#   s_ind      multi-order causal induction over W = 256, matched on the exact
#              two-token suffix and the exact one-token suffix, recency-decayed
#              and masked where a high-count global 6-gram already dominates;
#   s_copy24   strict 24-token IDF-weighted copy, unbind against chi(v);
#   s_copy256  extended 256-token IDF-weighted copy, same readout;
#   torque     a compact Contextual Mode-Phase Interferometer
#              Delta theta_m = a_m[bin] + b_m[slot(w_{t-1})] + c[slot, slot(v)]
#              (<= 128 scalars) fitted by the exact analytic Born cross-term
#              gradient with a backtracking line search and zero autograd.
#
# Every weight is chosen on an inner slice of >= 8,192 tokens carved from the
# 758,656-token TRAINING stream.  The 84,086-token held-out stream is never
# read here.
# ======================================================================


class OrthogonalExactCodebook:
    """A deterministic pseudo-orthogonal unit-phasor basis for exact identity.

    The phases come from a splitmix64-style integer hash of (token, lane), so
    the codebook is reproducible on any machine and any numpy version from
    (vocab_size, dim, seed) alone: no floating-point generator enters.

    Properties, all measured in ``audit``:

    * ``||chi(v)||_2 = 1`` for every v (unitary rows);
    * ``E|Re<chi(u), chi(v)>|^2 = 1 / D`` for u != v, the qFHRR cross-talk law;
    * the unitary DFT of a row preserves energy exactly (Parseval ratio
      1.000000).
    """

    GOLDEN = 0x9E3779B97F4A7C15
    MIX_A = 0xBF58476D1CE4E5B9
    MIX_B = 0x94D049BB133111EB

    def __init__(self, vocab_size: int, dim: int = 2048, seed: int = 20261203) -> None:
        self.vocab_size = int(vocab_size)
        self.dim = int(dim)
        self.seed = int(seed)
        token = np.arange(self.vocab_size, dtype=np.uint64)[:, None]
        lane = np.arange(self.dim, dtype=np.uint64)[None, :]
        mixed = (
            token * np.uint64(self.GOLDEN & 0xFFFFFFFFFFFFFFFF)
            + lane * np.uint64(self.MIX_A & 0xFFFFFFFFFFFFFFFF)
            + np.uint64(self.seed & 0xFFFFFFFFFFFFFFFF)
        )
        mixed = (mixed ^ (mixed >> np.uint64(30))) * np.uint64(
            self.MIX_A & 0xFFFFFFFFFFFFFFFF
        )
        mixed = (mixed ^ (mixed >> np.uint64(27))) * np.uint64(
            self.MIX_B & 0xFFFFFFFFFFFFFFFF
        )
        mixed = mixed ^ (mixed >> np.uint64(31))
        phases = (
            (mixed & np.uint64(0xFFFFFFFF)).astype(np.float64)
            * (2.0 * math.pi / 4294967296.0)
            - math.pi
        )
        self.phases = phases.astype(np.float32)
        rows = np.exp(1j * phases) / math.sqrt(float(self.dim))
        self.phasors = np.ascontiguousarray(rows.astype(np.complex64))
        self.conjugate = np.ascontiguousarray(np.conj(self.phasors))

    # ------------------------------------------------------------- algebra
    def superpose(self, counts: np.ndarray) -> np.ndarray:
        """S = sum_v counts[v] chi(v); a real non-negative weight vector."""
        weights = np.asarray(counts, dtype=np.float32)
        return np.asarray(self.phasors.T @ weights, dtype=np.complex64)

    def unbind(self, accumulator: np.ndarray) -> np.ndarray:
        """Re<chi(v), S> for every vocabulary token v."""
        state = np.asarray(accumulator, dtype=np.complex64)
        return np.asarray(np.real(self.conjugate @ state), dtype=np.float64)

    def project(self, counts: np.ndarray) -> np.ndarray:
        """The exact unbinding score of the superposed successor waves."""
        return self.unbind(self.superpose(counts))

    def coherence(self, left: int, right: int) -> float:
        return float(np.real(np.vdot(self.phasors[int(left)], self.phasors[int(right)])))

    # --------------------------------------------------------------- audit
    def audit(self, *, samples: int = 384, seed: int = 20261203) -> Dict[str, Any]:
        """Measure the unit-norm, Parseval and cross-talk laws."""
        generator = np.random.default_rng(int(seed))
        picked = generator.choice(self.vocab_size, size=min(int(samples), self.vocab_size), replace=False)
        rows = self.phasors[picked]
        norms = np.linalg.norm(rows, axis=1)
        spectra = np.fft.fft(rows, axis=1)
        parseval = np.square(np.abs(spectra)).sum(axis=1) / (
            float(self.dim) * np.square(norms) + 1e-30
        )
        gram = rows.conj() @ rows.T
        off = np.abs(gram - np.diag(np.diag(gram)))
        mask = ~np.eye(len(picked), dtype=bool)
        cross = np.abs(off[mask]) ** 2
        return {
            "vocab_size": int(self.vocab_size),
            "dim": int(self.dim),
            "seed": int(self.seed),
            "rows_sampled": int(len(picked)),
            "norm_min": float(norms.min()),
            "norm_max": float(norms.max()),
            "norm_error": float(np.max(np.abs(norms - 1.0))),
            "parseval_ratio": float(parseval.mean()),
            "parseval_error": float(np.max(np.abs(parseval - 1.0))),
            "crosstalk_mean_square": float(cross.mean()),
            "crosstalk_theory": float(1.0 / float(self.dim)),
            "crosstalk_ratio": float(cross.mean() * float(self.dim)),
            "crosstalk_rms": float(np.sqrt(cross.mean())),
            "crosstalk_max": float(np.abs(off[mask]).max()),
            "predicted_max_coherence": float(1.0 / math.sqrt(float(self.dim))),
        }


def document_frequencies(
    segments: Sequence[Sequence[int]], vocab_size: int
) -> Dict[str, Any]:
    """Per-token document frequency over the pinned training documents."""
    counts = np.zeros(int(vocab_size), dtype=np.float64)
    documents = 0
    for segment in segments:
        ids = np.unique(np.asarray([int(token) for token in segment], dtype=np.int64))
        ids = ids[(ids >= 0) & (ids < int(vocab_size))]
        if ids.size:
            counts[ids] += 1.0
        documents += 1
    return {"counts": counts, "documents": float(max(documents, 1))}


def inverse_document_frequencies(df: Dict[str, Any], vocab_size: int) -> np.ndarray:
    """idf(v) = log((1 + N) / (1 + df(v))) + 1, the smoothed Robertson-Sparck Jones law."""
    counts = np.asarray(df["counts"], dtype=np.float64)
    documents = float(df["documents"])
    idf = np.log((1.0 + documents) / (1.0 + counts)) + 1.0
    return idf[: int(vocab_size)].astype(np.float64)


class KroneckerInductionEgress(ContextSteeredEgress):
    """AC1 egress: exact-identity induction/copy over chi plus a phase torque.

    The semantic Bochner path is inherited unchanged.  Everything added here
    (the chi codebook, the multi-order causal induction, the IDF dual copy
    caches and the mode-phase torque) is calibrated on the inner slice with
    0.0 in every weight grid, so the model at all-zero weights is the AB1
    model bit for bit.
    """

    def __init__(
        self,
        projector,
        tokenizer,
        *,
        identity_dim: int = 2048,
        identity_seed: int = 20261203,
        mask_min_count: int = 2,
        entropy_bins: int = 4,
        induction_order_two: float = 2.0,
        induction_order_one: float = 1.0,
        idf_power: float = 1.0,
        surprise_bin: int = 2,
        surprise_quantile: float = 0.75,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("steer_rank", 1)
        super().__init__(projector, tokenizer, **kwargs)
        self.identity = OrthogonalExactCodebook(
            self.vocab_size, dim=int(identity_dim), seed=int(identity_seed)
        )
        self.identity_dim = int(identity_dim)
        self.mask_min_count = int(mask_min_count)
        self.induction_order_two = float(induction_order_two)
        self.induction_order_one = float(induction_order_one)
        self.induction_idf_power = float(idf_power)
        self.idf_power = float(idf_power)
        self.idf = np.ones(self.vocab_size, dtype=np.float64)
        self._role_array = np.asarray(
            [
                token_role(self.tokenizer.token_text(index))
                for index in range(self.vocab_size)
            ],
            dtype=np.int64,
        )
        # Per-channel IDF exponents, so the two horizons can differ.
        self.copy_fast_power = float(idf_power)
        self.copy_long_power = float(idf_power)
        # Contextual mode-phase torque (<= 128 scalars: 9*4 + 9*4 + 4*4 = 88).
        self.entropy_bins = int(entropy_bins)
        self.surprise_bin = int(surprise_bin)
        self.surprise_quantile = float(surprise_quantile)
        self.torque_a = np.zeros((self.phase_modes, self.entropy_bins), dtype=np.float64)
        self.torque_b = np.zeros((self.phase_modes, 4), dtype=np.float64)
        self.torque_c = np.zeros((self.phase_modes, 4), dtype=np.float64)
        self.torque0 = (
            self.torque_a.copy(),
            self.torque_b.copy(),
            self.torque_c.copy(),
        )
        # High-surprise token-specific phase step (the IDF-weighted AA1 table
        # restricted to diffuse suffix states).
        self.phase_high = np.zeros((self.phase_modes, self.vocab_size), dtype=np.float64)
        self.phase_high_weight = 0.0
        self.torque_report: Dict[str, Any] = {}
        self.surprise_report: Dict[str, Any] = {}
        self.ac_report: Dict[str, Any] = {}
        self.induction_matches = {"order_two": 0, "order_one": 0, "masked": 0}
        self._ac_induction_cache: Dict[
            Tuple[int, ...], Optional[Tuple[np.ndarray, np.ndarray]]
        ] = {}
        self._bin_cache: Dict[int, int] = {}

    # ------------------------------------------------------------ identity
    def set_idf(self, idf: Sequence[float]) -> None:
        values = np.asarray(idf, dtype=np.float64)
        self.idf = np.maximum(values[: self.vocab_size], 1e-6)

    def idf_of(self, token: int) -> float:
        if 0 <= int(token) < self.vocab_size:
            return float(self.idf[int(token)])
        return 1.0

    def set_floor_model(self, model) -> None:
        """Attach the calibrated 6/7-gram floor used by the non-redundancy mask."""
        self.base_counts = model
        self._ac_induction_cache.clear()

    # -------------------------------------------------------- phase torque
    def _entropy_bin_from_rows(self, rows: np.ndarray) -> int:
        power = np.square(np.asarray(rows, dtype=np.float64)).sum(axis=0)
        total = float(power.sum())
        if total <= 0.0:
            return 0
        probability = power / total
        entropy = float(
            -np.sum(probability * np.log(np.maximum(probability, 1e-12)))
        )
        ratio = entropy / math.log(float(self.vocab_size))
        index = int(ratio * float(self.entropy_bins))
        return int(min(self.entropy_bins - 1, max(0, index)))

    def torque_active(self) -> bool:
        return bool(
            float(np.abs(self.torque_a).sum())
            + float(np.abs(self.torque_b).sum())
            + float(np.abs(self.torque_c).sum())
            > 0.0
        )

    def torque_parameters(self) -> int:
        return int(self.torque_a.size + self.torque_b.size + self.torque_c.size)

    def phase_active(self) -> bool:
        return bool(self.phase_high_weight > 0.0)

    def _slot_of(self, token: int) -> int:
        index = int(token)
        if 0 <= index < self.vocab_size:
            return int(self._slot_ids[index])
        return 0

    def effective_phase(
        self, context: Sequence[int], rows: Optional[np.ndarray] = None
    ) -> np.ndarray:
        base = np.asarray(self.phase_table, dtype=np.float64)
        if not self.phase_active() and not self.torque_active():
            return base
        if rows is None:
            rows, _names = self._suffix_rows(context)
        bin_index = self._entropy_bin_from_rows(rows)
        if self.phase_active() and bin_index >= self.surprise_bin:
            base = base + float(self.phase_high_weight) * self.phase_high
        if not self.torque_active():
            return base
        last = int(context[-1]) if len(context) else 0
        slot_prev = self._slot_of(last)
        delta = (
            self.torque_a[:, bin_index][:, None]
            + self.torque_b[:, slot_prev][:, None]
            + self.torque_c[:, self._slot_ids]
        )
        return base + delta

    # ------------------------------------------------ exact-identity induction
    def _sixgram_dominated(self, window: Sequence[int]) -> Optional[np.ndarray]:
        """Tokens whose global 6-gram continuation count already dominates."""
        model = getattr(self, "base_counts", None)
        if model is None or int(self.mask_min_count) <= 0 or len(window) < 5:
            return None
        bucket = getattr(model, "sextgram", None)
        if not bucket:
            return None
        entry = bucket.get(tuple(int(token) for token in window[-5:]))
        if not entry:
            return None
        keys = np.fromiter(entry.keys(), dtype=np.int64, count=len(entry))
        values = np.fromiter(entry.values(), dtype=np.int64, count=len(entry))
        mask = np.zeros(self.vocab_size, dtype=bool)
        mask[keys[values >= int(self.mask_min_count)]] = True
        return mask

    def causal_induction_terms(
        self, context: Sequence[int]
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """(resonance, distribution) of the exact-identity causal induction head.

        Over the causal window W = 256 an in-window position j < t-1 supports
        the successor w_j when the exact two-token suffix (w_{j-2}, w_{j-1})
        equals (w_{t-2}, w_{t-1}) - an order-2 match, weighted above the exact
        one-token match w_{j-1} = w_{t-1}.  Every supporting successor is
        superposed as its orthogonal wave chi(w_j) with recency decay
        gamma^(t-j), and the superposition is unbound against chi(v).

        Candidates whose global 6-gram continuation count is at least
        ``mask_min_count`` are removed before the readout, so the channel is
        strictly non-redundant with the static corpus table.
        """
        window = [int(token) for token in context]
        length = len(window)
        if length < 4:
            return None
        key = tuple(window[-min(length, self.prefix_window + 1) :])
        cached = self._ac_induction_cache.get(key)
        if cached is not None:
            return cached
        counts = np.zeros(self.vocab_size, dtype=np.float64)
        mass = 0.0
        order_two = 0
        order_one = 0
        last_one = window[-1]
        last_two = window[-2]
        start = max(self.induction_low + 1, length - self.prefix_window - 1)
        for index in range(length - 2, start - 1, -1):
            if index <= 1:
                break
            successor = int(window[index])
            if not (0 <= successor < self.vocab_size):
                continue
            if int(window[index - 1]) != last_one:
                continue
            decay = float(self.induction_gamma) ** (length - 1 - index)
            if int(window[index - 2]) == last_two:
                weight = float(self.induction_order_two) * decay
                order_two += 1
            else:
                weight = float(self.induction_order_one) * decay
                order_one += 1
            weight *= float(self.idf[successor]) ** float(self.induction_idf_power)
            counts[successor] += weight
            mass += weight
        if mass <= 0.0:
            result: Optional[Tuple[np.ndarray, np.ndarray]] = None
        else:
            mask = self._sixgram_dominated(window)
            if mask is not None:
                masked_mass = float(counts[mask].sum())
                counts[mask] = 0.0
                mass -= masked_mass
                self.induction_matches["masked"] += 1
            if mass <= 0.0:
                result = None
            else:
                score = self.identity.project(counts)
                positive = np.maximum(score, 0.0)
                total = float(positive.sum())
                if total <= 0.0:
                    result = None
                else:
                    result = (positive / mass, positive / total)
                    self.induction_matches["order_two"] += order_two
                    self.induction_matches["order_one"] += order_one
        if len(self._ac_induction_cache) < 16384:
            self._ac_induction_cache[key] = result
        return result

    def induction_terms(
        self, context: Sequence[int]
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        return self.causal_induction_terms(context)

    # ------------------------------------------------ IDF dual copy channels
    def copy_terms(
        self,
        context: Sequence[int],
        *,
        window: int,
        gamma: float,
        boost: float = 1.0,
        power: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        """IDF-weighted exact copy readout over a strict causal window.

        counts[v] = sum_offsets gamma^(offset-low) idf(w_{t-offset})^p
                    * I[w_{t-offset} = v]
        and the returned distribution is the chi-unbinding of the superposed
        waves, clipped at zero.  The IDF exponent stops frequent syntax
        punctuation from swamping a rare local identifier.
        """
        tokens = [int(token) for token in context]
        length = len(tokens)
        if length <= self.copy_low:
            return None
        high = min(int(window), length - 1)
        if high < self.copy_low:
            return None
        exponent = float(self.idf_power if power is None else power)
        offsets = np.arange(self.copy_low, high + 1, dtype=np.int64)
        index = np.asarray([tokens[-int(offset)] for offset in offsets], dtype=np.int64)
        valid = (index >= 0) & (index < self.vocab_size)
        index = index[valid]
        offsets = offsets[valid]
        if index.size == 0:
            return None
        weights = np.power(float(gamma), (offsets - self.copy_low).astype(np.float64))
        weights = weights * np.power(np.maximum(self.idf[index], 1e-6), exponent)
        if float(boost) != 1.0:
            weights = weights * np.where(
                self._role_array[index] == _ROLE_WORD, float(boost), 1.0
            )
        counts = np.bincount(index, weights=weights, minlength=self.vocab_size)
        if float(counts.sum()) <= 0.0:
            return None
        score = self.identity.project(counts)
        positive = np.maximum(score, 0.0)
        total = float(positive.sum())
        if total <= 0.0:
            return None
        probability = positive / total
        if float(self.copy_sharpness) != 1.0:
            # A sharper readout keeps the copy peak narrow, so a large copy
            # weight can buy recall without diffusing the next-token decision.
            sharpened = np.power(probability, float(self.copy_sharpness))
            total_sharp = float(sharpened.sum())
            if total_sharp > 0.0:
                probability = sharpened / total_sharp
        return probability

    def copy_fast(self, context: Sequence[int]) -> Optional[np.ndarray]:
        return self.copy_terms(
            context,
            window=int(self.copy_fast_window),
            gamma=float(self.copy_fast_gamma),
            boost=float(self.copy_fast_boost),
            power=float(self.copy_fast_power),
        )

    def copy_long(self, context: Sequence[int]) -> Optional[np.ndarray]:
        return self.copy_terms(
            context,
            window=int(self.copy_long_window),
            gamma=float(self.copy_long_gamma),
            boost=float(self.copy_long_boost),
            power=float(self.copy_long_power),
        )

    # ------------------------------------------------- mode-phase torque fit
    def _torque_rows(
        self, ids: Sequence[int], positions: Sequence[int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rows = np.zeros(
            (len(positions), self.phase_modes, self.vocab_size), dtype=np.float32
        )
        bins = np.zeros(len(positions), dtype=np.int64)
        slots = np.zeros(len(positions), dtype=np.int64)
        targets = np.zeros(len(positions), dtype=np.int64)
        for row, position in enumerate(positions):
            context = ids[max(0, position - self.window) : position]
            state, _names = self._suffix_rows(context)
            rows[row] = state.astype(np.float32)
            bins[row] = self._entropy_bin_from_rows(state)
            last = int(context[-1]) if context else 0
            slots[row] = self._slot_of(last)
            targets[row] = int(ids[position])
        return rows, bins, slots, targets

    def _torque_theta(
        self,
        phases: np.ndarray,
        bins: np.ndarray,
        slots: np.ndarray,
        torque_a: np.ndarray,
        torque_b: np.ndarray,
        torque_c: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """theta[n, m, v] = phi[m, v] + a[m, bin] + b[m, slot] + c[m, slot(v)]."""
        term_a = np.transpose(np.asarray(torque_a)[:, bins], (1, 0))
        term_b = np.transpose(np.asarray(torque_b)[:, slots], (1, 0))
        term_c = np.asarray(torque_c)[:, self._slot_ids]
        theta = (
            np.asarray(phases)[None, :, :]
            + term_a[:, :, None]
            + term_b[:, :, None]
            + term_c[None, :, :]
        )
        return theta, term_a + term_b, term_c

    def _torque_forward(
        self,
        rows: np.ndarray,
        bins: np.ndarray,
        slots: np.ndarray,
        phases: np.ndarray,
        torque_a: np.ndarray,
        torque_b: np.ndarray,
        torque_c: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        theta, pair, term_c = self._torque_theta(phases, bins, slots, torque_a, torque_b, torque_c)
        base = np.exp(1j * np.asarray(phases))[None, :, :] * np.asarray(
            rows, dtype=np.complex128
        )
        pair_phase = np.exp(1j * pair)[:, :, None]
        candidate_phase = np.exp(1j * term_c)[None, :, :]
        amplitude = (base * pair_phase * candidate_phase).sum(axis=1)
        power = np.abs(amplitude) ** 2
        total = np.maximum(power.sum(axis=1), 1e-30)
        return {
            "theta": theta,
            "amplitude": amplitude,
            "power": power,
            "total": total,
        }

    def _torque_nll(
        self,
        rows: np.ndarray,
        bins: np.ndarray,
        slots: np.ndarray,
        targets: np.ndarray,
        phases: np.ndarray,
        torque_a: np.ndarray,
        torque_b: np.ndarray,
        torque_c: np.ndarray,
    ) -> Tuple[float, float]:
        state = self._torque_forward(rows, bins, slots, phases, torque_a, torque_b, torque_c)
        power = state["power"]
        total = state["total"]
        truth = power[np.arange(len(targets)), targets] / total
        return (
            float(-np.mean(np.log(np.maximum(truth, 1e-30)))),
            float(np.mean(np.argmax(power, axis=1) == targets)),
        )

    def _torque_gradient(
        self,
        rows: np.ndarray,
        bins: np.ndarray,
        slots: np.ndarray,
        targets: np.ndarray,
        phases: np.ndarray,
        torque_a: np.ndarray,
        torque_b: np.ndarray,
        torque_c: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Exact d(mean cross-entropy)/d(a, b, c) through the Born rule."""
        state = self._torque_forward(rows, bins, slots, phases, torque_a, torque_b, torque_c)
        theta = state["theta"]
        amplitude = state["amplitude"]
        power = state["power"]
        total = state["total"]
        angle = np.angle(amplitude)
        magnitude = np.abs(amplitude)
        truth = np.maximum(power[np.arange(len(targets)), targets], 1e-30)
        rows64 = np.asarray(rows, dtype=np.float64)
        tau = np.empty(theta.shape, dtype=np.float64)
        base = np.exp(1j * np.asarray(phases))[None, :, :]
        for mode in range(self.phase_modes):
            term = rows64[:, mode, :] * (
                2.0 * magnitude * np.sin(angle - theta[:, mode, :])
            )
            contribution = term / total[:, None]
            contribution[np.arange(len(targets)), targets] -= (
                term[np.arange(len(targets)), targets] / truth
            )
            tau[:, mode, :] = contribution
        tau /= float(len(targets))
        grad_a = np.zeros_like(np.asarray(torque_a, dtype=np.float64))
        grad_b = np.zeros_like(np.asarray(torque_b, dtype=np.float64))
        grad_c = np.zeros_like(np.asarray(torque_c, dtype=np.float64))
        summed_v = tau.sum(axis=2)
        for bin_index in range(self.entropy_bins):
            picked = bins == bin_index
            if picked.any():
                grad_a[:, bin_index] = summed_v[picked].sum(axis=0)
        for slot_index in range(4):
            picked = slots == slot_index
            if not bool(picked.any()):
                continue
            grad_b[:, slot_index] = summed_v[picked].sum(axis=0)
        per_mode_token = tau.sum(axis=0)
        for target_slot in range(4):
            columns = self._slot_ids == target_slot
            grad_c[:, target_slot] = per_mode_token[:, columns].sum(axis=1)
        return grad_a, grad_b, grad_c

    def fit_mode_phase_torque(
        self,
        ids: Sequence[int],
        *,
        epochs: int = 8,
        samples: int = 1600,
        inner_samples: int = 384,
        inner_fraction: float = 0.15,
        chunk: int = 200,
        base_step: float = 0.08,
        l2: float = 1e-3,
        min_gain: float = 2e-4,
        seed: int = 20261207,
    ) -> Dict[str, Any]:
        """Fit the Contextual Mode-Phase Interferometer by backtracking line search.

        The gradient is the exact analytic Born cross-term derivative.  A step
        is accepted only when the inner-slice cross-entropy falls, so the
        recorded curve is best-so-far and an overfit torque cannot be shipped.
        """
        sequence = [int(token) for token in ids]
        cut = int(len(sequence) * (1.0 - float(inner_fraction)))
        fit_ids, inner_ids = sequence[:cut], sequence[cut:]
        fit_positions = self._positions(fit_ids, int(samples))
        inner_positions = self._positions(inner_ids, int(inner_samples))
        if not fit_positions or not inner_positions:
            self.torque_report = {"epochs": 0}
            return dict(self.torque_report)
        rows, bins, slots, targets = self._torque_rows(fit_ids, fit_positions)
        inner_rows, inner_bins, inner_slots, inner_targets = self._torque_rows(
            inner_ids, inner_positions
        )
        phases = np.asarray(self.phase_table, dtype=np.float64).copy()
        torque_a, torque_b, torque_c = self.torque0
        torque_a = torque_a.copy()
        torque_b = torque_b.copy()
        torque_c = torque_c.copy()
        initial = self._torque_nll(
            inner_rows, inner_bins, inner_slots, inner_targets, phases, torque_a, torque_b, torque_c
        )
        best = (float(initial[0]), torque_a.copy(), torque_b.copy(), torque_c.copy())
        current = float(initial[0])
        curve: List[float] = [float(initial[0])]
        generator = np.random.default_rng(int(seed))
        accepted = 0
        steps = 0
        for _epoch in range(int(epochs)):
            grad_a = np.zeros_like(torque_a)
            grad_b = np.zeros_like(torque_b)
            grad_c = np.zeros_like(torque_c)
            permutation = generator.permutation(len(targets))
            for start in range(0, len(permutation), int(chunk)):
                index = permutation[start : start + int(chunk)]
                part_a, part_b, part_c = self._torque_gradient(
                    rows[index],
                    bins[index],
                    slots[index],
                    targets[index],
                    phases,
                    torque_a,
                    torque_b,
                    torque_c,
                )
                weight = float(len(index)) / float(len(permutation))
                grad_a += weight * part_a
                grad_b += weight * part_b
                grad_c += weight * part_c
            grad_a += float(l2) * torque_a
            grad_b += float(l2) * torque_b
            grad_c += float(l2) * torque_c
            scale = max(
                float(np.abs(grad_a).max()) if grad_a.size else 0.0,
                float(np.abs(grad_b).max()) if grad_b.size else 0.0,
                float(np.abs(grad_c).max()) if grad_c.size else 0.0,
                1e-12,
            )
            direction = (-grad_a / scale, -grad_b / scale, -grad_c / scale)
            improved = False
            for shrink in (1.0, 0.5, 0.25, 0.125, 0.0625):
                step = float(base_step) * shrink
                candidate_a = torque_a + step * direction[0]
                candidate_b = torque_b + step * direction[1]
                candidate_c = torque_c + step * direction[2]
                value = self._torque_nll(
                    inner_rows,
                    inner_bins,
                    inner_slots,
                    inner_targets,
                    phases,
                    candidate_a,
                    candidate_b,
                    candidate_c,
                )[0]
                steps += 1
                if value < current - 1e-7:
                    torque_a, torque_b, torque_c = candidate_a, candidate_b, candidate_c
                    current = float(value)
                    improved = True
                    break
            if improved:
                accepted += 1
                best = (float(value), torque_a.copy(), torque_b.copy(), torque_c.copy())
            curve.append(float(min(best[0], current)))
        gain = float(initial[0]) - float(best[0])
        if gain < float(min_gain):
            # The D-Guard refuses a torque whose inner gain is below the
            # declared margin: an 88-parameter field that moves the inner slice
            # by less than 2e-4 nats has not earned its variance.
            self.torque_a = np.zeros_like(torque_a)
            self.torque_b = np.zeros_like(torque_b)
            self.torque_c = np.zeros_like(torque_c)
            best = (float(initial[0]), self.torque_a.copy(), self.torque_b.copy(), self.torque_c.copy())
        else:
            self.torque_a, self.torque_b, self.torque_c = best[1], best[2], best[3]
        self.torque_a, self.torque_b, self.torque_c = best[1], best[2], best[3]
        final = self._torque_nll(
            inner_rows, inner_bins, inner_slots, inner_targets, phases,
            self.torque_a, self.torque_b, self.torque_c,
        )
        self.torque_report = {
            "epochs": int(epochs),
            "samples": int(len(fit_positions)),
            "inner_samples": int(len(inner_positions)),
            "parameters": int(self.torque_parameters()),
            "entropy_bins": int(self.entropy_bins),
            "inner_initial_nll": float(initial[0]),
            "inner_final_nll": float(best[0]),
            "inner_initial_top1": float(initial[1]),
            "inner_final_top1": float(final[1]),
            "accepted_steps": int(accepted),
            "line_search_evaluations": int(steps),
            "curve": curve,
            "torque_rms_rad": float(
                np.sqrt(
                    np.mean(
                        np.concatenate(
                            [
                                np.square(self.torque_a).ravel(),
                                np.square(self.torque_b).ravel(),
                                np.square(self.torque_c).ravel(),
                            ]
                        )
                    )
                )
            ),
            "inner_gain_nats": float(gain),
            "min_gain_nats": float(min_gain),
            "accepted_by_inner_slice": bool(gain >= float(min_gain)),
            "gradient": "exact analytic Born cross-term derivative, backtracking line search",
            "autograd_used": False,
        }
        return dict(self.torque_report)

    # ------------------------------------- high-surprise token phase step
    def _phase_gradient_weighted(
        self,
        rows: np.ndarray,
        targets: np.ndarray,
        phases: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        amplitude = np.einsum(
            "nmv,mv->nv", rows, np.exp(1j * phases).astype(np.complex128)
        )
        power = np.abs(amplitude) ** 2
        total = np.maximum(power.sum(axis=1), 1e-30)
        angle = np.angle(amplitude)
        magnitude = np.abs(amplitude)
        truth = np.maximum(power[np.arange(len(targets)), targets], 1e-30)
        gradient = np.zeros_like(phases)
        scale = np.asarray(weights, dtype=np.float64)
        scale = scale / max(float(scale.sum()), 1e-30)
        for mode in range(self.phase_modes):
            term = rows[:, mode, :] * (
                2.0 * magnitude * np.sin(angle - phases[mode][None, :])
            )
            contribution = term / total[:, None]
            contribution[np.arange(len(targets)), targets] -= (
                term[np.arange(len(targets)), targets] / truth
            )
            gradient[mode] = (contribution * scale[:, None]).sum(axis=0)
        # A phase added to EVERY mode of a token is a gauge: it multiplies
        # that token's chord by one unit phasor and leaves the Born power
        # unchanged.  Project the gauge direction out so the fitted step has a
        # physical effect.
        gradient = gradient - gradient.mean(axis=0, keepdims=True)
        return gradient

    def fit_surprise_phase(
        self,
        ids: Sequence[int],
        *,
        epochs: int = 10,
        samples: int = 1600,
        inner_samples: int = 384,
        inner_fraction: float = 0.15,
        learning_rate: float = 0.02,
        l2: float = 1e-3,
    ) -> Dict[str, Any]:
        """Fit the IDF-weighted token-specific phase step on high-surprise rows.

        Only positions whose suffix state lands in ``surprise_bin`` or above
        are used, and each row is weighted by idf(target): rare identifiers
        carry more of the gradient.  The step is applied inside
        ``effective_phase`` only for those diffuse contexts, so it cannot
        disturb the confident syntax states where AB1's static table already
        wins.
        """
        sequence = [int(token) for token in ids]
        cut = int(len(sequence) * (1.0 - float(inner_fraction)))
        fit_ids, inner_ids = sequence[:cut], sequence[cut:]
        fit_positions = self._positions(fit_ids, int(samples))
        inner_positions = self._positions(inner_ids, int(inner_samples))
        if not fit_positions or not inner_positions:
            self.surprise_report = {"epochs": 0}
            return dict(self.surprise_report)

        def entropy_of(state: np.ndarray) -> float:
            power = np.square(np.asarray(state, dtype=np.float64)).sum(axis=0)
            total = float(power.sum())
            if total <= 0.0:
                return 0.0
            probability = power / total
            return float(
                -np.sum(probability * np.log(np.maximum(probability, 1e-12)))
            )

        def select(sequence_ids, positions, threshold: float):
            rows_out = []
            targets_out = []
            weights_out = []
            for position in positions:
                context = sequence_ids[max(0, position - self.window) : position]
                state, _names = self._suffix_rows(context)
                if entropy_of(state) < float(threshold):
                    continue
                rows_out.append(state.astype(np.float32))
                targets_out.append(int(sequence_ids[position]))
                weights_out.append(float(self.idf_of(int(sequence_ids[position]))))
            if not rows_out:
                return None
            return (
                np.stack(rows_out),
                np.asarray(targets_out, dtype=np.int64),
                np.asarray(weights_out, dtype=np.float64),
            )

        fit_entropies = [
            entropy_of(
                self._suffix_rows(fit_ids[max(0, position - self.window) : position])[0]
            )
            for position in fit_positions
        ]
        threshold = float(np.quantile(np.asarray(fit_entropies), float(self.surprise_quantile)))
        fit = select(fit_ids, fit_positions, threshold)
        inner = select(inner_ids, inner_positions, threshold)
        if fit is None or inner is None:
            self.surprise_report = {"epochs": 0, "reason": "no_high_surprise_rows"}
            return dict(self.surprise_report)
        rows, targets, weights = fit
        inner_rows, inner_targets, inner_weights = inner
        phases = np.asarray(self.phase_table, dtype=np.float64).copy()
        delta = np.zeros_like(phases)
        base = np.asarray(self.phase_table, dtype=np.float64).copy()

        def weighted_nll(chunk_rows, chunk_targets, chunk_weights, candidate_delta):
            """Weighted cross-entropy, evaluated in bounded row blocks.

            The block size keeps the (rows, modes, vocab) intermediate small,
            so the fit never pages on a large vocabulary.
            """
            total_rows = int(len(chunk_targets))
            normaliser = max(float(chunk_weights.sum()), 1e-30)
            phasors = np.exp(1j * (base + candidate_delta)).astype(np.complex128)
            accumulated = 0.0
            for start in range(0, total_rows, 256):
                stop = min(start + 256, total_rows)
                amplitude = np.einsum("nmv,mv->nv", chunk_rows[start:stop], phasors)
                power = np.abs(amplitude) ** 2
                row_total = np.maximum(power.sum(axis=1), 1e-30)
                targets_block = chunk_targets[start:stop]
                truth = power[np.arange(len(targets_block)), targets_block] / row_total
                accumulated += float(
                    -np.sum(
                        chunk_weights[start:stop] * np.log(np.maximum(truth, 1e-30))
                    )
                )
            return accumulated / normaliser

        initial = weighted_nll(inner_rows, inner_targets, inner_weights, delta)
        best = (float(initial), delta.copy())
        curve: List[float] = [float(initial)]
        accepted = 0
        for _step in range(int(epochs)):
            gradient = self._phase_gradient_weighted(rows, targets, base + delta, weights)
            gradient = gradient + float(l2) * delta
            moved = False
            for shrink in (1.0, 0.5, 0.25, 0.125):
                candidate = delta - float(learning_rate) * shrink * gradient
                value = weighted_nll(inner_rows, inner_targets, inner_weights, candidate)
                if value < best[0] - 1e-9:
                    delta = candidate
                    best = (float(value), candidate.copy())
                    moved = True
                    accepted += 1
                    break
            curve.append(float(best[0]))
            if not moved:
                break
        self.phase_high = best[1].copy()
        self.phase_high_weight = 1.0
        self.surprise_report = {
            "epochs": int(epochs),
            "rows": int(len(targets)),
            "inner_rows": int(len(inner_targets)),
            "surprise_bin": int(self.surprise_bin),
            "surprise_quantile": float(self.surprise_quantile),
            "entropy_threshold": float(threshold),
            "entropy_bins": int(self.entropy_bins),
            "idf_weighted": True,
            "gauge_projected": True,
            "inner_initial_nll": float(initial),
            "inner_final_nll": float(best[0]),
            "accepted_steps": int(accepted),
            "rms_displacement_rad": float(np.sqrt(np.mean(np.square(self.phase_high)))),
            "curve": curve,
            "autograd_used": False,
        }
        return dict(self.surprise_report)

    def reset_phase_organs(self) -> None:
        """Return the phase block to the AA1 static table (for the marginal)."""
        self.phase_table = self.phase_table0.copy()
        self.phase_high = np.zeros_like(self.phase_high)
        self.phase_high_weight = 0.0
        self.torque_a, self.torque_b, self.torque_c = (
            self.torque0[0].copy(),
            self.torque0[1].copy(),
            self.torque0[2].copy(),
        )
        self._bright_cache.clear() if hasattr(self, "_bright_cache") else None

    def restore_phase_organs(self, saved: Dict[str, Any]) -> None:
        self.phase_table = np.asarray(saved["phase_table"], dtype=np.float64).copy()
        self.phase_high = np.asarray(saved["phase_high"], dtype=np.float64).copy()
        self.phase_high_weight = float(saved["phase_high_weight"])
        self.torque_a = np.asarray(saved["torque_a"], dtype=np.float64).copy()
        self.torque_b = np.asarray(saved["torque_b"], dtype=np.float64).copy()
        self.torque_c = np.asarray(saved["torque_c"], dtype=np.float64).copy()

    def snapshot_phase_organs(self) -> Dict[str, Any]:
        return {
            "phase_table": np.asarray(self.phase_table, dtype=np.float64).copy(),
            "phase_high": np.asarray(self.phase_high, dtype=np.float64).copy(),
            "phase_high_weight": float(self.phase_high_weight),
            "torque_a": self.torque_a.copy(),
            "torque_b": self.torque_b.copy(),
            "torque_c": self.torque_c.copy(),
        }

    # ------------------------------------------------ inner-slice D-Guard
    def _ac_positions(self, ids: Sequence[int], count: int, buffer: int = 260) -> List[int]:
        length = len(ids)
        low = int(buffer)
        high = length - 2
        if high <= low:
            return []
        span = high - low
        if span <= int(count):
            return list(range(low, high))
        stride = max(1, span // int(count))
        picked = list(range(low, high, stride))[: int(count)]
        return picked

    def _channel_arrays(
        self,
        sequence: Sequence[int],
        positions: Sequence[int],
        *,
        window: int,
        gamma: float,
        power: float,
        boost: float,
        chunk: int = 512,
    ) -> np.ndarray:
        """IDF copy distributions for many positions, built in bounded chunks."""
        out = np.zeros((len(positions), self.vocab_size), dtype=np.float32)
        for start in range(0, len(positions), int(chunk)):
            block = positions[start : start + int(chunk)]
            stack = np.zeros((len(block), self.vocab_size), dtype=np.float32)
            for row, position in enumerate(block):
                context = sequence[max(0, position - int(window)) : position]
                value = self.copy_terms(
                    context,
                    window=int(window),
                    gamma=float(gamma),
                    boost=float(boost),
                    power=float(power),
                )
                if value is not None:
                    stack[row] = value.astype(np.float32)
            out[start : start + len(block)] = stack
        return out

    def _induction_array(
        self,
        sequence: Sequence[int],
        positions: Sequence[int],
        mask: int,
        power: Optional[float] = None,
    ) -> np.ndarray:
        saved_mask = int(self.mask_min_count)
        saved_power = float(self.induction_idf_power)
        self.mask_min_count = int(mask)
        if power is not None:
            self.induction_idf_power = float(power)
        self._ac_induction_cache.clear()
        out = np.zeros((len(positions), self.vocab_size), dtype=np.float32)
        for row, position in enumerate(positions):
            context = sequence[max(0, position - self.prefix_window) : position]
            terms = self.causal_induction_terms(context)
            if terms is not None:
                out[row] = terms[1].astype(np.float32)
        self.mask_min_count = saved_mask
        self.induction_idf_power = saved_power
        self._ac_induction_cache.clear()
        return out

    def calibrate_ac_channels(
        self,
        ids: Sequence[int],
        *,
        tokens: int = 8192,
        tau_grid: Sequence[float] = (2.0, 2.5, 3.0),
        photonic_grid: Sequence[float] = (0.0, 0.05),
        induction_grid: Sequence[float] = (0.0, 0.02, 0.05, 0.10, 0.20, 0.40),
        prefix_grid: Sequence[float] = (0.0, 0.05, 0.10, 0.20),
        copy_grid: Sequence[float] = (0.0, 0.05, 0.10, 0.20, 0.30, 0.40),
        refinement: Sequence[float] = (1.0,),
        induction_options: Sequence[Tuple[int, float]] = (
            (0, 1.0),
            (2, 1.0),
            (2, 2.0),
            (4, 1.0),
        ),
        fast_shaping: Sequence[Tuple[float, float, float]] = (
            (0.94, 1.0, 1.0),
            (1.0, 1.0, 2.5),
            (1.0, 2.0, 1.0),
            (1.0, 2.0, 2.5),
        ),
        long_shaping: Sequence[Tuple[float, float, float]] = (
            (0.99, 1.0, 1.0),
            (0.996, 1.0, 2.5),
            (0.999, 1.0, 1.0),
            (0.999, 2.0, 2.5),
        ),
        couplings: Sequence[str] = ("mixture", "amplitude"),
    ) -> Dict[str, Any]:
        """Staged inner-slice selection of every long-range channel.

        The D-Guard slice is at least 8,192 tokens carved from the TRAINING
        stream.  Every weight grid contains 0.0, every stage scores the exact
        Born / mixture cross-entropy of the deployed readout, and the copy
        stages search their shaping (decay, IDF exponent, boost) by the same
        cross-entropy rather than by a proxy.
        """
        sequence = [int(token) for token in ids]
        positions = self._ac_positions(sequence, int(tokens))
        if not positions:
            self.ac_report = {"rows": 0, "selected": {}}
            return dict(self.ac_report)
        count = len(positions)
        chord = np.zeros((count, self.vocab_size), dtype=np.complex64)
        entropies = np.zeros(count, dtype=np.float64)
        targets = np.zeros(count, dtype=np.int64)
        for row, position in enumerate(positions):
            context = sequence[max(0, position - self.window) : position]
            state, _names = self._suffix_rows(context)
            theta = self.effective_phase(context, state)
            chord[row] = np.sum(
                np.exp(1j * theta) * state.astype(np.complex128), axis=0
            ).astype(np.complex64)
            weights, dists, _totals, _hp, _terms = self._order_waves_full(context, None)
            entropies[row] = self.suffix_entropy(context, weights=weights, dists=dists)
            targets[row] = int(sequence[position])
        coherent = np.abs(chord) ** 2
        coherent_total = np.maximum(coherent.sum(axis=1, keepdims=True), 1e-30)

        induction_arrays: Dict[Tuple[int, float], np.ndarray] = {}
        for mask, power in induction_options:
            induction_arrays[(int(mask), float(power))] = self._induction_array(
                sequence, positions, int(mask), float(power)
            )
        prefix_array = np.zeros((count, self.vocab_size), dtype=np.float32)
        for row, position in enumerate(positions):
            context = sequence[max(0, position - self.window) : position]
            terms = self.prefix_terms(context)
            if terms is not None:
                prefix_array[row] = terms[1].astype(np.float32)
        port_array = np.zeros((count, self.vocab_size), dtype=np.float32)
        if self.core is not None:
            for row, position in enumerate(positions):
                context = sequence[max(0, position - self.window) : position]
                port = self.photonic_port(context)
                if port is not None:
                    port_array[row] = port.astype(np.float32)

        rows_index = np.arange(len(targets))

        def score(
            combo: Dict[str, Any],
            option: Tuple[int, float],
            fast_array: Optional[np.ndarray] = None,
            long_array: Optional[np.ndarray] = None,
        ) -> Dict[str, Any]:
            gate = (entropies > float(combo["tau_H"])).astype(np.float64)
            induction = induction_arrays[(int(option[0]), float(option[1]))]
            weight_sum = (
                float(combo["w_photonic"])
                + float(combo["w_induction"])
                + float(combo["w_prefix"])
                + float(combo["w_copy_fast"])
                + float(combo["w_copy_long"])
            )
            if combo["coupling"] == "amplitude":
                extra = np.zeros_like(coherent)
                if combo["w_photonic"]:
                    extra += float(combo["w_photonic"]) * port_array
                if combo["w_induction"]:
                    extra += float(combo["w_induction"]) * induction
                if combo["w_prefix"]:
                    extra += float(combo["w_prefix"]) * prefix_array
                if combo["w_copy_fast"] and fast_array is not None:
                    extra += float(combo["w_copy_fast"]) * fast_array
                if combo["w_copy_long"] and long_array is not None:
                    extra += float(combo["w_copy_long"]) * long_array
                power = np.abs(chord + gate[:, None] * extra) ** 2
            else:
                residual = np.clip(1.0 - gate * weight_sum, 0.0, 1.0)
                power = residual[:, None] * coherent / coherent_total
                if combo["w_photonic"]:
                    power = power + gate[:, None] * float(combo["w_photonic"]) * port_array
                if combo["w_induction"]:
                    power = power + gate[:, None] * float(combo["w_induction"]) * induction
                if combo["w_prefix"]:
                    power = power + gate[:, None] * float(combo["w_prefix"]) * prefix_array
                if combo["w_copy_fast"] and fast_array is not None:
                    power = power + gate[:, None] * float(combo["w_copy_fast"]) * fast_array
                if combo["w_copy_long"] and long_array is not None:
                    power = power + gate[:, None] * float(combo["w_copy_long"]) * long_array
            total = np.maximum(power.sum(axis=1), 1e-30)
            truth = power[rows_index, targets] / total
            nll = float(-np.mean(np.log(np.maximum(truth, 1e-30))))
            return {
                "nll": nll,
                "bits_per_char": float(nll / math.log(2.0)),
                "top1": float(np.mean(np.argmax(power, axis=1) == targets)),
                "gated_positions": int(gate.sum()),
            }

        option_key: Tuple[int, float] = (int(induction_options[0][0]), float(induction_options[0][1]))
        current: Dict[str, Any] = {
            "tau_H": 2.0,
            "coupling": "mixture",
            "w_photonic": 0.0,
            "w_induction": 0.0,
            "w_prefix": 0.0,
            "w_copy_fast": 0.0,
            "w_copy_long": 0.0,
            "induction_option": list(option_key),
        }
        zeros = np.zeros((count, self.vocab_size), dtype=np.float32)
        stages: Dict[str, Any] = {}
        baseline = score(current, option_key, zeros, zeros)
        current.update(
            {key: baseline[key] for key in ("nll", "bits_per_char", "top1", "gated_positions")}
        )
        stages["baseline"] = dict(baseline)
        fast_array = zeros
        long_array = zeros

        def record(name: str, best: Dict[str, Any], measured: Dict[str, Any], tried: int) -> None:
            stages[name] = {
                "candidates": int(tried),
                "selected": {
                    key: best[key]
                    for key in (
                        "tau_H",
                        "coupling",
                        "w_photonic",
                        "w_induction",
                        "w_prefix",
                        "w_copy_fast",
                        "w_copy_long",
                        "induction_option",
                    )
                    if key in best
                },
                "nll": float(measured["nll"]),
                "bits_per_char": float(measured["bits_per_char"]),
                "top1": float(measured["top1"]),
                "gated_positions": int(measured["gated_positions"]),
            }

        # Stage A: gate, coupling, photonic port, exact-identity induction.
        tried = 0
        best = None
        best_score = None
        for tau in tau_grid:
            for coupling in couplings:
                for port in photonic_grid:
                    for weight in induction_grid:
                        for option in induction_options:
                            merged = dict(current)
                            merged.update(
                                {
                                    "tau_H": float(tau),
                                    "coupling": str(coupling),
                                    "w_photonic": float(port),
                                    "w_induction": float(weight),
                                    "induction_option": [int(option[0]), float(option[1])],
                                }
                            )
                            measured = score(
                                merged, (int(option[0]), float(option[1])), zeros, zeros
                            )
                            tried += 1
                            if best is None or (
                                measured["nll"],
                                -measured["top1"],
                            ) < (best_score["nll"], -best_score["top1"]):
                                best = merged
                                best_score = measured
        current.update(best)
        option_key = (int(best["induction_option"][0]), float(best["induction_option"][1]))
        record("A_induction", best, best_score, tried)

        # Stages B and C: the dual copy caches, shaping chosen by cross-entropy.
        def copy_stage(
            name: str,
            window: int,
            options: Sequence[Tuple[float, float, float]],
            key: str,
            other: np.ndarray,
        ) -> np.ndarray:
            nonlocal fast_array, long_array
            tried_local = 0
            best_local = None
            best_measured = None
            best_array = zeros
            best_shaping = (0.0, 0.0, 0.0)
            for gamma, power, boost in options:
                array = self._channel_arrays(
                    sequence,
                    positions,
                    window=int(window),
                    gamma=float(gamma),
                    power=float(power),
                    boost=float(boost),
                )
                for weight in copy_grid:
                    merged = dict(current)
                    merged[key] = float(weight)
                    if key == "w_copy_fast":
                        measured = score(merged, option_key, array, other)
                    else:
                        measured = score(merged, option_key, other, array)
                    tried_local += 1
                    if best_measured is None or (
                        measured["nll"],
                        -measured["top1"],
                    ) < (best_measured["nll"], -best_measured["top1"]):
                        best_measured = measured
                        best_local = merged
                        best_array = array
                        best_shaping = (float(gamma), float(power), float(boost))
            current.update(best_local)
            if key == "w_copy_fast":
                fast_array = best_array if float(best_local[key]) > 0.0 else zeros
            else:
                long_array = best_array if float(best_local[key]) > 0.0 else zeros
            row = dict(stages.get(name) or {})
            record(name, best_local, best_measured, tried_local)
            stages[name]["selected"]["gamma"] = best_shaping[0]
            stages[name]["selected"]["idf_power"] = best_shaping[1]
            stages[name]["selected"]["boost"] = best_shaping[2]
            _ = row
            return best_shaping

        fast_shaping_used = copy_stage(
            "B_fast_copy", int(self.copy_fast_window), fast_shaping, "w_copy_fast", zeros
        )
        long_shaping_used = copy_stage(
            "C_long_copy", int(self.copy_long_window), long_shaping, "w_copy_long", fast_array
        )

        # Stage D: role-prefix residual induction (kept as an offered channel).
        tried = 0
        best = None
        best_score = None
        for weight in prefix_grid:
            merged = dict(current)
            merged["w_prefix"] = float(weight)
            measured = score(merged, option_key, fast_array, long_array)
            tried += 1
            if best is None or (measured["nll"], -measured["top1"]) < (
                best_score["nll"],
                -best_score["top1"],
            ):
                best = merged
                best_score = measured
        current.update(best)
        record("D_prefix", best, best_score, tried)

        # Stage E2: local refinement of every admitted weight by a scale factor,
        # so a grid boundary cannot cap a channel that the D-Guard wants wider.
        best = None
        best_score = None
        tried = 0
        for factor in refinement:
            candidate = dict(current)
            for key in ("w_photonic", "w_induction", "w_prefix", "w_copy_fast", "w_copy_long"):
                candidate[key] = float(current[key]) * float(factor)
            measured = score(candidate, option_key, fast_array, long_array)
            tried += 1
            if best is None or (measured["nll"], -measured["top1"]) < (
                best_score["nll"],
                -best_score["top1"],
            ):
                best = candidate
                best_score = measured
        current.update(best)
        record("E2_refinement", best, best_score, tried)

        # Stage F: top-1 preservation.  Among the configurations whose inner
        # cross-entropy is within 1e-3 nats of the best, keep the one with the
        # highest inner top-1, so a diffuse channel tail cannot outvote a sharp
        # high-confidence floor argmax.
        reference_nll = float(current.get("nll", 0.0))
        best_top1 = None
        tried_top1 = 0
        for tau in tau_grid:
            for coupling in couplings:
                candidate = {key: current[key] for key in current}
                candidate["tau_H"] = float(tau)
                candidate["coupling"] = str(coupling)
                measured = score(candidate, option_key, fast_array, long_array)
                tried_top1 += 1
                if measured["nll"] > reference_nll + 1e-3:
                    continue
                if best_top1 is None or (
                    measured["top1"],
                    -measured["nll"],
                ) > (best_top1[0]["top1"], -best_top1[0]["nll"]):
                    best_top1 = (measured, candidate)
        if best_top1 is not None:
            current.update(best_top1[1])
            current.update(
                {
                    key: best_top1[0][key]
                    for key in ("nll", "bits_per_char", "top1", "gated_positions")
                }
            )
            record("F_top1_preservation", best_top1[1], best_top1[0], tried_top1)

        # Stage E: re-join the gate, coupling and port under the admitted channels.
        tried = 0
        best = None
        best_score = None
        for tau in tau_grid:
            for coupling in couplings:
                for port in photonic_grid:
                    merged = dict(current)
                    merged.update(
                        {
                            "tau_H": float(tau),
                            "coupling": str(coupling),
                            "w_photonic": float(port),
                        }
                    )
                    measured = score(merged, option_key, fast_array, long_array)
                    tried += 1
                    if best is None or (measured["nll"], -measured["top1"]) < (
                        best_score["nll"],
                        -best_score["top1"],
                    ):
                        best = merged
                        best_score = measured
        current.update(best)
        record("E_rejoin", best, best_score, tried)

        self.tau_gate_h = float(current["tau_H"])
        self.coupling = str(current["coupling"])
        self.w_photonic = float(current["w_photonic"])
        self.w_induction = float(current["w_induction"])
        self.w_prefix = float(current["w_prefix"])
        self.w_copy_fast = float(current["w_copy_fast"])
        self.w_copy_long = float(current["w_copy_long"])
        self.w_copy = 0.0
        self.mask_min_count = int(option_key[0])
        self.induction_idf_power = float(option_key[1])
        self.copy_fast_gamma = float(fast_shaping_used[0])
        self.copy_fast_power = float(fast_shaping_used[1])
        self.copy_fast_boost = float(fast_shaping_used[2])
        self.copy_long_gamma = float(long_shaping_used[0])
        self.copy_long_power = float(long_shaping_used[1])
        self.copy_long_boost = float(long_shaping_used[2])
        self._ac_induction_cache.clear()
        final = score(current, option_key, fast_array, long_array)
        self.channel_report = {
            "selected": {
                key: current[key]
                for key in (
                    "tau_H",
                    "coupling",
                    "w_photonic",
                    "w_induction",
                    "w_prefix",
                    "w_copy_fast",
                    "w_copy_long",
                    "induction_option",
                    "nll",
                    "bits_per_char",
                    "top1",
                )
                if key in current
            },
            "samples": int(count),
            "tokens": int(count),
            "stages": stages,
            "final": dict(final),
            "grids": {
                "tau_H": [float(value) for value in tau_grid],
                "w_photonic": [float(value) for value in photonic_grid],
                "w_induction": [float(value) for value in induction_grid],
                "w_prefix": [float(value) for value in prefix_grid],
                "w_copy": [float(value) for value in copy_grid],
                "induction_options": [[int(row[0]), float(row[1])] for row in induction_options],
                "fast_shaping": [list(map(float, row)) for row in fast_shaping],
                "long_shaping": [list(map(float, row)) for row in long_shaping],
                "coupling": [str(value) for value in couplings],
            },
            "copy_shaping": {
                "fast_gamma": float(self.copy_fast_gamma),
                "fast_boost": float(self.copy_fast_boost),
                "fast_idf_power": float(self.copy_fast_power),
                "long_gamma": float(self.copy_long_gamma),
                "long_boost": float(self.copy_long_boost),
                "long_idf_power": float(self.copy_long_power),
                "fast_window": int(self.copy_fast_window),
                "long_window": int(self.copy_long_window),
                "low": int(self.copy_low),
            },
            "zero_point_available": True,
            "identity_dim": int(self.identity_dim),
            "identity_audit": self.identity.audit(),
        }
        self.induction_report = dict(self.channel_report)
        self.ac_report = dict(self.channel_report)
        return dict(self.channel_report)


# ======================================================================
# Cycle AC closure: Zero-Phase Homodyne Optimality and the
# vocabulary-invariant relational / anti-phase law
# ======================================================================
#
# The AC1 ablation measured a physical law, not a tuning failure.  The count
# floor amplitude s_0(v|h) = sqrt(P_floor) is real and non-negative, and the
# exact-identity unbinding <w_j chi(v) + xi, chi(v)> = w_j + xi is real with
# expected phase 0 on every true match.  So Phi = 0 already sits at the global
# constructive maximum cos(0) = +1.  A token-indexed table Phi in R^{M x V} can
# only lower a finite training slice by fitting the accidental cross-talk phase
# arg xi (or by suppressing tokens that happen to be rare in the training
# documents); on unseen document tails that phase is independent, so the
# learned rotation is pure jitter and costs about +kappa sigma_phi^2 / 2 out of
# sample.  The measured cost was +0.0056 bpc for the table and +0.0041 bpc for
# the 88-parameter slot torque.
#
# What CAN transfer is interference whose domain is not the vocabulary:
#
#   (i)   an anti-phase (phi = pi) destructive term on an immediate
#         non-whitespace token repeat, which a non-negative mixture cannot
#         express because mixtures can only add mass;
#   (ii)  a single relative phase between the floor mode and the local modes,
#         conditioned on three token-ID-free scalars: the normalised floor
#         entropy, the exact-identity hit strength and the matched suffix
#         order;
#   (iii) a high-confidence floor preservation guard, so a diffuse copy tail
#         cannot outvote a sharp high-order floor argmax.
#
# The whole organ is nine scalars with zero per-token parameters, and every
# grid contains 0.0, so the D-Guard clamps it to the exact zero-phase optimum
# whenever the transfer guard is neutral.

_WHITESPACE_PIECES = frozenset(
    [" ", "\n", "\t", "  ", "\n\n", "\r\n", "    "]
)


class HomodyneEgress(KroneckerInductionEgress):
    """AC closure egress: zero token phase plus a vocabulary-invariant law.

    The deployed token phase table is identically zero and the slot torque is
    never fitted on the deployed path.  Nine scalars drive the invariant law:

    ``anti_phase_gain``   destructive weight of the ``phi = pi`` stutter term
    ``phase_a1..a3``      relative phase between floor and local modes
    ``floor_guard_tau``   floor-entropy threshold of the preservation guard
    ``floor_guard_weight`` blend weight of the preservation guard
    ``copy_alpha``        power-law exponent of the long-range copy kernel
    ``copy_tau0``         power-law time constant
    ``copy_scope_boost``  boost for identifiers in the most recent def header
    """

    def __init__(
        self,
        projector,
        tokenizer,
        *,
        phase_seed: float = 0.0,
        anti_phase_gain: float = 0.0,
        phase_a1: float = 0.0,
        phase_a2: float = 0.0,
        phase_a3: float = 0.0,
        floor_guard_tau: float = 1.0,
        floor_guard_weight: float = 0.0,
        floor_guard_min_order: int = 4,
        floor_guard_tau_z: float = 0.5,
        copy_alpha: float = 0.0,
        copy_tau0: float = 64.0,
        copy_scope_boost: float = 1.0,
        copy_sharpness: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(projector, tokenizer, phase_seed=float(phase_seed), **kwargs)
        self.anti_phase_gain = float(anti_phase_gain)
        self.phase_a1 = float(phase_a1)
        self.phase_a2 = float(phase_a2)
        self.phase_a3 = float(phase_a3)
        self.floor_guard_tau = float(floor_guard_tau)
        self.floor_guard_weight = float(floor_guard_weight)
        self.floor_guard_min_order = int(floor_guard_min_order)
        self.floor_guard_tau_z = float(floor_guard_tau_z)
        self.copy_alpha = float(copy_alpha)
        self.copy_tau0 = float(copy_tau0)
        self.copy_scope_boost = float(copy_scope_boost)
        self.copy_sharpness = float(copy_sharpness)
        self._text: Dict[int, str] = {}
        self._header_cache: Dict[Tuple[int, ...], frozenset] = {}
        self.homodyne_report: Dict[str, Any] = {}

    # ------------------------------------------------------------ vocabulary
    def _token_text(self, token: int) -> str:
        cached = self._text.get(int(token))
        if cached is None:
            try:
                cached = str(self.tokenizer.token_text(int(token)))
            except Exception:  # noqa: BLE001 - an unknown piece is not an indent
                cached = ""
            self._text[int(token)] = cached
        return cached

    def _is_indent_like(self, token: int) -> bool:
        return self._token_text(int(token)) in _WHITESPACE_PIECES

    # -------------------------------------------------- invariant phase scalars
    def invariant_scalars(self) -> int:
        return 9

    def invariant_active(self) -> bool:
        return bool(
            self.anti_phase_gain != 0.0
            or self.floor_guard_weight != 0.0
            or abs(self.phase_a1) + abs(self.phase_a2) + abs(self.phase_a3) > 0.0
        )

    def _floor_state(
        self, rows: np.ndarray
    ) -> Tuple[float, int, np.ndarray]:
        """(normalised floor entropy, matched suffix order, floor power)."""
        floor_power = np.square(np.asarray(rows[0], dtype=np.float64))
        total = float(floor_power.sum())
        if total <= 0.0:
            return 0.0, 0, floor_power
        probability = floor_power / total
        entropy = float(-np.sum(probability * np.log(np.maximum(probability, 1e-12))))
        matched = 0
        for order in self.orders:
            index = int(order)
            if index < rows.shape[0] and float(np.abs(rows[index]).sum()) > 0.0:
                matched = max(matched, index)
        return entropy, int(matched), floor_power

    def effective_phase(
        self, context: Sequence[int], rows: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Phi = the deployed table (zero), plus one floor/local relative phase."""
        base = np.array(self.phase_table, dtype=np.float64, copy=True)
        if self.phase_a1 == 0.0 and self.phase_a2 == 0.0 and self.phase_a3 == 0.0:
            return base
        if rows is None:
            rows, _names = self._suffix_rows(context)
        entropy, matched, _power = self._floor_state(rows)
        sharpness = entropy / math.log(float(self.vocab_size))
        hit = float(np.abs(rows[self.induction_row]).max()) if rows.shape[0] > self.induction_row else 0.0
        argument = (
            float(self.phase_a1) * (sharpness - 0.5)
            + float(self.phase_a2) * (min(hit, 1.0) - 1.0)
            + float(self.phase_a3) * (float(matched) / float(max(self.max_order, 1)) - 0.5)
        )
        delta = math.pi * math.tanh(argument)
        base[1:, :] = delta
        return base

    # ------------------------------------------------------- def-header scope
    def _def_header(self, window: Sequence[int]) -> frozenset:
        key = tuple(int(token) for token in window[-256:])
        cached = self._header_cache.get(key)
        if cached is not None:
            return cached
        header: set = set()
        last = -1
        for index, token in enumerate(key):
            text = self._token_text(int(token))
            if text.strip() == "def" or text.endswith("def") or text == "def ":
                last = int(index)
        if last >= 0:
            for offset in range(last + 1, min(len(key), last + 48)):
                piece = self._token_text(int(key[offset]))
                header.add(int(key[offset]))
                if ":" in piece:
                    break
        result = frozenset(header)
        if len(self._header_cache) < 4096:
            self._header_cache[key] = result
        return result

    # ---------------------------------------------------------- copy kernel
    def copy_terms(
        self,
        context: Sequence[int],
        *,
        window: int,
        gamma: float,
        boost: float = 1.0,
        power: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        """IDF copy with a power-law recency tail and a def-header scope boost.

        The exponential gamma^(offset) decays a parameter declared 200 tokens
        ago to gamma^200, which erases it; the power-law tail
        (1 + offset / tau0)^(-alpha) keeps a long, heavy-tailed memory of the
        enclosing signature, and identifiers inside the most recent def header
        inside the window carry an extra scope_boost.
        """
        tokens = [int(token) for token in context]
        length = len(tokens)
        if length <= self.copy_low:
            return None
        high = min(int(window), length - 1)
        if high < self.copy_low:
            return None
        exponent = float(self.idf_power if power is None else power)
        offsets = np.arange(self.copy_low, high + 1, dtype=np.int64)
        index = np.asarray([tokens[-int(offset)] for offset in offsets], dtype=np.int64)
        valid = (index >= 0) & (index < self.vocab_size)
        index = index[valid]
        offsets = offsets[valid]
        if index.size == 0:
            return None
        weights = np.power(float(gamma), (offsets - self.copy_low).astype(np.float64))
        weights = weights * np.power(np.maximum(self.idf[index], 1e-6), exponent)
        if float(self.copy_alpha) > 0.0:
            weights = weights * np.power(
                1.0 + (offsets - self.copy_low).astype(np.float64) / max(float(self.copy_tau0), 1.0),
                -float(self.copy_alpha),
            )
        if float(self.copy_scope_boost) != 1.0:
            header = self._def_header(context)
            if header:
                weights = weights * np.where(
                    np.isin(index, np.fromiter(header, dtype=np.int64, count=len(header))),
                    float(self.copy_scope_boost),
                    1.0,
                )
        if float(boost) != 1.0:
            weights = weights * np.where(
                self._role_array[index] == _ROLE_WORD, float(boost), 1.0
            )
        counts = np.bincount(index, weights=weights, minlength=self.vocab_size)
        if float(counts.sum()) <= 0.0:
            return None
        score = self.identity.project(counts)
        positive = np.maximum(score, 0.0)
        total = float(positive.sum())
        if total <= 0.0:
            return None
        return positive / total

    # ------------------------------------------------------------- readout
    def _readout(self, context: Sequence[int]) -> np.ndarray:
        """Born readout with the anti-phase stutter term and the floor guard."""
        rows, _names = self._suffix_rows(context)
        theta = self.effective_phase(context, rows)
        chord = np.sum(np.exp(1j * theta) * rows.astype(np.complex128), axis=0)
        power = np.abs(chord) ** 2
        total = float(power.sum())
        if total <= 0.0:
            return np.full(self.vocab_size, 1.0 / float(self.vocab_size))
        base = power / total
        gate = self.gate_weight(context, weights=None, dists=None)
        if gate <= 0.0:
            return base
        port = self.photonic_port(context) if float(self.w_photonic) > 0.0 else None
        induction = self.induction_terms(context) if float(self.w_induction) > 0.0 else None
        prefix = self.prefix_terms(context) if float(self.w_prefix) > 0.0 else None
        fast = self.copy_fast(context) if float(self.w_copy_fast) > 0.0 else None
        long = self.copy_long(context) if float(self.w_copy_long) > 0.0 else None
        extra = np.zeros(self.vocab_size, dtype=np.float64)
        w_port = float(self.w_photonic) if port is not None else 0.0
        w_ind = float(self.w_induction) if induction is not None else 0.0
        w_prefix = float(self.w_prefix) if prefix is not None else 0.0
        w_fast = float(self.w_copy_fast) if fast is not None else 0.0
        w_long = float(self.w_copy_long) if long is not None else 0.0
        if port is not None:
            extra += w_port * np.asarray(port, dtype=np.float64)
        if induction is not None:
            extra += w_ind * np.asarray(induction[0], dtype=np.float64)
        if prefix is not None:
            extra += w_prefix * np.asarray(prefix[0], dtype=np.float64)
        if fast is not None:
            extra += w_fast * np.asarray(fast, dtype=np.float64)
        if long is not None:
            extra += w_long * np.asarray(long, dtype=np.float64)
        if float(self.anti_phase_gain) > 0.0 and len(context) > 0:
            last = int(context[-1])
            if (
                0 <= last < self.vocab_size
                and not self._is_indent_like(last)
                and (fast is not None or long is not None)
            ):
                # phi = pi on the immediate repeat: flip a controlled fraction
                # of the copy mass that this candidate would otherwise receive.
                extra[last] *= 1.0 - 2.0 * float(self.anti_phase_gain)
        weight_sum = 0.0
        power_total = 0.0
        if self.coupling == "amplitude":
            mixed = np.abs(chord + gate * extra) ** 2
        else:
            weight_sum = w_port + w_ind + w_prefix + w_fast + w_long
            residual = float(min(max(1.0 - gate * weight_sum, 0.0), 1.0))
            mixed = residual * power
            if port is not None:
                mixed = mixed + gate * w_port * np.asarray(port, dtype=np.float64)
            if induction is not None:
                mixed = mixed + gate * w_ind * np.asarray(induction[1], dtype=np.float64)
            if prefix is not None:
                mixed = mixed + gate * w_prefix * np.asarray(prefix[1], dtype=np.float64)
            if fast is not None:
                mixed = mixed + gate * w_fast * np.asarray(fast, dtype=np.float64)
            if long is not None:
                mixed = mixed + gate * w_long * np.asarray(long, dtype=np.float64)
        power_total = float(mixed.sum())
        if power_total <= 0.0:
            return base
        probability = mixed / power_total
        if float(self.floor_guard_weight) > 0.0:
            entropy, matched, floor_power = self._floor_state(rows)
            hit = (
                1.0
                if induction is not None
                and float(np.max(induction[1])) > float(self.floor_guard_tau_z)
                else 0.0
            )
            floor_total = float(floor_power.sum())
            if (
                floor_total > 0.0
                and matched >= int(self.floor_guard_min_order)
                and entropy <= float(self.floor_guard_tau)
                and hit < 0.5
            ):
                # Preserve the sharp high-confidence floor argmax: a diffuse
                # copy tail must not outvote it, so the guard mixes a point
                # mass on that one token rather than the whole floor row.
                best = int(np.argmax(floor_power))
                weight = float(min(max(self.floor_guard_weight, 0.0), 1.0))
                preserved = np.array(probability, dtype=np.float64, copy=True)
                preserved *= 1.0 - weight
                preserved[best] += weight
                total_preserved = float(preserved.sum())
                if total_preserved > 0.0:
                    return preserved / total_preserved
        return probability
