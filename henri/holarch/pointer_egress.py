"""Cycle Z1: pointer-resonant photonic egress at D = 2048.

The Y1 measurement left two exact physical defects:

1. A context-invariant scalar fusion weight averages over the ~90 percent of
   positions where the six-order suffix floor already peaks, so the
   cross-entropy-optimal weight is zero.  The fix is an ENTROPY-GATED weight:
   the photonic long-range pointer is admitted only where the suffix entropy
   exceeds tau_H.
2. A scalar per-slot phase phi[m, s] is a U(1) gauge rotation inside one
   slot: every co-slot competitor rotates with the target, so the normalised
   likelihood gradient vanishes (measured 0.0000 rad in Y1).  The fix is a
   FREQUENCY-SELECTIVE rotor theta[m, s, b] over B_freq sub-bands tiling
   d_slot, applied in the complex wave domain before the Hermitian inner
   product.  Different tokens in the same slot load different sub-bands, so
   the rotor is not a gauge rotation and its torque is non-zero.

This module also implements the long-range POINTER: the position-unbound
resonance of the aligned post-photonic port at offsets 7..24.  A six-order
suffix bank is structurally blind to those offsets, so the pointer is the
only channel that can copy an identifier seen 7 to 24 tokens back.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from henri.holarch.wave_egress import WaveEgressHead, _EPS


class PointerEgress(WaveEgressHead):
    """Wave egress with an entropy-gated long-range pointer and a band rotor."""

    def __init__(
        self,
        projector,
        tokenizer,
        *,
        core=None,
        cartridge=None,
        freq_bands: int = 32,
        pointer_low: int = 7,
        pointer_high: int = 24,
        rotor_modes: Optional[Sequence[int]] = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("quadrature", True)
        super().__init__(projector, tokenizer, **kwargs)
        self.core = core
        self.cartridge = cartridge
        self.freq_bands = int(freq_bands)
        self.pointer_low = int(pointer_low)
        self.pointer_high = int(pointer_high)
        self.pointer_slot = int(self.phase.shape[0])
        self.rotor = np.zeros(
            (self.pointer_slot + 1, 4, self.freq_bands), dtype=np.float64
        )
        # Deterministic small incommensurate seed, the same symmetry break as
        # the Y1 phase vector, per band.
        for mode in range(self.rotor.shape[0]):
            for slot in range(4):
                for band in range(self.freq_bands):
                    value = (
                        (mode + 1) * math.sqrt(2.0)
                        + (slot + 1) * math.sqrt(3.0)
                        + (band + 1) * math.sqrt(5.0)
                    )
                    self.rotor[mode, slot, band] = 0.10 * (value - math.floor(value))
        self.rotor0 = self.rotor.copy()
        # The rotor applies to the suffix modes by default.  Mode 0 (the
        # count prior) is dense over the vocabulary, so a band rotation there
        # costs V * d_slot per position without adding a distinct channel.
        self.rotor_modes = tuple(
            int(value) for value in (rotor_modes if rotor_modes is not None else range(1, self.pointer_slot))
        )
        # A mixing frame, not the identity: with U = I the Born power is the
        # incoherent sum of the modes and EVERY phase is gauge, which is the
        # measured Y1 defect.  The DFT frame makes the modes interfere from
        # step 0, so a band phase has leverage.  fit(qr_step=0) keeps it.
        modes = int(self.adapter.shape[0])
        index = np.arange(modes)
        self.adapter = (
            np.exp(2j * math.pi * np.outer(index, index) / float(modes))
            / math.sqrt(float(modes))
        ).astype(np.complex128)
        self.mixer_seeded = True
        self.w_low = 0.0
        self.w_high = 0.0
        self.tau_h = 2.0
        self.fusion_report: Dict[str, Any] = {}
        self.rotor_report: Dict[str, Any] = {}
        self._pointer_cache: Dict[Tuple[int, ...], Tuple[np.ndarray, np.ndarray]] = {}
        self.pointer_contexts = 0

    # ------------------------------------------------------------- portrait
    def _position_power(self) -> np.ndarray:
        cache = getattr(self, "_position_cache", None)
        if cache is None:
            cache = self.projector.position_wave()
            self._position_cache = cache
        return cache

    def _aligned_port(self, context: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
        """The cartridge-aligned bright port and the context wave."""
        import torch

        key = tuple(int(token) for token in context[-24:])
        cached = self._pointer_cache.get(key)
        if cached is not None:
            return cached
        wave = self.context_wave(key)
        if self.core is None:
            return wave, wave
        psi = torch.from_numpy(np.asarray(wave, dtype=np.complex64)).reshape(1, -1)
        with torch.no_grad():
            state = self.core(psi)
            bright = state.bright[0].detach().cpu().numpy().astype(np.complex64)
            dark = state.dark[0].detach().cpu().numpy().astype(np.complex64)
        if self.cartridge is not None:
            bright, dark = self.cartridge.adapt_arrays(bright, dark)
        self.pointer_contexts += 1
        if len(self._pointer_cache) < 8192:
            self._pointer_cache[key] = (bright, dark)
        return bright, dark

    def pointer_readout(
        self, context: Sequence[int], *, support_size: int = 64
    ) -> Optional[np.ndarray]:
        """Per-position unbound resonance at offsets 7..24, on a support.

        The dense full-vocabulary resonance is diffuse: 18 superposed unbound
        positions put 18 different identifiers into one readout, so the target
        is never the argmax.  The pointer therefore resolves each offset
        separately and keeps the per-offset power on a CANDIDATE support (the
        top of the base mixture).  That is the copy prior the suffix bank
        cannot compute, restricted to the tokens the local model already
        admits.
        """
        bright, _dark = self._aligned_port(context)
        if bright is None:
            return None
        length = min(int(self.pointer_high), max(0, len(context) - 1))
        if length < int(self.pointer_low):
            return None
        base = self.base_distribution(context)
        support = np.argsort(-np.asarray(base))[: int(support_size)]
        support = support[np.asarray(base)[support] > 0.0]
        if support.size == 0:
            return None
        position = self._position_power()
        phase = np.angle(position)
        rows = np.asarray(self.phi[support], dtype=np.complex128)   # (k, D)
        field = np.asarray(bright, dtype=np.complex128)
        power = np.zeros(support.size, dtype=np.float64)
        for offset in range(int(self.pointer_low), int(length) + 1):
            unbound = field * np.exp(-1j * phase * float(offset))
            overlap = np.abs(rows.conj() @ unbound) ** 2
            # MAX, not sum: 18 superposed offsets put 18 different identifiers
            # into one readout, so a sum ranks the strongest offset and loses
            # the target.  The max asks "is this token present at ANY of the
            # long-range offsets", which is the copy prior itself.
            power = np.maximum(power, overlap)
        mass = float(power.sum())
        if mass <= 0.0:
            return None
        out = np.zeros(self.vocab_size, dtype=np.float64)
        out[support] = power / mass
        return out

    # --------------------------------------------------------------- rotor
    def rotor_displacement(self) -> float:
        difference = np.asarray(self.rotor) - np.asarray(self.rotor0)
        return float(np.linalg.norm(difference))

    def _mode_phasor(
        self, index: int, wave: np.ndarray, support: np.ndarray
    ) -> np.ndarray:
        """Band-resolved overlap: exp(i arg sum_b o_b exp(i theta_m,s,b))."""
        if index not in self.rotor_modes or support.size == 0:
            return super()._mode_phasor(index, wave, support)
        phasor = np.ones(self.vocab_size, dtype=np.complex128)
        d_slot = self.dim // 4
        width = d_slot // self.freq_bands
        if width * self.freq_bands != d_slot:
            return super()._mode_phasor(index, wave, support)
        slot_ids = self._slot_ids[support]
        for slot in range(4):
            index_rows = support[slot_ids == slot]
            if index_rows.size == 0:
                continue
            start = slot * d_slot
            stop = start + d_slot
            wave_slot = np.asarray(wave[start:stop], dtype=np.complex128)
            chi = np.asarray(
                self.phi[index_rows][:, start:stop], dtype=np.complex128
            )
            product = chi.conj() * wave_slot[None, :]
            bands = product.reshape(index_rows.size, self.freq_bands, width).sum(axis=2)
            theta = self.rotor[index, slot]
            overlap = bands @ np.exp(1j * theta)
            magnitude = np.abs(overlap)
            safe = magnitude > 1e-6
            unit = np.ones_like(overlap)
            unit[safe] = overlap[safe] / magnitude[safe]
            phasor[index_rows] = unit
        return phasor

    # -------------------------------------------------------------- fusion
    def suffix_entropy(self, context: Sequence[int]) -> float:
        # The base mixture, without the pointer: calling self._order_waves_full
        # here would recurse through the fusion weight it is meant to compute.
        weights, dists, _totals, _hp, _terms = super()._order_waves_full(context, None)
        mixture = np.zeros(self.vocab_size, dtype=np.float64)
        for weight, dist in zip(weights, dists):
            mixture += float(weight) * np.asarray(dist, dtype=np.float64)
        mixture = np.maximum(mixture, 1e-12)
        return float(-np.sum(mixture * np.log(mixture)))

    def fusion_weight(self, context: Sequence[int], *, force_low: bool = False) -> float:
        if force_low:
            return float(self.w_low)
        entropy = self.suffix_entropy(context)
        return float(self.w_high if entropy > float(self.tau_h) else self.w_low)

    def _order_waves_full(self, context: Sequence[int], model=None):
        weights, dists, totals, has_photonic, terms = super()._order_waves_full(
            context, model
        )
        if self.core is None:
            return weights, dists, totals, has_photonic, terms
        weight = self.fusion_weight(context)
        if weight <= 0.0:
            return weights, dists, totals, has_photonic, terms
        pointer = self.pointer_readout(context)
        if pointer is None:
            return weights, dists, totals, has_photonic, terms
        # Append the pointer mode and renormalise the whole mixture.
        extended = [float(value) for value in weights] + [float(weight)]
        mass = float(sum(extended))
        if mass <= 0.0:
            return weights, dists, totals, has_photonic, terms
        return (
            [value / mass for value in extended],
            list(dists) + [pointer],
            list(totals) + [1.0],
            has_photonic,
            list(terms) + [None],
        )

    # -------------------------------------------------------- rotor learner
    def _band_overlaps(
        self, mode: int, wave: np.ndarray, token: int
    ) -> Optional[np.ndarray]:
        slot = int(self._slot_ids[token])
        d_slot = self.dim // 4
        width = d_slot // self.freq_bands
        start = slot * d_slot
        stop = start + d_slot
        wave_slot = np.asarray(wave[start:stop], dtype=np.complex128)
        chi = np.asarray(self.phi[token][start:stop], dtype=np.complex128)
        product = chi.conj() * wave_slot
        return product.reshape(self.freq_bands, width).sum(axis=1)

    def fit_rotor(
        self,
        ids: Sequence[int],
        *,
        epochs: int = 6,
        samples: int = 240,
        inner_fraction: float = 0.15,
        step: float = 0.05,
        floor_fn=None,
        modes: Optional[Sequence[int]] = None,
    ) -> Dict[str, Any]:
        """Analytic contrastive Born-rule torque on theta[m, s, b].

        For a mode resonance a_m(v) = |a| exp(i arg o(v)) with the band
        decomposition o(v) = sum_b o_b(v) exp(i theta_b), the exact derivative
        of the Born power with respect to one band phase is

            d|A(v)|^2 / d theta_b = -2 Im( sum_m c_m a_m(v) g_b(m) conj(A(v)) )

        with c_m the Stiefel-mixer column sum and

            g_b(m) = Im( i exp(i theta_b) o_b(v) / o(v) ).

        The target's slot is pushed up and the floor's wrong competitor's
        slot is pushed down.  A step is accepted only when the inner slice
        cross-entropy falls, so the recorded curve is best-so-far.
        """
        sequence = [int(token) for token in ids]
        if self.base_counts is None:
            from henri.holarch.code_gen import CountLanguageModel

            self.base_counts = CountLanguageModel(self.vocab_size)
            self.base_counts.fit(sequence)
        cut = int(len(sequence) * (1.0 - float(inner_fraction)))
        fit_ids, inner_ids = sequence[:cut], sequence[cut:]
        if len(fit_ids) <= self.max_order + 2 or len(inner_ids) <= self.max_order + 2:
            return {"epochs": 0, "updates": 0}
        count = int(min(int(samples), max(1, len(fit_ids) - self.max_order - 2)))
        stride = max(1, (len(fit_ids) - self.max_order - 2) // count)
        positions = list(range(self.max_order, len(fit_ids) - 1, stride))[:count]
        inner_positions = list(
            range(self.max_order, len(inner_ids) - 1, max(1, len(inner_ids) // 160))
        )[:160]
        active = tuple(int(value) for value in (modes if modes is not None else self.rotor_modes))

        def inner_nll() -> float:
            total = 0.0
            seen = 0
            for position in inner_positions:
                context = inner_ids[max(0, position - self.window) : position]
                target = int(inner_ids[position])
                probability = self.born(self.amplitudes(context, photonic=False)[0])
                total -= float(np.log(max(float(probability[target]), 1e-12)))
                seen += 1
            return float(total / max(seen, 1))

        initial = inner_nll()
        best = (float(initial), self.rotor.copy())
        curve: List[float] = [float(initial)]
        updates = 0
        for _epoch in range(int(epochs)):
            gradient = np.zeros_like(self.rotor)
            touched = 0
            for position in positions:
                context = fit_ids[max(0, position - self.window) : position]
                target = int(fit_ids[position])
                if not (0 <= target < self.vocab_size):
                    continue
                state, info = self.amplitudes(context, photonic=False)
                pre = info.get("pre_mixer")
                if pre is None:
                    continue
                modes_present = int(state.shape[0])
                mixer = np.asarray(self.adapter[:modes_present, :modes_present], dtype=np.complex128)
                column = mixer.sum(axis=0)
                power = float(np.sum(np.abs(state) ** 2)) + _EPS
                competitor = -1
                if floor_fn is not None:
                    try:
                        floor = floor_fn(context)
                        if floor is not None:
                            competitor = int(np.argmax(floor))
                    except Exception:  # noqa: BLE001
                        competitor = -1
                for token, sign in ((target, 1.0), (competitor, -1.0)):
                    if not (0 <= token < self.vocab_size) or token == target and sign < 0:
                        continue
                    slot = int(self._slot_ids[token])
                    amplitude = complex(state[:, token].sum())
                    for mode in active:
                        if mode >= modes_present:
                            continue
                        wave = self._mode_wave(mode, context)
                        bands = self._band_overlaps(mode, wave, token)
                        if bands is None:
                            continue
                        theta = self.rotor[mode, slot]
                        rotated = bands * np.exp(1j * theta)
                        total_overlap = complex(np.sum(rotated))
                        if abs(total_overlap) <= 1e-9:
                            continue
                        g = np.imag(1j * rotated / total_overlap)
                        # a_m(token) is pre[mode, token]
                        torque = -2.0 * float(
                            np.imag(
                                np.sum(
                                    column[mode] * pre[mode, token] * g
                                ) * np.conj(amplitude)
                            )
                        ) / power
                        gradient[mode, slot] += sign * torque
                    touched += 1
            if touched == 0:
                break
            # Gauge fixing: a common shift of every band phase of one mode is
            # the exact U(1) rotation Y1 measured as invisible.  Only the
            # RELATIVE band phases carry leverage, so the uniform component is
            # projected out before the step.
            for mode in active:
                if mode >= gradient.shape[0]:
                    continue
                for slot in range(gradient.shape[1]):
                    row = gradient[mode, slot]
                    if float(np.any(row)):
                        gradient[mode, slot] = row - float(np.mean(row))
            scale = float(np.max(np.abs(gradient)))
            if scale <= 0.0:
                break
            direction = gradient / scale
            trial = self.rotor + float(step) * direction
            self.rotor = trial
            value = inner_nll()
            if value < best[0] - 1e-12:
                best = (float(value), trial.copy())
                updates += int(touched)
                curve.append(float(value))
            else:
                self.rotor = best[1].copy()
                step *= 0.5
                curve.append(float(min(best[0], value)))
                if step < 1e-4:
                    break
        self.rotor = best[1]
        self.rotor_report = {
            "inner_nll_initial": float(initial),
            "inner_nll_final": float(best[0]),
            "inner_nll_gain": float(initial - best[0]),
            "displacement": float(self.rotor_displacement()),
            "updates": int(updates),
            "epochs": int(epochs),
            "bands": int(self.freq_bands),
            "modes": list(active),
            "curve": [float(value) for value in curve],
        }
        return dict(self.rotor_report)

    # ------------------------------------------------------- fusion learner
    def calibrate_pointer_fusion(
        self,
        ids: Sequence[int],
        *,
        samples: int = 120,
        low_grid: Sequence[float] = (0.0, 0.05, 0.10),
        high_grid: Sequence[float] = (0.0, 0.10, 0.20, 0.30),
        tau_grid: Sequence[float] = (1.5, 2.0, 2.5, 3.0),
        context: int = 24,
    ) -> Dict[str, Any]:
        """Select (w_low, w_high, tau_H) by inner-slice cross-entropy.

        The grid contains (0.0, 0.0) for every tau, so the D-Guard identity
        is available: the calibrator can always choose to switch the pointer
        off, and it does so unless the cross-entropy falls.
        """
        sequence = [int(token) for token in ids]
        indices = list(range(context, len(sequence) - 1))
        stride = max(1, len(indices) // max(int(samples), 1))
        indices = indices[::stride][: int(samples)]
        rows: List[Dict[str, float]] = []
        best: Optional[Dict[str, float]] = None
        saved = (self.w_low, self.w_high, self.tau_h)
        try:
            for tau in tau_grid:
                self.tau_h = float(tau)
                for low in low_grid:
                    for high in high_grid:
                        self.w_low = float(low)
                        self.w_high = float(high)
                        nll = 0.0
                        top1 = 0
                        for index in indices:
                            window = sequence[max(0, index - context) : index]
                            truth = int(sequence[index])
                            probability = self.distribution(window, None)
                            nll -= float(np.log(max(float(probability[truth]), 1e-12)))
                            top1 += int(int(np.argmax(probability)) == truth)
                        tokens = max(len(indices), 1)
                        row = {
                            "tau_H": float(tau),
                            "w_low": float(low),
                            "w_high": float(high),
                            "nll": float(nll / tokens),
                            "bits_per_char": float(nll / tokens / math.log(2.0)),
                            "top1": float(top1 / tokens),
                        }
                        rows.append(row)
                        if best is None or (row["nll"], -row["top1"]) < (
                            best["nll"], -best["top1"]
                        ):
                            best = row
        finally:
            if best is None:
                self.w_low, self.w_high, self.tau_h = saved
        self.w_low = float(best["w_low"]) if best else saved[0]
        self.w_high = float(best["w_high"]) if best else saved[1]
        self.tau_h = float(best["tau_H"]) if best else saved[2]
        self.fusion_report = {
            "selected": dict(best or {}),
            "rows": len(rows),
            "grid": {
                "low": [float(value) for value in low_grid],
                "high": [float(value) for value in high_grid],
                "tau": [float(value) for value in tau_grid],
            },
            "zero_point_available": True,
        }
        return dict(self.fusion_report)


def core_model(core):
    """A minimal model view so the egress can mount a bare photonic core."""

    class _CoreModel:
        zone_b = core

    return _CoreModel()
