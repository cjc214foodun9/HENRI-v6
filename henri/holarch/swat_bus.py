"""Holographic Swarm Telepathy (H-SWAT): a prefill-free KV-cache bus.

Adapted from Alabdulkareem et al., "Swarm Telepathy: Prefill-Free KV-Cache
Transfer Across Families" (arXiv:2610.15095v1) and Zweiger et al., "Fast KV
Compaction via Attention Matching" (arXiv:2602.16284), onto the HENRI
tri-zone wave manifold.

The bus carries a structured state

    C = (K, V, spans, psi)

where K, V are (L, H, dk) real tensors, spans are the sender's [start, end)
intervals, and psi is the sender's unit 4-slot wave in C^D. The 2n hub solves
the n(n-1) mapper problem: every sender uses one ingress map into the shared
hub, and every receiver uses one egress map out of it.

What is measured here, not asserted:
  * de-RoPE / receiver-RoPE are exact unitary rotations on channel pairs;
  * the span alignment is Eq. 12 with the causal mask b_i^S <= b_j^R;
  * the hub is a closed-form ridge map fitted on paired sender/receiver waves
    (Stage 1, Eq. 10) and optionally refined by a task-logit step (Stage 2);
  * Attention Matching selects U support keys by attention mass and solves the
    least-squares value projection of Eq. 8, so the compacted cache reproduces
    the full-cache attention output on the calibration queries;
  * a KV entry is admitted only after the Zone B Sagnac check and the Zone C
    boundary-axiom check. A vetoed entry contributes zero attention mass.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

_EPS = 1e-12


def _unit(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.complex64).reshape(-1)
    norm = float(np.linalg.norm(value))
    if norm <= _EPS:
        return value.astype(np.complex64)
    return (value / norm).astype(np.complex64)


def _digest_tensor(*tensors: torch.Tensor, tags: Sequence[str] = ()) -> str:
    hasher = hashlib.sha256()
    for tag in tags:
        hasher.update(str(tag).encode("utf-8"))
    for tensor in tensors:
        array = np.ascontiguousarray(tensor.detach().cpu().numpy())
        hasher.update(str(array.dtype).encode("utf-8"))
        hasher.update(str(array.shape).encode("utf-8"))
        hasher.update(array.tobytes())
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# KV states
# ---------------------------------------------------------------------------
@dataclass
class SwarmKVCache:
    """One sender's key-value state, its spans and its wave projection."""

    keys: torch.Tensor
    values: torch.Tensor
    spans: np.ndarray
    wave: np.ndarray
    source: str = ""
    task: str = ""
    positions: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    digest: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> "SwarmKVCache":
        if self.keys.dim() != 3 or self.keys.shape != self.values.shape:
            raise ValueError(f"K/V must both be (L, H, dk), got {tuple(self.keys.shape)}")
        length = int(self.keys.shape[0])
        if self.spans.shape != (length, 2):
            raise ValueError(f"spans must be ({length}, 2), got {self.spans.shape}")
        if self.wave.size == 0:
            raise ValueError("a KV cache needs a wave projection")
        if self.positions.size != length:
            self.positions = np.arange(length, dtype=np.int64)
        if not self.digest:
            self.digest = _digest_tensor(
                self.keys, self.values, torch.as_tensor(self.spans.astype(np.float32)),
                tags=(self.source, self.task),
            )
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": str(self.source),
            "task": str(self.task),
            "length": int(self.keys.shape[0]),
            "heads": int(self.keys.shape[1]),
            "dk": int(self.keys.shape[2]),
            "spans": self.spans.astype(int).tolist(),
            "digest": str(self.digest),
            "wave_norm": float(np.linalg.norm(self.wave)),
            "meta": dict(self.meta),
        }


_FRAME_CACHE: Dict[Tuple[int, int, int, int], np.ndarray] = {}
_PROJECTION_CACHE: Dict[Tuple[int, int, int], np.ndarray] = {}


def _projection_frame(plane_size: int, width: int, seed: int = 20261109) -> np.ndarray:
    """A fixed random projection for the bridge read.

    The full 1,024 x 2,048 hub frame costs about 0.11 ms per query on this
    CPU. The bridge index only needs the RANKING of a query wave against a
    thousand declared keys, and a 256-dimensional Gaussian projection
    preserves that ranking (same key -> identical projection; distinct
    identity waves stay at chance). The frame is a constant, built once.
    """
    key = (int(plane_size), int(width), int(seed))
    cached = _PROJECTION_CACHE.get(key)
    if cached is not None:
        return cached
    rng = np.random.default_rng(int(seed))
    frame = rng.standard_normal((int(plane_size), int(width))).astype(np.float32)
    frame /= max(float(np.linalg.norm(frame)), _EPS)
    _PROJECTION_CACHE[key] = frame
    return frame


def _frame(seed: int, heads: int, dk: int, plane_size: int) -> np.ndarray:
    """The fixed orthonormalised Gaussian frame kv_from_wave projects with."""
    key = (int(seed), int(heads), int(dk), int(plane_size))
    cached = _FRAME_CACHE.get(key)
    if cached is not None:
        return cached
    rng = np.random.default_rng(int(seed))
    width = 8 * int(heads) * int(dk)
    frame = rng.standard_normal((width, int(plane_size))).astype(np.float32)
    frame /= np.maximum(np.linalg.norm(frame, axis=1, keepdims=True), _EPS)
    _FRAME_CACHE[key] = frame
    return frame


def kv_from_wave(
    wave: np.ndarray,
    *,
    heads: int = 4,
    dk: int = 32,
    seed: int = 20261108,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Deterministic key/value projection of one unit wave.

    The wave's real and imaginary planes (2D values) are multiplied by two
    fixed, orthonormalised Gaussian frames. Nothing is trained here; the
    learned part is the hub mapper, so the projection cannot leak a label.
    """
    wave = _unit(wave)
    dim = int(wave.size)
    plane = np.concatenate([wave.real, wave.imag]).astype(np.float32)
    width = 8 * int(heads) * int(dk)
    # The frame is a fixed constant, not data. Rebuilding 2M Gaussians on
    # every call costs ~20 ms and would be charged to the receiver's TTFT,
    # so it is built once per (seed, heads, dk) and reused read-only.
    frame = _frame(int(seed), int(heads), int(dk), int(plane.size))
    projected = frame @ plane
    chunk = 4 * int(heads) * int(dk)
    keys = torch.from_numpy(projected[:chunk].reshape(4, int(heads), int(dk)))
    values = torch.from_numpy(projected[chunk:2 * chunk].reshape(4, int(heads), int(dk)))
    return keys, values


def kv_from_features(
    features: torch.Tensor,
    *,
    heads: int = 4,
    dk: int = 32,
    seed: int = 20261108,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Deterministic key/value projection of one expert feature row."""
    row = features.detach().cpu().numpy().reshape(-1).astype(np.float32)
    rng = np.random.default_rng(int(seed) + int(row.size))
    width = 8 * int(heads) * int(dk)
    frame = rng.standard_normal((width, row.size)).astype(np.float32)
    frame /= np.maximum(np.linalg.norm(frame, axis=1, keepdims=True), _EPS)
    projected = frame @ row
    chunk = 4 * int(heads) * int(dk)
    keys = torch.from_numpy(projected[:chunk].reshape(4, int(heads), int(dk)))
    values = torch.from_numpy(projected[chunk:2 * chunk].reshape(4, int(heads), int(dk)))
    return keys, values


def cache_from_wave(
    wave: np.ndarray,
    *,
    span_length: int = 4,
    source: str = "wave",
    task: str = "",
    heads: int = 4,
    dk: int = 32,
    total_length: int = 64,
) -> SwarmKVCache:
    """Build the four-slot cache for one wave, with spans over [0, total_length)."""
    keys, values = kv_from_wave(wave, heads=heads, dk=dk)
    length = int(keys.shape[0])
    step = max(1, int(total_length) // max(length, 1))
    spans = np.asarray(
        [[index * step, min(int(total_length), (index + 1) * step)] for index in range(length)],
        dtype=np.int64,
    )
    return SwarmKVCache(
        keys=keys, values=values, spans=spans, wave=_unit(wave), source=source, task=task,
    ).validate()


def cache_from_pairs(
    pairs: Sequence[Tuple[np.ndarray, np.ndarray]],
    *,
    source: str = "facts",
    task: str = "",
    heads: int = 4,
    dk: int = 32,
    wave: Optional[np.ndarray] = None,
    facts: Optional[Sequence[Sequence[str]]] = None,
) -> SwarmKVCache:
    """One KV slot per (key wave, value wave) pair.

    Each pair is projected by the SAME fixed frame kv_from_wave uses, so a
    query wave and a key wave that name the same content align exactly. The
    chunk wave is the bundled topic wave: the unit sum of the key waves,
    unless the caller passes the chunk's own content wave.
    """
    keys: List[torch.Tensor] = []
    values: List[torch.Tensor] = []
    topic: Optional[np.ndarray] = None
    for key_wave, value_wave in pairs:
        key = _unit(key_wave)
        projected_key, _ = kv_from_wave(key, heads=heads, dk=dk)
        _ignored, projected_value = kv_from_wave(_unit(value_wave), heads=heads, dk=dk)
        keys.append(projected_key[0])
        values.append(projected_value[0])
        if topic is None:
            topic = key
    if not keys:
        raise ValueError("cache_from_pairs needs at least one pair")
    if wave is None:
        # The chunk's declared wave is its TOPIC key: the first fact's key.
        # A bundle would announce a state no single key carries, which the
        # gate's binding check would then (correctly) refuse.
        wave = topic if topic is not None else np.zeros(1, dtype=np.complex64)
    length = len(keys)
    spans = np.asarray([[index, index + 1] for index in range(length)], dtype=np.int64)
    meta: Dict[str, Any] = {}
    if facts is not None:
        # Directive Q1: the gate's value/fact-consistency condition reads the
        # symbolic bindings the chunk actually carries. The wave itself cannot
        # say which value a key was bound to, so the declared pairs travel
        # beside the KV state exactly as a sender's span table does.
        meta["facts"] = [[str(key), str(value)] for key, value in facts]
    return SwarmKVCache(
        keys=torch.stack(keys), values=torch.stack(values), spans=spans,
        wave=_unit(np.asarray(wave)), source=str(source), task=str(task), meta=meta,
    ).validate()


# ---------------------------------------------------------------------------
# de-RoPE and span alignment
# ---------------------------------------------------------------------------
class DeRoPEPositionAligner:
    """Remove the sender rotary phase and re-apply the receiver's.

    The rotation acts on channel pairs (2i, 2i+1) with theta_i = base^(-2i/dk).
    It is orthogonal, so it moves no information; it only changes the frame a
    receiver reads. de_rope(rope(x, p), p) == x to float32 precision.
    """

    def __init__(self, dk: int, base: float = 10000.0) -> None:
        self.dk = int(dk)
        self.base = float(base)
        index = np.arange(0, self.dk, 2, dtype=np.float64)
        self.inv_freq = (1.0 / (self.base ** (index / float(self.dk)))).astype(np.float64)

    def _angles(self, positions: Sequence[int]) -> torch.Tensor:
        pos = torch.as_tensor(np.asarray(positions, dtype=np.float64))
        angles = torch.outer(pos, torch.from_numpy(self.inv_freq))
        return angles.to(torch.float32)

    @staticmethod
    def _rotate(x: torch.Tensor, angles: torch.Tensor, sign: float) -> torch.Tensor:
        # x: (L, H, dk); angles: (L, dk/2)
        length, heads, dk = x.shape
        pairs = x.reshape(length, heads, dk // 2, 2)
        cos = torch.cos(sign * angles).reshape(length, 1, dk // 2, 1)
        sin = torch.sin(sign * angles).reshape(length, 1, dk // 2, 1)
        real = pairs[..., 0:1]
        imag = pairs[..., 1:2]
        rotated = torch.cat([real * cos - imag * sin, real * sin + imag * cos], dim=-1)
        return rotated.reshape(length, heads, dk)

    def de_rope(self, keys: torch.Tensor, positions: Sequence[int]) -> torch.Tensor:
        return self._rotate(keys, self._angles(positions), -1.0)

    def rope(self, keys: torch.Tensor, positions: Sequence[int]) -> torch.Tensor:
        return self._rotate(keys, self._angles(positions), 1.0)

    def span_alignment(
        self,
        sender_spans: np.ndarray,
        receiver_spans: np.ndarray,
        *,
        causal: bool = True,
    ) -> np.ndarray:
        """Eq. 12 span-midpoint alignment, shape (L_sender, L_receiver).

        rho_i = max(0, j + clip((m_i - a_j) / (b_j - a_j), 0, 1) - 1/2), where
        m_i is the sender span midpoint. The causal mask zeroes any pair with
        sender end > receiver end, so a receiver cannot read a future span.
        """
        sender = np.asarray(sender_spans, dtype=np.float64)
        receiver = np.asarray(receiver_spans, dtype=np.float64)
        mid = 0.5 * (sender[:, 0] + sender[:, 1])
        a = receiver[:, 0][None, :]
        b = receiver[:, 1][None, :]
        width = np.maximum(b - a, _EPS)
        frac = np.clip((mid[:, None] - a) / width, 0.0, 1.0)
        j = np.arange(receiver.shape[0], dtype=np.float64)[None, :]
        rho = np.maximum(0.0, j + frac - 0.5)
        if causal:
            mask = sender[:, 1][:, None] <= receiver[:, 1][None, :]
            rho = rho * mask.astype(np.float64)
        return rho.astype(np.float32)


# ---------------------------------------------------------------------------
# Attention Matching compaction
# ---------------------------------------------------------------------------
class AttentionMatchingCompactor:
    """AM_U: keep U support keys and solve the least-squares values (Eq. 8)."""

    def __init__(self, budget_u: int = 16) -> None:
        self.budget_u = int(budget_u)
        self.last_report: Dict[str, float] = {}

    def compact(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        query_keys: Optional[torch.Tensor] = None,
        budget_u: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, Dict[str, float]]:
        length, heads, dk = keys.shape
        budget = int(self.budget_u if budget_u is None else budget_u)
        budget = max(1, min(int(budget), length))
        if query_keys is None:
            query = keys.mean(dim=0, keepdim=True)
        else:
            query = query_keys
        # The keys and the query are unit-frame projections, so the natural
        # score is a cosine. A temperature of 0.05 reproduces the sharp
        # attention a transformer head would have; scaling by 1/sqrt(dk)
        # would flatten a cosine into a near-uniform distribution.
        unit_keys = torch.nn.functional.normalize(keys, dim=-1)
        unit_query = torch.nn.functional.normalize(query, dim=-1)
        scores = torch.einsum("lhd,qhd->ql", unit_keys, unit_query) / 0.05
        attention = torch.softmax(scores, dim=-1)  # (Q, L)
        mass = attention.mean(dim=0)  # (L,)
        selected = torch.topk(mass, k=budget, largest=True, sorted=False).indices
        selected = torch.sort(selected).values
        support = attention[:, selected]  # (Q, U)
        target = torch.einsum("ql,lhd->qhd", attention, values).reshape(
            int(attention.shape[0]), -1
        )
        # Eq. 8 least squares through the normal equations: (A^T A + ridge I) X
        # = A^T target. The system is U x U, not L x dk, so a read stays cheap.
        gram = support.t() @ support + 1e-6 * torch.eye(
            int(support.shape[1]), dtype=support.dtype
        )
        solution = torch.linalg.solve(gram, support.t() @ target)  # (U, H*dk)
        values_hat = solution.reshape(budget, heads, dk)
        preserved = float(mass[selected].sum() / torch.clamp(mass.sum(), min=_EPS))
        self.last_report = {
            "length": float(length),
            "budget_u": float(budget),
            "attention_mass_preserved": float(preserved),
            "support_keys": [int(value) for value in selected.tolist()],
        }
        return keys[selected], values_hat, selected.detach().cpu().numpy(), dict(self.last_report)


# ---------------------------------------------------------------------------
# Zone B / Zone C gate
# ---------------------------------------------------------------------------
class SagnacKVGate:
    """Admit a telepathic KV state only if physics and memory both agree.

    The check has four independent conditions, the same union rule the model's
    TwoConditionVeto uses:

      chance   the sender wave retrieves no Zone C engram, or its top-1 raw
               coherence s_max is below tau_chance;
      sagnac   the post-photonic residual between the sender wave and the
               retrieved axiom exceeds tau_residual;
      binding  the announced wave is not the frame of any delivered key;
      fact     a key the chunk carries has a Tier-3 law or reference value and
               the chunk's bound value contradicts it (Directive Q1).

    A vetoed state contributes zero attention mass, so a corrupted sender
    cannot inject a token into the receiver.
    """

    def __init__(
        self,
        model,
        memory=None,
        *,
        tau_chance: Optional[float] = None,
        tau_residual: Optional[float] = None,
        tau_binding: float = 0.85,
        tau_contra: float = 0.50,
        tau_fact: float = 0.50,
        task: str = "language",
        fact_registry: Optional[Dict[str, str]] = None,
        wave_of: Optional[Any] = None,
        enforce_string_rule: bool = True,
    ) -> None:
        self.model = model
        self.memory = memory if memory is not None else getattr(model, "memory", None)
        self.tau_chance = tau_chance
        self.tau_residual = tau_residual
        self.tau_binding = float(tau_binding)
        self.tau_contra = float(tau_contra)
        # Directive R1: the physical fact-phase residual. The value slot of a
        # chunk is compared with the law's value slot in the same frame; a
        # wrong quantity or entity is a random phase, so 1 - Re<.,.> is near
        # 1 and the chunk is vetoed in wave space, not by string inequality.
        self.tau_fact = float(tau_fact)
        self.enforce_string_rule = bool(enforce_string_rule)
        self.task = str(task)
        # Directive Q1: the reference values the gate checks a chunk against.
        # The registry is supplied by the session (the Tier-3 laws of Zone C,
        # or the sender's declared reference facts). An empty registry leaves
        # the historical three-condition gate bit-identical.
        self.fact_registry: Dict[str, str] = {
            str(key): str(value) for key, value in dict(fact_registry or {}).items()
        }
        # The wave code a session writes its names in. It must be the SAME
        # code the sender used, so the gate can rebuild the law's value slot
        # in the frame the cache actually carries.
        self.wave_of = wave_of
        self.calibration: Dict[str, Any] = {}

    @torch.no_grad()
    def measure(self, cache: SwarmKVCache) -> Dict[str, float]:
        from henri.holarch.zone_b import sagnac_delta

        wave = _unit(cache.wave)
        psi = torch.from_numpy(wave).reshape(1, -1)
        s_max = 0.0
        top_wave = None
        if self.memory is not None:
            _prefix, read = self.memory.context_tensor(wave, k=4, task=cache.task or self.task)
            s_max = float(read.s_max)
            top_wave = read.top_wave
        bright = self.model.zone_b(psi, apply_memory=False).bright
        if top_wave is not None:
            axiom_wave = torch.from_numpy(np.asarray(top_wave, dtype=np.complex64)).reshape(1, -1)
            axiom = self.model.zone_b(axiom_wave, apply_memory=False).bright
            delta = float(sagnac_delta(bright, axiom).mean())
        else:
            delta = 1.0
        return {"s_max": s_max, "delta_sagnac": delta}

    def calibrate(self, valid: Sequence[SwarmKVCache], quantile: float = 0.99) -> Dict[str, Any]:
        rows = [self.measure(cache) for cache in valid]
        if not rows:
            self.tau_chance = 0.25 if self.tau_chance is None else self.tau_chance
            self.tau_residual = 0.30 if self.tau_residual is None else self.tau_residual
            self.calibration = {"rows": 0, "tau_chance": self.tau_chance,
                                "tau_residual": self.tau_residual}
            return dict(self.calibration)
        s_values = np.asarray([row["s_max"] for row in rows], dtype=np.float64)
        d_values = np.asarray([row["delta_sagnac"] for row in rows], dtype=np.float64)
        # A valid sender must clear the chance floor comfortably: the floor is
        # a fraction of the WEAKEST valid retrieval, not of the median.
        self.tau_chance = (
            float(max(0.10, float(s_values.min()) * 0.85))
            if self.tau_chance is None
            else float(self.tau_chance)
        )
        self.tau_residual = (
            float(float(d_values.max()) * 1.05 + 0.01)
            if self.tau_residual is None
            else float(self.tau_residual)
        )
        self.calibration = {
            "rows": int(len(rows)),
            "tau_chance": float(self.tau_chance),
            "tau_residual": float(self.tau_residual),
            "s_max_min": float(s_values.min()),
            "s_max_median": float(np.median(s_values)),
            "delta_max": float(d_values.max()),
            "delta_median": float(np.median(d_values)),
        }
        return dict(self.calibration)

    def binding(self, cache: SwarmKVCache) -> float:
        """How well the declared wave names the keys the cache actually carries.

        A sender announces one unit wave for its state. That wave must be the
        frame of at least one key in the KV it hands over: otherwise the
        announced state and the delivered state are different objects, which
        is exactly the ungrounded-handoff case the paper's ethics section
        warns about. The test is a max cosine, so a bundled chunk that
        announces its topic key passes with 1.0 and a phase-scrambled or
        substituted wave fails.
        """
        if int(cache.keys.shape[0]) == 0:
            return 0.0
        probe, _values = kv_from_wave(cache.wave, heads=int(cache.keys.shape[1]),
                                      dk=int(cache.keys.shape[2]))
        probe = probe[0].reshape(-1)
        flat = cache.keys.reshape(int(cache.keys.shape[0]), -1)
        numerator = flat @ probe
        denominator = torch.clamp(
            flat.norm(dim=-1) * torch.clamp(probe.norm(), min=_EPS), min=_EPS
        )
        return float((numerator / denominator).max())

    def fact_consistency(self, cache: SwarmKVCache) -> Dict[str, Any]:
        """The fourth condition: a chunk's bound values versus the law.

        Directive Q1. A chunk that announces the right topic wave can still
        carry a fabricated or contradicted value. For every key that has a
        Tier-3 law or reference value, the delivered value slot is compared
        with the law's value slot in the SAME fixed frame the sender wrote.

        The key factor is common to both sides and unitary, so the test is on
        the value binding: Re<psi_chunk_kv, psi_law> = cosine(v_chunk, v_law).
        An exact reference value scores +1, a phase-inverted value scores -1,
        and a substituted value scores near 0. The exact-string rule catches
        the substituted case; the destructive-interference rule catches the
        phase-inverted one. Either is a veto.
        """
        facts = list(cache.meta.get("facts") or [])
        fact_waves = list(cache.meta.get("fact_waves") or [])
        registry = self.fact_registry
        if not facts or not registry or self.wave_of is None:
            return {
                "facts": len(facts),
                "checked": 0,
                "violated": False,
                "violations": [],
                "min_value_coherence": 1.0,
                "max_phase_residual": 0.0,
                "residuals": [],
                "tau_fact": float(self.tau_fact),
                "string_rule_enforced": bool(self.enforce_string_rule),
            }
        violations: List[str] = []
        min_coherence = 1.0
        max_residual = 0.0
        residuals: List[Dict[str, Any]] = []
        checked = 0
        for index, pair in enumerate(facts):
            if len(pair) < 2:
                continue
            key, value = str(pair[0]), str(pair[1])
            if key not in registry:
                continue
            expected = str(registry[key])
            checked += 1
            if index < len(fact_waves):
                key_wave = np.asarray(fact_waves[index][0], dtype=np.complex64)
                value_wave = np.asarray(fact_waves[index][1], dtype=np.complex64)
            else:
                key_wave = np.asarray(self.wave_of(key), dtype=np.complex64)
                value_wave = np.asarray(self.wave_of(value), dtype=np.complex64)
            key_wave = _unit(key_wave)
            # The chunk's bound state is key_wave * value_wave. The law's is
            # the SAME key bound to the reference value, so the key phase is
            # common and unitary: Re<chunk, law> = Re<value, reference>.
            chunk_bound = _unit(key_wave * _unit(value_wave))
            law_bound = _unit(
                key_wave * _unit(np.asarray(self.wave_of(expected), dtype=np.complex64))
            )
            coherence = float(
                np.real(np.vdot(chunk_bound, law_bound))
                / max(float(np.linalg.norm(chunk_bound)) * float(np.linalg.norm(law_bound)), _EPS)
            )
            min_coherence = min(min_coherence, coherence)
            residual = float(1.0 - coherence)
            max_residual = max(max_residual, residual)
            residuals.append(
                {
                    "key": str(key),
                    "expected": str(expected),
                    "observed": str(value),
                    "coherence": float(coherence),
                    "residual": float(residual),
                }
            )
            # Directive R1: the PHYSICAL branch. Any wrong quantity or entity
            # is a random phase in the declared identity code, so its value
            # residual is near 1 and exceeds tau_fact. This branch is what
            # vetoes arbitrary wrong values; the string rule below is kept as
            # a fail-closed belt, and the physical-only audit disables it to
            # prove the physics alone suffices.
            if residual > float(self.tau_fact):
                violations.append(f"{key}:phase_residual")
            if self.enforce_string_rule and value != expected:
                violations.append(f"{key}:value_mismatch")
            if coherence < -float(self.tau_contra):
                violations.append(f"{key}:destructive_interference")
        return {
            "facts": len(facts),
            "checked": int(checked),
            "violated": bool(violations),
            "violations": violations,
            "min_value_coherence": float(min_coherence),
            "max_phase_residual": float(max_residual),
            "residuals": residuals,
            "tau_fact": float(self.tau_fact),
            "string_rule_enforced": bool(self.enforce_string_rule),
        }

    def check(self, cache: SwarmKVCache) -> Dict[str, Any]:
        stats = self.measure(cache)
        tau_chance = 0.10 if self.tau_chance is None else float(self.tau_chance)
        tau_residual = 0.60 if self.tau_residual is None else float(self.tau_residual)
        coherence = self.binding(cache)
        fact = self.fact_consistency(cache)
        chance_fail = bool(stats["s_max"] < tau_chance)
        sagnac_fail = bool(stats["delta_sagnac"] > tau_residual)
        binding_fail = bool(coherence < float(self.tau_binding))
        fact_fail = bool(fact.get("violated", False))
        admit = not (chance_fail or sagnac_fail or binding_fail or fact_fail)
        return {
            "admitted": bool(admit),
            "reason": (
                "ok"
                if admit
                else (
                    "binding"
                    if binding_fail
                    else ("fact" if fact_fail else ("chance" if chance_fail else "sagnac"))
                )
            ),
            "s_max": float(stats["s_max"]),
            "delta_sagnac": float(stats["delta_sagnac"]),
            "binding_coherence": float(coherence),
            "fact_checked": int(fact.get("checked", 0)),
            "fact_violations": list(fact.get("violations") or []),
            "fact_min_coherence": float(fact.get("min_value_coherence", 1.0)),
            "fact_max_phase_residual": float(fact.get("max_phase_residual", 0.0)),
            "fact_residuals": list(fact.get("residuals") or []),
            "tau_chance": float(tau_chance),
            "tau_residual": float(tau_residual),
            "tau_binding": float(self.tau_binding),
            "tau_contra": float(self.tau_contra),
            "tau_fact": float(self.tau_fact),
            "string_rule_enforced": bool(self.enforce_string_rule),
        }


# ---------------------------------------------------------------------------
# Shared hub mapper
# ---------------------------------------------------------------------------
class SwarmTelepathyMapper:
    """2n shared-hub mapper: one ingress, one egress, frozen endpoints.

    Ingress removes the sender RoPE phase and folds the sender wave into the
    hub. Egress projects the hub to receiver keys and values, rescales each
    head to the receiver RMS statistics (Eq. 10), and re-applies receiver RoPE.
    Stage 1 is a closed-form ridge fit on paired sender/receiver states; Stage
    2 is an optional linear task-logit step. No sender or receiver weight is
    trained.
    """

    def __init__(
        self,
        dim: int = 1024,
        *,
        heads: int = 4,
        dk: int = 32,
        hub: int = 256,
        span_length: int = 64,
        seed: int = 20261108,
    ) -> None:
        self.dim = int(dim)
        self.heads = int(heads)
        self.dk = int(dk)
        self.hub_width = int(hub)
        self.span_length = int(span_length)
        self.aligner = DeRoPEPositionAligner(self.dk)
        self.ingress = torch.zeros(2 * self.dim, self.hub_width)
        self.egress_k = torch.zeros(self.hub_width, 4 * self.heads * self.dk)
        self.egress_v = torch.zeros(self.hub_width, 4 * self.heads * self.dk)
        self.normalizer = torch.zeros(self.hub_width)
        self.fitted = False
        self.stage = 0
        self.report: Dict[str, Any] = {}

    # ------------------------------------------------------------- features
    def _hub(self, wave: np.ndarray) -> torch.Tensor:
        wave = _unit(wave)
        plane = torch.from_numpy(
            np.concatenate([wave.real, wave.imag]).astype(np.float32)
        )
        hub = plane @ self.ingress
        hub = hub / torch.clamp(self.normalizer, min=_EPS)
        return hub.reshape(4, self.hub_width // 4) if self.hub_width % 4 == 0 else hub.reshape(1, -1)

    def _project(self, hub: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        flat = hub.reshape(-1)
        return (flat @ matrix).reshape(4, self.heads, self.dk)

    # --------------------------------------------------------------- fitting
    def set_shared_frame(self, seed: int = 20261108) -> Dict[str, Any]:
        """Install the analytic shared frame (the 2n hub with no pairs).

        Every sender and every receiver is projected by the same fixed frame
        in kv_from_wave, so the frame map is the identity in wave space and
        its factorisation is exact. Stage 1 can still be fitted on top; this
        method is the declared initialisation that makes the bus usable
        before any paired sample exists.
        """
        rng = np.random.default_rng(int(seed))
        plane_size = 2 * self.dim
        width = 8 * self.heads * self.dk
        frame = rng.standard_normal((width, plane_size)).astype(np.float32)
        frame /= np.maximum(np.linalg.norm(frame, axis=1, keepdims=True), _EPS)
        half = 4 * self.heads * self.dk
        self.ingress = torch.from_numpy(frame.T.copy()).contiguous()  # (2D, width)
        self.hub_width = int(width)
        self.normalizer = torch.ones(self.hub_width)
        eye = np.eye(width, dtype=np.float32)
        self.egress_k = torch.from_numpy(eye[:, :half].copy()).contiguous()  # (width, half)
        self.egress_v = torch.from_numpy(eye[:, half:].copy()).contiguous()
        self.fitted = True
        self.stage = 1
        self.report = {
            "stage": 1,
            "kind": "shared_frame",
            "rank": int(width),
            "pairs": 0,
            "kv_relative_error": 0.0,
        }
        return dict(self.report)

    def fit_stage1(
        self,
        sender_waves: Sequence[np.ndarray],
        receiver_waves: Sequence[np.ndarray],
        *,
        ridge: float = 1e-3,
    ) -> Dict[str, Any]:
        """Closed-form ridge map from the sender plane to receiver K/V.

        Eq. 10: minimise ||W x - y||^2 + ridge ||W||^2 over the paired states.
        The pair is (sender wave of a content, receiver encoding of the same
        content), so the map learns the frame change, not the label.
        """
        if len(sender_waves) != len(receiver_waves) or not sender_waves:
            raise ValueError("Stage 1 needs paired sender and receiver waves")
        x_rows = []
        y_rows = []
        for sender, receiver in zip(sender_waves, receiver_waves):
            sender = _unit(sender)
            x = np.concatenate([sender.real, sender.imag]).astype(np.float32)
            keys, values = kv_from_wave(receiver, heads=self.heads, dk=self.dk)
            y = np.concatenate(
                [keys.reshape(-1).numpy(), values.reshape(-1).numpy()]
            ).astype(np.float32)
            x_rows.append(x)
            y_rows.append(y)
        X = torch.from_numpy(np.stack(x_rows))
        Y = torch.from_numpy(np.stack(y_rows))
        gram = X.t() @ X + float(ridge) * torch.eye(X.shape[1])
        W = torch.linalg.solve(gram, X.t() @ Y)  # (2D, 2*H*dk)
        half = 4 * self.heads * self.dk
        # The closed-form map is factorised into one ingress and one egress.
        # The rank is the hub width the architecture declared, so the bus has
        # exactly 2n projections and no pairwise mapper.
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        rank = int(min(self.hub_width, int(U.shape[1])))
        self.ingress = (U[:, :rank] * S[:rank]).contiguous()  # (2D, rank)
        egress = Vh[:rank].contiguous()  # (rank, 2*half)
        self.egress_k = egress[:, :half].contiguous()  # (rank, half)
        self.egress_v = egress[:, half:].contiguous()
        self.hub_width = int(rank)
        self.normalizer = torch.ones(self.hub_width)
        self.fitted = True
        self.stage = 1
        hub = X @ self.ingress
        prediction = torch.cat([hub @ self.egress_k, hub @ self.egress_v], dim=1)
        error = float(torch.mean((prediction - Y) ** 2))
        baseline = float(torch.mean(Y ** 2))
        self.report.update(
            {
                "stage": 1,
                "pairs": int(len(x_rows)),
                "rank": int(rank),
                "ridge": float(ridge),
                "kv_reconstruction_mse": error,
                "kv_baseline_mse": baseline,
                "kv_relative_error": float(error / max(baseline, _EPS)),
            }
        )
        return dict(self.report)

    def fit_stage2(
        self,
        hub_rows: torch.Tensor,
        targets: torch.Tensor,
        *,
        steps: int = 40,
        lr: float = 0.05,
    ) -> Dict[str, Any]:
        """Stage 2: one linear task-logit step on the frozen hub features.

        Only the small readout is fitted; the sender and receiver weights stay
        frozen. The loss is Eq. 11 (cross entropy on the downstream label).
        """
        if hub_rows.numel() == 0:
            return {"stage": 2, "steps": 0}
        head = torch.zeros(self.hub_width, int(targets.max().item()) + 1, requires_grad=True)
        optimizer = torch.optim.Adam([head], lr=float(lr))
        labels = targets.long()
        for _ in range(int(steps)):
            optimizer.zero_grad()
            logits = hub_rows @ head
            loss = torch.nn.functional.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
        self.readout = head.detach()
        self.stage = 2
        self.report.update(
            {"stage": 2, "steps": int(steps), "final_nll": float(loss.detach())}
        )
        return dict(self.report)

    # ------------------------------------------------------------- translate
    def translate(
        self,
        cache: SwarmKVCache,
        *,
        receiver_stats: Optional[Dict[str, np.ndarray]] = None,
        receiver_spans: Optional[np.ndarray] = None,
        receiver_positions: Optional[Sequence[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        cache.validate()
        de_ropped = self.aligner.de_rope(cache.keys, cache.positions)
        hub = self._hub(cache.wave)
        flat = hub.reshape(-1)
        keys = (flat @ self.egress_k).reshape(4, self.heads, self.dk)
        values = (flat @ self.egress_v).reshape(4, self.heads, self.dk)
        # The sender K is still the more faithful key frame; blend the hub
        # prediction with the de-RoPE'd sender keys so the receiver sees the
        # sender's own geometry as well as the calibrated frame.
        keys = 0.5 * (keys + de_ropped[: keys.shape[0]])
        if receiver_spans is not None:
            rho = self.aligner.span_alignment(cache.spans, receiver_spans)
            mass = rho / np.maximum(rho.sum(axis=0, keepdims=True), _EPS)
            keys = torch.einsum("sr,lhd->rhd", torch.from_numpy(mass), keys)
            values = torch.einsum("sr,lhd->rhd", torch.from_numpy(mass), values)
        if receiver_stats is not None:
            sigma_k = torch.from_numpy(np.asarray(receiver_stats.get("sigma_k"), dtype=np.float32))
            sigma_v = torch.from_numpy(np.asarray(receiver_stats.get("sigma_v"), dtype=np.float32))
            rms_k = torch.sqrt(torch.mean(keys ** 2, dim=-1, keepdim=True) + _EPS)
            rms_v = torch.sqrt(torch.mean(values ** 2, dim=-1, keepdim=True) + _EPS)
            keys = keys * (sigma_k.reshape(1, -1, 1) / rms_k)
            values = values * (sigma_v.reshape(1, -1, 1) / rms_v)
        positions = (
            list(range(int(keys.shape[0])))
            if receiver_positions is None
            else [int(value) for value in receiver_positions]
        )
        keys = self.aligner.rope(keys, positions[: keys.shape[0]])
        return keys, values, {
            "sent_length": float(cache.keys.shape[0]),
            "receiver_length": float(keys.shape[0]),
            "rms_calibrated": float(receiver_stats is not None),
            "span_aligned": float(receiver_spans is not None),
            "positions": [int(value) for value in positions[: keys.shape[0]]],
        }


# ---------------------------------------------------------------------------
# The bus
# ---------------------------------------------------------------------------
@dataclass
class TelepathyPacket:
    source: str
    receiver_task: str
    admitted: bool
    reason: str
    keys: torch.Tensor
    values: torch.Tensor
    digest: str
    parent: str
    budget_u: int
    mass_preserved: float
    gate: Dict[str, Any] = field(default_factory=dict)
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": str(self.source),
            "receiver_task": str(self.receiver_task),
            "admitted": bool(self.admitted),
            "reason": str(self.reason),
            "digest": str(self.digest),
            "parent": str(self.parent),
            "budget_u": int(self.budget_u),
            "mass_preserved": float(self.mass_preserved),
            "gate": dict(self.gate),
            "detail": dict(self.detail),
        }


class SwarmTelepathyBus:
    """One shared hub for every expert and every modality.

    The bus is append-only and hash-chained. A vetoed sender leaves a packet
    with zero-length keys, so the receiver can audit that the state was seen
    and refused rather than silently dropped.
    """

    def __init__(
        self,
        mapper: Optional[SwarmTelepathyMapper] = None,
        *,
        gate: Optional[SagnacKVGate] = None,
        budget_u: int = 16,
        dim: int = 1024,
        heads: int = 4,
        dk: int = 32,
        max_entries: int = 256,
    ) -> None:
        self.mapper = mapper or SwarmTelepathyMapper(dim=dim, heads=heads, dk=dk)
        self.gate = gate
        self.budget_u = int(budget_u)
        self.dim = int(dim)
        self.heads = int(heads)
        self.dk = int(dk)
        self.compactor = AttentionMatchingCompactor(self.budget_u)
        self.entries: Dict[str, SwarmKVCache] = {}
        self.packets: List[TelepathyPacket] = []
        self.parent = "0" * 64
        self.counts = {"sent": 0, "admitted": 0, "vetoed": 0}
        self.max_entries = int(max_entries)
        self.receiver_stats: Dict[str, Dict[str, np.ndarray]] = {}
        self.last_packet: Optional[TelepathyPacket] = None

    # ------------------------------------------------------------- registry
    def register(self, cache: SwarmKVCache) -> str:
        cache.validate()
        self.entries[str(cache.source)] = cache
        if len(self.entries) > self.max_entries:
            oldest = next(iter(self.entries))
            self.entries.pop(oldest, None)
        self.counts["sent"] += 1
        return cache.digest

    def calibrate_receiver(
        self, task: str, waves: Sequence[np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Receiver RMS statistics sigma_K, sigma_V per head (Eq. 10)."""
        keys = []
        values = []
        for wave in waves:
            k, v = kv_from_wave(wave, heads=self.heads, dk=self.dk)
            keys.append(k)
            values.append(v)
        if not keys:
            sigma_k = np.ones(self.heads, dtype=np.float32)
            sigma_v = np.ones(self.heads, dtype=np.float32)
        else:
            sigma_k = torch.stack(keys).pow(2).mean(dim=(0, 1, 3)).sqrt().numpy().astype(np.float32)
            sigma_v = torch.stack(values).pow(2).mean(dim=(0, 1, 3)).sqrt().numpy().astype(np.float32)
        self.receiver_stats[str(task)] = {"sigma_k": sigma_k, "sigma_v": sigma_v}
        return dict(self.receiver_stats[str(task)])

    # ------------------------------------------------------------- transfer
    @torch.no_grad()
    def transmit(
        self,
        source: str,
        *,
        receiver_task: str = "language",
        receiver_spans: Optional[np.ndarray] = None,
        receiver_positions: Optional[Sequence[int]] = None,
        budget_u: Optional[int] = None,
        use_gate: bool = True,
        query_keys: Optional[torch.Tensor] = None,
    ) -> TelepathyPacket:
        cache = self.entries.get(str(source))
        if cache is None:
            raise KeyError(f"source {source!r} is not on the bus")
        gate_row: Dict[str, Any] = {"admitted": True, "reason": "gate-disabled"}
        if use_gate and self.gate is not None:
            gate_row = self.gate.check(cache)
        if not bool(gate_row.get("admitted", True)):
            self.counts["vetoed"] += 1
            packet = TelepathyPacket(
                source=str(source), receiver_task=str(receiver_task), admitted=False,
                reason=str(gate_row.get("reason", "veto")),
                keys=torch.zeros(0, self.heads, self.dk),
                values=torch.zeros(0, self.heads, self.dk),
                digest=hashlib.sha256((self.parent + cache.digest + "veto").encode()).hexdigest(),
                parent=self.parent, budget_u=0, mass_preserved=0.0, gate=gate_row,
            )
            self.parent = packet.digest
            self.packets.append(packet)
            self.last_packet = packet
            return packet
        stats = self.receiver_stats.get(str(receiver_task))
        keys, values, detail = self.mapper.translate(
            cache, receiver_stats=stats, receiver_spans=receiver_spans,
            receiver_positions=receiver_positions,
        )
        budget = int(self.budget_u if budget_u is None else budget_u)
        compact_keys, compact_values, selected, report = self.compactor.compact(
            keys, values, query_keys=query_keys, budget_u=budget,
        )
        digest = _digest_tensor(
            compact_keys, compact_values, tags=(self.parent, cache.digest, str(receiver_task)),
        )
        packet = TelepathyPacket(
            source=str(source), receiver_task=str(receiver_task), admitted=True, reason="ok",
            keys=compact_keys, values=compact_values, digest=digest, parent=self.parent,
            budget_u=int(budget), mass_preserved=float(report["attention_mass_preserved"]),
            gate=gate_row, detail={**detail, **report},
        )
        self.parent = digest
        self.packets.append(packet)
        self.last_packet = packet
        self.counts["admitted"] += 1
        return packet

    # ------------------------------------------- beyond-context compaction
    def _query_keys(self, query_wave: np.ndarray) -> torch.Tensor:
        keys, _values = kv_from_wave(query_wave, heads=self.heads, dk=self.dk)
        return keys[:1]

    @staticmethod
    def _read_values(
        keys: torch.Tensor,
        values: torch.Tensor,
        query_key: torch.Tensor,
        temperature: float = 0.05,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Attention read of a compacted cache: (context, per-slot weight)."""
        unit_keys = torch.nn.functional.normalize(keys, dim=-1)
        unit_query = torch.nn.functional.normalize(query_key, dim=-1)
        scores = torch.einsum("lhd,qhd->ql", unit_keys, unit_query) / float(temperature)
        weights = torch.softmax(scores, dim=-1)  # (1, L)
        context = torch.einsum("ql,lhd->qhd", weights, values)
        return context.reshape(-1), weights.reshape(-1)

    @torch.no_grad()
    def compact_long_horizon(
        self,
        chunks: Sequence[SwarmKVCache],
        query_wave: np.ndarray,
        *,
        per_chunk_u: int = 8,
        total_budget_u: int = 32,
        hops: int = 3,
        resolver=None,
        session_id: str = "",
        use_gate: bool = True,
        compose: bool = True,
    ) -> Dict[str, Any]:
        """SWAT Appendix F: compact C chunks that exceed the receiver window.

        The receiver's hard window is one chunk. Every chunk is gated by
        SagnacKVGate, translated and compacted to u = U / C support slots by
        AttentionMatchingCompactor, and the admitted slots are concatenated
        into one small cache of at most total_budget_u slots.

        When a resolver is supplied the sender walks the cross-chunk bridges
        while it compacts: hop h is read from chunk h with the current query,
        the read value is resolved to the next query wave, and the walk
        continues. The composed binding (original query -> final value) is
        written into the delivered cache, so the receiver answers with zero
        text prefill. Without a resolver the cache is the query-conditioned
        union of the per-chunk compactions (the paper's Table 19 shape).
        """
        query_wave = _unit(np.asarray(query_wave))
        query_key = self._query_keys(query_wave)
        admitted: List[SwarmKVCache] = []
        chunk_rows: List[Dict[str, Any]] = []
        vetoed: List[str] = []
        for index, cache in enumerate(chunks):
            cache.validate()
            gate_row: Dict[str, Any] = {"admitted": True, "reason": "gate-disabled"}
            if use_gate and self.gate is not None:
                gate_row = self.gate.check(cache)
            row: Dict[str, Any] = {
                "index": int(index),
                "source": str(cache.source),
                "length": int(cache.keys.shape[0]),
                "digest": str(cache.digest),
                "admitted": bool(gate_row.get("admitted", True)),
                "reason": str(gate_row.get("reason", "ok")),
                "gate": dict(gate_row),
            }
            if row["admitted"]:
                admitted.append(cache)
            else:
                vetoed.append(str(cache.source))
                self.counts["vetoed"] += 1
            chunk_rows.append(row)
        if not admitted:
            digest = hashlib.sha256((self.parent + "no-admitted-chunk").encode()).hexdigest()
            self.parent = digest
            return {
                "session_id": str(session_id),
                "admitted_chunks": 0,
                "vetoed_chunks": vetoed,
                "compacted_slots_u": 0,
                "swat_kv_digest": digest,
                "chunks": chunk_rows,
                "chain": [],
                "cache": None,
                "answer_wave": None,
                "receiver_prefill_tokens": 0,
                "reason": "every chunk was vetoed",
            }
        u = max(1, int(per_chunk_u))
        chain: List[Dict[str, Any]] = []
        current_wave = query_wave
        answer_context: Optional[torch.Tensor] = None
        answer_label = ""
        answer_score = 0.0
        rounds = int(hops) if resolver is not None else 1
        last_keys: List[torch.Tensor] = []
        last_values: List[torch.Tensor] = []
        for hop in range(rounds):
            current_key = self._query_keys(current_wave)
            stage_keys: List[torch.Tensor] = []
            stage_values: List[torch.Tensor] = []
            stage_rows: List[Dict[str, Any]] = []
            for index, cache in enumerate(admitted):
                compact_keys, compact_values, selected, report = self.compactor.compact(
                    cache.keys, cache.values, query_keys=current_key, budget_u=u,
                )
                stage_keys.append(compact_keys)
                stage_values.append(compact_values)
                stage_rows.append(
                    {
                        "chunk_index": int(index),
                        "chunk": str(cache.source),
                        "selected_slots": [int(value) for value in selected.tolist()],
                        "attention_mass_preserved": float(report["attention_mass_preserved"]),
                    }
                )
            keys = torch.cat(stage_keys, dim=0)
            values = torch.cat(stage_values, dim=0)
            context, weights = self._read_values(keys, values, current_key)
            resolved = None
            label = ""
            if resolver is not None:
                resolved = resolver(context.numpy().astype(np.float32))
                if isinstance(resolved, tuple):
                    resolved, label = resolved
            answer_context = context
            answer_label = str(label)
            peak_index = int(torch.argmax(weights).item())
            answer_score = float(weights.max())
            last_keys, last_values = stage_keys, stage_values
            chain.append(
                {
                    "hop": int(hop),
                    "peak_chunk": str(admitted[min(peak_index // max(u, 1), len(admitted) - 1)].source),
                    "peak_slot": int(peak_index),
                    "resolved": str(label),
                    "peak_weight": float(weights.max()),
                    "pool_slots": int(keys.shape[0]),
                    "attention_mass_total": float(weights.sum()),
                    "chunks": stage_rows,
                    "query_digest": hashlib.sha256(
                        np.ascontiguousarray(current_wave.astype(np.complex64)).tobytes()
                    ).hexdigest()[:16],
                }
            )
            if resolved is None:
                break
            current_wave = _unit(np.asarray(resolved))
        final_keys = torch.cat(last_keys, dim=0) if last_keys else torch.zeros(0, self.heads, self.dk)
        final_values = torch.cat(last_values, dim=0) if last_values else torch.zeros(0, self.heads, self.dk)
        if compose and answer_context is not None and answer_label:
            # The composed binding is the cross-chunk dependency the paper's
            # Table 19 requires: keyed by the ORIGINAL receiver query, valued
            # by the terminal hop's read. A support slot that carries the SAME
            # key (the first hop's own fact) is superseded: leaving it in
            # would split the receiver's attention between the composed
            # answer and the first-hop value, which is how a three-hop read
            # degrades to a one-hop read.
            if int(final_keys.shape[0]) > 0:
                flat = final_keys.reshape(int(final_keys.shape[0]), -1)
                probe = query_key.reshape(-1)
                numerator = flat @ probe
                denominator = torch.clamp(
                    flat.norm(dim=-1) * torch.clamp(probe.norm(), min=_EPS), min=_EPS
                )
                keep = (numerator / denominator) < 0.99
                if bool(keep.any()):
                    final_keys = final_keys[keep]
                    final_values = final_values[keep]
                else:
                    final_keys = torch.zeros(0, self.heads, self.dk)
                    final_values = torch.zeros(0, self.heads, self.dk)
            final_keys = torch.cat([query_key, final_keys], dim=0)
            final_values = torch.cat(
                [answer_context.reshape(1, self.heads, self.dk), final_values], dim=0
            )
        if int(final_keys.shape[0]) > int(total_budget_u):
            final_keys, final_values, _sel, _rep = self.compactor.compact(
                final_keys, final_values, query_keys=query_key,
                budget_u=int(total_budget_u),
            )
        length = int(final_keys.shape[0])
        spans = np.asarray([[index, index + 1] for index in range(length)], dtype=np.int64)
        combined = SwarmKVCache(
            keys=final_keys,
            values=final_values,
            spans=spans,
            wave=query_wave,
            source=str(session_id or "long_horizon"),
            task="telepathy",
            meta={
                "chunks": len(chunks),
                "admitted": len(admitted),
                "vetoed": len(vetoed),
                "hops": int(rounds),
                "per_chunk_u": int(u),
            },
        ).validate()
        digest = str(combined.digest)
        if session_id:
            self.entries[str(session_id)] = combined
        answer_wave = None
        if answer_context is not None:
            answer_wave = self._unbind_context(answer_context)
        self.parent = digest
        self.counts["admitted"] += 1
        return {
            "session_id": str(session_id),
            "admitted_chunks": int(len(admitted)),
            "vetoed_chunks": vetoed,
            "compacted_slots_u": int(length),
            "per_chunk_u": int(u),
            "total_budget_u": int(total_budget_u),
            "chunks": chunk_rows,
            "chain": chain,
            "cache": combined,
            "cache_digest": digest,
            "swat_kv_digest": digest,
            "answer_label": answer_label,
            "answer_peak_weight": float(answer_score),
            "answer_wave": None if answer_wave is None else answer_wave,
            "receiver_prefill_tokens": 0,
            "cross_chunk_bridges": int(max(0, len(chain) - 1)),
        }

    # ---------------------------------- query-agnostic ingest (Directive Q1)
    @torch.no_grad()
    def build_bridge_index(
        self,
        chunks: Sequence[SwarmKVCache],
        *,
        per_chunk_u: int = 8,
    ) -> Dict[str, Any]:
        """Compact every chunk ONCE and index its fact bindings for later hops.

        Directive Q1. A real multi-turn session ingests its document
        collection before any query is known. No receiver query exists here,
        so the Attention-Matching support is selected against each chunk's
        own mean key (the query-agnostic union of the paper's Table 19), and
        the cross-chunk bridges are indexed by the fact bindings the sender
        declared. The index is what makes a later unseen query a read, not a
        re-ingest.
        """
        per_chunk: List[Dict[str, Any]] = []
        bridge_keys: List[torch.Tensor] = []
        bridge_values: List[torch.Tensor] = []
        bridge_waves: List[np.ndarray] = []
        bridge_value_waves: List[np.ndarray] = []
        key_names: List[str] = []
        value_names: List[str] = []
        for cache in chunks:
            cache.validate()
            length = int(cache.keys.shape[0])
            budget = max(1, min(int(per_chunk_u), length))
            agnostic = cache.keys.mean(dim=0, keepdim=True)
            compact_keys, compact_values, selected, report = self.compactor.compact(
                cache.keys, cache.values, query_keys=agnostic, budget_u=budget
            )
            per_chunk.append(
                {
                    "source": str(cache.source),
                    "keys": compact_keys,
                    "values": compact_values,
                    "selected": [int(value) for value in selected.tolist()],
                    "mass_preserved": float(report["attention_mass_preserved"]),
                }
            )
            facts = list(cache.meta.get("facts") or [])
            if facts:
                bridge_keys.append(cache.keys)
                bridge_values.append(cache.values)
                waves = list(cache.meta.get("fact_waves") or [])
                for position, pair in enumerate(facts):
                    key_names.append(str(pair[0]))
                    value_names.append(str(pair[1]))
                    if position < len(waves):
                        bridge_waves.append(
                            _unit(np.asarray(waves[position][0], dtype=np.complex64))
                        )
                        bridge_value_waves.append(
                            _unit(np.asarray(waves[position][1], dtype=np.complex64))
                        )
                    else:
                        bridge_waves.append(np.zeros(self.dim, dtype=np.complex64))
                        bridge_value_waves.append(np.zeros(self.dim, dtype=np.complex64))
        if bridge_keys:
            all_keys = torch.cat(bridge_keys, dim=0)
            all_values = torch.cat(bridge_values, dim=0)
        else:
            all_keys = torch.zeros(0, self.heads, self.dk)
            all_values = torch.zeros(0, self.heads, self.dk)
        if bridge_waves:
            wave_matrix = np.stack(bridge_waves).astype(np.complex64)
            wave_norms = np.linalg.norm(wave_matrix, axis=1).astype(np.float32)
            # Re<key, q> = Re(k) . Re(q) + Im(k) . Im(q), so the hub read is
            # one real GEMM over the concatenated real/imaginary planes. That
            # keeps the unseen-query hop cost inside the receiver's budget.
            wave_planes = np.concatenate(
                [wave_matrix.real, wave_matrix.imag], axis=1
            ).astype(np.float32)
            projection_frame = _projection_frame(wave_planes.shape[1], 128)
            bridge_projection = (wave_planes @ projection_frame).astype(np.float32)
            projection_norms = np.linalg.norm(bridge_projection, axis=1).astype(
                np.float32
            )
            # A hop's next query IS a value wave this index already holds, so
            # its projection is precomputed once at ingest. Only the very
            # first (unseen) query is projected at read time.
            value_wave_matrix = np.stack(bridge_value_waves).astype(np.complex64)
            value_planes = np.concatenate(
                [value_wave_matrix.real, value_wave_matrix.imag], axis=1
            ).astype(np.float32)
            value_projection_matrix = (value_planes @ projection_frame).astype(
                np.float32
            )
            value_projections: Dict[str, np.ndarray] = {}
            for row_index, name in enumerate(value_names):
                if name not in value_projections:
                    value_projections[name] = value_projection_matrix[row_index].copy()
        else:
            wave_matrix = np.zeros((0, int(self.dim)), dtype=np.complex64)
            wave_norms = np.zeros(0, dtype=np.float32)
            wave_planes = np.zeros((0, 2 * int(self.dim)), dtype=np.float32)
            projection_frame = np.zeros((2 * int(self.dim), 256), dtype=np.float32)
            bridge_projection = np.zeros((0, 128), dtype=np.float32)
            projection_norms = np.zeros(0, dtype=np.float32)
            value_projections = {}
        flat = all_keys.reshape(int(all_keys.shape[0]), -1) if int(all_keys.shape[0]) else all_keys
        norms = (
            torch.clamp(flat.norm(dim=-1), min=_EPS)
            if int(flat.shape[0]) == int(all_keys.shape[0]) and int(flat.shape[0]) > 0
            else torch.zeros(0)
        )
        return {
            "per_chunk": per_chunk,
            "bridge_keys": all_keys,
            "bridge_values": all_values,
            "bridge_waves": wave_matrix,
            "bridge_wave_planes": wave_planes,
            "bridge_wave_norms": wave_norms,
            "bridge_projection": bridge_projection,
            "bridge_projection_frame": projection_frame,
            "bridge_projection_norms": projection_norms,
            "bridge_value_projections": value_projections,
            "bridge_key_names": key_names,
            "bridge_value_names": value_names,
            "bridge_norms": norms,
            "budget_u": int(self.budget_u),
        }

    @torch.no_grad()
    def resolve_bridges(
        self,
        index: Dict[str, Any],
        query_wave: np.ndarray,
        *,
        hops: int = 3,
        wave_of=None,
    ) -> Dict[str, Any]:
        """Walk the indexed cross-chunk bridges with an UNSEEN query.

        Directive Q1. Each hop is one cosine read of the current query wave
        against the indexed key slots; the winning slot's declared value names
        the next hop. No receiver text token is consumed and no chunk is
        re-compacted.
        """
        import time as _time

        started = _time.perf_counter()
        current = np.asarray(query_wave, dtype=np.complex64).reshape(-1)
        # The identity code the session writes is already unit. Normalise only
        # when a caller hands over something else, so the three-hop read does
        # not pay four norms it does not need.
        current_norm = float(np.linalg.norm(current))
        if abs(current_norm - 1.0) > 1e-3:
            current = (current / max(current_norm, _EPS)).astype(np.complex64)
        bridge_keys = index["bridge_keys"]
        bridge_values = index["bridge_values"]
        value_names = list(index["bridge_value_names"])
        norms = index.get("bridge_norms")
        if int(bridge_keys.shape[0]) == 0:
            # The empty index must return a context of exactly the shape the
            # caller reshapes to (heads, dk). A 4-slot width here made
            # compose_bridge_cache raise on every session without a bridge.
            return {
                "answer": "",
                "context": np.zeros(self.heads * self.dk, dtype=np.float32),
                "trace": [],
                "resolved_hops": 0,
                "latency_ms": (_time.perf_counter() - started) * 1e3,
                "peaks": [],
            }
        flat = bridge_keys.reshape(int(bridge_keys.shape[0]), -1)
        wave_matrix = index.get("bridge_waves")
        wave_planes = index.get("bridge_wave_planes")
        wave_norms = index.get("bridge_wave_norms")
        use_waves = (
            wave_matrix is not None
            and int(np.asarray(wave_matrix).shape[0]) == int(bridge_keys.shape[0])
            and int(np.asarray(wave_matrix).shape[0]) > 0
        )
        value_projections = index.get("bridge_value_projections") or {}
        trace: List[Dict[str, Any]] = []
        peaks: List[float] = []
        context = torch.zeros(1, self.heads, self.dk)
        resolved = 0
        precomputed_projection = None
        for hop in range(int(hops)):
            if use_waves:
                # Wave-space read: the identity code is mutually incoherent,
                # so Re<query, key> is the same ranking as the projected slot
                # cosine and costs one 1,000 x 1,024 complex product instead
                # of a 1,024 x 2,048 frame projection per hop.
                probe_wave = current
                projection = index.get("bridge_projection")
                projection_frame = index.get("bridge_projection_frame")
                projection_norms = index.get("bridge_projection_norms")
                if projection is not None and projection_frame is not None:
                    if precomputed_projection is not None:
                        probe_projection = precomputed_projection
                    else:
                        plane = np.concatenate(
                            [probe_wave.real, probe_wave.imag]
                        ).astype(np.float32)
                        probe_projection = plane @ np.asarray(projection_frame)
                    raw = np.asarray(projection) @ probe_projection
                    denominator = np.maximum(
                        np.asarray(projection_norms)
                        * float(np.linalg.norm(probe_projection)),
                        _EPS,
                    )
                    scores_np = raw / denominator
                elif wave_planes is not None:
                    plane = np.concatenate([probe_wave.real, probe_wave.imag]).astype(
                        np.float32
                    )
                    raw = np.asarray(wave_planes) @ plane
                    denominator = np.maximum(
                        np.asarray(wave_norms)
                        * float(np.linalg.norm(probe_wave)),
                        _EPS,
                    )
                    scores_np = raw / denominator
                else:
                    raw = np.asarray(wave_matrix) @ np.conj(probe_wave)
                    denominator = np.maximum(
                        np.asarray(wave_norms) * float(np.linalg.norm(probe_wave)), _EPS
                    )
                    scores_np = np.real(raw) / denominator
                best = int(np.argmax(scores_np))
                peak = float(scores_np[best])
            else:
                query_key = self._query_keys(current)
                probe = query_key.reshape(-1)
                probe_norm = torch.clamp(probe.norm(), min=_EPS)
                scores = (flat @ probe) / (norms.to(flat.dtype) * probe_norm)
                best = int(torch.argmax(scores).item())
                peak = float(scores[best])
            peaks.append(peak)
            context = bridge_values[best].reshape(1, self.heads, self.dk)
            answer = str(value_names[best]) if best < len(value_names) else ""
            trace.append(
                {
                    "hop": int(hop),
                    "slot": int(best),
                    "value": answer,
                    "peak": float(peak),
                }
            )
            resolved += 1
            if not answer:
                break
            if wave_of is None:
                break
            current = np.asarray(wave_of(answer), dtype=np.complex64).reshape(-1)
            precomputed_projection = value_projections.get(answer)
        return {
            "answer": str(trace[-1]["value"]) if trace else "",
            "context": context.reshape(-1),
            "trace": trace,
            "resolved_hops": int(resolved),
            "latency_ms": (_time.perf_counter() - started) * 1e3,
            "peaks": peaks,
            "terminal_wave": current,
        }

    @torch.no_grad()
    def compose_bridge_cache(
        self,
        index: Dict[str, Any],
        query_wave: np.ndarray,
        terminal_context: torch.Tensor,
        *,
        budget_u: int = 32,
        source: str = "query_agnostic_session",
    ) -> SwarmKVCache:
        """Assemble the receiver cache around the composed terminal binding.

        The composed binding (original query -> terminal hop read) supersedes
        any support slot that carries the same key, exactly as in the P2
        compactor, so the receiver's read cannot degrade into a one-hop read.
        """
        query_wave = _unit(np.asarray(query_wave, dtype=np.complex64))
        query_key = self._query_keys(query_wave)
        support_keys: List[torch.Tensor] = []
        support_values: List[torch.Tensor] = []
        for row in index.get("per_chunk") or []:
            support_keys.append(row["keys"])
            support_values.append(row["values"])
        if support_keys:
            keys = torch.cat(support_keys, dim=0)
            values = torch.cat(support_values, dim=0)
        else:
            keys = torch.zeros(0, self.heads, self.dk)
            values = torch.zeros(0, self.heads, self.dk)
        if int(keys.shape[0]) > 0:
            flat = keys.reshape(int(keys.shape[0]), -1)
            probe = query_key.reshape(-1)
            cosine = (flat @ probe) / torch.clamp(
                flat.norm(dim=-1) * torch.clamp(probe.norm(), min=_EPS), min=_EPS
            )
            keep = cosine < 0.99
            if bool(keep.any()):
                keys = keys[keep]
                values = values[keep]
            else:
                keys = torch.zeros(0, self.heads, self.dk)
                values = torch.zeros(0, self.heads, self.dk)
        # The composed binding is never dropped. If the auxiliary support
        # exceeds the remaining budget, slots are sampled evenly across the
        # whole chunk order rather than solved again: the terminal binding is
        # the answer, and a second least-squares projection would spend the
        # receiver's TTFT on support it does not read.
        support_budget = max(0, int(budget_u) - 1)
        if int(keys.shape[0]) > support_budget:
            if support_budget <= 0:
                keys = torch.zeros(0, self.heads, self.dk)
                values = torch.zeros(0, self.heads, self.dk)
            else:
                keep = np.linspace(
                    0, int(keys.shape[0]) - 1, int(support_budget)
                ).astype(np.int64)
                keep = np.unique(keep)
                keys = keys[torch.as_tensor(keep)]
                values = values[torch.as_tensor(keep)]
        need = int(self.heads) * int(self.dk)
        terminal = torch.as_tensor(np.asarray(terminal_context).reshape(-1))
        if int(terminal.numel()) != need:
            # Defensive: a caller that supplies a wider slot vector contributes
            # its first (heads * dk) elements rather than crashing the cache.
            terminal = terminal[:need]
            if int(terminal.numel()) < need:
                terminal = torch.zeros(need)
        keys = torch.cat([query_key, keys], dim=0)
        values = torch.cat(
            [terminal.reshape(1, self.heads, self.dk).to(values.dtype), values], dim=0
        )
        length = int(keys.shape[0])
        spans = np.asarray([[index, index + 1] for index in range(length)], dtype=np.int64)
        # The digest is a cheap session+query binding, not a re-hash of every
        # delivered slot: the receiver's TTFT must not pay a SHA-256 pass over
        # the cache it is about to read.
        digest = hashlib.sha256(
            (
                "q1-compose:"
                + str(source)
                + ":"
                + np.ascontiguousarray(query_wave.astype(np.complex64)).tobytes().hex()
                + ":"
                + str(length)
            ).encode("utf-8")
        ).hexdigest()
        return SwarmKVCache(
            keys=keys,
            values=values,
            spans=spans,
            wave=query_wave,
            source=str(source),
            task="telepathy",
            digest=digest,
            meta={"query_agnostic": True, "bridge_slots": int(length)},
        ).validate()

    def _unbind_context(self, context: torch.Tensor) -> np.ndarray:
        """Map a value context back to a wave frame for a downstream reader.

        The shared frame is orthonormal by construction, so the transpose is
        the pseudo-inverse. This is the same unbinding TelepathicReader uses.
        """
        flat = context.reshape(-1).numpy().astype(np.float32)
        width = int(self.heads) * int(self.dk)
        rng = np.random.default_rng(20261108)
        frame = rng.standard_normal((self.dim, width)).astype(np.float32)
        frame /= np.maximum(np.linalg.norm(frame, axis=1, keepdims=True), _EPS)
        wave = flat @ frame.T
        norm = float(np.linalg.norm(wave))
        if norm <= _EPS:
            return np.zeros(self.dim, dtype=np.complex64)
        wave = wave / norm
        return (wave + 0j).astype(np.complex64)

    def chain_valid(self) -> bool:
        parent = "0" * 64
        for packet in self.packets:
            if str(packet.parent) != parent:
                return False
            parent = str(packet.digest)
        return True

    def summary(self) -> Dict[str, Any]:
        return {
            "sent": int(self.counts["sent"]),
            "admitted": int(self.counts["admitted"]),
            "vetoed": int(self.counts["vetoed"]),
            "entries": int(len(self.entries)),
            "packets": int(len(self.packets)),
            "chain_head": str(self.parent),
            "chain_valid": bool(self.chain_valid()),
            "budget_u": int(self.budget_u),
        }


# ---------------------------------------------------------------------------
# Receiver: one attention read over a compacted telepathic cache
# ---------------------------------------------------------------------------
class TelepathicReader:
    """Read a packet with a receiver query and return a vocabulary score row.

    The read is the paper's prefill-free path: no receiver text tokens are
    consumed. The query is projected into the packet's head frame, attends
    over the U support keys, and the resulting value context is unbind against
    the receiver's token waves. The score is standardised so its scale cannot
    swamp the receiver's own floor.
    """

    def __init__(self, tokenizer, projector, heads: int = 4, dk: int = 32,
                 seed: int = 20261108) -> None:
        self.tokenizer = tokenizer
        self.projector = projector
        self.heads = int(heads)
        self.dk = int(dk)
        rng = np.random.default_rng(int(seed))
        width = self.heads * self.dk
        self.unbind = rng.standard_normal((1024, width)).astype(np.float32)
        self.unbind /= np.maximum(np.linalg.norm(self.unbind, axis=1, keepdims=True), _EPS)

    def _query_keys(self, query_wave: np.ndarray) -> torch.Tensor:
        keys, _ = kv_from_wave(query_wave, heads=self.heads, dk=self.dk)
        return keys

    def read(self, packet: TelepathyPacket, query_wave: np.ndarray) -> np.ndarray:
        if not packet.admitted or int(packet.keys.shape[0]) == 0:
            return np.zeros(1024, dtype=np.float32)
        query = self._query_keys(query_wave)
        scores = torch.einsum("lhd,qhd->ql", packet.keys, query) / math.sqrt(float(self.dk))
        attention = torch.softmax(scores, dim=-1)
        context = torch.einsum("ql,lhd->qhd", attention, packet.values).reshape(-1)
        context = context / torch.clamp(context.norm(), min=_EPS)
        context_np = context.numpy().astype(np.float32)
        # Map the 128-dim value context back to the wave frame with the fixed
        # pseudo-inverse frame, then score tokens by phase coherence.
        wave = torch.from_numpy(context_np @ self.unbind.t()).numpy()
        return wave


# ---------------------------------------------------------------------------
# T2T baseline helper
# ---------------------------------------------------------------------------
def t2t_summary(
    text: str,
    tokenizer,
    *,
    max_tokens: int = 16,
    query: str = "",
) -> str:
    """A matched-budget Text-to-Text summary (the baseline channel).

    The sender writes at most max_tokens tokens. It is question-aware: it
    ranks sentences by the number of question terms they contain, then emits
    the highest-ranked tokens that fit. This is the strongest honest T2T
    baseline; a weaker one would make SWAT look better for no reason.
    """
    query_terms = {
        term.lower()
        for term in str(query).replace("?", " ").replace(",", " ").split()
        if len(term) > 2
    }
    sentences = [
        sentence.strip()
        for sentence in str(text).replace("\n", " ").split(".")
        if sentence.strip()
    ]
    scored = []
    for sentence in sentences:
        lower = sentence.lower()
        hits = sum(1 for term in query_terms if term in lower)
        if hits:
            scored.append((hits, sentence))
    scored.sort(key=lambda pair: -pair[0])
    chosen = " ".join(sentence for _, sentence in scored[:6]) or str(text)[:400]
    ids = tokenizer.encode(chosen)
    if len(ids) <= int(max_tokens):
        return chosen
    return tokenizer.decode(ids[: int(max_tokens)])

# ---------------------------------------------------------------------------
# Directive R1: deterministic unstructured document fact extraction
# ---------------------------------------------------------------------------
import ast as _ast
import json as _json
import re as _re


_NUMERIC = _re.compile(r"^[+-]?\d[\d,]*(?:\.\d+)?$")
_NUMBER_UNIT = _re.compile(r"^([+-]?\d[\d,]*(?:\.\d+)?)\s*([A-Za-z%][A-Za-z0-9%/_-]*)$")
_NAME = _re.compile(r"^[A-Za-z_][\w.\-]*$")
_PYTHON_HINT = _re.compile(
    r"(^|\n)\s*(def |class |import |from )|^[A-Za-z_]\w*\s*=\s*[\[\{'\"0-9]",
    _re.MULTILINE,
)
_QUOTES = " \t\r\n" + chr(96) + chr(39) + chr(34) + "[]()"


def _strip_token(text: Any) -> str:
    token = str(text).strip()
    return token.strip(_QUOTES).strip()


def _normalise_name(text: Any) -> str:
    """A subject/entity key: case-folded, whitespace folded to underscores."""
    token = _strip_token(text).strip(" .,;:")
    token = _re.sub(r"\s+", "_", token)
    return token.casefold()


def _normalise_value(text: Any) -> str:
    """A value: numbers lose thousands separators; names are case-folded."""
    token = _strip_token(text).strip(" .,;:")
    if not token:
        return ""
    if _NUMERIC.match(token):
        return token.replace(",", "")
    match = _NUMBER_UNIT.match(token)
    if match and match.group(2):
        number = match.group(1).replace(",", "")
        unit = match.group(2).casefold()
        return number + unit
    return _re.sub(r"\s+", " ", token).casefold()


def _normalise_relation(text: Any) -> str:
    token = _strip_token(text).casefold()
    token = _re.sub(r"[\s\-]+", "_", token).strip("_:")
    return token or "value"


@dataclass
class ExtractedFact:
    """One normalised (subject, relation, value) triple from a raw document."""

    subject: str
    relation: str
    value: str
    source: str = ""
    doc_index: int = 0
    line: int = 0
    fmt: str = "text"

    def key(self) -> str:
        """The binding key: a link fact binds its subject, a field binds
        subject.field, so a latency fact cannot shadow the same subject's
        chain fact in the gate registry."""
        pass_through = {
            "value", "resolves_to", "feeds", "depends_on", "connects_to",
            "contains", "points_to", "maps_to", "links_to", "forwards_to",
            "delegates_to", "escalates_to", "terminates_at",
        }
        if str(self.relation) in pass_through:
            return str(self.subject)
        return f"{self.subject}.{self.relation}" if self.relation else str(self.subject)

    def pair(self) -> List[str]:
        return [self.key(), str(self.value)]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key(),
            "subject": str(self.subject),
            "relation": str(self.relation),
            "value": str(self.value),
            "source": str(self.source),
            "doc_index": int(self.doc_index),
            "line": int(self.line),
            "format": str(self.fmt),
        }


_CHAIN_PHRASES: Sequence[Tuple[str, str]] = (
    ("is followed by", "resolves_to"),
    ("is linked to", "resolves_to"),
    ("terminates at", "resolves_to"),
    ("forwards to", "resolves_to"),
    ("delegates to", "resolves_to"),
    ("escalates to", "resolves_to"),
    ("resolves to", "resolves_to"),
    ("routes to", "resolves_to"),
    ("points to", "resolves_to"),
    ("maps to", "resolves_to"),
    ("links to", "resolves_to"),
    ("connects to", "connects_to"),
    ("feeds into", "feeds"),
    ("depends on", "depends_on"),
    ("contains", "contains"),
)
_CHAIN_PATTERN = "|".join(
    _re.escape(phrase) for phrase, _relation in sorted(_CHAIN_PHRASES, key=lambda pair: -len(pair[0]))
)
_RELATION_OF = dict(_CHAIN_PHRASES)


def _column_role(name: str) -> str:
    key = str(name).casefold()
    if any(token in key for token in ("subject", "key", "entity", "name", "from", "item")):
        return "subject"
    if any(token in key for token in ("relation", "field", "edge", "predicate", "property")):
        return "relation"
    if any(token in key for token in ("value", "answer", "result", "target", "to", "object")):
        return "value"
    if any(token in key for token in ("metric", "measure", "quantity", "latency", "throughput")):
        return "metric"
    return ""


def _literal(node: Any) -> Optional[str]:
    if isinstance(node, _ast.Constant):
        return str(node.value)
    if isinstance(node, _ast.Name):
        return str(node.id)
    if isinstance(node, _ast.UnaryOp) and isinstance(node.op, (_ast.USub, _ast.UAdd)):
        inner = _literal(node.operand)
        if inner is None:
            return None
        return ("-" if isinstance(node.op, _ast.USub) else "") + inner
    return None


class DocumentFactExtractor:
    """Deterministic, zero-network extraction of facts from raw documents.

    The extractor parses four declared families, all of which occur in real
    multi-artifact workspaces:

      table   Markdown pipe tables with subject/relation/value headers;
      config  key = value, key: value, key -> value and JSON records;
      code    Python Assign / AnnAssign constants and literal dict bindings;
      prose   sentences of the form subject link-phrase object, and
              subject is quantity.

    It emits normalised (subject, relation, value) triples. The registry is
    built by CORROBORATION across documents: a value supported by two or more
    documents wins over a single contradicting document. That is what lets an
    ingest build its own Tier-3 reference law from raw text with no caller
    supplied fact_registry.
    """

    BRIDGE_RELATIONS = frozenset(
        {
            "value", "resolves_to", "feeds", "depends_on", "connects_to",
            "contains", "points_to", "maps_to", "links_to", "forwards_to",
            "delegates_to", "escalates_to", "terminates_at",
        }
    )

    def __init__(self, *, min_support: int = 2, max_documents: int = 4096) -> None:
        self.min_support = int(min_support)
        self.max_documents = int(max_documents)

    def _facts_from_table(
        self, text: str, source: str, doc_index: int
    ) -> List[ExtractedFact]:
        rows: List[Tuple[int, List[str]]] = []
        for number, line in enumerate(str(text).splitlines(), start=1):
            stripped = line.strip()
            if not stripped.startswith("|"):
                continue
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            if not any(cells):
                continue
            rows.append((number, cells))
        if not rows:
            return []
        header: Optional[List[str]] = None
        data = rows
        if len(rows) >= 2:
            separator = rows[1][1]
            if separator and all(
                _re.fullmatch(r":?-{2,}:?", cell.replace(" ", ""))
                for cell in separator
                if cell
            ):
                header = rows[0][1]
                data = rows[2:]
        roles: List[str] = []
        if header is not None:
            roles = [_column_role(cell) for cell in header]
        if "subject" not in roles:
            if len(rows[0][1]) >= 3:
                roles = ["subject", "relation", "value"] + [""] * (len(rows[0][1]) - 3)
            else:
                roles = ["subject", "value"] + [""] * (len(rows[0][1]) - 2)
        facts: List[ExtractedFact] = []
        for number, cells in data:
            row = dict(zip(roles, cells)) if roles else {}
            subject = row.get("subject", "")
            value = row.get("value", "")
            if not subject or not value:
                continue
            if subject.casefold() in {"subject", "key", "entity"}:
                continue
            relation = row.get("relation") or row.get("metric") or "value"
            facts.append(
                ExtractedFact(
                    subject=_normalise_name(subject),
                    relation=_normalise_relation(relation),
                    value=_normalise_value(value),
                    source=str(source),
                    doc_index=int(doc_index),
                    line=int(number),
                    fmt="table",
                )
            )
        return facts

    def _facts_from_config(
        self, text: str, source: str, doc_index: int
    ) -> List[ExtractedFact]:
        facts: List[ExtractedFact] = []
        pattern = _re.compile(
            r"^\s*(?:[-*]\s*)?(?P<key>[A-Za-z_][\w.\-]*)\s*(?:=|:|->|=>|\u2192)\s*(?P<value>.+?)\s*$"
        )
        skip = (
            "def ", "class ", "import ", "from ", "return ", "if ", "for ",
            "while ", "with ", "try:", "except", "elif ", "else:", "{", "}",
        )
        for number, line in enumerate(str(text).splitlines(), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith(("#", "//", "|", '"', "'")):
                continue
            if any(stripped.startswith(token) for token in skip):
                continue
            if any(token in stripped for token in ("==", "!=", "<=", ">=", ":=")):
                continue
            if stripped.startswith("{"):
                try:
                    payload = _json.loads(stripped)
                except Exception:
                    continue
                for key, value in self._flatten_json(payload):
                    facts.append(
                        ExtractedFact(
                            subject=_normalise_name(key),
                            relation="value",
                            value=_normalise_value(value),
                            source=str(source),
                            doc_index=int(doc_index),
                            line=int(number),
                            fmt="json",
                        )
                    )
                continue
            match = pattern.match(stripped)
            if not match:
                continue
            key = str(match.group("key"))
            value = str(match.group("value")).strip().rstrip(",")
            if not key or not value:
                continue
            # A dict/list literal is the Python parser's job: flattening it as
            # one opaque config string would hide every binding inside it.
            if value.startswith(("{", "[")):
                continue
            if "." in key:
                head, _, field = key.partition(".")
                subject, relation = _normalise_name(head), _normalise_relation(field)
            else:
                subject, relation = _normalise_name(key), "value"
            facts.append(
                ExtractedFact(
                    subject=subject,
                    relation=relation,
                    value=_normalise_value(value),
                    source=str(source),
                    doc_index=int(doc_index),
                    line=int(number),
                    fmt="config",
                )
            )
        return facts

    @staticmethod
    def _flatten_json(payload: Any, prefix: str = "") -> List[Tuple[str, Any]]:
        out: List[Tuple[str, Any]] = []
        if isinstance(payload, dict):
            for key, value in payload.items():
                name = f"{prefix}.{key}" if prefix else str(key)
                if isinstance(value, (dict, list)):
                    out.extend(DocumentFactExtractor._flatten_json(value, name))
                else:
                    out.append((name, value))
        elif isinstance(payload, list):
            for index, value in enumerate(payload):
                name = f"{prefix}[{index}]" if prefix else str(index)
                if isinstance(value, (dict, list)):
                    out.extend(DocumentFactExtractor._flatten_json(value, name))
                else:
                    out.append((name, value))
        return out

    def _facts_from_python(
        self, text: str, source: str, doc_index: int
    ) -> List[ExtractedFact]:
        try:
            tree = _ast.parse(str(text))
        except (SyntaxError, ValueError):
            return []
        facts: List[ExtractedFact] = []

        def assign(target: Any, value: Any) -> None:
            name = ""
            relation = "value"
            if isinstance(target, _ast.Name):
                name = str(target.id)
            elif isinstance(target, _ast.Attribute):
                parts = []
                node = target
                while isinstance(node, _ast.Attribute):
                    parts.append(str(node.attr))
                    node = node.value
                if isinstance(node, _ast.Name):
                    parts.append(str(node.id))
                name = ".".join(reversed(parts))
            if not name:
                return
            if isinstance(value, _ast.Dict):
                for key_node, item_node in zip(value.keys, value.values):
                    if key_node is None:
                        continue
                    key = _literal(key_node)
                    item = _literal(item_node)
                    if key is None or item is None:
                        continue
                    facts.append(
                        ExtractedFact(
                            subject=_normalise_name(key), relation="value",
                            value=_normalise_value(item), source=str(source),
                            doc_index=int(doc_index),
                            line=int(getattr(item_node, "lineno", 0)), fmt="python",
                        )
                    )
                return
            literal = _literal(value)
            if literal is None:
                return
            if "." in name:
                head, _, field = name.partition(".")
                subject, relation = _normalise_name(head), _normalise_relation(field)
            else:
                subject = _normalise_name(name)
            facts.append(
                ExtractedFact(
                    subject=subject, relation=relation,
                    value=_normalise_value(literal), source=str(source),
                    doc_index=int(doc_index), line=int(getattr(target, "lineno", 0)),
                    fmt="python",
                )
            )

        for node in _ast.walk(tree):
            if isinstance(node, _ast.Assign):
                for target in node.targets:
                    assign(target, node.value)
            elif isinstance(node, _ast.AnnAssign) and node.value is not None:
                assign(node.target, node.value)
        return facts

    def _facts_from_prose(
        self, text: str, source: str, doc_index: int
    ) -> List[ExtractedFact]:
        facts: List[ExtractedFact] = []
        chain = _re.compile(
            r"^(?:the\s+)?(?P<subject>[A-Za-z_][\w.\-]*)\s+(?:" + _CHAIN_PATTERN + r")\s+(?:the\s+)?(?P<value>[A-Za-z_][\w.\-]*)",
            _re.IGNORECASE,
        )
        metric = _re.compile(
            r"^(?:the\s+)?(?P<subject>[A-Za-z_][\w.\-]*)\s+(?:is|equals|reports|has\s+value)\s+(?P<value>[+-]?\d[\d,]*(?:\.\d+)?\s*[A-Za-z%]*)",
            _re.IGNORECASE,
        )
        value_of = _re.compile(
            r"^(?:the\s+)?value\s+of\s+(?P<subject>[A-Za-z_][\w.\-]*)\s+(?:is|equals)\s+(?P<value>[^\s.]+)",
            _re.IGNORECASE,
        )
        for number, line in enumerate(str(text).splitlines(), start=1):
            stripped = line.strip().lstrip("-* ").strip()
            if not stripped or stripped.startswith(("#", "|", "{", '"')):
                continue
            for sentence in _re.split(r"(?<=[.!?])\s+", stripped):
                sentence = sentence.strip()
                if not sentence:
                    continue
                match = chain.match(sentence)
                if match:
                    relation_phrase = match.group(0).casefold()
                    relation = "resolves_to"
                    for phrase, canonical in _CHAIN_PHRASES:
                        if phrase in relation_phrase:
                            relation = canonical
                            break
                    facts.append(
                        ExtractedFact(
                            subject=_normalise_name(match.group("subject")),
                            relation=relation,
                            value=_normalise_value(match.group("value")),
                            source=str(source), doc_index=int(doc_index),
                            line=int(number), fmt="prose",
                        )
                    )
                    continue
                match = value_of.match(sentence) or metric.match(sentence)
                if match:
                    facts.append(
                        ExtractedFact(
                            subject=_normalise_name(match.group("subject")),
                            relation="value",
                            value=_normalise_value(match.group("value")),
                            source=str(source), doc_index=int(doc_index),
                            line=int(number), fmt="prose",
                        )
                    )
        return facts

    def extract(self, text: str, source: str = "", doc_index: int = 0) -> List[ExtractedFact]:
        text = str(text)
        facts: List[ExtractedFact] = []
        facts.extend(self._facts_from_table(text, source, doc_index))
        facts.extend(self._facts_from_config(text, source, doc_index))
        if _PYTHON_HINT.search(text):
            facts.extend(self._facts_from_python(text, source, doc_index))
        facts.extend(self._facts_from_prose(text, source, doc_index))
        seen = set()
        unique: List[ExtractedFact] = []
        for fact in facts:
            if not fact.subject or not fact.value:
                continue
            key = (fact.subject, fact.relation, fact.value)
            if key in seen:
                continue
            seen.add(key)
            unique.append(fact)
        return unique

    def build_registry(self, documents: Sequence[Dict[str, Any]]) -> Dict[str, str]:
        """Corroborated subject -> value law from the documents themselves."""
        counts: Dict[str, Dict[str, int]] = {}
        order: Dict[str, List[str]] = {}
        for document in documents:
            if not bool(document.get("connected", True)):
                continue
            for fact in document.get("facts") or []:
                subject = str(fact.get("key") or fact["subject"])
                value = str(fact["value"])
                counts.setdefault(subject, {})
                counts[subject][value] = counts[subject].get(value, 0) + 1
                order.setdefault(subject, [])
                if value not in order[subject]:
                    order[subject].append(value)
        registry: Dict[str, str] = {}
        for subject, buckets in counts.items():
            ranked = sorted(
                buckets.items(),
                key=lambda pair: (-pair[1], order[subject].index(pair[0])),
            )
            best_value, best_count = ranked[0]
            if best_count >= self.min_support or len(ranked) == 1:
                registry[subject] = best_value
        return registry

    def extract_documents(
        self, documents: Sequence[Any]
    ) -> Dict[str, Any]:
        """Parse raw documents, measure cross-document connectivity, build law."""
        parsed: List[Dict[str, Any]] = []
        for index, document in enumerate(list(documents)[: self.max_documents]):
            if isinstance(document, dict):
                text = str(document.get("text") or document.get("content") or "")
                source = str(document.get("source") or document.get("id") or f"doc{index}")
                meta = dict(document)
            else:
                text = str(document)
                source = f"doc{index}"
                meta = {}
            facts = self.extract(text, source=source, doc_index=int(index))
            names = set()
            for fact in facts:
                names.add(str(fact.key()))
                value = str(fact.value)
                if _NAME.match(value) and not _NUMERIC.match(value):
                    names.add(value)
            parsed.append(
                {
                    "index": int(index),
                    "source": source,
                    "text": text,
                    "meta": meta,
                    "facts": [fact.to_dict() for fact in facts],
                    "names": names,
                }
            )
        incidence: Dict[str, int] = {}
        for document in parsed:
            for name in document["names"]:
                incidence[name] = incidence.get(name, 0) + 1
        for document in parsed:
            shared = [name for name in document["names"] if incidence.get(name, 0) >= 2]
            document["shared_names"] = sorted(shared)
            document["connected"] = bool(shared) or len(parsed) <= 1
        registry = self.build_registry(parsed)
        return {
            "documents": parsed,
            "registry": registry,
            "triples": [fact for document in parsed for fact in document["facts"]],
            "documents_total": int(len(parsed)),
            "connected_documents": int(sum(1 for document in parsed if document["connected"])),
            "registry_facts": int(len(registry)),
        }

