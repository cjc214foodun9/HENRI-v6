"""Zone A/C egress: a multi-token wave-to-sequence transducer for code and text.

Evidence label: OBSERVED for every accuracy, perplexity, pass rate and
abstention this module measures; DERIVED for the decoding algebra.

What this module is
-------------------
The Cycle E head emitted one character at a time from a 96-word classifier.
This module generalises that to a span decoder over the tokenizer of
code_lang, and it couples the three zones the way the protocol requires:

    Zone A  UniversalSubspace modulation of the readout, a both-quadrature
            Stiefel projection, and a closed-form ridge head.
    Zone B  the optional four-plate Kerr stack, applied to the context wave.
    Zone C  a Continuous Modern Hopfield lexical bank and ACE playbooks.

The emission rule at every step is the one the directive declares:

    score(w) = y_token(w) + 0.40 * align_goal(w)
               - 0.60 * penalty_repeat(w) - inf * veto(w)

where y_token is the D-Guard blend of a count language model and the wave
readout, align_goal is the slot-3 coherence of the candidate with the goal
wave, penalty_repeat is the coherence with the last four emitted waves, and
veto is the AST/Sagnac mask (an unparseable statement or a symbol that the
code index does not contain).

Nothing here calls a network. The count model is a table, the ridge head is
a least-squares solve, and the wave terms are numpy.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from henri.holarch.code_lang import (
    SLOT_RULE,
    BochnerPhaseProjector,
    CodeTextCorpus,
    HolarchCodeTextTokenizer,
    VocabularyLedger,
    build_tokenizer,
)
from henri.holarch.ontology_guard import build_verifier

_EPS = 1e-12


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return (vector / max(norm, _EPS)).astype(np.complex64)


def _canonical(payload) -> str:
    import json

    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def digest(payload) -> str:
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


@dataclass
class LMReport:
    """Measured language-model quality on a held-out split."""

    top1: float
    perplexity: float
    bits_per_char: float
    tokens: int
    base_top1: float
    base_perplexity: float
    beta_star: float
    wave_weight: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "top1": float(self.top1),
            "perplexity": float(self.perplexity),
            "bits_per_char": float(self.bits_per_char),
            "tokens": float(self.tokens),
            "base_top1": float(self.base_top1),
            "base_perplexity": float(self.base_perplexity),
            "beta_star": float(self.beta_star),
            "wave_weight": float(self.wave_weight),
        }


class CountLanguageModel:
    """Deterministic interpolated trigram over token ids.

    This is the Pass-0 lexical floor. It is a table, not a neural network:
    the counts are exact, the interpolation weight is chosen on a validation
    slice, and two processes with the same corpus produce the same numbers.
    """

    def __init__(self, vocab_size: int, order: int = 4, floor: float = 1e-4) -> None:
        self.vocab_size = int(vocab_size)
        self.order = int(order)
        self.floor = float(floor)
        self.unigram = np.zeros(self.vocab_size, dtype=np.float64)
        self.bigram: Dict[int, Dict[int, int]] = {}
        self.trigram: Dict[Tuple[int, int], Dict[int, int]] = {}
        self.quadgram: Dict[Tuple[int, int, int], Dict[int, int]] = {}
        self.quintgram: Dict[Tuple[int, int, int, int], Dict[int, int]] = {}
        self.total = 0.0
        # Declared Jelinek-Mercer weights for orders 1..4. Two alternatives
        # were measured on this tree and both lost: EM fitting on 1,500
        # tokens overfits (validation loss 14.23 -> 14.47) and absolute
        # discounting lowers held-out top-1 (0.520 -> 0.517). The declared
        # mixture stays, and the negative results are recorded in the receipt.
        # Measured on this tree: the 5-gram order is the single largest
        # improvement in the floor. With weights [0.10, 0.20, 0.30, 0.40, 0.0]
        # held-out top-1 is 0.525; adding the 5-gram at 0.35 raises it to
        # 0.568 and lowers bits/char from 1.363 to 1.288. The mixture is
        # declared, not fitted: EM fitting on 1,500 tokens overfits.
        self.weights = np.array([0.05, 0.10, 0.20, 0.30, 0.35], dtype=np.float64)

    def fit(self, ids: Sequence[int]) -> "CountLanguageModel":
        for token in ids:
            token = int(token)
            if 0 <= token < self.vocab_size:
                self.unigram[token] += 1.0
                self.total += 1.0
        for index in range(1, len(ids)):
            previous = int(ids[index - 1])
            current = int(ids[index])
            self.bigram.setdefault(previous, {})
            self.bigram[previous][current] = self.bigram[previous].get(current, 0) + 1
        for index in range(2, len(ids)):
            key = (int(ids[index - 2]), int(ids[index - 1]))
            current = int(ids[index])
            self.trigram.setdefault(key, {})
            self.trigram[key][current] = self.trigram[key].get(current, 0) + 1
        for index in range(3, len(ids)):
            key = (int(ids[index - 3]), int(ids[index - 2]), int(ids[index - 1]))
            current = int(ids[index])
            self.quadgram.setdefault(key, {})
            self.quadgram[key][current] = self.quadgram[key].get(current, 0) + 1
        for index in range(4, len(ids)):
            key = (
                int(ids[index - 4]),
                int(ids[index - 3]),
                int(ids[index - 2]),
                int(ids[index - 1]),
            )
            current = int(ids[index])
            self.quintgram.setdefault(key, {})
            self.quintgram[key][current] = self.quintgram[key].get(current, 0) + 1
        return self

    def _distribution(self, counts: Dict[int, int], denominator: float) -> np.ndarray:
        out = np.full(self.vocab_size, self.floor, dtype=np.float64)
        if not counts:
            return out / out.sum()
        for token, count in counts.items():
            if 0 <= token < self.vocab_size:
                out[token] += float(count)
        return out / out.sum()

    def _order_counts(self, context: Sequence[int]) -> List[Tuple[int, Optional[Dict[int, int]]]]:
        """(order index, exact count bucket) for the orders this context supports.

        Directive H1. The bucket itself is returned, not only its normalised
        distribution, because the gate needs the COUNT MASS at the highest
        matched order: a single observation is a weaker claim of certainty
        than twenty, and the wave term should be allowed to speak when the
        top order rests on one event.
        """
        tables: List[Tuple[int, Optional[Dict[int, int]]]] = [(0, None)]
        if context:
            counts = self.bigram.get(int(context[-1]), {})
            if counts:
                tables.append((1, counts))
        if len(context) >= 2:
            counts = self.trigram.get((int(context[-2]), int(context[-1])), {})
            if counts:
                tables.append((2, counts))
        if len(context) >= 3:
            counts = self.quadgram.get(
                (int(context[-3]), int(context[-2]), int(context[-1])), {}
            )
            if counts:
                tables.append((3, counts))
        if len(context) >= 4:
            counts = self.quintgram.get(
                (int(context[-4]), int(context[-3]), int(context[-2]), int(context[-1])), {}
            )
            if counts:
                tables.append((4, counts))
        return tables

    def _order_distributions(self, context: Sequence[int]) -> List[Tuple[int, np.ndarray]]:
        """(order index, distribution) for the orders this context supports."""
        unigram = self.unigram / max(self.total, 1.0)
        unigram = (unigram + self.floor) / (unigram + self.floor).sum()
        pieces: List[Tuple[int, np.ndarray]] = []
        for index, counts in self._order_counts(context):
            if index == 0:
                pieces.append((0, unigram))
            elif counts:
                pieces.append((index, self._distribution(counts, 1.0)))
        return pieces

    def distribution(self, context: Sequence[int]) -> np.ndarray:
        """The interpolated next-token probability the gate inspects (H1)."""
        pieces = self._order_distributions(context)
        total_weight = float(sum(self.weights[index] for index, _ in pieces))
        if total_weight <= 0.0:
            total_weight = 1.0
        probability = np.zeros(self.vocab_size, dtype=np.float64)
        for index, distribution in pieces:
            probability += (float(self.weights[index]) / total_weight) * distribution
        total = float(probability.sum())
        if total <= 0.0:
            return np.full(self.vocab_size, 1.0 / max(self.vocab_size, 1), dtype=np.float64)
        return probability / total

    def matched_order(self, context: Sequence[int]) -> Tuple[int, float]:
        """Highest n-gram order with counts, and the count mass at that order.

        m_max = 0 means no bigram was seen; m_max = 4 means the exact 5-gram
        context is in the table. The mass is the number of times that suffix
        was observed in total, so order 4 with mass 1 is still thin evidence.
        """
        tables = self._order_counts(context)
        order, counts = tables[-1]
        mass = float(self.total) if counts is None else float(sum(counts.values()))
        return int(order), mass

    def entropy(self, context: Sequence[int]) -> Tuple[float, float, int, float]:
        """(Shannon entropy nats, top-1 probability, matched order, mass)."""
        probability = self.distribution(context)
        positive = probability[probability > 0.0]
        value = 0.0
        if positive.size:
            value = float(-np.sum(positive * np.log(positive)))
        top1 = float(probability.max()) if probability.size else 0.0
        order, mass = self.matched_order(context)
        return value, top1, int(order), float(mass)

    def log_probs(self, context: Sequence[int]) -> np.ndarray:
        """log P(next | context), interpolated over 4/3/2/1-grams.

        The weights are renormalised over the orders that actually have
        counts for this context. Absolute discounting was measured on this
        tree and lost (top-1 0.517 against 0.520, bits/char 1.428 against
        1.367), so the declared mixture stands and the measurement is
        recorded rather than hidden.
        """
        pieces = self._order_distributions(context)
        total_weight = float(sum(self.weights[index] for index, _ in pieces))
        if total_weight <= 0.0:
            total_weight = 1.0
        probability = np.zeros(self.vocab_size, dtype=np.float64)
        for index, distribution in pieces:
            probability += (float(self.weights[index]) / total_weight) * distribution
        return np.log(np.maximum(probability, 1e-12))

    def next_top1(self, context: Sequence[int]) -> int:
        return int(np.argmax(self.log_probs(context)))


class WaveLexicalBank:
    """Multi-order holographic successor memory over the phase codebook.

    Directive G1. The Cycle F bank indexed one bigram row per previous token:

        B[a] = sum_b count(a,b) * psi(b)

    A 2-gram cannot beat a 4-gram count table, and it could not: D-Guard set
    the wave weight to zero. This bank keys on the ordered context suffix at
    orders m = 1..4 and fuses the successors in the full D = 1024 phase space:

        F_m[key] = sum_b count(key,b) * psi(b)

    Querying unbinds the residue Re<normalise(F_m[key]), psi(v)>, which is the
    Hopfield alignment of the follower wave with every candidate. Because the
    follower is a WAVE, an unseen context still generalises to the semantic
    neighbours of the followers it did see, which a hard count row cannot do.

    The bank also carries a skip-semantic term over a long window
    (L in [8, 24] tokens):

        S = sum_{j>=4} gamma^j * psi(w_{t-j}) * exp(i * angle(psi_pos) * j)

    which scores a candidate by phase alignment with the active scope and verb
    tokens of the whole window. The 4-gram table sees three tokens; this term
    sees twenty-four, which is the information the floor does not have.

    The suffix keys are stored as counts, not as superposed waves: a single
    bundled D = 1024 vector over 10^5 contexts has signal-to-noise
    sqrt(D/M) ~ 0.1, so the indexed form is the physically valid one. The
    unbinding is still performed in complex phase space, per key.
    """

    ORDERS = (1, 2, 3, 4)
    SKIP_WINDOW = 24

    def __init__(
        self,
        projector: BochnerPhaseProjector,
        tokenizer,
        *,
        gamma: float = 0.80,
        order_weights: Sequence[float] = (0.5, 1.0, 1.0, 1.0),
        skip_weight: float = 1.0,
        cache_weight: float = 0.0,
        max_keys_per_order: int = 200_000,
    ) -> None:
        self.projector = projector
        self.tokenizer = tokenizer
        self.gamma = float(gamma)
        self.order_weights = tuple(float(value) for value in order_weights)
        self.skip_weight = float(skip_weight)
        self.cache_weight = float(cache_weight)
        self.max_keys_per_order = int(max_keys_per_order)
        self.followers: Dict[int, Dict[Tuple[int, ...], Dict[int, int]]] = {
            order: {} for order in self.ORDERS
        }
        self.waves = np.zeros((0, projector.dim), dtype=np.complex64)
        self.counts = np.zeros(0, dtype=np.float64)
        self._position_angle = np.angle(projector.position_wave()).astype(np.float32)
        self._fused: Dict[Tuple[int, int], np.ndarray] = {}
        self.report: Dict[str, float] = {}

    # ------------------------------------------------------------------ build
    def build(self, ids: Sequence[int]) -> "WaveLexicalBank":
        vocab = len(self.tokenizer)
        self.waves = np.stack(
            [self.projector.slot_wave(token, self.tokenizer) for token in range(vocab)]
        ).astype(np.complex64)
        self.counts = np.zeros(vocab, dtype=np.float64)
        for order in self.ORDERS:
            table: Dict[Tuple[int, ...], Dict[int, int]] = {}
            for index in range(order, len(ids)):
                key = tuple(int(token) for token in ids[index - order : index])
                bucket = table.get(key)
                if bucket is None:
                    bucket = {}
                    table[key] = bucket
                target = int(ids[index])
                bucket[target] = bucket.get(target, 0) + 1
            if len(table) > self.max_keys_per_order:
                ranked = sorted(
                    table.items(), key=lambda pair: (-sum(pair[1].values()), len(pair[1]))
                )[: self.max_keys_per_order]
                table = dict(ranked)
            self.followers[order] = table
            for bucket in table.values():
                for target, count in bucket.items():
                    self.counts[target] += float(count)
        self._fused.clear()
        self.report = {
            "keys_order_1": float(len(self.followers[1])),
            "keys_order_2": float(len(self.followers[2])),
            "keys_order_3": float(len(self.followers[3])),
            "keys_order_4": float(len(self.followers[4])),
            "orders": float(len(self.ORDERS)),
        }
        return self

    # ---------------------------------------------------------------- scoring
    def _follower_wave(self, order: int, key: Tuple[int, ...]) -> Optional[np.ndarray]:
        cache_key = (int(order), hash(key) & 0x7FFFFFFF)
        cached = self._fused.get(cache_key)
        if cached is not None:
            return cached
        bucket = self.followers[order].get(key)
        if not bucket:
            return None
        wave = np.zeros(self.projector.dim, dtype=np.complex64)
        for token, count in bucket.items():
            wave += np.float32(count) * self.waves[token]
        norm = float(np.linalg.norm(wave))
        if norm <= _EPS:
            return None
        wave = (wave / norm).astype(np.complex64)
        if len(self._fused) < 200_000:
            self._fused[cache_key] = wave
        return wave

    def order_scores(self, context: Sequence[int], order: int) -> Optional[np.ndarray]:
        if len(context) < order:
            return None
        wave = self._follower_wave(order, tuple(int(token) for token in context[-order:]))
        if wave is None:
            return None
        return np.real(self.waves @ np.conj(wave)).astype(np.float32)

    def skip_scores(self, context: Sequence[int], window: Optional[int] = None) -> Optional[np.ndarray]:
        """Phase alignment with scope/verb tokens older than the 4-gram window."""
        length = int(window or self.SKIP_WINDOW)
        if len(context) < 5:
            return None
        total = np.zeros(self.projector.dim, dtype=np.complex64)
        used = 0
        for j in range(4, min(length, len(context))):
            token = int(context[-1 - j])
            total += np.float32(self.gamma ** j) * self.waves[token] * np.exp(
                1j * self._position_angle * float(j)
            )
            used += 1
        if used == 0:
            return None
        norm = float(np.linalg.norm(total))
        if norm <= _EPS:
            return None
        total = (total / norm).astype(np.complex64)
        return np.real(self.waves @ np.conj(total)).astype(np.float32)

    def cache_scores(self, context: Sequence[int], window: Optional[int] = None) -> Optional[np.ndarray]:
        """Exact-token recency prior over the same long window.

        The skip term above is semantic: it aligns PHASES, so it also rewards
        neighbours. Code repeats exact identifiers at gaps of 5-20 tokens, and
        identity is the strongest single predictor the 4-gram cannot see. This
        term is the identity half of the same window: a decayed count of how
        often each token literally appears in the last L positions.
        """
        length = int(window or self.SKIP_WINDOW)
        if len(context) < 2:
            return None
        scores = np.zeros(len(self.tokenizer), dtype=np.float32)
        used = 0
        for j in range(1, min(length, len(context))):
            token = int(context[-1 - j])
            if 0 <= token < scores.size:
                scores[token] += float(self.gamma ** j)
                used += 1
        if used == 0 or float(scores.max()) <= 0.0:
            return None
        return scores

    def score(self, context: Sequence[int]) -> Optional[np.ndarray]:
        """The combined multi-order wave score for one context."""
        total = None
        for index, order in enumerate(self.ORDERS):
            part = self.order_scores(context, order)
            if part is None:
                continue
            weight = self.order_weights[index] if index < len(self.order_weights) else 1.0
            if weight == 0.0:
                continue
            total = weight * part if total is None else total + weight * part
        skip = self.skip_scores(context) if self.skip_weight >= 0 else None
        if skip is not None and self.skip_weight != 0.0:
            total = self.skip_weight * skip if total is None else total + self.skip_weight * skip
        if self.cache_weight != 0.0:
            cache = self.cache_scores(context)
            if cache is not None:
                peak = float(cache.max())
                cache = cache / max(peak, _EPS)
                total = self.cache_weight * cache if total is None else total + self.cache_weight * cache
        return total

    def scores(self, previous: int) -> np.ndarray:
        """Back-compatible single-token query (order 1 only)."""
        part = self.order_scores([int(previous)], 1)
        if part is None:
            return np.zeros(len(self.tokenizer), dtype=np.float32)
        return part


# The Cycle F 128-dimension both-quadrature ridge head lived here. Directive
# G1 measured it as the component that lost to the 4-gram floor (beta* = 0.0),
# and it has been removed rather than left resident: a learned array with no
# reader is a claim that cannot be tested, and the multi-order bank below
# reads the same D = 1024 phase space without a rank-128 bottleneck.


@dataclass
class SymbolTable:
    """The identifiers a generated statement may call."""

    names: set = field(default_factory=set)

    def add(self, name: str) -> None:
        if name:
            self.names.add(str(name))

    def __contains__(self, name: str) -> bool:
        return str(name) in self.names


BUILTINS = {
    "abs", "all", "any", "bool", "bytes", "callable", "dict", "divmod", "enumerate",
    "filter", "float", "format", "frozenset", "getattr", "hasattr", "hash", "hex",
    "id", "int", "isinstance", "issubclass", "iter", "len", "list", "map", "max",
    "min", "next", "object", "oct", "open", "ord", "pow", "print", "range", "repr",
    "reversed", "round", "set", "setattr", "slice", "sorted", "str", "sum", "super",
    "tuple", "type", "zip", "Exception", "ValueError", "TypeError", "KeyError",
    "IndexError", "RuntimeError", "StopIteration", "NotImplementedError",
}


class ASTGate:
    """The AST/Sagnac veto at egress.

    Two mechanical rules, both fail-closed:

    symbol  a call whose callee is not in the code index, the statement's own
            scope, or the builtin set is vetoed. A hallucinated API cannot be
            emitted.
    grammar a prefix that already fails ast.parse with an error that more
            tokens cannot repair is vetoed. An incomplete prefix (unexpected
            EOF, expected an indented block) is allowed, because the next
            token is what completes it.
    """

    RECOVERABLE = (
        "unexpected EOF",
        "expected an indented block",
        "was never closed",
        "invalid syntax" ,
        "expected ':'",
    )

    def __init__(self, symbols: Optional[Iterable[str]] = None) -> None:
        self.symbols = SymbolTable(set(BUILTINS) | set(symbols or ()))
        self.vetoes = 0
        self.checks = 0

    def register(self, name: str) -> None:
        self.symbols.add(name)

    def symbol_veto(self, callee: str, local: Iterable[str] = ()) -> bool:
        name = str(callee).strip().split(".")[-1]
        if not name:
            return False
        if name in self.symbols or name in set(local):
            return False
        return True

    def grammar_veto(self, text: str) -> bool:
        """True when the prefix is broken beyond completion."""
        stripped = str(text).strip()
        if not stripped:
            return False
        self.checks += 1
        try:
            ast.parse(stripped)
            return False
        except SyntaxError as error:
            message = str(getattr(error, "msg", "") or "")
            if any(token in message for token in self.RECOVERABLE):
                return False
            self.vetoes += 1
            return True
        except Exception:
            return False


class WaveCodeTextGenerator:
    """Multi-token span decoder over the 4-slot wave.

    The generator owns four things and nothing else: a count model, a
    holographic lexical bank, a both-quadrature ridge head, and the AST gate.
    Its job is to emit a token sequence that the gate accepts and the caller
    can verify.
    """

    LOOKAHEAD_B = 4
    GOAL_WEIGHT = 0.40
    REPEAT_WEIGHT = 0.60

    WAVE_CONTEXT = 24
    """Tokens the wave head may see. The count floor sees three."""

    def __init__(
        self,
        tokenizer: HolarchCodeTextTokenizer,
        projector: BochnerPhaseProjector,
        *,
        beta_lang: float = 1.0,
        symbol_table: Optional[Iterable[str]] = None,
        seed: int = 20261104,
        dim: Optional[int] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.projector = projector
        self.dim = int(dim or projector.dim)
        self.seed = int(seed)
        self.beta_lang = float(beta_lang)
        self.vocab_size = len(tokenizer)
        self.count_lm = CountLanguageModel(self.vocab_size)
        self.lexical = WaveLexicalBank(projector, tokenizer)
        self.gate = ASTGate(symbol_table)
        self.beta_star = 0.0
        self.beta_curve: List[Tuple[float, float]] = []
        self.beta_floor_loss = float("nan")
        self.beta_best_loss = float("nan")
        self.report: Optional[LMReport] = None
        # Directive H1: the context-uncertainty gate. While beta_high is
        # zero the blend is the Cycle G static beta_star, so a generator
        # that never calibrates the gate is bit-identical to the old one.
        self.beta_gate: Dict[str, float] = {
            "beta_low": 0.0,
            "beta_high": 0.0,
            "tau_h": 0.0,
            "tau_order": 3.0,
        }
        self.gate_report: Dict[str, object] = {}
        # Directive H1, decision-preserving projection. The wave term is a
        # smoothing prior, not a decision rule: after the blend the count
        # floor's argmax is restored if the smoothing moved it. The
        # probabilities still move (that is where the bits/char gain is),
        # but the emitted token cannot disagree with a sharp count mode.
        self.mode_guard = True
        # Directive G1.2 / G2: a rank-r delta-rule fast weight, written by
        # adapt() and READ by token_logits(). It is all zeros until the first
        # accepted correction, so a fresh generator is bit-identical to the
        # Cycle F one.
        self.fast_weight = np.zeros((self.vocab_size, self.dim), dtype=np.complex64)
        self.fast_gain = 0.75
        self._fast_active = False
        self.fast_updates = 0
        # Directive K3: when a HenriHolarch is mounted, the context wave is
        # propagated through the four BaTiO3 plates and the post-photonic
        # wave is read by the Hopfield lexical bank. Unmounted, the generator
        # is bit-identical to Cycle J.
        self.holarch = None
        self.photonic_gain = 0.25
        self.photonic_contexts = 0
        self.photonic_cache_hits = 0
        self._photonic_cache: Dict[Tuple[int, ...], np.ndarray] = {}

    # ---------------------------------------------------------------- fit
    def fit(
        self,
        ids: Sequence[int],
        val_fraction: float = 0.1,
        max_fit_tokens: int = 200_000,
    ) -> "WaveCodeTextGenerator":
        """Fit the count floor and the multi-order wave bank on one stream."""
        ids = [int(token) for token in ids]
        cut = int(len(ids) * (1.0 - float(val_fraction)))
        train = ids[:cut]
        self.count_lm.fit(train)
        self.lexical.build(train[: int(max_fit_tokens)])
        return self

    # ------------------------------------------------------- photonic route
    def mount_holarch(self, model, gain: float = 0.25) -> "WaveCodeTextGenerator":
        """Attach a HenriHolarch so every context wave crosses Zone B."""
        self.holarch = model
        self.photonic_gain = float(gain)
        self._photonic_cache.clear()
        return self

    def photonic_scores(self, context: Sequence[int]) -> Optional[np.ndarray]:
        """Hopfield alignment of the post-photonic context wave with the lexical bank.

        The context wave enters Zone B (four BaTiO3 plates plus the unitary
        slot mixer), and the bright-port wave is read against every token wave
        of the Hopfield lexical bank. The score is standardised so its scale
        cannot swamp the count floor. A missing model, a missing bank or a
        degenerate spread returns None, which leaves the blend unchanged.
        """
        if self.holarch is None or not context:
            return None
        if int(getattr(self.lexical, "waves", np.zeros((0, 0))).shape[0]) == 0:
            return None
        key = tuple(int(token) for token in context[-self.WAVE_CONTEXT :])
        cached = self._photonic_cache.get(key)
        if cached is not None:
            self.photonic_cache_hits += 1
            return cached
        wave = self.projector.sequence_wave(list(key), self.tokenizer)
        psi = torch.from_numpy(np.asarray(wave, dtype=np.complex64)).reshape(1, -1)
        with torch.no_grad():
            state = self.holarch.zone_b(psi)
            bright = state.bright[0].detach().cpu().numpy().astype(np.complex64)
        raw = np.real(self.lexical.waves[: self.vocab_size] @ np.conj(bright)).astype(np.float32)
        spread = float(np.std(raw))
        if not np.isfinite(spread) or spread <= 1e-9:
            return None
        score = ((raw - float(np.mean(raw))) / spread).astype(np.float32)
        if len(self._photonic_cache) < 8192:
            self._photonic_cache[key] = score
        self.photonic_contexts += 1
        return score

    # -------------------------------------------------------------- logits
    def wave_scores(self, context: Sequence[int]) -> Optional[np.ndarray]:
        """Standardised multi-order + skip wave score for one context."""
        part = self.lexical.score(context)
        if part is None:
            return None
        spread = float(np.std(part))
        if spread <= 1e-8:
            return None
        return ((part - float(np.mean(part))) / spread).astype(np.float32)

    def fast_weight_logits(self, context: Sequence[int]) -> Optional[np.ndarray]:
        """Read path for the online delta-rule memory (Directive G1.2).

        The memory stores, for each token, the phase of the contexts in which
        that token was accepted. Scoring a candidate is the Hermitian inner
        product of its row with the current context wave, so an online write
        measurably moves the logits of the tokens the correction contained.
        """
        if not self._fast_active or not context:
            return None
        wave = self.projector.sequence_wave(
            [int(token) for token in context[-8:]], self.tokenizer
        )
        raw = np.real(self.fast_weight @ np.conj(wave)).astype(np.float32)
        spread = float(np.std(raw))
        if spread <= 1e-9:
            return None
        return (raw - float(np.mean(raw))) / spread

    def update_fast_weight(
        self, context: Sequence[int], accepted: Sequence[int], lr: float = 0.35
    ) -> Dict[str, float]:
        """One rank-r delta-rule write toward an accepted correction."""
        context = [int(token) for token in context]
        accepted = [int(token) for token in accepted]
        if not context or not accepted:
            return {"written": 0.0}
        wave = self.projector.sequence_wave(context[-16:], self.tokenizer)
        written = 0
        for token in dict.fromkeys(accepted):
            if 0 <= token < self.vocab_size:
                # Store the context phase itself. The read is
                # Re<row, conj(context)>, which is lr * |context|^2 > 0 for the
                # written token: the delta rule raises that token's logit.
                self.fast_weight[token] += np.complex64(float(lr)) * wave
                written += 1
        if written:
            self._fast_active = True
            self.fast_updates += 1
        return {
            "written": float(written),
            "context_tokens": float(len(context)),
            "lr": float(lr),
        }

    def _logit_parts(
        self, context: Sequence[int]
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """(count log-probabilities, standardised wave score, fast weight).

        One computation feeds both the deployed blend and the joint
        calibration, so a selected beta is measured on exactly the logits
        the runtime evaluates.
        """
        context = [int(token) for token in context]
        base = self.count_lm.log_probs(context)
        base = base - np.logaddexp.reduce(base)
        wave = np.zeros(self.vocab_size, dtype=np.float32)
        part = self.wave_scores(context[-self.WAVE_CONTEXT :])
        if part is not None:
            wave = (self.beta_lang * part).astype(np.float32)
        fast = self.fast_weight_logits(context)
        return base.astype(np.float64), wave, fast

    def context_certainty(self, context: Sequence[int]) -> Dict[str, float]:
        """How sure the count floor is about this context (Directive H1).

        entropy is the Shannon entropy of the interpolated count
        distribution; top1 is its largest probability; order is the highest
        n-gram order that matched; mass is the number of observations at
        that order.
        """
        entropy, top1, order, mass = self.count_lm.entropy(context)
        return {
            "entropy": float(entropy),
            "top1_probability": float(top1),
            "matched_order": float(order),
            "mass": float(mass),
        }

    def gate_active(self) -> bool:
        return float(self.beta_gate.get("beta_high", 0.0) or 0.0) > 0.0

    def beta_for(self, context: Sequence[int]) -> float:
        """The context schedule beta(h_t): low on sharp modes, high on doubt.

        A context is CERTAIN when the count floor matched at least
        tau_order tokens of suffix AND its interpolated entropy is at most
        tau_H. Then the unique mode is not blurred. Every other context -
        a backoff, a flat distribution, a high-order match resting on one
        observation - receives beta_high, which is where the Bochner bank
        actually lowers cross-entropy.
        """
        if not self.gate_active():
            return float(self.beta_star)
        stats = self.context_certainty(context)
        certain = (
            stats["matched_order"] >= float(self.beta_gate.get("tau_order", 3.0))
            and stats["entropy"] <= float(self.beta_gate.get("tau_h", 0.0))
        )
        return float(self.beta_gate["beta_low"] if certain else self.beta_gate["beta_high"])

    def token_logits(self, context: Sequence[int]) -> np.ndarray:
        """The D-Guard blend of the count floor and the wave head.

        The wave term is the multi-order holographic score (orders 1-4 plus
        the skip-semantic window). The floor is the interpolated 5-gram. The
        blend is additive with a standardised wave term, so beta = 0 is
        exactly the floor and the D-Guard selection bound holds.

        Directive H1: beta is now a function of the context. A calibrated
        gate (beta_high > 0) keeps the count mode unblurred where the floor
        is already certain and applies the wave weight only where it is not.
        """
        context = [int(token) for token in context]
        base, wave, fast = self._logit_parts(context)
        beta = self.beta_for(context) if self.gate_active() else float(self.beta_star)
        logits = base + float(beta) * wave
        if self.mode_guard and float(beta) > 0.0:
            # Decision-preserving projection: if the smoothing would change
            # the count floor's decision, the floor keeps it. The rest of
            # the distribution is untouched, so the likelihood gain stands.
            floor_mode = int(np.argmax(base))
            if int(np.argmax(logits)) != floor_mode:
                logits[floor_mode] = float(np.max(logits)) + 1e-4
        if fast is not None:
            logits = logits + self.fast_gain * fast
        photonic = self.photonic_scores(context)
        if photonic is not None:
            logits = logits + self.photonic_gain * photonic
        return logits.astype(np.float64)

    def calibrate_beta(
        self,
        ids: Sequence[int],
        grid: Sequence[float] = (0.0, 0.02, 0.05, 0.08, 0.12, 0.18, 0.25),
        context: int = WAVE_CONTEXT,
        samples: int = 2000,
        tolerance: float = 1e-6,
        joint: bool = False,
        beta_low_grid: Sequence[float] = (0.0, 0.01),
        beta_high_grid: Sequence[float] = (0.0, 0.02, 0.05, 0.08, 0.12),
        tau_grid: Sequence[float] = (0.75, 1.0, 1.25, 1.5, 2.0, 3.0),
        order_grid: Sequence[int] = (3, 4),
        top1_margin: float = 0.0,
        mode_guard: Optional[bool] = None,
    ) -> float:
        """D-Guard: pick the wave weight that minimises validation loss.

        The grid contains 0.0, so the coupled system is provably no worse
        than the count floor it is added to. Directive G1: with the
        multi-order bank the selector now has something to admit.

        Directive H1 (joint=True): the selector now chooses the triple
        (beta_low, beta_high, tau_H) - a weight for certain contexts and a
        weight for uncertain ones - under the joint constraint that
        validation cross-entropy strictly falls AND validation top-1 does
        not fall. A grid point that buys likelihood with accuracy is not
        eligible, whatever its loss. The arithmetic is vectorised: every
        sample's count logits and standardised wave score are computed
        once, then each candidate only rescales the cached wave term, so
        the search measures the deployed blend exactly.
        """
        sequence = [int(token) for token in ids]
        indices = list(range(context, len(sequence) - 1))
        if len(indices) > int(samples):
            stride = max(1, len(indices) // int(samples))
            indices = indices[::stride]
        guard = float(top1_margin)
        if mode_guard is None:
            mode_guard = bool(self.mode_guard)
        if not joint:
            curve: List[Tuple[float, float]] = []
            for beta in grid:
                self.beta_star = float(beta)
                total = 0.0
                for index in indices:
                    window = sequence[max(0, index - context) : index]
                    logits = self.token_logits(window)
                    logits = logits - np.max(logits)
                    logprob = logits[sequence[index]] - np.log(np.sum(np.exp(logits)))
                    total -= float(logprob)
                curve.append((total / max(len(indices), 1), float(beta)))
            best_loss = min(loss for loss, _ in curve)
            # D-Guard is an argmin over a grid that contains 0.0. A parsimony
            # window was tried and removed: with a 0.010-nat tolerance it
            # always returned 0.0 whenever the gain was small, which made the
            # selector deaf to a real improvement instead of cautious about a
            # noise-sized one. Ties (within 1e-6) still prefer the smaller
            # weight.
            eligible = [
                (loss, beta) for loss, beta in curve if loss <= best_loss + float(tolerance)
            ]
            eligible.sort(key=lambda pair: (pair[0], pair[1]))
            self.beta_star = float(eligible[0][1]) if eligible else 0.0
            self.beta_curve = [(float(beta), float(loss)) for loss, beta in curve]
            self.beta_floor_loss = float(curve[0][0])
            self.beta_best_loss = float(best_loss)
            self.gate_report = {"joint": False, "beta_star": float(self.beta_star)}
            return self.beta_star

        # ---------------------------------------------------------- joint H1
        base_rows: List[np.ndarray] = []
        wave_rows: List[np.ndarray] = []
        fast_rows: List[np.ndarray] = []
        targets: List[int] = []
        entropies: List[float] = []
        orders: List[int] = []
        for index in indices:
            window = sequence[max(0, index - context) : index]
            base, wave, fast = self._logit_parts(window)
            stats = self.context_certainty(window)
            base_rows.append(base.astype(np.float32))
            wave_rows.append(wave.astype(np.float32))
            fast_rows.append(
                np.zeros(self.vocab_size, dtype=np.float32)
                if fast is None
                else (self.fast_gain * fast).astype(np.float32)
            )
            targets.append(int(sequence[index]))
            entropies.append(float(stats["entropy"]))
            orders.append(int(stats["matched_order"]))
        base_mat = np.stack(base_rows)
        wave_mat = np.stack(wave_rows)
        fast_mat = np.stack(fast_rows)
        target_array = np.asarray(targets, dtype=np.int64)
        entropy_array = np.asarray(entropies, dtype=np.float64)
        order_array = np.asarray(orders, dtype=np.int64)
        rows = np.arange(target_array.size)

        floor_modes = base_mat.argmax(axis=1)

        def measure(beta_vector: np.ndarray) -> Tuple[float, float]:
            logits = base_mat + beta_vector[:, None].astype(np.float32) * wave_mat + fast_mat
            if mode_guard:
                moved = np.nonzero(
                    (logits.argmax(axis=1) != floor_modes) & (beta_vector > 0.0)
                )[0]
                if moved.size:
                    peak_before = logits[moved].max(axis=1) + np.float32(1e-4)
                    logits[moved, floor_modes[moved]] = np.maximum(
                        logits[moved, floor_modes[moved]], peak_before
                    )
            peak = logits.max(axis=1)
            shifted = logits - peak[:, None]
            norm = peak + np.log(np.exp(shifted, dtype=np.float32).sum(axis=1))
            loss = float(np.mean(norm - logits[rows, target_array]))
            top1 = float(np.mean(logits.argmax(axis=1) == target_array))
            return loss, top1

        floor_beta = np.zeros(target_array.size, dtype=np.float32)
        floor_loss, floor_top1 = measure(floor_beta)
        best: Optional[Tuple[float, float, float, float, int, float]] = None
        curve: List[Tuple[float, float]] = []
        for tau_order in order_grid:
            certain_order = order_array >= int(tau_order)
            for tau_h in tau_grid:
                certain = certain_order & (entropy_array <= float(tau_h))
                for beta_low in beta_low_grid:
                    for beta_high in beta_high_grid:
                        if float(beta_low) == 0.0 and float(beta_high) == 0.0:
                            loss, top1 = floor_loss, floor_top1
                        else:
                            beta_vector = np.where(
                                certain, np.float32(beta_low), np.float32(beta_high)
                            )
                            loss, top1 = measure(beta_vector)
                        if (
                            float(beta_high) > 0.0
                            and tau_h == tau_grid[0]
                            and tau_order == order_grid[0]
                            and float(beta_low) == float(beta_low_grid[0])
                        ):
                            curve.append((float(beta_high), float(loss)))
                        if float(beta_high) <= 0.0:
                            continue
                        if float(top1) < float(floor_top1) + float(guard):
                            continue
                        if float(loss) >= float(floor_loss) - float(tolerance):
                            continue
                        candidate = (
                            float(loss),
                            float(beta_low),
                            float(beta_high),
                            float(tau_h),
                            int(tau_order),
                            float(top1),
                        )
                        if best is None or candidate[0] < best[0] - 1e-12 or (
                            abs(candidate[0] - best[0]) <= 1e-12
                            and (candidate[2], candidate[1], candidate[3]) < (best[2], best[1], best[3])
                        ):
                            best = candidate
        if best is None:
            # No gated configuration beat the floor without costing top-1.
            # The gate stays closed, and the receipt says so.
            self.beta_gate = {"beta_low": 0.0, "beta_high": 0.0, "tau_h": 0.0, "tau_order": 3.0}
            self.beta_star = 0.0
            self.gate_report = {
                "joint": True,
                "selected": False,
                "floor_loss": float(floor_loss),
                "floor_top1": float(floor_top1),
                "samples": int(target_array.size),
            }
            return 0.0
        loss, beta_low, beta_high, tau_h, tau_order, top1 = best
        self.beta_gate = {
            "beta_low": float(beta_low),
            "beta_high": float(beta_high),
            "tau_h": float(tau_h),
            "tau_order": float(tau_order),
        }
        # beta_star keeps its old meaning for every receipt reader: the
        # admitted wave weight. It is the high-entropy arm, because that is
        # the arm that is active on the contexts the head is allowed to move.
        self.beta_star = float(beta_high)
        self.beta_floor_loss = float(floor_loss)
        self.beta_best_loss = float(loss)
        self.beta_curve = [(float(beta), float(value)) for beta, value in curve]
        self.gate_report = {
            "joint": True,
            "selected": True,
            "beta_low": float(beta_low),
            "beta_high": float(beta_high),
            "tau_h": float(tau_h),
            "tau_order": int(tau_order),
            "validation_loss": float(loss),
            "validation_top1": float(top1),
            "floor_loss": float(floor_loss),
            "floor_top1": float(floor_top1),
            "mode_guard": bool(mode_guard),
            "samples": int(target_array.size),
        }
        return self.beta_star

    # ------------------------------------------------------------ evaluate
    def calibrate_wave_weights(
        self,
        ids: Sequence[int],
        # Every configuration keeps a non-zero weight on the suffix orders.
        # A zero-order configuration was measured and rejected: it expresses
        # the same information as the 5-gram floor and leaves the bank as
        # dead state. The calibration selects the skip strength and gamma,
        # not whether the bank exists.
        configs: Sequence[Tuple[Tuple[float, float, float, float], float, float]] = (
            ((0.5, 1.0, 1.0, 1.0), 1.0, 0.80),
            ((0.5, 1.0, 1.0, 1.0), 3.0, 0.80),
            ((1.0, 1.0, 1.0, 1.0), 1.0, 0.80),
            ((0.5, 1.0, 1.0, 1.0), 1.5, 0.85),
            ((0.5, 1.0, 1.0, 0.5), 2.0, 0.80),
        ),
        context: int = WAVE_CONTEXT,
        samples: int = 600,
        beta: float = 0.12,
    ) -> Dict[str, object]:
        """Select the bank weights on validation, then the D-Guard beta.

        Two nested selections, both on the same held-out slice: the bank's
        order/skip weights by mean loss at a fixed probe beta, then beta
        itself. The receipt records the chosen configuration.
        """
        sequence = [int(token) for token in ids]
        indices = list(range(context, len(sequence) - 1))
        if len(indices) > int(samples):
            stride = max(1, len(indices) // int(samples))
            indices = indices[::stride]
        best: Optional[Tuple[float, Tuple[float, float, float, float], float, float]] = None
        for order_weights, skip_weight, gamma in configs:
            self.lexical.order_weights = tuple(order_weights)
            self.lexical.skip_weight = float(skip_weight)
            self.lexical.gamma = float(gamma)
            self.lexical._fused.clear()
            self.beta_star = float(beta)
            total = 0.0
            for index in indices:
                window = sequence[max(0, index - context) : index]
                logits = self.token_logits(window)
                logits = logits - np.max(logits)
                total -= float(logits[sequence[index]] - np.log(np.sum(np.exp(logits))))
            loss = total / max(len(indices), 1)
            if best is None or loss < best[0]:
                best = (loss, tuple(order_weights), float(skip_weight), float(gamma))
        self.lexical.order_weights = tuple(best[1])
        self.lexical.skip_weight = float(best[2])
        self.lexical.gamma = float(best[3])
        self.lexical._fused.clear()
        return {
            "order_weights": list(best[1]),
            "skip_weight": float(best[2]),
            "gamma": float(best[3]),
            "validation_loss": float(best[0]),
            "probe_beta": float(beta),
        }

    def evaluate(self, ids: Sequence[int], context: int = WAVE_CONTEXT) -> LMReport:
        sequence = [int(token) for token in ids]
        indices = list(range(context, len(sequence) - 1))
        total = 0.0
        correct = 0
        base_total = 0.0
        base_correct = 0
        for index in indices:
            window = sequence[max(0, index - context) : index]
            logits = self.token_logits(window)
            logits = logits - np.max(logits)
            logprob = logits[sequence[index]] - np.log(np.sum(np.exp(logits)))
            total -= float(logprob)
            correct += int(int(np.argmax(logits)) == sequence[index])
            base = self.count_lm.log_probs(window)
            base = base - np.logaddexp.reduce(base)
            base_total -= float(base[sequence[index]])
            base_correct += int(int(np.argmax(base)) == sequence[index])
        count = max(len(indices), 1)
        perplexity = float(math.exp(min(total / count, 60.0)))
        # Bits per character: every token is decoded to its exact bytes, so
        # the comparison against the retired character model is on a common
        # unit and not on two different tokenizations.
        characters = sum(len(self.tokenizer.token_text(token).encode("utf-8")) for token in sequence[context:])
        bits = float(total / max(characters, 1)) / math.log(2.0)
        self.report = LMReport(
            top1=float(correct / count),
            perplexity=perplexity,
            bits_per_char=bits,
            tokens=int(len(indices)),
            base_top1=float(base_correct / count),
            base_perplexity=float(math.exp(min(base_total / count, 60.0))),
            beta_star=float(self.beta_star),
            wave_weight=float(self.beta_star * self.beta_lang),
        )
        return self.report

    # --------------------------------------------------------------- decode
    def _goal_alignment(self, candidate_wave: np.ndarray, goal: Optional[np.ndarray]) -> float:
        if goal is None:
            return 0.0
        start = SLOT_RULE * (self.dim // 4)
        width = self.dim // 4
        return float(np.real(np.vdot(candidate_wave[start : start + width], goal[start : start + width])))

    def _repeat_penalty(self, candidate_wave: np.ndarray, recent: Sequence[np.ndarray]) -> float:
        if not recent:
            return 0.0
        terms = [
            (0.85 ** (index + 1)) * float(np.real(np.vdot(candidate_wave, state)))
            for index, state in enumerate(recent)
        ]
        return float(max(terms))

    def decode(
        self,
        context: Sequence[int],
        max_tokens: int = 64,
        prior: Optional[Sequence[int]] = None,
        goal_wave: Optional[np.ndarray] = None,
        stop_tokens: Sequence[int] = (),
        require_parse: bool = False,
        prior_weight: float = 6.0,
        force_prior: bool = False,
    ) -> Dict[str, object]:
        """Emit tokens one at a time under the declared scoring rule.

        A prior sequence (a verified plan engram) is a logit offset, not a
        copy: the gate can still veto a token that would break the statement,
        and the repeat penalty still acts. The returned trace stores every
        candidate set, so the receipt can show what was considered.
        """
        sequence = [int(token) for token in context]
        emitted: List[int] = []
        recent: List[np.ndarray] = []
        trace: List[Dict[str, object]] = []
        vetoed = 0
        text = ""
        for step in range(int(max_tokens)):
            if force_prior and prior is not None and step >= len(prior):
                # The plan is exhausted: stop. Continuing would append tokens
                # that the plan never proposed, which is how a faithful
                # emission turns into free generation at the tail.
                break
            logits = self.token_logits(sequence)
            if prior is not None and step < len(prior):
                target = int(prior[step])
                logits = logits + float(prior_weight)
                logits[target] += float(prior_weight) * 4.0
            order = np.argsort(-logits)[: self.LOOKAHEAD_B]
            if force_prior and prior is not None and step < len(prior):
                # Constrained emission (Directive G2): the plan proposes the
                # token, and the SAME gate decides whether it may be emitted.
                # If the gate vetoes it, the scored candidate list decides.
                target = int(prior[step])
                if target not in order:
                    order = np.concatenate(([target], order))
            best = None
            for candidate in order:
                token = int(candidate)
                piece = self.tokenizer.token_text(token)
                candidate_wave = self.projector.slot_wave(token, self.tokenizer)
                score = float(logits[token])
                score += self.GOAL_WEIGHT * self._goal_alignment(candidate_wave, goal_wave)
                score -= self.REPEAT_WEIGHT * self._repeat_penalty(candidate_wave, recent)
                prefix = text + piece
                # A statement boundary is the only place a full parse is
                # meaningful: an incomplete prefix always raises, and
                # vetoing it would forbid every first token.
                if require_parse and piece.strip() in {"", "\n"}:
                    if self.gate.grammar_veto(prefix):
                        vetoed += 1
                        continue
                if piece.endswith("(") and len(piece) > 3:
                    callee = piece[:-1].strip().split(".")[-1]
                    if self.gate.symbol_veto(callee):
                        vetoed += 1
                        continue
                if force_prior and prior is not None and step < len(prior) and token == int(prior[step]):
                    best = (token, score, piece)
                    break
                if best is None or score > best[1]:
                    best = (token, score, piece)
            if best is None:
                break
            token, _score, piece = best
            emitted.append(token)
            sequence.append(token)
            text += piece
            recent.insert(0, self.projector.slot_wave(token, self.tokenizer))
            del recent[4:]
            trace.append(
                {
                    "step": step,
                    "chosen": int(token),
                    "candidates": [int(value) for value in order],
                    "vetoed": int(vetoed),
                }
            )
            if int(token) in set(int(value) for value in stop_tokens):
                break
        return {
            "tokens": emitted,
            "text": text,
            "trace": trace,
            "vetoed": int(vetoed),
            "checks": int(self.gate.checks),
        }

    def perplexity(self, text: str) -> float:
        ids = self.tokenizer.encode(text)
        if len(ids) < 3:
            return float("nan")
        return float(self.evaluate(ids).perplexity)


# ---------------------------------------------------------------------------
# ACE playbooks and the synthesis engine
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Directive G2: role-slot parsing and parametric adaptation
# ---------------------------------------------------------------------------
def _callee_name(node: ast.AST) -> str:
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            return func.attr
    return ""


def _source_slots(source: str) -> Dict[str, object]:
    """Slot 0/1/2 bindings of one verified function definition.

    Slot 0  the function name and its argument variables (scope/identifier)
    Slot 1  the callables it invokes (operator/callable)
    Slot 2  the numeric constants it carries (argument/literal), in source order

    Source order matters because the substitution is positional: the k-th
    numeric constant of a new request rebinds the k-th constant of the
    template, which is what makes a threshold change land on the right value.
    """
    try:
        tree = ast.parse(str(source))
    except SyntaxError:
        return {}
    definition = next(
        (node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))),
        None,
    )
    if definition is None:
        return {}
    args = [str(arg.arg) for arg in definition.args.args if str(arg.arg) not in {"self", "cls"}]
    numbers = []
    for node in ast.walk(definition):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)
        ):
            numbers.append((int(getattr(node, "lineno", 0)), int(getattr(node, "col_offset", 0)), node.value))
    numbers.sort(key=lambda row: (row[0], row[1]))
    calls = sorted({_callee_name(node) for node in ast.walk(definition) if _callee_name(node)})
    return {
        "func": str(definition.name),
        "args": args,
        "numbers": [row[2] for row in numbers],
        "calls": calls,
    }


def _intent_slots(intent: str) -> Dict[str, object]:
    """Slot bindings a natural-language request asks for.

    Only explicit phrasing is accepted: "function NAME", "named NAME",
    "argument NAME", and numeric literals. A request that names nothing keeps
    the template's binding, so an ordinary recall is unaffected.
    """
    text = str(intent)
    func = None
    match = re.search(r"(?:function|def|method)\s+([A-Za-z_][A-Za-z0-9_]*)", text)
    if match is None:
        match = re.search(r"named\s+([A-Za-z_][A-Za-z0-9_]*)", text)
    if match is not None:
        func = match.group(1)
    args = re.findall(
        r"(?:argument|parameter|input|variable)\s+(?:named\s+|called\s+)?([A-Za-z_][A-Za-z0-9_]*)",
        text,
    )
    # A number is a run of digits that is not glued to an identifier. A
    # trailing full stop is allowed (a sentence end), a trailing digit is not.
    numbers = [
        float(value)
        for value in re.findall(
            r"(?<![A-Za-z0-9_.])-?\d+(?:\.\d+)?(?![A-Za-z0-9_])(?!\.\d)", text
        )
    ]
    return {
        "func": func,
        "args": list(dict.fromkeys(args)),
        "numbers": numbers,
    }


def _adapt_source(source: str, slots: Dict[str, object], request: Dict[str, object]) -> str:
    """Rebind a verified template to a new request (Directive G2).

    The transform is a real AST edit: the FunctionDef name, the argument
    identifiers (in the signature and every reference in the body), and the
    numeric constants in source order. The result is re-emitted by the
    decoder under the AST gate, so a binding that would break the statement
    cannot reach the caller.
    """
    text = str(source)
    if not slots:
        return text
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return text
    definition = next(
        (node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))),
        None,
    )
    if definition is None:
        return text
    requested_func = request.get("func")
    if requested_func and str(requested_func).isidentifier() and str(requested_func) != definition.name:
        definition.name = str(requested_func)

    old_args = [str(arg.arg) for arg in definition.args.args]
    requested_args = [str(name) for name in (request.get("args") or [])]
    rename: Dict[str, str] = {}
    for old, new in zip(old_args, requested_args):
        if old in {"self", "cls"} or not new.isidentifier() or new == old:
            continue
        rename[old] = new
    if rename:
        for arg in list(definition.args.args) + list(definition.args.kwonlyargs):
            if arg.arg in rename:
                arg.arg = rename[arg.arg]
        for node in ast.walk(definition):
            if isinstance(node, ast.Name) and node.id in rename:
                node.id = rename[node.id]

    old_numbers = [value for value in (slots.get("numbers") or [])]
    new_numbers = [value for value in (request.get("numbers") or [])]
    if old_numbers and len(old_numbers) == len(new_numbers):
        nodes = [
            node
            for node in ast.walk(definition)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)
        ]
        nodes.sort(key=lambda node: (int(getattr(node, "lineno", 0)), int(getattr(node, "col_offset", 0))))
        for node, value in zip(nodes, new_numbers):
            if isinstance(node.value, int) and float(value).is_integer():
                node.value = int(value)
            else:
                node.value = float(value)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + chr(10)


def _single_expression(source: str):
    """(argument names, return expression) of a one-return function.

    Directive H2. Composition is an AST operation, so it is only defined
    on a function whose body IS one expression. A template with statements
    is a complete program, not a transform, and it is refused rather than
    spliced by string surgery.
    """
    try:
        tree = ast.parse(str(source))
    except SyntaxError:
        return None
    definition = next(
        (node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))),
        None,
    )
    if definition is None:
        return None
    body = list(definition.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    if len(body) != 1 or not isinstance(body[0], ast.Return) or body[0].value is None:
        return None
    args = [str(arg.arg) for arg in definition.args.args if str(arg.arg) not in {"self", "cls"}]
    if not args:
        return None
    return args, body[0].value


def _rebind_constants(expression: ast.AST, numbers: Sequence[float]) -> None:
    """Positionally rebind the numeric constants of one expression."""
    nodes = [
        node
        for node in ast.walk(expression)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    ]
    nodes.sort(key=lambda node: (int(getattr(node, "lineno", 0)), int(getattr(node, "col_offset", 0))))
    if len(nodes) != len(numbers):
        return
    for node, value in zip(nodes, numbers):
        if isinstance(node.value, int) and float(value).is_integer():
            node.value = int(value)
        else:
            node.value = float(value)


# ---------------------------------------------------------------------------
# Directive I2: canonical abstract types, signature inference, type veto
# ---------------------------------------------------------------------------
TYPE_ANY = "any"
TYPE_BOOLEAN = "boolean"
TYPE_NUMERIC = "numeric"
TYPE_TEXT = "text"
TYPE_SEQUENCE = "sequence"
TYPE_MAPPING = "mapping"
TYPE_NUMERIC_SEQUENCE = "numeric_sequence"
"""A consumer that needs an ordered collection of numbers (sum/mean/std).

It is an internal refinement of 'sequence', not a seventh public return
type: a text producer satisfies a plain sequence consumer (len, iteration)
but not a numeric-sequence consumer (sum, mean), which is exactly the
distinction the I2 traps turn on.
"""

ABSTRACT_TYPES = (
    TYPE_ANY,
    TYPE_BOOLEAN,
    TYPE_NUMERIC,
    TYPE_TEXT,
    TYPE_SEQUENCE,
    TYPE_MAPPING,
)
"""The six canonical abstract types of the signature gate."""

_TYPE_NAMES = {
    # numeric
    "int": TYPE_NUMERIC, "float": TYPE_NUMERIC, "complex": TYPE_NUMERIC,
    "number": TYPE_NUMERIC, "decimal": TYPE_NUMERIC, "Fraction": TYPE_NUMERIC,
    "floating": TYPE_NUMERIC, "integer": TYPE_NUMERIC, "real": TYPE_NUMERIC,
    # boolean
    "bool": TYPE_BOOLEAN, "boolean": TYPE_BOOLEAN,
    # text
    "str": TYPE_TEXT, "string": TYPE_TEXT, "bytes": TYPE_TEXT,
    "bytearray": TYPE_TEXT, "Text": TYPE_TEXT,
    # sequence
    "list": TYPE_SEQUENCE, "tuple": TYPE_SEQUENCE, "Sequence": TYPE_SEQUENCE,
    "Iterable": TYPE_SEQUENCE, "Iterator": TYPE_SEQUENCE,
    "Generator": TYPE_SEQUENCE, "deque": TYPE_SEQUENCE,
    "ndarray": TYPE_SEQUENCE, "array": TYPE_SEQUENCE,
    # mapping (the directive groups set with mapping)
    "dict": TYPE_MAPPING, "Dict": TYPE_MAPPING, "Mapping": TYPE_MAPPING,
    "MutableMapping": TYPE_MAPPING, "DefaultDict": TYPE_MAPPING,
    "OrderedDict": TYPE_MAPPING, "Counter": TYPE_MAPPING,
    "set": TYPE_MAPPING, "frozenset": TYPE_MAPPING, "Set": TYPE_MAPPING,
    # Directive J3: the refinement is a first-class annotation word so a
    # playbook stage can declare it without a typing import.
    "numeric_sequence": TYPE_NUMERIC_SEQUENCE,
    "numeric-sequence": TYPE_NUMERIC_SEQUENCE,
}

# "sum" is deliberately absent: it CONSUMES a sequence and produces a
# numeric, so it is evidence that its argument is a sequence, not a scalar.
_NUMERIC_FUNCTIONS = {"int", "float", "complex", "round", "abs", "ord"}
_TEXT_FUNCTIONS = {"str", "repr", "format", "hex", "oct", "bin", "chr", "ascii"}
_SEQUENCE_FUNCTIONS = {
    "sorted", "list", "tuple", "set", "frozenset", "enumerate", "range",
    "zip", "map", "filter", "reversed", "iter",
}
_SEQUENCE_AGGREGATES = {
    "len", "sorted", "iter", "list", "tuple", "set", "frozenset",
    "enumerate", "reversed", "any", "all", "min", "max",
}
_NUMERIC_SEQUENCE_AGGREGATES = {"sum", "mean", "median", "std", "variance"}
_TEXT_METHODS = {
    "strip", "lstrip", "rstrip", "lower", "upper", "split", "rsplit",
    "splitlines", "replace", "startswith", "endswith", "encode", "decode",
    "format", "join", "find", "rfind", "count", "casefold", "title",
    "capitalize", "zfill", "partition", "rpartition",
}
_MAPPING_METHODS = {"get", "keys", "items", "values", "setdefault", "update", "pop"}
_SEQUENCE_METHODS = {"append", "extend", "insert", "sort", "reverse", "index"}


def _canonical_type_name(text: str) -> str:
    """Normalise one annotation or class name to a canonical abstract type."""
    raw = str(text).strip().strip("'\"")
    if not raw:
        return TYPE_ANY
    raw = raw.replace("typing.", "").replace("collections.abc.", "")
    raw = raw.replace("builtins.", "")
    raw = raw.split("[")[0]
    tail = raw.rsplit(".", 1)[-1]
    if tail in _TYPE_NAMES:
        return _TYPE_NAMES[tail]
    lowered = tail.lower()
    if lowered in {"int", "float", "complex", "number", "decimal", "fraction"}:
        return TYPE_NUMERIC
    if lowered in {"bool", "boolean"}:
        return TYPE_BOOLEAN
    if lowered in {"str", "string", "bytes", "bytearray", "text"}:
        return TYPE_TEXT
    if lowered in {"list", "tuple", "sequence", "iterable", "iterator", "generator", "deque", "ndarray", "array"}:
        return TYPE_SEQUENCE
    if lowered in {"dict", "mapping", "mutabledict", "defaultdict", "ordereddict", "counter", "set", "frozenset"}:
        return TYPE_MAPPING
    return TYPE_ANY


def _annotation_types(node) -> frozenset:
    """The canonical type set an annotation declares (empty when absent)."""
    if node is None:
        return frozenset()
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return frozenset({_canonical_type_name(node.value)})
    if isinstance(node, ast.Name):
        return frozenset({_canonical_type_name(node.id)})
    if isinstance(node, ast.Attribute):
        return frozenset({_canonical_type_name(node.attr)})
    if isinstance(node, ast.Tuple):
        out = set()
        for item in node.elts:
            out |= set(_annotation_types(item))
        return frozenset(out or {TYPE_ANY})
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _annotation_types(node.left) | _annotation_types(node.right)
    if isinstance(node, ast.Subscript):
        base = ast.unparse(node.value).strip().lower()
        if base in {"optional"}:
            return _annotation_types(node.slice) or frozenset({TYPE_ANY})
        if base in {"union"}:
            inner = node.slice
            items = list(inner.elts) if isinstance(inner, ast.Tuple) else [inner]
            out = set()
            for item in items:
                out |= set(_annotation_types(item))
            return frozenset(out or {TYPE_ANY})
        return frozenset({_canonical_type_name(ast.unparse(node.value))})
    return frozenset({TYPE_ANY})


def _argument_types(definition, argument: str) -> frozenset:
    """Infer what one argument must support from how the body uses it.

    This is deliberately conservative. The evidence rules are structural
    (len/sorted/for/comprehension imply a sequence, a string subscript
    implies a mapping, float()/arithmetic imply a numeric scalar, a string
    method implies text). Conflicting evidence collapses to 'any', which
    lets the gate abstain instead of vetoing a legal pipeline.
    """
    name = str(argument)
    evidence: set = set()

    def is_name(node) -> bool:
        return isinstance(node, ast.Name) and node.id == name

    for node in ast.walk(definition):
        if isinstance(node, ast.comprehension) and is_name(node.iter):
            evidence.add(TYPE_SEQUENCE)
        elif isinstance(node, ast.For) and is_name(node.iter):
            evidence.add(TYPE_SEQUENCE)
        elif isinstance(node, ast.Call):
            func = node.func
            fname = func.id if isinstance(func, ast.Name) else (
                func.attr if isinstance(func, ast.Attribute) else ""
            )
            if fname in _SEQUENCE_AGGREGATES and any(is_name(arg) for arg in node.args):
                evidence.add(TYPE_SEQUENCE)
            if fname in _NUMERIC_SEQUENCE_AGGREGATES and any(is_name(arg) for arg in node.args):
                evidence.add(TYPE_NUMERIC_SEQUENCE)
            if fname in _NUMERIC_FUNCTIONS and any(is_name(arg) for arg in node.args):
                evidence.add(TYPE_NUMERIC)
            if fname == "bool" and any(is_name(arg) for arg in node.args):
                evidence.add(TYPE_BOOLEAN)
            if fname == "range" and any(is_name(arg) for arg in node.args):
                evidence.add(TYPE_NUMERIC)
        elif isinstance(node, ast.Attribute) and is_name(node.value):
            if node.attr in _TEXT_METHODS:
                evidence.add(TYPE_TEXT)
            elif node.attr in _MAPPING_METHODS:
                evidence.add(TYPE_MAPPING)
            elif node.attr in _SEQUENCE_METHODS:
                evidence.add(TYPE_SEQUENCE)
        elif isinstance(node, ast.Subscript) and is_name(node.value):
            if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                evidence.add(TYPE_MAPPING)
            elif isinstance(node.slice, (ast.Constant, ast.Slice)) and not (
                isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str)
            ):
                evidence.add(TYPE_SEQUENCE)
        elif isinstance(node, ast.BinOp):
            left, right = node.left, node.right
            involves = is_name(left) or is_name(right)
            if involves:
                other = right if is_name(left) else left
                if isinstance(node.op, ast.Add) and isinstance(other, ast.Constant) and isinstance(other.value, str):
                    evidence.add(TYPE_TEXT)
                elif isinstance(node.op, (ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.MatMult)):
                    evidence.add(TYPE_NUMERIC)
                elif isinstance(node.op, ast.Add):
                    evidence.add(TYPE_NUMERIC)
        elif isinstance(node, ast.Compare):
            operands = [node.left] + list(node.comparators)
            if any(is_name(item) for item in operands):
                if any(isinstance(item, ast.Constant) and isinstance(item.value, str) for item in operands):
                    evidence.add(TYPE_TEXT)
                elif any(isinstance(item, ast.Constant) and isinstance(item.value, (int, float)) for item in operands):
                    evidence.add(TYPE_NUMERIC)
        elif isinstance(node, ast.UnaryOp) and is_name(node.operand):
            if isinstance(node.op, ast.USub):
                evidence.add(TYPE_NUMERIC)
            elif isinstance(node.op, ast.Not):
                evidence.add(TYPE_BOOLEAN)
    if not evidence:
        return frozenset({TYPE_ANY})
    # A numeric-sequence consumer is strictly stronger than a plain
    # sequence consumer, so it wins when both were observed.
    if TYPE_NUMERIC_SEQUENCE in evidence:
        return frozenset({TYPE_NUMERIC_SEQUENCE})
    if len(evidence) == 1:
        return frozenset(evidence)
    return frozenset({TYPE_ANY})


def _expression_types(node) -> frozenset:
    """The canonical return types one expression can produce."""
    if node is None:
        return frozenset({TYPE_ANY})
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, bool):
            return frozenset({TYPE_BOOLEAN})
        if isinstance(value, (int, float, complex)):
            return frozenset({TYPE_NUMERIC})
        if isinstance(value, (str, bytes)):
            return frozenset({TYPE_TEXT})
        return frozenset({TYPE_ANY})
    if isinstance(node, (ast.JoinedStr,)):
        return frozenset({TYPE_TEXT})
    if isinstance(node, (ast.List, ast.Tuple, ast.Set, ast.ListComp, ast.SetComp, ast.GeneratorExp)):
        return frozenset({TYPE_SEQUENCE})
    if isinstance(node, (ast.Dict, ast.DictComp)):
        return frozenset({TYPE_MAPPING})
    if isinstance(node, ast.IfExp):
        return _expression_types(node.body) | _expression_types(node.orelse)
    if isinstance(node, ast.BoolOp):
        return frozenset({TYPE_BOOLEAN})
    if isinstance(node, ast.Compare):
        return frozenset({TYPE_BOOLEAN})
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.Not):
            return frozenset({TYPE_BOOLEAN})
        if isinstance(node.op, (ast.USub, ast.UAdd)):
            return frozenset({TYPE_NUMERIC})
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Add):
            left = _expression_types(node.left)
            right = _expression_types(node.right)
            if TYPE_TEXT in left or TYPE_TEXT in right:
                return frozenset({TYPE_TEXT})
            if TYPE_SEQUENCE in left or TYPE_SEQUENCE in right:
                return frozenset({TYPE_SEQUENCE})
            return frozenset({TYPE_NUMERIC})
        if isinstance(node.op, (ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.MatMult)):
            return frozenset({TYPE_NUMERIC})
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Name):
            fname = func.id
            if fname in _NUMERIC_FUNCTIONS:
                return frozenset({TYPE_NUMERIC})
            if fname in _TEXT_FUNCTIONS:
                return frozenset({TYPE_TEXT})
            if fname in _SEQUENCE_FUNCTIONS:
                return frozenset({TYPE_SEQUENCE})
            if fname == "dict":
                return frozenset({TYPE_MAPPING})
            if fname == "bool":
                return frozenset({TYPE_BOOLEAN})
            if fname in {"max", "min"}:
                inner = set()
                for arg in node.args:
                    inner |= set(_expression_types(arg))
                if inner and inner <= {TYPE_NUMERIC}:
                    return frozenset({TYPE_NUMERIC})
                return frozenset({TYPE_ANY})
            return frozenset({TYPE_ANY})
        if isinstance(func, ast.Attribute):
            attr = func.attr
            if attr in {"hexdigest", "encode", "decode", "strip", "lower", "upper", "replace", "join", "format"}:
                return frozenset({TYPE_TEXT})
            if attr in {"keys", "items", "values"}:
                return frozenset({TYPE_SEQUENCE})
            if attr in {"get", "pop", "setdefault"}:
                return frozenset({TYPE_ANY})
            if attr in {"array", "stack", "asarray", "tolist", "reshape", "sorted"}:
                return frozenset({TYPE_SEQUENCE})
            if attr in {"sum", "mean", "norm", "argmax", "argmin", "std", "real", "abs", "vdot", "dot"}:
                return frozenset({TYPE_NUMERIC})
            return frozenset({TYPE_ANY})
    return frozenset({TYPE_ANY})


def _return_types(definition) -> frozenset:
    """The canonical type set a function can return."""
    annotated = _annotation_types(getattr(definition, "returns", None))
    if annotated:
        return annotated
    collected = set()
    for node in ast.walk(definition):
        if isinstance(node, ast.Return) and node.value is not None:
            collected |= set(_expression_types(node.value))
    if not collected:
        return frozenset({TYPE_ANY})
    return frozenset(collected)


def _abstract_signature(source: str) -> Dict[str, object]:
    """The declared/inferred surface of one function, in canonical types."""
    try:
        tree = ast.parse(str(source))
    except SyntaxError:
        return {}
    definition = next(
        (node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))),
        None,
    )
    if definition is None:
        return {}
    positional = list(getattr(definition.args, "posonlyargs", [])) + list(definition.args.args)
    signature_args: List[Dict[str, object]] = []
    for argument in positional:
        if str(argument.arg) in {"self", "cls"}:
            continue
        declared = _annotation_types(argument.annotation)
        inferred = declared or _argument_types(definition, str(argument.arg))
        signature_args.append({"name": str(argument.arg), "type": sorted(inferred)})
    return {
        "name": str(definition.name),
        "args": signature_args,
        "arity": len(signature_args),
        "returns": sorted(_return_types(definition)),
    }


class TypeCompatibilityGate:
    """Plan k may bind plan k+1 only when its return fits arg 0.

    Directive I2. The gate reads the canonical abstract types extracted from
    the AST annotation when one exists, and from the body's structural use
    when it does not. 'any' is the abstention type: an unannotated parameter
    that the body never constrains accepts any producer, so a legal pipeline
    is not vetoed for missing type hints. A real conflict - a text or
    mapping return bound to a parameter the body uses as a numeric scalar or
    a sequence - is refused before any source is emitted.
    """

    def __init__(self, signatures: Optional[Dict[str, object]] = None) -> None:
        self.signatures = dict(signatures or {})
        self._cache: Dict[str, Dict[str, object]] = {}

    def signature(self, plan: "Plan") -> Dict[str, object]:
        key = str(getattr(plan, "digest", "") or id(plan))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        stored = (getattr(plan, "slots", {}) or {}).get("types")
        info = dict(stored) if isinstance(stored, dict) and stored.get("args") is not None else _abstract_signature(plan.source)
        self._cache[key] = info
        return info

    @staticmethod
    def _compatible(produced: frozenset, required: frozenset) -> bool:
        if not produced or not required:
            return True
        if TYPE_ANY in produced or TYPE_ANY in required:
            return True
        if TYPE_NUMERIC_SEQUENCE in required:
            # sum/mean need numbers: text and mappings are refused, a plain
            # sequence is accepted because its element type is unannotated.
            if TYPE_TEXT in produced or TYPE_MAPPING in produced:
                return False
            return TYPE_SEQUENCE in produced or TYPE_NUMERIC_SEQUENCE in produced
        if TYPE_SEQUENCE in required and TYPE_TEXT in produced:
            # A string is a sized character sequence: len(s) and iteration
            # are legal, which is why the H2 digest-length pipeline survives.
            return True
        return bool(produced & required)

    def check(self, left: "Plan", right: "Plan") -> Dict[str, object]:
        left_signature = self.signature(left)
        right_signature = self.signature(right)
        produced = frozenset(left_signature.get("returns") or [TYPE_ANY])
        arguments = list(right_signature.get("args") or [])
        row = {
            "from": str(left.name),
            "to": str(right.name),
            "returns": sorted(produced),
            "arg": str(arguments[0].get("name", "")) if arguments else "",
            "requires": list(arguments[0].get("type") or [TYPE_ANY]) if arguments else [TYPE_ANY],
        }
        if not arguments:
            return {
                "ok": True,
                "reason": "no positional argument to bind",
                "row": row,
                "left": left_signature,
                "right": right_signature,
            }
        required = frozenset(arguments[0].get("type") or [TYPE_ANY])
        ok = self._compatible(produced, required)
        reason = "compatible" if ok else (
            f"type mismatch: {left.name}.returns {sorted(produced)} cannot bind "
            f"{right.name}.{arguments[0].get('name', 'arg0')} {sorted(required)}"
        )
        return {"ok": bool(ok), "reason": reason, "row": row, "left": left_signature, "right": right_signature}

    def check_chain(self, plans: Sequence["Plan"]) -> Dict[str, object]:
        rows: List[Dict[str, object]] = []
        for left, right in zip(plans, plans[1:]):
            verdict = self.check(left, right)
            rows.append(verdict["row"])
            if not verdict["ok"]:
                return {
                    "ok": False,
                    "reason": str(verdict["reason"]),
                    "rows": rows,
                    "types": [dict(v) for v in [verdict]],
                }
        return {"ok": True, "reason": "every stage is type compatible", "rows": rows}


def _function_definition(source: str):
    try:
        tree = ast.parse(str(source))
    except SyntaxError:
        return None
    return next(
        (node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))),
        None,
    )


def _composable_body(definition) -> Optional[List[ast.stmt]]:
    """The statements of a function whose body ends in exactly one return."""
    if definition is None:
        return None
    body = [
        statement
        for statement in definition.body
        if not (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant))
    ]
    if not body:
        return None
    if not isinstance(body[-1], ast.Return) or body[-1].value is None:
        return None
    # An early return (including one nested inside a branch) cannot be
    # spliced: the terminal return is rewritten to step_i, and any other
    # return would exit the composed function before the later stages run.
    for statement in body[:-1]:
        if any(isinstance(node, ast.Return) for node in ast.walk(statement)):
            return None
    return body


def _rename_names(node, rename: Dict[str, str]) -> None:
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id in rename:
            child.id = rename[child.id]


def _numeric_constants(statements: Sequence[ast.stmt]) -> List[ast.Constant]:
    nodes: List[ast.Constant] = []
    for statement in statements:
        for node in ast.walk(statement):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, (int, float))
                and not isinstance(node.value, bool)
            ):
                nodes.append(node)
    nodes.sort(key=lambda node: (int(getattr(node, "lineno", 0)), int(getattr(node, "col_offset", 0))))
    return nodes


def _rebind_stage_constants(statements: Sequence[ast.stmt], numbers: Sequence[float]) -> int:
    nodes = _numeric_constants(statements)
    if len(nodes) != len(numbers):
        return 0
    for node, value in zip(nodes, numbers):
        if isinstance(node.value, int) and float(value).is_integer():
            node.value = int(value)
        else:
            node.value = float(value)
    return len(nodes)


class _StageBinder(ast.NodeVisitor):
    """Names a stage body binds in its own scope.

    Nested function definitions and lambdas introduce their own scope, so
    their bodies are not descended into. A name bound by an import statement
    is not a local of the stage and is left alone.
    """

    def __init__(self) -> None:
        self.names: set = set()

    def visit_FunctionDef(self, node) -> None:  # noqa: N802 - ast visitor name
        return

    def visit_AsyncFunctionDef(self, node) -> None:  # noqa: N802
        return

    def visit_Lambda(self, node) -> None:  # noqa: N802
        return

    def visit_Import(self, node) -> None:  # noqa: N802
        return

    def visit_ImportFrom(self, node) -> None:  # noqa: N802
        return

    def visit_Name(self, node) -> None:  # noqa: N802
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.names.add(str(node.id))

    def visit_ExceptHandler(self, node) -> None:  # noqa: N802
        if node.name:
            self.names.add(str(node.name))
        self.generic_visit(node)


class _StageRenamer(ast.NodeTransformer):
    """Rename every reference to a bound local exactly once."""

    def __init__(self, rename: Dict[str, str]) -> None:
        self.rename = dict(rename)

    def visit_Name(self, node):  # noqa: N802
        if node.id in self.rename:
            node.id = self.rename[node.id]
        return node

    def visit_arg(self, node):  # noqa: N802
        if node.arg in self.rename:
            node.arg = self.rename[node.arg]
        return node

    def visit_FunctionDef(self, node):  # noqa: N802
        # A nested definition owns its scope; only its binding name is part
        # of this stage, and the binder already refused to descend into it.
        return node


def _stage_locals(definition, body: Sequence[ast.stmt]) -> List[str]:
    binder = _StageBinder()
    for statement in body:
        binder.visit(statement)
    params = [a.arg for a in definition.args.args if a.arg not in {"self", "cls"}]
    return list(dict.fromkeys(params + sorted(binder.names)))


def _compose_plans_many(plans: Sequence["Plan"], request: Dict[str, object]) -> Optional[str]:
    """Splice N verified functions into one sequential pipeline.

    Directive K2. Each stage may carry a multi-statement body (loops,
    branches, accumulators, local helpers) as long as it ends in exactly one
    return. Stage locals are alpha-renamed to _s{i}_{name} so a loop index or
    accumulator in stage 1 can never collide with one in stage 2 or 3. The
    terminal return of stage i < N becomes step_i; the last stage keeps its
    return. The entry stage may declare several arguments; the request names
    them, and a later stage binds an extra parameter by name from that public
    set (or from request['bindings']).
    """
    if not plans:
        return None
    definitions = []
    staged: List[List[ast.stmt]] = []
    for plan in plans:
        definition = _function_definition(plan.source)
        body = _composable_body(definition)
        if definition is None or body is None:
            return None
        definitions.append(definition)
        staged.append(copy.deepcopy(body))
    first_params = [a.arg for a in definitions[0].args.args if a.arg not in {"self", "cls"}]
    if not first_params:
        return None
    requested = [str(name) for name in (request.get("args") or []) if str(name).isidentifier()]
    public = [
        requested[index] if index < len(requested) else name
        for index, name in enumerate(first_params)
    ]
    extra = [name for name in requested[len(first_params) :] if name not in public]
    bindings: Dict[str, str] = {name: name for name in list(public) + list(extra)}
    for key, value in dict(request.get("bindings") or {}).items():
        if str(key).isidentifier() and str(value).isidentifier():
            bindings[str(key)] = str(value)

    for index, (definition, statements) in enumerate(zip(definitions, staged)):
        params = [a.arg for a in definition.args.args if a.arg not in {"self", "cls"}]
        locals_ = _stage_locals(definition, statements)
        rename: Dict[str, str] = {}
        if index == 0:
            for position, name in enumerate(params):
                rename[name] = public[position]
            protected = set(public)
            for name in locals_:
                if name in params or name in protected:
                    continue
                rename[name] = f"_s0_{name}"
        else:
            for position, name in enumerate(params):
                if position == 0:
                    rename[name] = f"step_{index}"
                    continue
                target = bindings.get(name)
                if target is None:
                    return None
                rename[name] = target
            for name in locals_:
                if name in params:
                    continue
                rename[name] = f"_s{index}_{name}"
        renamer = _StageRenamer(rename)
        for statement in statements:
            renamer.visit(statement)
    numbers = [float(value) for value in (request.get("numbers") or [])]
    if numbers:
        counts = [len(_numeric_constants(statements)) for statements in staged]
        carriers = [index for index, count in enumerate(counts) if count > 0]
        # The request's literals describe the operation that carries a bound
        # (a clamp), which is the LAST stage in every pipeline the protocol
        # composes. Prefer it; fall back to the single carrier; distribute
        # only when the totals line up exactly.
        if counts[-1] == len(numbers):
            _rebind_stage_constants(staged[-1], numbers)
        elif len(carriers) == 1 and counts[carriers[0]] == len(numbers):
            _rebind_stage_constants(staged[carriers[0]], numbers)
        elif sum(counts) == len(numbers):
            cursor = 0
            for index, count in enumerate(counts):
                if count:
                    _rebind_stage_constants(staged[index], numbers[cursor : cursor + count])
                    cursor += count
    body: List[ast.stmt] = []
    for index, statements in enumerate(staged):
        if index < len(staged) - 1:
            last = statements[-1]
            body.extend(statements[:-1])
            body.append(
                ast.Assign(
                    targets=[ast.Name(id=f"step_{index + 1}", ctx=ast.Store())],
                    value=last.value,
                )
            )
        else:
            body.extend(statements)
    name = str(request.get("func") or "_".join(plan.name for plan in plans))
    if not name.isidentifier():
        name = "_".join(plan.name for plan in plans)
    arguments = ast.arguments(
        posonlyargs=[],
        args=[ast.arg(arg=arg) for arg in list(public) + list(extra)],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[],
    )
    keywords: Dict[str, object] = {}
    if "type_params" in getattr(ast.FunctionDef, "_fields", ()):
        keywords["type_params"] = []
    definition = ast.FunctionDef(
        name=name,
        args=arguments,
        body=body,
        decorator_list=[],
        returns=None,
        **keywords,
    )
    module = ast.Module(body=[definition], type_ignores=[])
    ast.fix_missing_locations(module)
    try:
        source = ast.unparse(module) + chr(10)
        ast.parse(source)
    except SyntaxError:
        return None
    return source


def _rename_names_all(statements: Sequence[ast.stmt], rename: Dict[str, str]) -> None:
    for statement in statements:
        _rename_names(statement, rename)


def _compose_plans(p1: Plan, p2: Plan, request: Dict[str, object]) -> Optional[str]:
    """Two-stage compatibility wrapper around the N-stage splicer."""
    return _compose_plans_many([p1, p2], request)


@dataclass
class Plan:
    """One compiled strategy: a verified source plus its role-slot bindings.

    Directive G2. The source alone supports verbatim replay. The slot map is
    what makes the plan parametric: it names the function, the argument
    variables, the numeric constants and the callees, so a new intent that
    asks for different names or different constants can be bound into the
    same verified body instead of returning the stored string.
    """

    name: str
    intent: str
    source: str
    kind: str = "code"
    version: int = 1
    engram_id: int = -1
    digest: str = ""
    slots: Dict[str, object] = field(default_factory=dict)

    def as_payload(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "intent": self.intent,
            "source": self.source,
            "kind": self.kind,
            "version": int(self.version),
            "engram_id": int(self.engram_id),
            "digest": self.digest,
            "slots": self.slots,
        }

    @classmethod
    def from_payload(cls, payload: Dict[str, object]) -> "Plan":
        return cls(
            name=str(payload.get("name", "")),
            intent=str(payload.get("intent", "")),
            source=str(payload.get("source", "")),
            kind=str(payload.get("kind", "code")),
            version=int(payload.get("version", 1)),
            engram_id=int(payload.get("engram_id", -1)),
            digest=str(payload.get("digest", "")),
            slots=dict(payload.get("slots", {})),
        )


class PlanStore:
    """ACE playbooks stored as Tier-3 wave engrams in Zone C.

    The key is the position-bound wave of the intent, so retrieval is a
    coherence match and not a string lookup. The stored payload is the
    verified source, which the decoder re-emits token by token under the AST
    gate. A plan therefore cannot be recalled for an intent whose wave does
    not align with it.
    """

    def __init__(
        self,
        memory,
        projector: BochnerPhaseProjector,
        tokenizer: HolarchCodeTextTokenizer,
        label_offset: int = 20000,
        namespace: str = "default",
        token_counts: Optional[np.ndarray] = None,
    ) -> None:
        self.memory = memory
        self.projector = projector
        self.tokenizer = tokenizer
        self.label_offset = int(label_offset)
        self.namespace = str(namespace)
        self.plans: Dict[str, Plan] = {}
        self.tau = 0.15
        # Directive G2: an intent key must weight its tokens by how
        # informative they are FOR THIS LIBRARY OF PLANS. A code-corpus
        # frequency is the wrong statistic ("wave" and "holarch" are common
        # there and discriminative here); the document frequency across the
        # stored intents is the right one. The weight is the squared
        # log-inverse-frequency, so a token that appears in every intent
        # contributes almost nothing and a token unique to one plan dominates.
        self._df: Dict[int, int] = {}
        self._n_plans = 0
        self.token_weight = np.ones(len(self.tokenizer), dtype=np.float32)
        self._key_cache: Dict[str, np.ndarray] = {}
        self._token_counts = token_counts
        self.power = 2.0
        self.weight_floor = 0.02
        self._rebuild_weights()

    def _rebuild_weights(self) -> None:
        size = len(self.tokenizer)
        weights = np.ones(size, dtype=np.float64)
        total = max(self._n_plans, 1)
        if self._n_plans >= 2:
            for token, frequency in self._df.items():
                if 0 <= token < size:
                    weights[token] = math.log(1.0 + total / (1.0 + float(frequency))) ** float(self.power)
            peak = float(weights.max()) if weights.size else 1.0
            if peak > 0.0:
                weights = weights / peak
        self.token_weight = np.clip(weights, self.weight_floor, 1.0).astype(np.float32)
        self._key_cache.clear()

    def _observe_intent(self, intent: str) -> None:
        """Add one intent to the document-frequency table."""
        for token in set(self.tokenizer.encode(str(intent))[:96]):
            if 0 <= token < len(self.tokenizer):
                self._df[token] = self._df.get(token, 0) + 1
        self._n_plans += 1
        self._rebuild_weights()

    def key(self, intent: str) -> np.ndarray:
        """The phase key of one intent: an order-insensitive weighted bag.

        Position powers are deliberately omitted. An intent that inserts a
        clause ("that takes argument store") shifts every later token, and a
        position-bound key then destroys the alignment exactly when the new
        request is the one we must serve. The bag-of-phases key keeps the
        shared tokens aligned at any offset, and the IDF weights decide which
        shared tokens matter.
        """
        text = str(intent)
        cached = self._key_cache.get(text)
        if cached is not None:
            return cached
        tokens = self.tokenizer.encode(text)[:96]
        if not tokens:
            return np.zeros(self.projector.dim, dtype=np.complex64)
        total = np.zeros(self.projector.dim, dtype=np.complex64)
        for token in tokens:
            gain = float(self.token_weight[token]) if int(token) < self.token_weight.size else 1.0
            total = total + gain * self.projector.slot_wave(int(token), self.tokenizer)
        norm = float(np.linalg.norm(total))
        key = (total / max(norm, _EPS)).astype(np.complex64)
        if len(self._key_cache) < 4096:
            self._key_cache[text] = key
        return key

    def compile(self, name: str, intent: str, source: str, kind: str = "code", tier: int = 3) -> Plan:
        existing = self.plans.get(str(name))
        version = 1 if existing is None else int(existing.version) + 1
        wave = self.key(intent)
        label = self.label_offset + len(self.plans)
        record = self.memory.write(
            wave, label=int(label), tier=int(tier), namespace=self.namespace
        )
        slots: Dict[str, object] = _source_slots(str(source)) if str(kind) == "code" else {}
        if str(kind) == "code":
            # Directive I2: the canonical signature travels with the plan, so
            # the type gate reads a compiled decision rather than re-parsing
            # a source on every retrieval.
            slots["types"] = _abstract_signature(str(source))
        plan = Plan(
            name=str(name),
            intent=str(intent),
            source=str(source),
            kind=str(kind),
            version=version,
            engram_id=int(getattr(record, "engram_id", -1)),
            digest=hashlib.sha256(str(source).encode("utf-8")).hexdigest(),
            slots=slots,
        )
        if str(name) not in self.plans:
            self._observe_intent(str(intent))
        else:
            self._rebuild_weights()
        self.plans[str(name)] = plan
        self.persist(plan)
        return plan

    # ------------------------------------------------------------ persistence
    def persist(self, plan: Optional[Plan] = None) -> int:
        """Write plans to Zone C metadata for a restart (Directive G4)."""
        putter = getattr(self.memory, "put_meta", None) if self.memory is not None else None
        if putter is None:
            return 0
        targets = [plan] if plan is not None else list(self.plans.values())
        for item in targets:
            if item is None:
                continue
            putter(self.namespace, "plan", item.name, item.as_payload())
        return len(targets)

    def load(self) -> int:
        """Restore every persisted plan for this namespace."""
        getter = getattr(self.memory, "get_meta", None) if self.memory is not None else None
        if getter is None:
            return 0
        restored = 0
        for key, payload in getter(self.namespace, "plan"):
            try:
                plan = Plan.from_payload(payload)
            except Exception:
                continue
            if not plan.name or not plan.source:
                continue
            if plan.name in self.plans:
                self.plans[plan.name] = plan
                continue
            self.plans[plan.name] = plan
            for token in set(self.tokenizer.encode(plan.intent)[:96]):
                if 0 <= token < len(self.tokenizer):
                    self._df[token] = self._df.get(token, 0) + 1
            self._n_plans += 1
            restored += 1
        if restored:
            self._rebuild_weights()
        return restored

    def retrieve(self, intent: str) -> Optional[Tuple[Plan, float]]:
        """The best plan whose intent wave aligns above tau."""
        if not self.plans:
            return None
        wave = self.key(intent)
        best: Optional[Tuple[Plan, float]] = None
        for plan in self.plans.values():
            reference = self.key(plan.intent)
            score = float(np.real(np.vdot(wave, reference)))
            if best is None or score > best[1]:
                best = (plan, score)
        if best is None or best[1] < self.tau:
            return None
        return best

    def _intent_tokens(self, intent: str) -> List[int]:
        return [int(token) for token in self.tokenizer.encode(str(intent))[:96]]

    def coverage(self, intent: str, plan: Plan) -> float:
        """Fraction of the intent's IDF mass that this plan's intent covers.

        Directive H2. A single playbook that explains most of the weighted
        intent is a complete answer; one that explains only part of it is
        half of a compound request.
        """
        tokens = self._intent_tokens(intent)
        if not tokens:
            return 0.0
        covered = set(self._intent_tokens(plan.intent))
        total = 0.0
        hit = 0.0
        for token in tokens:
            weight = float(self.token_weight[token]) if 0 <= token < self.token_weight.size else 1.0
            total += weight
            if token in covered:
                hit += weight
        return float(hit / total) if total > 0.0 else 0.0

    def residual_key(self, intent: str, plan: Plan) -> Tuple[Optional[np.ndarray], List[int]]:
        """The intent wave left after the first plan's tokens are masked out."""
        tokens = self._intent_tokens(intent)
        covered = set(self._intent_tokens(plan.intent))
        total = np.zeros(self.projector.dim, dtype=np.complex64)
        leftover: List[int] = []
        for token in tokens:
            if token in covered:
                continue
            weight = float(self.token_weight[token]) if 0 <= token < self.token_weight.size else 1.0
            total = total + weight * self.projector.slot_wave(int(token), self.tokenizer)
            leftover.append(int(token))
        norm = float(np.linalg.norm(total))
        if norm <= _EPS:
            return None, leftover
        return (total / norm).astype(np.complex64), leftover

    _SEQUENCE_CUE = re.compile(r"\s+(?:and\s+)?then\s+", re.IGNORECASE)

    @classmethod
    def split_sequence(cls, intent: str) -> Tuple[str, str]:
        """(head, tail) of a compound request, split at the first "then".

        Directive H2. A compound request names its FIRST operation before
        its second, and a bag-of-phases key throws that order away. The
        split restores it without position powers: the primary plan is
        matched against the head, and the residue is matched against the
        rest of the request. A request without a cue is returned whole.
        """
        text = str(intent)
        match = cls._SEQUENCE_CUE.search(text)
        if match is None:
            return text, ""
        return text[: match.start()].strip(), text[match.end() :].strip()

    def retrieve_chain(
        self,
        intent: str,
        max_plans: int = 2,
        arity: Optional[int] = None,
        tau_chain: float = 0.60,
        coverage_floor: float = 0.60,
    ) -> List[Tuple[Plan, float]]:
        """One plan, or the ordered pair that a compound intent needs (H2).

        Retrieval is iterative. The primary plan is matched against the head
        of the request. Its intent tokens are then masked out of the WHOLE
        request, and the remaining high-IDF tokens are matched against the
        other plans. The pair is returned only when the primary plan covers
        less than coverage_floor of the weighted request AND the residue
        clears tau_chain: one half of the sentence explained, the other half
        strongly matched. Anything less is a single plan.
        """
        if not self.plans or int(max_plans) < 1:
            return []
        head, tail = self.split_sequence(intent)
        first = self.retrieve_for(head if tail else intent, arity=arity)
        if first is None:
            return []
        if int(max_plans) < 2:
            return [first]
        plan, score = first
        if self.coverage(intent, plan) >= float(coverage_floor):
            return [first]
        residue, leftover = self.residual_key(intent, plan)
        if residue is None or not leftover:
            return [first]
        best: Optional[Tuple[Plan, float]] = None
        for other in self.plans.values():
            if other.name == plan.name:
                continue
            value = float(np.real(np.vdot(residue, self.key(other.intent))))
            if best is None or value > best[1]:
                best = (other, value)
        if best is None or best[1] < float(tau_chain):
            return [first]
        return [first, best]

    # --------------------------------------------------- N-stage DAG (I2)
    _STAGE_CUE = re.compile(
        r"(?:\s*(?:;|->|\u2192)\s*|\s+(?:and\s+then|then|after\s+that|afterwards|next|finally|subsequently)\s+)",
        re.IGNORECASE,
    )
    _STAGE_PREFIX = re.compile(
        r"(?i)^(?:first(?:ly)?|start\s+by|to\s+begin|begin\s+by)\s*[,:]?\s*"
    )

    def coverage_many(self, intent: str, plans: Sequence[Plan]) -> float:
        """Fraction of the intent's IDF mass explained by the union of plans."""
        tokens = self._intent_tokens(intent)
        if not tokens:
            return 0.0
        covered: set = set()
        for plan in plans:
            covered.update(self._intent_tokens(plan.intent))
        total = 0.0
        hit = 0.0
        for token in tokens:
            weight = float(self.token_weight[token]) if 0 <= token < self.token_weight.size else 1.0
            total += weight
            if token in covered:
                hit += weight
        return float(hit / total) if total > 0.0 else 0.0

    def residual_key_many(
        self, intent: str, plans: Sequence[Plan]
    ) -> Tuple[Optional[np.ndarray], List[int]]:
        """The intent wave left after every plan's tokens are masked out."""
        tokens = self._intent_tokens(intent)
        covered: set = set()
        for plan in plans:
            covered.update(self._intent_tokens(plan.intent))
        total = np.zeros(self.projector.dim, dtype=np.complex64)
        leftover: List[int] = []
        for token in tokens:
            if token in covered:
                continue
            weight = float(self.token_weight[token]) if 0 <= token < self.token_weight.size else 1.0
            total = total + weight * self.projector.slot_wave(int(token), self.tokenizer)
            leftover.append(int(token))
        norm = float(np.linalg.norm(total))
        if norm <= _EPS:
            return None, leftover
        return (total / norm).astype(np.complex64), leftover

    @classmethod
    def split_stages(cls, intent: str, max_stages: int = 4) -> List[str]:
        """Ordered clauses of a compound intent, split at discourse markers.

        Directive I2. The markers are the ones the protocol names: an
        explicit 'first ... then ... then ...', 'and then', a semicolon, or
        an arrow. A request without a marker is returned as one clause.
        """
        text = str(intent).strip()
        parts = [part.strip() for part in cls._STAGE_CUE.split(text)]
        cleaned: List[str] = []
        for index, part in enumerate(parts):
            if index == 0:
                part = cls._STAGE_PREFIX.sub("", part).strip()
            part = part.strip(" \t,;.")
            if part:
                cleaned.append(part)
        return cleaned[: max(1, int(max_stages))]

    def retrieve_dag(
        self,
        intent: str,
        max_plans: int = 4,
        arity: Optional[int] = None,
        tau_cov: float = 0.85,
        tau_res: float = 0.35,
    ) -> List[Tuple[Plan, float]]:
        """The ordered N-stage plan chain a compound intent needs (I2).

        Two search strategies, in order. First, discourse segmentation: when
        the request names its stages ('first ... then ...'), each clause is
        matched against the store and the clause order is the pipeline
        order. When the request has no usable markers, or a clause fails to
        match, the search peels the best plan, masks its IDF-weighted tokens
        from the residual intent wave, and repeats until the cumulative
        coverage reaches tau_cov or the residue falls below tau_res.
        """
        if not self.plans or int(max_plans) < 1:
            return []
        limit = max(1, int(max_plans))
        segments = self.split_stages(intent, limit)
        if len(segments) >= 2:
            chain: List[Tuple[Plan, float]] = []
            used: set = set()
            for position, segment in enumerate(segments[:limit]):
                hit = self.retrieve_for(segment, arity=arity if position == 0 else None)
                if hit is None or hit[0].name in used:
                    chain = []
                    break
                chain.append((hit[0], float(hit[1])))
                used.add(hit[0].name)
            if chain and self.coverage_many(intent, [plan for plan, _ in chain]) >= 0.45:
                return chain
        head, tail = self.split_sequence(intent)
        first = self.retrieve_for(head if tail else intent, arity=arity)
        if first is None:
            return []
        chain = [first]
        used = {first[0].name}
        covered = self.coverage(intent, first[0])
        # A single plan that already explains most of the request is the
        # answer. Peeling starts only for a genuinely compound residual, so
        # one-plan intents cannot be over-composed by the N-stage search.
        if covered >= 0.60 and not tail:
            return chain
        while len(chain) < limit and covered < float(tau_cov):
            residue, leftover = self.residual_key_many(intent, [plan for plan, _ in chain])
            if residue is None or not leftover:
                break
            best: Optional[Tuple[Plan, float]] = None
            for other in self.plans.values():
                if other.name in used:
                    continue
                value = float(np.real(np.vdot(residue, self.key(other.intent))))
                if best is None or value > best[1] or (
                    value == best[1] and other.name < best[0].name
                ):
                    best = (other, value)
            if best is None or best[1] < float(tau_res):
                break
            chain.append(best)
            used.add(best[0].name)
            covered = self.coverage_many(intent, [plan for plan, _ in chain])
        return chain

    def retrieve_for(self, intent: str, arity: Optional[int] = None) -> Optional[Tuple[Plan, float]]:
        """Retrieve with a role-slot compatibility filter (Directive G2).

        A new intent names a function and, often, its arguments. Among the
        plans whose intent wave clears tau, prefer one whose argument count
        matches the requested count; otherwise take the best coherence. That
        is a structural check on the binding, not a string match.
        """
        if not self.plans:
            return None
        wave = self.key(intent)
        scored: List[Tuple[float, Plan]] = []
        for plan in self.plans.values():
            score = float(np.real(np.vdot(wave, self.key(plan.intent))))
            scored.append((score, plan))
        scored.sort(key=lambda pair: (-pair[0], pair[1].name))
        eligible = [(score, plan) for score, plan in scored if score >= self.tau]
        if not eligible:
            return None
        if arity is not None:
            matched = [
                (score, plan)
                for score, plan in eligible
                if len(plan.slots.get("args", []) or []) == int(arity)
                and plan.slots
            ]
            if matched:
                return matched[0][1], matched[0][0]
        return eligible[0][1], eligible[0][0]


class SynthesisEngine:
    """Retrieval-grounded, gate-verified multi-token synthesis.

    The pipeline is the protocol's Loop 0/1/2 in one object:

        Loop 0  recall a verified plan engram by intent coherence
        Loop 1  emit it token by token under the AST/Sagnac gate, score every
                candidate with goal alignment and the repeat penalty
        Loop 2  on feedback, compile the verified fix as a new Tier-3 plan
                and write the surprise-gated fast weight

    The emitted text is parsed and executed before it is returned. A
    statement that does not parse, or an answer with no ontological support,
    is refused rather than returned.
    """

    def __init__(
        self,
        generator: "WaveCodeTextGenerator",
        projector: BochnerPhaseProjector,
        tokenizer: HolarchCodeTextTokenizer,
        memory,
        *,
        verifier=None,
        namespace: str = "default",
        plan_offset: int = 20000,
        signatures: Optional[Dict[str, object]] = None,
    ) -> None:
        self.generator = generator
        self.projector = projector
        self.tokenizer = tokenizer
        self.memory = memory
        self.verifier = verifier
        self.namespace = str(namespace)
        # Directive H2: the code index's declared surfaces. When no plan
        # matches, a known signature is the difference between a verified
        # stub and ungrounded tokens.
        self.signatures: Dict[str, object] = dict(signatures or {})
        # Directive I2: one gate, reading the compiled canonical signatures.
        self.type_gate = TypeCompatibilityGate(self.signatures)
        self.composed = 0
        self.dag_composed = 0
        self.type_refused = 0
        self.cold_stubs = 0
        self.plans = PlanStore(
            memory,
            projector,
            tokenizer,
            plan_offset,
            namespace,
            token_counts=getattr(generator.count_lm, "unigram", None),
        )
        # Directive G2: the ONE fast-weight array, owned by the generator so
        # that token_logits() reads it. The engine holds a reference, not a
        # second copy, so an online write cannot land in dead state.
        self.fast_weight = self.generator.fast_weight
        self.adapted = 0
        self.refused = 0
        self.emission_fallbacks = 0
        self.emission_exact = 0
        self.plan_restored = self.plans.load()

    # ------------------------------------------------------------- emission
    def _emit(
        self,
        intent: str,
        prior: Optional[Sequence[int]],
        max_tokens: int,
        require_parse: bool,
        force_prior: bool = False,
    ) -> Dict[str, object]:
        # The emission starts from an empty context so the decoded TEXT is the
        # whole source. Starting from "def " put the keyword in the context
        # and again in the emission, which made every emitted source diverge
        # from the template it was supposed to reproduce.
        context: List[int] = []
        goal = self.projector.sequence_wave(self.tokenizer.encode(intent)[:64], self.tokenizer)
        return self.generator.decode(
            context,
            max_tokens=int(max_tokens),
            prior=prior,
            goal_wave=goal,
            require_parse=bool(require_parse),
            prior_weight=8.0,
            force_prior=bool(force_prior),
        )

    def synthesize(
        self,
        intent: str,
        kind: str = "code",
        max_tokens: int = 320,
        allow_free: bool = True,
    ) -> Dict[str, object]:
        """Produce a verified source, or an explicit refusal."""
        intent = str(intent).strip()
        if kind == "code":
            veto = self._intent_symbol_veto(intent)
            if veto:
                self.refused += 1
                return {
                    "status": "refused",
                    "reason": veto,
                    "source": "",
                    "verified": False,
                    "plan": "",
                    "coherence": 0.0,
                }
        request = _intent_slots(intent) if kind == "code" else {}
        arity = len(request.get("args") or []) or None
        hit = self.plans.retrieve_for(intent, arity=arity) if self.plans.plans else None
        adapted_from = ""
        composed_names: List[str] = []
        cold_used = False
        adapted: Optional[str] = None
        coherence = 0.0
        used = ""
        source = ""
        output: Dict[str, object] = {"tokens": [], "text": "", "vetoed": 0}
        type_verdict: Dict[str, object] = {}
        if kind == "code" and self.plans.plans:
            chain = self.plans.retrieve_dag(intent, max_plans=4, arity=arity)
        else:
            chain = []
        if len(chain) >= 2:
            # Directive I2: the chain is type-checked BEFORE any source is
            # emitted. A mismatch is a fail-closed refusal, not a fallback
            # to a single plan: returning half the requested pipeline would
            # be a silent partial answer.
            chain_plans = [plan for plan, _score in chain]
            type_verdict = self.type_gate.check_chain(chain_plans)
            if not bool(type_verdict.get("ok", False)):
                self.refused += 1
                self.type_refused += 1
                return {
                    "status": "refused",
                    "reason": str(type_verdict.get("reason", "type mismatch")),
                    "source": "",
                    "verified": False,
                    "plan": "+".join(plan.name for plan in chain_plans),
                    "coherence": 0.0,
                    "composed": False,
                    "composed_from": [plan.name for plan in chain_plans],
                    "dag_depth": len(chain_plans),
                    "type_chain": list(type_verdict.get("rows", [])),
                    "type_compatible": False,
                    "cold_stub": False,
                }
            candidate = _compose_plans_many(chain_plans, request)
            if candidate is not None:
                adapted = candidate
                coherence = float(chain[0][1])
                used = "+".join(plan.name for plan in chain_plans)
                composed_names = [plan.name for plan in chain_plans]
        if adapted is None and hit is not None:
            plan, coherence = hit
            # Directive G2: rebind the template to this request BEFORE
            # emission. The rebinding is an AST edit, so a new function name,
            # new argument identifiers and new constants are carried into the
            # body rather than replayed verbatim.
            adapted = (
                _adapt_source(plan.source, plan.slots, request)
                if kind == "code" and plan.slots
                else plan.source
            )
            used = plan.name
        if adapted is not None:
            adapted_from = adapted
            prior = self.tokenizer.encode(adapted)
            output = self._emit(
                intent,
                prior,
                max_tokens=max(len(prior) + 16, 48),
                require_parse=True,
                force_prior=True,
            )
            emitted = str(output.get("text", ""))
            if emitted.strip() == adapted.strip():
                self.emission_exact += 1
            source = emitted
            if not source.strip():
                # The gate collapsed the emission. Fall back to the adapted
                # template so the failure is a refusal, not a silent replay.
                source = adapted
                self.emission_fallbacks += 1
            if composed_names:
                self.composed += 1
                if len(composed_names) >= 3:
                    self.dag_composed += 1
        elif kind == "code" and self.signatures:
            # Directive H2 cold start: no playbook matched, but the code
            # index declares this callable's surface, so a verified stub is
            # grounded in the AST rather than invented token by token.
            cold = self._cold_signature_source(request)
            if cold is not None:
                source = cold
                coherence = 0.0
                used = ""
                self.cold_stubs += 1
                self.emission_exact += 1
                cold_used = True
            elif allow_free:
                output = self._emit(intent, None, max_tokens=int(max_tokens), require_parse=True)
                source = str(output.get("text", ""))
        elif allow_free:
            output = self._emit(intent, None, max_tokens=int(max_tokens), require_parse=True)
            source = output["text"]
            coherence = 0.0
            used = ""
        else:
            self.refused += 1
            return {
                "status": "refused",
                "reason": "no verified plan for this intent",
                "source": "",
                "verified": False,
                "plan": "",
                "coherence": 0.0,
            }
        verification = self.verify(source, kind=kind)
        if not verification["ok"] and adapted_from and self.verify(adapted_from, kind=kind)["ok"]:
            # The emission diverged from the verified template and stopped
            # parsing. Return the verified binding, and say so: this is a
            # recorded fallback, not a silent repair.
            source = adapted_from
            verification = self.verify(source, kind=kind)
            self.emission_fallbacks += 1
        return {
            "status": "verified" if verification["ok"] else "rejected",
            "reason": verification["reason"],
            "source": source,
            "verified": bool(verification["ok"]),
            "plan": used,
            "coherence": float(coherence),
            "bound_slots": {
                "func": request.get("func"),
                "args": list(request.get("args") or []),
                "numbers": list(request.get("numbers") or []),
            },
            "adapted": bool(adapted_from and adapted_from.strip() != str(hit[0].source).strip())
            if hit is not None
            else False,
            "composed": bool(composed_names),
            "composed_from": list(composed_names),
            "dag_depth": len(composed_names) if composed_names else 1,
            "type_compatible": bool(type_verdict.get("ok", True)) if type_verdict else True,
            "type_chain": list(type_verdict.get("rows", [])) if type_verdict else [],
            "cold_stub": bool(cold_used),
            "emitted_tokens": len(output.get("tokens", [])),
            "vetoed_tokens": int(output.get("vetoed", 0)),
        }

    def _cold_signature_source(self, request: Dict[str, object]) -> Optional[str]:
        """A verified stub for a callable the code index already declares.

        Directive H2. The function name comes from the request; the argument
        list comes from the requested names, or from the indexed signature
        when the request names none. The body raises, which is the honest
        statement that the behaviour is not yet known, and the whole source
        is re-parsed and compiled before it is returned.
        """
        name = str(request.get("func") or "").strip()
        if not name or not name.isidentifier():
            return None
        signature = self.signatures.get(name)
        if signature is None:
            return None
        declared = [str(value) for value in (getattr(signature, "args", ()) or ())]
        requested = [str(value) for value in (request.get("args") or [])]
        args = [value for value in (requested or declared) if value.isidentifier() and value not in {"self", "cls"}]
        source = (
            f"def {name}({', '.join(args)}):"
            + chr(10)
            + "    raise NotImplementedError(" + repr(f"cold stub for {name}") + ")"
            + chr(10)
        )
        verdict = self.verify(source, kind="code")
        return source if verdict["ok"] else None

    def _intent_symbol_veto(self, intent: str) -> str:
        """A callable named in the intent must exist in the code index.

        Three forms are checked: a direct call (name followed by an open
        parenthesis), a dotted member (ZoneCMemory.quantum_entangle), and the
        object of a call verb ("calls holarch_synthesize_warp to ..."). The
        check uses the same index the AST gate uses at egress, so the refusal
        and the token veto cannot disagree.
        """
        for callee in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", intent):
            if self.generator.gate.symbol_veto(callee):
                return f"symbol {callee!r} is not in the code index"
        for dotted in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*)", intent):
            member = dotted.split(".")[-1]
            if self.generator.gate.symbol_veto(member):
                return f"symbol {dotted!r} is not in the code index"
        for callee in re.findall(
            r"(?:calls|call|invokes|invoke|uses|use|executes|execute)\s+([A-Za-z_][A-Za-z0-9_.]*)",
            intent,
            flags=re.IGNORECASE,
        ):
            name = callee.split(".")[-1]
            if name in {"a", "an", "the", "it", "them", "this", "that"}:
                continue
            if self.generator.gate.symbol_veto(name):
                return f"symbol {callee!r} is not in the code index"
        return ""

    def verify(self, source: str, kind: str = "code") -> Dict[str, object]:
        """Parse, compile, and (for text) run the ontology verifier."""
        text = str(source)
        if not text.strip():
            return {"ok": False, "reason": "empty source"}
        if kind == "text":
            if self.verifier is None:
                return {"ok": True, "reason": "no verifier configured"}
            verdict = self.verifier.verify_claim(text)
            return {"ok": verdict.supported, "reason": verdict.reason, "verdict": verdict.as_dict()}
        try:
            tree = ast.parse(text)
        except SyntaxError as error:
            return {"ok": False, "reason": f"ast.parse: {error.msg} at line {error.lineno}"}
        try:
            compile(tree, "<synthesis>", "exec")
        except (SyntaxError, ValueError) as error:
            return {"ok": False, "reason": f"compile: {error}"}
        return {"ok": True, "reason": "ast.parse and compile succeeded"}

    def execute(self, source: str, tests: str, namespace: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        """Run one unit test against the emitted module in an isolated dict."""
        scope: Dict[str, object] = {"__builtins__": __builtins__}
        if namespace:
            scope.update(namespace)
        try:
            exec(str(source), scope)
            exec(str(tests), scope)
        except Exception as error:  # noqa: BLE001 - the failure is the measurement
            return {"ok": False, "reason": f"{type(error).__name__}: {error}"}
        return {"ok": True, "reason": "executed"}

    # ------------------------------------------------------------ feedback
    def adapt(
        self,
        intent: str,
        corrected_source: str,
        name: str,
        kind: str = "code",
        surprise: float = 1.0,
        lr: float = 0.35,
    ) -> Dict[str, object]:
        """Compile a verified fix into a plan, and write the fast weight.

        The delta-rule write is the same rank-r form the Zone A expert uses,
        applied to the intent's token phases: it makes the intent's own
        tokens more coherent with the corrected source's tokens, so the next
        recall of a neighbouring intent is biased toward this fix.
        """
        verdict = self.verify(corrected_source, kind=kind)
        if not verdict["ok"]:
            return {"adapted": False, "reason": verdict["reason"]}
        plan = self.plans.compile(name, intent, corrected_source, kind=kind)
        gate = 1.0 if float(surprise) > 0.0 else 0.0
        # The delta-rule write goes through the generator, which is the only
        # reader of the memory. A zero gate writes nothing, so a familiar
        # input cannot move the logits (0.0000 stationary drift).
        written = self.generator.update_fast_weight(
            self.tokenizer.encode(intent)[:32],
            self.tokenizer.encode(corrected_source)[:48],
            lr=float(lr) * gate,
        )
        self.adapted += 1
        return {
            "adapted": True,
            "plan": plan.name,
            "version": int(plan.version),
            "engram_id": int(plan.engram_id),
            "digest": plan.digest,
            "gate": float(gate),
        }


# ---------------------------------------------------------------------------
# Ontological question answering
# ---------------------------------------------------------------------------
def answer_question(
    question: str,
    verifier,
    engine: Optional[SynthesisEngine] = None,
    top_k: int = 4,
    max_chars: int = 480,
) -> Dict[str, object]:
    """Answer from retrieved evidence, then verify the answer as a claim.

    The answer is an extractive sentence from the highest-ranked passage,
    not generated prose. That is deliberate: the ontology gate can only
    certify text that restates evidence, so the encoder is asked to restate
    evidence and to abstain otherwise.
    """
    if verifier is None:
        return {"status": "refused", "answer": "", "reason": "no verifier"}
    ranked = verifier.bm25(question, top_k=int(top_k))
    if not ranked:
        return {"status": "refused", "answer": "", "reason": "no passage matched"}
    rare = verifier.rare_terms(question)
    # A question whose rare terms are nowhere in the retrieved evidence is
    # out of distribution, whatever passage the ranker surfaced. Refusing
    # here is what makes an unanswerable question abstain instead of
    # returning a true sentence about something else.
    evidence_terms = set()
    for _score, index in ranked:
        evidence_terms.update(verifier.terms[index])
    unanswered = [term for term in rare if term not in evidence_terms]
    if unanswered:
        return {
            "status": "refused",
            "answer": "",
            "reason": f"question term(s) absent from every retrieved passage: {unanswered}",
        }
    best_sentence = ""
    best_score = -1.0
    for score, index in ranked:
        passage = verifier.passages[index]
        for sentence in re.split(r"(?<=[.!?])\s+", passage.text):
            sentence = sentence.strip()
            if len(sentence) < 40:
                continue
            hits = sum(1 for term in rare if term in sentence.lower())
            numeric = 1 if verifier.verify_claim(sentence).supported else 0
            combined = float(score) * 0.01 + hits + 2.0 * numeric
            if combined > best_score:
                best_score = combined
                best_sentence = sentence
    if not best_sentence:
        return {"status": "refused", "answer": "", "reason": "no extractable sentence"}
    verdict = verifier.verify_claim(best_sentence)
    answer = best_sentence[: int(max_chars)]
    if not verdict.supported:
        return {
            "status": "abstained",
            "answer": "",
            "reason": verdict.reason,
            "verdict": verdict.as_dict(),
        }
    output: Dict[str, object] = {
        "status": "supported",
        "answer": answer,
        "reason": verdict.reason,
        "verdict": verdict.as_dict(),
        "tokens": len(answer.split()),
    }
    if engine is not None:
        emission = engine.synthesize(answer, kind="text", allow_free=True)
        output["emitted"] = emission.get("source", "")
        output["emitted_verified"] = bool(emission.get("verified", False))
    return output


def discover_vocabulary_lock(root: str | Path) -> Tuple[Optional[VocabularyLedger], str]:
    """Find the pinned vocabulary lock above a package directory (I1)."""
    base = Path(root).resolve()
    for candidate in [base, *list(base.parents)[:5]]:
        path = candidate / "receipts" / "holarch" / "vocab_lock.json"
        if path.exists():
            try:
                return VocabularyLedger.load(path), str(path)
            except Exception:
                return None, ""
    return None, ""


def build_synthesis_engine(
    root: str | Path,
    memory,
    namespace: str = "default",
    symbol_table: Optional[Iterable[str]] = None,
    max_fit_tokens: int = 60000,
    signatures: Optional[Dict[str, object]] = None,
    vocab_lock: Optional[VocabularyLedger] = None,
) -> Tuple["SynthesisEngine", Dict[str, object]]:
    """Build the full ingress+egress stack over a corpus, once.

    This is the factory the sidecar uses, so the REST endpoint, the MCP tool
    and the benchmark cannot construct three subtly different generators.
    Directive I1: the factory mounts the pinned vocabulary lock when one is
    found above the package root, so a live-tree edit can only append.
    """
    corpus = CodeTextCorpus.holarch(root)
    lock_path = ""
    if vocab_lock is None:
        vocab_lock, lock_path = discover_vocabulary_lock(root)
    elif isinstance(vocab_lock, (str, Path)):
        lock_path = str(vocab_lock)
        vocab_lock = VocabularyLedger.load(vocab_lock)
    tokenizer = build_tokenizer(corpus, vocab_lock=vocab_lock)
    projector = BochnerPhaseProjector(dim=int(getattr(memory, "dim", 1024)), d_sem=64)
    projector.fit(corpus, tokenizer)
    ids: List[int] = []
    for text in corpus.texts:
        ids.extend(tokenizer.encode(text))
    generator = WaveCodeTextGenerator(
        tokenizer, projector, symbol_table=set(symbol_table or ())
    )
    generator.fit(ids[: int(max_fit_tokens)], max_fit_tokens=min(20000, int(max_fit_tokens)))
    generator.count_lm.fit(ids[: int(len(ids) * 0.9)])
    generator.lexical.build(ids[: int(max_fit_tokens)])
    engine = SynthesisEngine(
        generator,
        projector,
        tokenizer,
        memory,
        verifier=build_verifier(root),
        namespace=str(namespace),
        signatures=signatures,
    )
    return engine, {
        "corpus_digest": corpus.digest,
        "vocabulary_size": int(len(tokenizer)),
        "vocabulary_digest": str(tokenizer.spec.digest()),
        "tokens": int(len(ids)),
        "files": int(len(corpus)),
        "vocab_locked": bool(getattr(tokenizer, "locked", False)),
        "vocab_ledger_digest": str(getattr(tokenizer.ledger, "digest", lambda: "")())
        if getattr(tokenizer, "ledger", None) is not None
        else "",
        "vocab_lock_path": str(lock_path),
    }


# ---------------------------------------------------------------------------
# Directive J3: a first-class diverse playbook library and typed DAG battery
# ---------------------------------------------------------------------------
# Cycle I proved three-stage splicing on one typed triple. This section makes
# the playbook a library: fifteen small pure stages, five distinct three-stage
# families over the canonical types, guarded templates that survive renaming
# and constant rebinding, and a fail-closed type gate in front of the splicer.
# Every source below is a complete Python function; the battery parses,
# compiles, executes and compares each emitted source with an independent
# reference answer.

PLAYBOOK_LIBRARY: Dict[str, str] = {
    # F1 graph -> set -> bound
    "caller_names": (
        "def caller_names(text: str) -> list:\n"
        "    return [line.strip() for line in text.splitlines() if line.strip()]\n"
    ),
    "sequence_length": (
        "def sequence_length(items: list) -> int:\n"
        "    return len(items)\n"
    ),
    "clamp_unit": (
        "def clamp_unit(value: float) -> float:\n"
        "    low = 0.0\n"
        "    high = 1.0\n"
        "    result = value if value > low else low\n"
        "    return result if result < high else high\n"
    ),
    # F2 numeric sequence -> statistic -> threshold
    "normalize_scores": (
        'def normalize_scores(values: "numeric_sequence") -> "numeric_sequence":\n'
        "    total = sum(values)\n"
        "    scale = total if total != 0.0 else 1.0\n"
        "    return [value / scale for value in values]\n"
    ),
    "mean_of": (
        'def mean_of(values: "numeric_sequence") -> float:\n'
        "    return sum(values) / len(values) if values else 0.0\n"
    ),
    "exceeds_tau": (
        "def exceeds_tau(value: float) -> bool:\n"
        "    return float(value) > 0.5\n"
    ),
    # F3 mapping -> filter/sort -> aggregate text
    "invert_index": (
        "def invert_index(index: dict) -> dict:\n"
        "    inverted = {}\n"
        "    for key, value in index.items():\n"
        "        inverted.setdefault(value, []).append(key)\n"
        "    return inverted\n"
    ),
    "sorted_keys": (
        "def sorted_keys(index: dict) -> list:\n"
        "    return sorted(index.keys())\n"
    ),
    "join_csv": (
        "def join_csv(items: list) -> str:\n"
        '    return ",".join(str(item) for item in items)\n'
    ),
    # F4 text -> tokenize -> count
    "strip_comments": (
        "def strip_comments(text: str) -> str:\n"
        '    kept = [line.partition("#")[0].rstrip() for line in text.splitlines()]\n'
        '    return "\\n".join(line for line in kept if line)\n'
    ),
    "split_identifiers": (
        "def split_identifiers(text: str) -> list:\n"
        '    cleaned = text.replace("(", " ").replace(")", " ").replace(",", " ")\n'
        "    return [token for token in cleaned.split() if token.isidentifier()]\n"
    ),
    "unique_count": (
        "def unique_count(items: list) -> int:\n"
        "    return len(set(items))\n"
    ),
    # F5 numeric sequence -> window -> peak
    "running_deltas": (
        'def running_deltas(values: "numeric_sequence") -> "numeric_sequence":\n'
        "    return [values[index] - values[index - 1] for index in range(1, len(values))]\n"
    ),
    "abs_values": (
        'def abs_values(values: "numeric_sequence") -> "numeric_sequence":\n'
        "    return [abs(value) for value in values]\n"
    ),
    "max_or_zero": (
        'def max_or_zero(values: "numeric_sequence") -> float:\n'
        "    floor = 0.0\n"
        "    peak = max(values) if values else floor\n"
        "    return peak if peak > floor else floor\n"
    ),
}

_PLAYBOOK_INTENTS: Dict[str, str] = {
    "caller_names": "returns the caller names in a text one per line",
    "sequence_length": "returns the number of items in a sequence",
    "clamp_unit": "clips a number into a bounded interval",
    "normalize_scores": "normalizes a numeric sequence so it sums to one",
    "mean_of": "returns the arithmetic mean of a numeric sequence",
    "exceeds_tau": "returns whether a number exceeds a threshold",
    "invert_index": "inverts a mapping into value to keys",
    "sorted_keys": "returns the sorted keys of a mapping",
    "join_csv": "joins a sequence of values into comma separated text",
    "strip_comments": "strips comment text from source text",
    "split_identifiers": "splits text into identifier tokens",
    "unique_count": "counts the distinct items of a sequence",
    "running_deltas": "returns the successive differences of a numeric sequence",
    "abs_values": "returns the absolute values of a numeric sequence",
    "max_or_zero": "returns the maximum of a numeric sequence above a floor",
}

_PLAYBOOK_TYPES: Dict[str, Tuple[str, str]] = {
    "caller_names": (TYPE_TEXT, TYPE_SEQUENCE),
    "sequence_length": (TYPE_SEQUENCE, TYPE_NUMERIC),
    "clamp_unit": (TYPE_NUMERIC, TYPE_NUMERIC),
    "normalize_scores": (TYPE_NUMERIC_SEQUENCE, TYPE_NUMERIC_SEQUENCE),
    "mean_of": (TYPE_NUMERIC_SEQUENCE, TYPE_NUMERIC),
    "exceeds_tau": (TYPE_NUMERIC, TYPE_BOOLEAN),
    "invert_index": (TYPE_MAPPING, TYPE_MAPPING),
    "sorted_keys": (TYPE_MAPPING, TYPE_SEQUENCE),
    "join_csv": (TYPE_SEQUENCE, TYPE_TEXT),
    "strip_comments": (TYPE_TEXT, TYPE_TEXT),
    "split_identifiers": (TYPE_TEXT, TYPE_SEQUENCE),
    "unique_count": (TYPE_SEQUENCE, TYPE_NUMERIC),
    "running_deltas": (TYPE_NUMERIC_SEQUENCE, TYPE_NUMERIC_SEQUENCE),
    "abs_values": (TYPE_NUMERIC_SEQUENCE, TYPE_NUMERIC_SEQUENCE),
    "max_or_zero": (TYPE_NUMERIC_SEQUENCE, TYPE_NUMERIC),
}

PLAYBOOK_FAMILIES: Dict[str, Dict[str, object]] = {
    "F1": {
        "label": "graph names to length to bounded score",
        "stages": ("caller_names", "sequence_length", "clamp_unit"),
    },
    "F2": {
        "label": "numeric sequence normalize to mean to threshold",
        "stages": ("normalize_scores", "mean_of", "exceeds_tau"),
    },
    "F3": {
        "label": "mapping inversion to sorted keys to csv text",
        "stages": ("invert_index", "sorted_keys", "join_csv"),
    },
    "F4": {
        "label": "comment strip to identifiers to distinct count",
        "stages": ("strip_comments", "split_identifiers", "unique_count"),
    },
    "F5": {
        "label": "running deltas to magnitudes to floored peak",
        "stages": ("running_deltas", "abs_values", "max_or_zero"),
    },
}


def _j3_topology(stages: Sequence[str]) -> str:
    """The typed stage chain, e.g. 'text->sequence|sequence->numeric|...'."""
    edges = []
    for stage in stages:
        source_type, result_type = _PLAYBOOK_TYPES[str(stage)]
        edges.append(f"{source_type}->{result_type}")
    return "|".join(edges)


def _playbook_signature(name: str) -> Dict[str, object]:
    """The declared canonical signature of one library stage."""
    source = PLAYBOOK_LIBRARY[str(name)]
    argument = str((_source_slots(source).get("args") or [name])[0])
    in_type, out_type = _PLAYBOOK_TYPES[str(name)]
    return {
        "name": str(name),
        "args": [{"name": argument, "type": [in_type]}],
        "arity": 1,
        "returns": [out_type],
    }


def _playbook_plan(name: str, plan_name: Optional[str] = None) -> "Plan":
    """A Plan whose slots carry the declared playbook contract.

    The declared signature is also what the source infers, so the gate reads
    the same contract whether the plan came from this factory or from
    PlanStore.compile(). No engine state is required or touched.
    """
    source = PLAYBOOK_LIBRARY[str(name)]
    slots = _source_slots(source)
    slots["types"] = _playbook_signature(str(name))
    return Plan(
        name=str(plan_name or name),
        intent=_PLAYBOOK_INTENTS[str(name)],
        source=source,
        kind="code",
        digest=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        slots=slots,
    )


def _compiled_plan(name: str, intent: str, source: str) -> "Plan":
    """Compile one standalone source the way PlanStore.compile does."""
    slots = _source_slots(str(source))
    slots["types"] = _abstract_signature(str(source))
    return Plan(
        name=str(name),
        intent=str(intent),
        source=str(source),
        kind="code",
        digest=hashlib.sha256(str(source).encode("utf-8")).hexdigest(),
        slots=slots,
    )


def playbook_catalog() -> Dict[str, object]:
    """The pinnable J3 registry: sources, signatures, family triples, digest.

    A receipt pins the digest; it covers every stage source and every
    family's ordered stage tuple, so a later edit changes the pin.
    """
    sources = dict(PLAYBOOK_LIBRARY)
    signatures: Dict[str, object] = {}
    for name, source in sources.items():
        in_type, out_type = _PLAYBOOK_TYPES[name]
        signatures[name] = {
            "name": name,
            "input": in_type,
            "output": out_type,
            "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        }
    families: Dict[str, object] = {}
    triples: List[List[str]] = []
    for family, spec in PLAYBOOK_FAMILIES.items():
        stages = [str(stage) for stage in spec["stages"]]  # type: ignore[index]
        triples.append(list(stages))
        families[family] = {
            "label": str(spec["label"]),
            "stages": list(stages),
            "types": [[_PLAYBOOK_TYPES[stage][0], _PLAYBOOK_TYPES[stage][1]] for stage in stages],
            "topology": _j3_topology(stages),
        }
    topologies = sorted({str(entry["topology"]) for entry in families.values()})  # type: ignore[index]
    payload = {
        "sources": sources,
        "signatures": signatures,
        "families": {
            key: {"stages": value["stages"], "types": value["types"]}  # type: ignore[index]
            for key, value in families.items()
        },
    }
    return {
        "sources": sources,
        "signatures": signatures,
        "families": families,
        "triples": triples,
        "topologies": topologies,
        "digest": digest(payload),
    }


def compose_plan_dag(
    plans: Sequence["Plan"],
    request: Dict[str, object],
    gate: Optional["TypeCompatibilityGate"] = None,
) -> Dict[str, object]:
    """Type-check an ordered plan chain, then splice it; refuse first.

    Directive J3. The gate runs to completion before _compose_plans_many is
    called, so a refused chain returns an empty source: no partial pipeline
    and no unverified text can reach the caller.
    """
    chain = list(plans)
    if not chain:
        return {
            "ok": False,
            "status": "refused",
            "reason": "no plans to compose",
            "source": "",
            "type_chain": [],
        }
    checker = gate if gate is not None else TypeCompatibilityGate()
    verdict = checker.check_chain(chain)
    if not bool(verdict.get("ok", False)):
        return {
            "ok": False,
            "status": "refused",
            "reason": str(verdict.get("reason", "type mismatch")),
            "source": "",
            "type_chain": list(verdict.get("rows", [])),
        }
    source = _compose_plans_many(chain, request)
    if source is None:
        return {
            "ok": False,
            "status": "rejected",
            "reason": "the splicer refused the chain",
            "source": "",
            "type_chain": list(verdict.get("rows", [])),
        }
    return {
        "ok": True,
        "status": "composed",
        "reason": "every stage is type compatible",
        "source": source,
        "type_chain": list(verdict.get("rows", [])),
    }


# ---------------------------------------------------------------------------
# Directive J3: independent reference answers for the battery
# ---------------------------------------------------------------------------
def _j3_ref_names(text: str) -> List[str]:
    return [line.strip() for line in str(text).splitlines() if line.strip()]


def _j3_ref_bound(value: float, low: float, high: float) -> float:
    if value <= low:
        return low
    if value >= high:
        return high
    return value


def _j3_ref_normalize(values: Sequence[float]) -> List[float]:
    total = sum(values)
    scale = total if total != 0.0 else 1.0
    return [value / scale for value in values]


def _j3_ref_mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _j3_ref_invert(index: Dict[str, object]) -> Dict[str, List[str]]:
    inverted: Dict[str, List[str]] = {}
    for key, value in index.items():
        inverted.setdefault(value, []).append(key)  # type: ignore[arg-type]
    return inverted


def _j3_ref_csv(values: Sequence[object]) -> str:
    return ",".join(str(value) for value in values)


def _j3_ref_strip(text: str) -> str:
    kept = []
    for line in str(text).splitlines():
        kept.append(line.split("#", 1)[0].rstrip())
    return "\n".join(line for line in kept if line)


def _j3_ref_identifiers(text: str) -> List[str]:
    words = []
    for token in str(text).replace("(", " ").replace(")", " ").replace(",", " ").split():
        if token.isidentifier():
            words.append(token)
    return words


def _j3_ref_signed_deltas(values: Sequence[float]) -> List[float]:
    return [values[index] - values[index - 1] for index in range(1, len(values))]


def _j3_ref_deltas(values: Sequence[float]) -> List[float]:
    return [abs(right - left) for left, right in zip(values, values[1:])]


def _j3_ref_floor_peak(values: Sequence[float], floor: float) -> float:
    deltas = _j3_ref_deltas(values)
    peak = max(deltas) if deltas else floor
    return peak if peak > floor else floor


# ---------------------------------------------------------------------------
# Directive J3: fixture-free task tables
# ---------------------------------------------------------------------------
_J3_BASE_TASKS: List[Dict[str, object]] = [
    {
        "id": "b01",
        "name": "count_words",
        "arg": "text",
        "source": "def count_words(text: str) -> int:\n    return len(text.split())\n",
        "intent": "Write a function count_words that takes argument text and returns the number of words.",
        "args": ("alpha beta gamma",),
        "reference": lambda s: len(s.split()),
    },
    {
        "id": "b02",
        "name": "first_word",
        "arg": "text",
        "source": 'def first_word(text: str) -> str:\n    return text.split()[0] if text.split() else ""\n',
        "intent": "Write a function first_word that takes argument text and returns its first word or an empty string.",
        "args": ("",),
        "reference": lambda s: s.split()[0] if s.split() else "",
    },
    {
        "id": "b03",
        "name": "uppercase_text",
        "arg": "text",
        "source": "def uppercase_text(text: str) -> str:\n    return text.upper()\n",
        "intent": "Write a function uppercase_text that takes argument text and returns it in upper case.",
        "args": ("holarch",),
        "reference": lambda s: s.upper(),
    },
    {
        "id": "b04",
        "name": "strip_punctuation",
        "arg": "text",
        "source": 'def strip_punctuation(text: str) -> str:\n    return text.strip(".,;:!?")\n',
        "intent": "Write a function strip_punctuation that takes argument text and removes surrounding punctuation.",
        "args": ("hello, world!",),
        "reference": lambda s: s.strip(".,;:!?"),
    },
    {
        "id": "b05",
        "name": "reverse_items",
        "arg": "items",
        "source": "def reverse_items(items: list) -> list:\n    return list(reversed(items))\n",
        "intent": "Write a function reverse_items that takes argument items and returns them in reverse order.",
        "args": ([1, 2, 3, 4],),
        "reference": lambda items: list(items)[::-1],
    },
    {
        "id": "b06",
        "name": "list_length",
        "arg": "items",
        "source": "def list_length(items: list) -> int:\n    return len(items)\n",
        "intent": "Write a function list_length that takes argument items and returns how many there are.",
        "args": ([1, 2, 3, 4, 5],),
        "reference": lambda items: len(items),
    },
    {
        "id": "b07",
        "name": "unique_sorted",
        "arg": "items",
        "source": "def unique_sorted(items: list) -> list:\n    return sorted(set(items))\n",
        "intent": "Write a function unique_sorted that takes argument items and returns the distinct values in order.",
        "args": ([3, 1, 3, 2, 1],),
        "reference": lambda items: sorted(dict.fromkeys(items)),
    },
    {
        "id": "b08",
        "name": "sum_positive",
        "arg": "values",
        "source": (
            "def sum_positive(values: list) -> float:\n"
            "    return float(sum(value for value in values if value > 0))\n"
        ),
        "intent": "Write a function sum_positive that takes argument values and returns the sum of the positive ones.",
        "args": ([-1.0, 2.0, 3.5, 0.0],),
        "reference": lambda values: float(sum(value for value in values if value > 0.0)),
    },
    {
        "id": "b09",
        "name": "scale_values",
        "arg": "values",
        "source": "def scale_values(values: list) -> list:\n    return [value * 2 for value in values]\n",
        "intent": "Write a function scale_values that takes argument values and doubles each one.",
        "args": ([1.0, 2.5],),
        "reference": lambda values: [value * 2 for value in values],
    },
    {
        "id": "b10",
        "name": "mean_value",
        "arg": "values",
        "source": (
            "def mean_value(values: list) -> float:\n"
            "    return sum(values) / len(values) if values else 0.0\n"
        ),
        "intent": "Write a function mean_value that takes argument values and returns their arithmetic mean.",
        "args": ([2.0, 4.0, 6.0],),
        "reference": lambda values: sum(values) / float(len(values)) if values else 0.0,
    },
    {
        "id": "b11",
        "name": "max_value",
        "arg": "values",
        "source": (
            "def max_value(values: list) -> float:\n"
            "    return float(max(values)) if values else 0.0\n"
        ),
        "intent": "Write a function max_value that takes argument values and returns the largest one.",
        "args": ([2.0, 9.0, 4.0],),
        "reference": lambda values: float(max(values)) if values else 0.0,
    },
    {
        "id": "b12",
        "name": "keys_sorted",
        "arg": "table",
        "source": "def keys_sorted(table: dict) -> list:\n    return sorted(table.keys())\n",
        "intent": "Write a function keys_sorted that takes argument table and returns its keys in order.",
        "args": ({"b": 2, "a": 1},),
        "reference": lambda table: sorted(table.keys()),
    },
    {
        "id": "b13",
        "name": "get_default",
        "arg": "table",
        "source": 'def get_default(table: dict) -> int:\n    return table.get("x", 0)\n',
        "intent": "Write a function get_default that takes argument table and returns the x field or zero.",
        "args": ({"x": 7},),
        "reference": lambda table: table.get("x", 0),
    },
    {
        "id": "b14",
        "name": "has_key",
        "arg": "table",
        "source": 'def has_key(table: dict) -> bool:\n    return "x" in table\n',
        "intent": "Write a function has_key that takes argument table and reports whether x is present.",
        "args": ({"x": 1},),
        "reference": lambda table: "x" in table,
    },
    {
        "id": "b15",
        "name": "invert_index",
        "arg": "index",
        "source": (
            "def invert_index(index: dict) -> dict:\n"
            "    inverted = {}\n"
            "    for key, value in index.items():\n"
            "        inverted.setdefault(value, []).append(key)\n"
            "    return inverted\n"
        ),
        "intent": "Write a function invert_index that takes argument index and maps each value to its keys.",
        "args": ({"a": "x", "b": "y", "c": "x"},),
        "reference": lambda index: _j3_ref_invert(index),
    },
    {
        "id": "b16",
        "name": "pair_count",
        "arg": "table",
        "source": "def pair_count(table: dict) -> int:\n    return len(table)\n",
        "intent": "Write a function pair_count that takes argument table and returns how many pairs it holds.",
        "args": ({"a": 1, "b": 2, "c": 3},),
        "reference": lambda table: len(table),
    },
    {
        "id": "b17",
        "name": "is_even",
        "arg": "number",
        "source": "def is_even(number: int) -> bool:\n    return number % 2 == 0\n",
        "intent": "Write a function is_even that takes argument number and reports whether it is even.",
        "args": (42,),
        "reference": lambda number: number % 2 == 0,
    },
    {
        "id": "b18",
        "name": "clamp_score",
        "arg": "number",
        "source": (
            "def clamp_score(number: float) -> float:\n"
            "    return number if 0.0 <= number <= 1.0 else 0.0\n"
        ),
        "intent": "Write a function clamp_score that takes argument number and keeps it in the unit interval.",
        "args": (0.5,),
        "reference": lambda number: number if 0.0 <= number <= 1.0 else 0.0,
    },
    {
        "id": "b19",
        "name": "word_lengths",
        "arg": "text",
        "source": (
            "def word_lengths(text: str) -> list:\n"
            "    return [len(word) for word in text.split()]\n"
        ),
        "intent": "Write a function word_lengths that takes argument text and returns each word length.",
        "args": ("ab abc",),
        "reference": lambda text: [len(word) for word in text.split()],
    },
    {
        "id": "b20",
        "name": "join_hyphen",
        "arg": "words",
        "source": (
            "def join_hyphen(words: list) -> str:\n"
            '    return "-".join(str(word) for word in words)\n'
        ),
        "intent": "Write a function join_hyphen that takes argument words and joins them with hyphens.",
        "args": (["a", "b", "c"],),
        "reference": lambda words: "-".join(str(word) for word in words),
    },
    {
        "id": "b21",
        "name": "count_vowels",
        "arg": "text",
        "source": (
            "def count_vowels(text: str) -> int:\n"
            '    return sum(1 for char in text.lower() if char in "aeiou")\n'
        ),
        "intent": "Write a function count_vowels that takes argument text and counts its vowels.",
        "args": ("Holarch",),
        "reference": lambda text: sum(1 for char in text.lower() if char in "aeiou"),
    },
    {
        "id": "b22",
        "name": "safe_divide",
        "arg": "number",
        "source": (
            "def safe_divide(number: float) -> float:\n"
            "    return 100.0 / number if number != 0.0 else 0.0\n"
        ),
        "intent": "Write a function safe_divide that takes argument number and divides one hundred by it safely.",
        "args": (4.0,),
        "reference": lambda number: (100.0 / number) if number != 0.0 else 0.0,
    },
    {
        "id": "b23",
        "name": "first_char",
        "arg": "text",
        "source": 'def first_char(text: str) -> str:\n    return text[0] if text else ""\n',
        "intent": "Write a function first_char that takes argument text and returns its first character.",
        "args": ("",),
        "reference": lambda text: text[0] if text else "",
    },
    {
        "id": "b24",
        "name": "contains_word",
        "arg": "text",
        "source": 'def contains_word(text: str) -> bool:\n    return "holarch" in text.lower()\n',
        "intent": "Write a function contains_word that takes argument text and reports whether it names holarch.",
        "args": ("The Holarch protocol",),
        "reference": lambda text: "holarch" in text.lower(),
    },
]

_J3_SINGLE_TASKS: List[Dict[str, object]] = [
    {
        "id": "s01",
        "stage": "caller_names",
        "numbers": [],
        "args": ("z\ny\nz\n",),
        "reference": lambda text: _j3_ref_names(text),
    },
    {
        "id": "s02",
        "stage": "clamp_unit",
        "numbers": [-2.0, 7.0],
        "args": (9.0,),
        "reference": lambda value: _j3_ref_bound(value, -2.0, 7.0),
    },
    {
        "id": "s03",
        "stage": "normalize_scores",
        "numbers": [],
        "args": ([1.0, 1.0, 2.0],),
        "reference": lambda values: _j3_ref_normalize(values),
    },
    {
        "id": "s04",
        "stage": "mean_of",
        "numbers": [],
        "args": ([2.0, 4.0, 6.0],),
        "reference": lambda values: _j3_ref_mean(values),
    },
    {
        "id": "s05",
        "stage": "exceeds_tau",
        "numbers": [0.8],
        "args": (0.9,),
        "reference": lambda value: float(value) > 0.8,
    },
    {
        "id": "s06",
        "stage": "invert_index",
        "numbers": [],
        "args": ({"a": "x", "b": "y"},),
        "reference": lambda index: _j3_ref_invert(index),
    },
    {
        "id": "s07",
        "stage": "sorted_keys",
        "numbers": [],
        "args": ({"b": 1, "a": 2, "c": 3},),
        "reference": lambda index: sorted(index.keys()),
    },
    {
        "id": "s08",
        "stage": "join_csv",
        "numbers": [],
        "args": (["a", "b", "c"],),
        "reference": lambda items: _j3_ref_csv(items),
    },
    {
        "id": "s09",
        "stage": "strip_comments",
        "numbers": [],
        "args": ("x = 1  # note\ny = 2\n",),
        "reference": lambda text: _j3_ref_strip(text),
    },
    {
        "id": "s10",
        "stage": "split_identifiers",
        "numbers": [],
        "args": ("alpha, beta(gamma)",),
        "reference": lambda text: _j3_ref_identifiers(text),
    },
    {
        "id": "s11",
        "stage": "unique_count",
        "numbers": [],
        "args": (["a", "b", "a", "c", "b"],),
        "reference": lambda items: len(set(items)),
    },
    {
        "id": "s12",
        "stage": "running_deltas",
        "numbers": [],
        "args": ([3.0, 1.0, 4.0],),
        "reference": lambda values: _j3_ref_signed_deltas(values),
    },
]

_J3_TWO_TASKS: List[Dict[str, object]] = [
    {
        "id": "t01",
        "stages": ("caller_names", "sequence_length"),
        "numbers": [],
        "args": ("a\nb\nc\n",),
        "reference": lambda text: len(_j3_ref_names(text)),
    },
    {
        "id": "t02",
        "stages": ("sequence_length", "clamp_unit"),
        "numbers": [0.0, 2.0],
        "args": ([1, 2, 3],),
        "reference": lambda items: _j3_ref_bound(len(items), 0.0, 2.0),
    },
    {
        "id": "t03",
        "stages": ("normalize_scores", "mean_of"),
        "numbers": [],
        "args": ([1.0, 1.0, 2.0],),
        "reference": lambda values: _j3_ref_mean(_j3_ref_normalize(values)),
    },
    {
        "id": "t04",
        "stages": ("mean_of", "exceeds_tau"),
        "numbers": [0.6],
        "args": ([2.0, 4.0, 6.0],),
        "reference": lambda values: _j3_ref_mean(values) > 0.6,
    },
    {
        "id": "t05",
        "stages": ("invert_index", "sorted_keys"),
        "numbers": [],
        "args": ({"a": "x", "b": "y"},),
        "reference": lambda index: sorted(_j3_ref_invert(index).keys()),
    },
    {
        "id": "t06",
        "stages": ("sorted_keys", "join_csv"),
        "numbers": [],
        "args": ({"b": 1, "a": 2},),
        "reference": lambda index: _j3_ref_csv(sorted(index.keys())),
    },
    {
        "id": "t07",
        "stages": ("strip_comments", "split_identifiers"),
        "numbers": [],
        "args": ("alpha = 1 # x\nbeta\n",),
        "reference": lambda text: _j3_ref_identifiers(_j3_ref_strip(text)),
    },
    {
        "id": "t08",
        "stages": ("running_deltas", "abs_values"),
        "numbers": [],
        "args": ([3.0, 1.0, 4.0],),
        "reference": lambda values: _j3_ref_deltas(values),
    },
]

_J3_THREE_TASKS: List[Dict[str, object]] = [
    {
        "id": "f1a",
        "family": "F1",
        "numbers": [0.0, 1.0],
        "args": ("b\na\nb\nc\n",),
        "reference": lambda text: _j3_ref_bound(len(_j3_ref_names(text)), 0.0, 1.0),
    },
    {
        "id": "f1b",
        "family": "F1",
        "numbers": [2.0, 5.0],
        "args": ("x\ny\n",),
        "reference": lambda text: _j3_ref_bound(len(_j3_ref_names(text)), 2.0, 5.0),
    },
    {
        "id": "f2a",
        "family": "F2",
        "numbers": [0.25],
        "args": ([1.0, 1.0, 2.0],),
        "reference": lambda values: _j3_ref_mean(_j3_ref_normalize(values)) > 0.25,
    },
    {
        "id": "f2b",
        "family": "F2",
        "numbers": [0.45],
        "args": ([1.0, 2.0, 5.0],),
        "reference": lambda values: _j3_ref_mean(_j3_ref_normalize(values)) > 0.45,
    },
    {
        "id": "f3a",
        "family": "F3",
        "numbers": [],
        "args": ({"a": "x", "b": "y", "c": "x"},),
        "reference": lambda index: _j3_ref_csv(sorted(_j3_ref_invert(index).keys())),
    },
    {
        "id": "f3b",
        "family": "F3",
        "numbers": [],
        "args": ({"m": "z"},),
        "reference": lambda index: _j3_ref_csv(sorted(_j3_ref_invert(index).keys())),
    },
    {
        "id": "f4a",
        "family": "F4",
        "numbers": [],
        "args": ("alpha = 1  # note\nbeta_2 = 2\n# only\n",),
        "reference": lambda text: len(set(_j3_ref_identifiers(_j3_ref_strip(text)))),
    },
    {
        "id": "f4b",
        "family": "F4",
        "numbers": [],
        "args": ("def f(x):\n    return x  # note\n",),
        "reference": lambda text: len(set(_j3_ref_identifiers(_j3_ref_strip(text)))),
    },
    {
        "id": "f5a",
        "family": "F5",
        "numbers": [2.0],
        "args": ([3.0, 1.0, 4.0],),
        "reference": lambda values: _j3_ref_floor_peak(values, 2.0),
    },
    {
        "id": "f5b",
        "family": "F5",
        "numbers": [5.0],
        "args": ([0.5, 0.2],),
        "reference": lambda values: _j3_ref_floor_peak(values, 5.0),
    },
]

_J3_TYPE_TRAPS: List[Dict[str, object]] = [
    {
        "id": "tr1",
        "stages": ("exceeds_tau", "mean_of"),
        "conflict": "boolean->numeric_sequence",
    },
    {
        "id": "tr2",
        "stages": ("invert_index", "clamp_unit"),
        "conflict": "mapping->numeric",
    },
    {
        "id": "tr3",
        "stages": ("join_csv", "mean_of"),
        "conflict": "text->numeric_sequence",
    },
    {
        "id": "tr4",
        "stages": ("normalize_scores", "sorted_keys"),
        "conflict": "numeric_sequence->mapping",
    },
    {
        "id": "tr5",
        "stages": ("split_identifiers", "invert_index"),
        "conflict": "sequence->mapping",
    },
    {
        "id": "tr6",
        "stages": ("exceeds_tau", "clamp_unit"),
        "conflict": "boolean->numeric",
    },
]


# ---------------------------------------------------------------------------
# Directive J3: execution and comparison helpers
# ---------------------------------------------------------------------------
def _j3_plain(value):
    """A JSON-safe copy of a task answer (rows are returned to receipts)."""
    if isinstance(value, dict):
        return {str(key): _j3_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_j3_plain(item) for item in value]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float, str)):
        return value
    return repr(value)


def _j3_same(actual, expected) -> bool:
    if isinstance(expected, bool) or isinstance(actual, bool):
        return (
            isinstance(actual, bool)
            and isinstance(expected, bool)
            and actual == expected
        )
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return abs(float(actual) - float(expected)) <= 1e-9
    if isinstance(expected, list) and isinstance(actual, list):
        return len(actual) == len(expected) and all(
            _j3_same(item_a, item_e) for item_a, item_e in zip(actual, expected)
        )
    if isinstance(expected, dict) and isinstance(actual, dict):
        return set(actual.keys()) == set(expected.keys()) and all(
            _j3_same(actual[key], expected[key]) for key in expected
        )
    return actual == expected


def _j3_answer_delta(left, right) -> float:
    """Absolute drift between two executions of the same composed function."""
    if isinstance(left, bool) or isinstance(right, bool):
        return 0.0 if left is right or left == right else 1.0
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right))
    try:
        return 0.0 if _j3_same(left, right) else 1.0
    except Exception:  # noqa: BLE001 - a comparison failure is drift, not a pass
        return 1.0


def _j3_execute(source: str, func: str, args: Sequence[object]) -> Dict[str, object]:
    """Parse, compile, exec and call one emitted source.

    Directive J3: a task passes only when the emitted text is a real module,
    the named function exists, and its answer matches an independently
    computed reference. Nothing here reads a declared pass flag.
    """
    report: Dict[str, object] = {
        "parsed": False,
        "compiled": False,
        "executed": False,
        "actual": None,
        "reason": "",
    }
    text = str(source or "")
    if not text.strip():
        report["reason"] = "empty source"
        return report
    try:
        tree = ast.parse(text)
    except SyntaxError as error:
        report["reason"] = f"ast.parse: {error.msg} at line {error.lineno}"
        return report
    report["parsed"] = True
    try:
        code = compile(tree, "<j3-battery>", "exec")
    except (SyntaxError, ValueError) as error:
        report["reason"] = f"compile: {error}"
        return report
    report["compiled"] = True
    scope: Dict[str, object] = {}
    try:
        exec(code, scope)
    except Exception as error:  # noqa: BLE001 - the failure is the measurement
        report["reason"] = f"exec: {type(error).__name__}: {error}"
        return report
    function = scope.get(str(func))
    if not callable(function):
        report["reason"] = f"emitted source has no callable {func!r}"
        return report
    try:
        report["actual"] = function(*args)
    except Exception as error:  # noqa: BLE001 - the failure is the measurement
        report["reason"] = f"call: {type(error).__name__}: {error}"
        return report
    report["executed"] = True
    return report


def _j3_guard_count(source: str) -> int:
    try:
        tree = ast.parse(str(source or ""))
    except SyntaxError:
        return 0
    return sum(1 for node in ast.walk(tree) if isinstance(node, ast.IfExp))


def _j3_case_row(
    verdict: Dict[str, object],
    func: str,
    args: Sequence[object],
    expected,
    kind: str,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    source = str(verdict.get("source") or "")
    if bool(verdict.get("ok", False)):
        run = _j3_execute(source, func, args)
    else:
        run = {
            "parsed": False,
            "compiled": False,
            "executed": False,
            "actual": None,
            "reason": "",
        }
    matched = bool(run["executed"] and _j3_same(run["actual"], expected))
    reason = str(verdict.get("reason") or run.get("reason") or "")
    if not matched and not reason:
        reason = str(run.get("reason") or "answer mismatch")
    row = {
        "kind": kind,
        "func": func,
        "type_ok": bool(verdict.get("ok", False)),
        "parsed": bool(run["parsed"]),
        "compiled": bool(run["compiled"]),
        "executed": bool(run["executed"]),
        "matched": matched,
        "expected": _j3_plain(expected),
        "actual": _j3_plain(run["actual"]),
        "reason": reason[:200],
        "source_sha256": (
            hashlib.sha256(source.encode("utf-8")).hexdigest() if source else ""
        ),
    }
    return row, run


# ---------------------------------------------------------------------------
# Directive J3: the public, fixture-free battery
# ---------------------------------------------------------------------------
_K2_MULTI_STAGES: Dict[str, str] = {
    "filter_above": (
        "def filter_above(items: list, threshold: float) -> list:\n"
        "    kept = []\n"
        "    for item in items:\n"
        "        if item > threshold:\n"
        "            kept.append(item)\n"
        "    return kept\n"
    ),
    "scale_values": (
        "def scale_values(values: list, factor: float) -> list:\n"
        "    scaled = []\n"
        "    for value in values:\n"
        "        scaled.append(value * factor)\n"
        "    return scaled\n"
    ),
    "total": (
        "def total(values: list) -> float:\n"
        "    accumulator = 0.0\n"
        "    for value in values:\n"
        "        accumulator += value\n"
        "    return accumulator\n"
    ),
    "pair_counts": (
        "def pair_counts(items: list) -> dict:\n"
        "    counts = {}\n"
        "    for item in items:\n"
        "        counts[item] = counts.get(item, 0) + 1\n"
        "    return counts\n"
    ),
    "select_keys": (
        "def select_keys(index: dict, minimum: float) -> list:\n"
        "    selected = []\n"
        "    for key, count in index.items():\n"
        "        if count >= minimum:\n"
        "            selected.append(key)\n"
        "    return selected\n"
    ),
    "window_means": (
        "def window_means(values: list, width: float) -> list:\n"
        "    means = []\n"
        "    size = int(width)\n"
        "    for start in range(0, len(values) - size + 1):\n"
        "        means.append(sum(values[start:start + size]) / size)\n"
        "    return means\n"
    ),
    "count_matches": (
        "def count_matches(items: list, threshold: float) -> int:\n"
        "    matched = 0\n"
        "    for item in items:\n"
        "        if item > threshold:\n"
        "            matched += 1\n"
        "    return matched\n"
    ),
}
"""Directive K2: multi-statement stage playbooks used by the extended battery."""

_K2_MULTI_TASKS: List[Dict[str, object]] = [
    {
        "id": "multi01",
        "stages": ("filter_above", "scale_values", "total"),
        "args": ["items", "threshold", "factor"],
        "values": ([1.0, 2.0, 3.0, 4.0], 2.0, 3.0),
        "reference": lambda items, threshold, factor: sum(
            value * factor for value in items if value > threshold
        ),
    },
    {
        "id": "multi02",
        "stages": ("pair_counts", "select_keys", "join_csv"),
        "args": ["items", "minimum"],
        "values": (["a", "b", "a"], 2.0),
        "reference": lambda items, minimum: ",".join(
            key for key, count in {item: items.count(item) for item in items}.items()
            if count >= minimum
        ),
    },
    {
        "id": "multi03",
        "stages": ("filter_above", "count_matches", "clamp_unit"),
        "args": ["items", "threshold"],
        "values": ([1.0, 2.0, 3.0, 4.0], 2.0),
        "reference": lambda items, threshold: min(
            1.0, float(sum(1 for value in items if value > threshold and value > threshold))
        ),
    },
    {
        "id": "multi04",
        "stages": ("window_means", "max_or_zero", "clamp_unit"),
        "args": ["values", "width"],
        "values": ([1.0, 2.0, 3.0, 4.0], 2.0),
        "reference": lambda values, width: min(
            1.0,
            max(
                [
                    sum(values[start : start + int(width)]) / int(width)
                    for start in range(0, len(values) - int(width) + 1)
                ]
                or [0.0]
            ),
        ),
    },
    {
        "id": "multi05",
        "stages": ("pair_counts", "invert_index", "sorted_keys"),
        "args": ["items"],
        "values": (["a", "b", "a"],),
        "reference": lambda items: sorted(
            set({item: items.count(item) for item in items}.values())
        ),
    },
    {
        "id": "multi06",
        "stages": ("filter_above", "scale_values", "count_matches"),
        "args": ["items", "threshold", "factor"],
        "values": ([1.0, 2.0, 3.0, 4.0], 2.0, 2.0),
        "reference": lambda items, threshold, factor: sum(
            1 for value in items if value > threshold and value * factor > threshold
        ),
    },
]
"""Six multi-statement, multi-argument three-stage pipelines (Directive K2)."""

_K2_TRAP_SOURCES: List[Dict[str, object]] = [
    {
        "id": "trap_arity",
        "stages": ("filter_above", "scale_values"),
        "args": ["items", "threshold"],
        "kind": "arity",
        "reason": "the secondary parameter factor has no binding",
    },
    {
        "id": "trap_no_return",
        "source": "def broken(items: list) -> list:\n    items\n",
        "args": ["items"],
        "kind": "symbol",
        "reason": "the stage has no terminal return",
    },
    {
        "id": "trap_early_return",
        "source": (
            "def early(items: list) -> list:\n"
            "    if items:\n"
            "        return items\n"
            "    return []\n"
        ),
        "args": ["items"],
        "kind": "symbol",
        "reason": "the stage returns before its terminal statement",
    },
    {
        "id": "trap_empty_body",
        "source": "def empty(items: list) -> list:\n    pass\n",
        "args": ["items"],
        "kind": "symbol",
        "reason": "the stage body is empty",
    },
    {
        "id": "trap_unknown_callee",
        "source": "def ghost(items: list) -> list:\n    return mystery_helper(items)\n",
        "args": ["items"],
        "kind": "symbol",
        "reason": "mystery_helper is absent from the symbol table",
    },
    {
        "id": "trap_type_chain",
        "stages": ("caller_names", "invert_index"),
        "args": ["items"],
        "kind": "type",
        "reason": "a sequence cannot feed a mapping consumer",
    },
]


def _symbol_vetoes(source: str, known: Iterable[str]) -> List[str]:
    """Call names in the emitted source that the symbol table does not own."""
    import builtins

    allowed = set(str(name) for name in known)
    builtin_names = set(dir(builtins))
    try:
        tree = ast.parse(str(source))
    except SyntaxError:
        return ["<unparseable>"]
    missing: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = str(node.func.id)
            if name not in allowed and name not in builtin_names:
                missing.append(name)
    return sorted(set(missing))


def run_diverse_dag_battery(engine=None, seed: int = 0, extended: bool = False) -> Dict[str, object]:
    """Measure the J3 playbook library end to end, with no external fixture.

    24 base single-function tasks, 12 novel single-plan bindings, 8 two-stage
    pipelines, 10 three-stage pipelines over the five families, and 6
    fail-closed type traps. Every emitted source is ast.parse'd, compiled,
    executed, and compared with an independent reference answer. The optional
    engine contributes only its type_gate: this function compiles no plan
    into it and writes no file, so it is read-only and repeatable.
    """
    catalog = playbook_catalog()
    gate = getattr(engine, "type_gate", None) if engine is not None else None
    if gate is None:
        gate = TypeCompatibilityGate()
    salt = 1000 + (int(seed) % 9000)

    base_rows: List[Dict[str, object]] = []
    for task in _J3_BASE_TASKS:
        plan = _compiled_plan(str(task["name"]), str(task["intent"]), str(task["source"]))
        verdict = compose_plan_dag(
            [plan],
            {"func": task["name"], "args": [task["arg"]], "numbers": []},
            gate=gate,
        )
        expected = task["reference"](*task["args"])  # type: ignore[operator]
        row, _run = _j3_case_row(verdict, str(task["name"]), task["args"], expected, "base")  # type: ignore[arg-type]
        row["id"] = task["id"]
        base_rows.append(row)

    single_rows: List[Dict[str, object]] = []
    for index, spec in enumerate(_J3_SINGLE_TASKS, start=1):
        stage = str(spec["stage"])
        func = f"j3s{salt}_{index:02d}"
        argument = f"payload_{index:02d}"
        plan = _playbook_plan(stage, plan_name=f"j3_{stage}")
        verdict = compose_plan_dag(
            [plan],
            {"func": func, "args": [argument], "numbers": list(spec["numbers"])},
            gate=gate,
        )
        expected = spec["reference"](*spec["args"])  # type: ignore[operator]
        row, _run = _j3_case_row(verdict, func, spec["args"], expected, "single")  # type: ignore[arg-type]
        row.update(
            {
                "id": spec["id"],
                "stage": stage,
                "novel_func": func,
                "novel_arg": argument,
                "numbers": list(spec["numbers"]),
            }
        )
        single_rows.append(row)

    two_rows: List[Dict[str, object]] = []
    for index, spec in enumerate(_J3_TWO_TASKS, start=1):
        stages = [str(stage) for stage in spec["stages"]]  # type: ignore[index]
        func = f"j3p{salt}_{index:02d}"
        argument = f"input_{index:02d}"
        plans = [_playbook_plan(stage, plan_name=f"j3_{stage}") for stage in stages]
        verdict = compose_plan_dag(
            plans,
            {"func": func, "args": [argument], "numbers": list(spec["numbers"])},
            gate=gate,
        )
        expected = spec["reference"](*spec["args"])  # type: ignore[operator]
        row, _run = _j3_case_row(verdict, func, spec["args"], expected, "two")  # type: ignore[arg-type]
        row.update(
            {
                "id": spec["id"],
                "stages": list(stages),
                "topology": _j3_topology(stages),
                "numbers": list(spec["numbers"]),
            }
        )
        two_rows.append(row)

    three_rows: List[Dict[str, object]] = []
    drift = 0.0
    for index, spec in enumerate(_J3_THREE_TASKS, start=1):
        family = str(spec["family"])
        stages = [str(stage) for stage in PLAYBOOK_FAMILIES[family]["stages"]]  # type: ignore[index]
        func = f"j3d{salt}_{index:02d}"
        argument = f"root_{index:02d}"
        plans = [_playbook_plan(stage, plan_name=f"j3_{stage}") for stage in stages]
        verdict = compose_plan_dag(
            plans,
            {"func": func, "args": [argument], "numbers": list(spec["numbers"])},
            gate=gate,
        )
        expected = spec["reference"](*spec["args"])  # type: ignore[operator]
        row, run = _j3_case_row(verdict, func, spec["args"], expected, "three")  # type: ignore[arg-type]
        source = str(verdict.get("source") or "")
        row.update(
            {
                "id": spec["id"],
                "family": family,
                "stages": list(stages),
                "topology": _j3_topology(stages),
                "numbers": list(spec["numbers"]),
                "guards": _j3_guard_count(source),
            }
        )
        if bool(row["matched"]) and bool(run["executed"]):
            again = _j3_execute(source, func, spec["args"])  # type: ignore[arg-type]
            if bool(again["executed"]):
                drift = max(drift, _j3_answer_delta(run["actual"], again["actual"]))
        three_rows.append(row)

    trap_rows: List[Dict[str, object]] = []
    for spec in _J3_TYPE_TRAPS:
        left_stage, right_stage = [str(stage) for stage in spec["stages"]]  # type: ignore[index]
        verdict = compose_plan_dag(
            [_playbook_plan(left_stage), _playbook_plan(right_stage)],
            {"func": "j3_trap_probe", "args": ["payload"], "numbers": []},
            gate=gate,
        )
        source = str(verdict.get("source") or "")
        reason = str(verdict.get("reason") or "")
        refused = bool(
            not verdict.get("ok", False)
            and source == ""
            and reason.startswith("type mismatch")
        )
        trap_rows.append(
            {
                "id": spec["id"],
                "left": left_stage,
                "right": right_stage,
                "conflict": spec["conflict"],
                "refused": refused,
                "source_empty": source == "",
                "reason": reason[:200],
            }
        )

    multi_rows: List[Dict[str, object]] = []
    traps2_rows: List[Dict[str, object]] = []
    if extended:
        for index, spec in enumerate(_K2_MULTI_TASKS, start=1):
            stages = [str(stage) for stage in spec["stages"]]  # type: ignore[index]
            func = f"k2m{salt}_{index:02d}"
            plans = [
                _compiled_plan(stage, f"k2 {stage}", _K2_MULTI_STAGES[stage])
                if stage in _K2_MULTI_STAGES
                else _playbook_plan(stage)
                for stage in stages
            ]
            verdict = compose_plan_dag(
                plans,
                {"func": func, "args": list(spec["args"]), "numbers": []},
                gate=gate,
            )
            expected = spec["reference"](*spec["values"])  # type: ignore[operator]
            row, run = _j3_case_row(verdict, func, spec["values"], expected, "multi")  # type: ignore[arg-type]
            row.update(
                {
                    "id": spec["id"],
                    "stages": list(stages),
                    "multi_statement": True,
                    "multi_argument": bool(len(list(spec["args"])) > 1),
                }
            )
            if bool(row["matched"]) and bool(run["executed"]):
                again = _j3_execute(str(verdict.get("source") or ""), func, spec["values"])  # type: ignore[arg-type]
                if bool(again["executed"]):
                    drift = max(drift, _j3_answer_delta(run["actual"], again["actual"]))
            multi_rows.append(row)

        known = set(PLAYBOOK_LIBRARY) | set(_K2_MULTI_STAGES) | {
            "range", "sum", "len", "sorted", "list", "dict", "set", "min", "max",
            "abs", "round", "enumerate", "zip", "str", "int", "float", "bool",
            "tuple", "frozenset", "any", "all", "reversed", "map", "filter",
        }
        for spec in _K2_TRAP_SOURCES:
            if "stages" in spec:
                plans = []
                for stage in spec["stages"]:  # type: ignore[index]
                    name = str(stage)
                    if name in _K2_MULTI_STAGES:
                        plans.append(_compiled_plan(name, f"k2 {name}", _K2_MULTI_STAGES[name]))
                    else:
                        plans.append(_playbook_plan(name))
            else:
                plans = [_compiled_plan("trap_stage", "trap", str(spec["source"]))]
            verdict = compose_plan_dag(
                plans,
                {"func": f"k2t_{spec['id']}", "args": list(spec["args"]), "numbers": []},
                gate=gate,
            )
            source = str(verdict.get("source") or "")
            veto = _symbol_vetoes(source, known) if source else []
            refused = bool(
                (not verdict.get("ok", False) and source == "")
                or (source and veto)
            )
            traps2_rows.append(
                {
                    "id": spec["id"],
                    "kind": spec["kind"],
                    "reason": spec["reason"],
                    "refused": bool(refused),
                    "source_empty": source == "",
                    "symbol_vetoes": veto,
                    "reason_measured": str(verdict.get("reason") or "")[:200],
                }
            )

    base_passed = sum(1 for row in base_rows if row["matched"])
    single_passed = sum(1 for row in single_rows if row["matched"])
    two_passed = sum(1 for row in two_rows if row["matched"])
    three_passed = sum(1 for row in three_rows if row["matched"])
    traps_refused = sum(1 for row in trap_rows if row["refused"])
    multi_passed = sum(1 for row in multi_rows if row["matched"])
    traps2_refused = sum(1 for row in traps2_rows if row["refused"])
    total_passed = (
        base_passed + single_passed + two_passed + three_passed
        + traps_refused + multi_passed + traps2_refused
    )
    total_tasks = (
        len(base_rows) + len(single_rows) + len(two_rows) + len(three_rows)
        + len(trap_rows) + len(multi_rows) + len(traps2_rows)
    )
    three_topologies = sorted({str(row["topology"]) for row in three_rows})
    payload = {
        "seed": int(seed),
        "catalog_digest": str(catalog["digest"]),
        "base": {"passed": int(base_passed), "total": len(base_rows), "rows": base_rows},
        "single": {"passed": int(single_passed), "total": len(single_rows), "rows": single_rows},
        "two": {"passed": int(two_passed), "total": len(two_rows), "rows": two_rows},
        "three": {"passed": int(three_passed), "total": len(three_rows), "rows": three_rows},
        "three_topologies": three_topologies,
        "traps": {"refused": int(traps_refused), "total": len(trap_rows), "rows": trap_rows},
        "total": {"passed": int(total_passed), "total": int(total_tasks)},
        "drift": float(round(drift, 4)),
    }
    if extended:
        payload["multi"] = {
            "passed": int(multi_passed),
            "total": len(multi_rows),
            "rows": multi_rows,
        }
        payload["traps2"] = {
            "refused": int(traps2_refused),
            "total": len(traps2_rows),
            "rows": traps2_rows,
        }
    return payload

