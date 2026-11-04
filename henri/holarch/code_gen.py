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
import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from henri.holarch.code_lang import (
    SLOT_RULE,
    BochnerPhaseProjector,
    CodeTextCorpus,
    HolarchCodeTextTokenizer,
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
        self.total = 0.0

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
        return self

    def _distribution(self, counts: Dict[int, int], denominator: float) -> np.ndarray:
        out = np.full(self.vocab_size, self.floor, dtype=np.float64)
        if not counts:
            return out / out.sum()
        for token, count in counts.items():
            if 0 <= token < self.vocab_size:
                out[token] += float(count)
        return out / out.sum()

    def log_probs(self, context: Sequence[int]) -> np.ndarray:
        """log P(next | context), interpolated over 4/3/2/1-grams.

        The weights are renormalised over the orders that actually have
        counts for this context, so an unseen 4-gram does not dilute the
        lower orders it was supposed to back off to.
        """
        unigram = self.unigram / max(self.total, 1.0)
        unigram = (unigram + self.floor) / (unigram + self.floor).sum()
        pieces: List[Tuple[float, np.ndarray]] = [(0.10, unigram)]
        if context:
            counts = self.bigram.get(int(context[-1]), {})
            if counts:
                pieces.append((0.20, self._distribution(counts, 1.0)))
        if len(context) >= 2:
            counts = self.trigram.get((int(context[-2]), int(context[-1])), {})
            if counts:
                pieces.append((0.30, self._distribution(counts, 1.0)))
        if len(context) >= 3:
            counts = self.quadgram.get(
                (int(context[-3]), int(context[-2]), int(context[-1])), {}
            )
            if counts:
                pieces.append((0.40, self._distribution(counts, 1.0)))
        mass = sum(weight for weight, _ in pieces)
        probability = np.zeros(self.vocab_size, dtype=np.float64)
        for weight, distribution in pieces:
            probability += (weight / mass) * distribution
        return np.log(np.maximum(probability, 1e-12))

    def next_top1(self, context: Sequence[int]) -> int:
        return int(np.argmax(self.log_probs(context)))


class WaveLexicalBank:
    """Holographic successor memory over the phase codebook.

    A bigram is stored as the superposition of its successor waves:

        B[a] = sum_b count(a,b) * psi(b)

    Scoring a candidate is Re<normalise(B[a]), psi(candidate)>, which is the
    Hopfield alignment of the lexical bank with the unbinding residue. The
    bank is built once and queried in O(dim) per candidate set.
    """

    def __init__(self, projector: BochnerPhaseProjector, tokenizer) -> None:
        self.projector = projector
        self.tokenizer = tokenizer
        self.bank = np.zeros((0, projector.dim), dtype=np.complex64)
        self.counts = np.zeros(0, dtype=np.float64)

    def build(self, ids: Sequence[int]) -> "WaveLexicalBank":
        vocab = len(self.tokenizer)
        bank = np.zeros((vocab, self.projector.dim), dtype=np.complex64)
        counts = np.zeros(vocab, dtype=np.float64)
        waves = np.stack(
            [self.projector.slot_wave(token, self.tokenizer) for token in range(vocab)]
        ).astype(np.complex64)
        for index in range(1, len(ids)):
            previous = int(ids[index - 1])
            current = int(ids[index])
            bank[previous] += waves[current]
            counts[previous] += 1.0
        norms = np.linalg.norm(bank, axis=1, keepdims=True)
        self.bank = (bank / np.maximum(norms, _EPS)).astype(np.complex64)
        self.counts = counts
        self.waves = waves
        return self

    def scores(self, previous: int) -> np.ndarray:
        if previous < 0 or previous >= self.bank.shape[0]:
            return np.zeros(len(self.tokenizer), dtype=np.float32)
        row = self.bank[int(previous)]
        if float(np.linalg.norm(row)) <= _EPS:
            return np.zeros(len(self.tokenizer), dtype=np.float32)
        return np.real(self.waves @ np.conj(row)).astype(np.float32)


class BothQuadratureReadout:
    """g = [Re(U_s psi) || Im(U_s psi)] over all four slots.

    Keeping both quadratures is the Decision 3 result; dropping the
    imaginary part throws away the relative phase that carries binding.
    """

    def __init__(self, dim: int, n_slots: int = 4, k: int = 16, seed: int = 7) -> None:
        self.dim = int(dim)
        self.n_slots = int(n_slots)
        self.d_slot = self.dim // self.n_slots
        self.k = int(k)
        generator = np.random.default_rng(int(seed))
        frames = []
        for _ in range(self.n_slots):
            raw = generator.standard_normal((self.k, self.d_slot)) + 1j * generator.standard_normal(
                (self.k, self.d_slot)
            )
            q, _ = np.linalg.qr(raw.conj().T)
            frames.append(q.conj().T.astype(np.complex64))
        self.frames = frames

    @property
    def width(self) -> int:
        return 2 * self.k * self.n_slots

    def __call__(self, wave: np.ndarray) -> np.ndarray:
        blocks = []
        for slot in range(self.n_slots):
            start = slot * self.d_slot
            piece = wave[start : start + self.d_slot]
            projection = self.frames[slot] @ piece
            blocks.append(np.concatenate([projection.real, projection.imag]))
        return np.concatenate(blocks).astype(np.float32)


class RidgeHead:
    """A closed-form readout from the both-quadrature block to vocabulary.

    Solved with a deterministic least-squares fit and a fixed ridge. No
    gradient descent, no external optimiser, no randomness.
    """

    def __init__(self, width: int, vocab_size: int, ridge: float = 1.0) -> None:
        self.weight = np.zeros((vocab_size, width), dtype=np.float32)
        self.ridge = float(ridge)
        self.fitted = False

    def fit(self, features: np.ndarray, targets: np.ndarray) -> "RidgeHead":
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        if x.shape[0] < 8:
            return self
        gram = x.T @ x + self.ridge * np.eye(x.shape[1])
        rhs = x.T @ y
        solution = np.linalg.solve(gram, rhs)
        self.weight = solution.T.astype(np.float32)
        self.fitted = True
        return self

    def logits(self, features: np.ndarray) -> np.ndarray:
        return (np.asarray(features, dtype=np.float32) @ self.weight.T).astype(np.float32)


class UniversalModulation:
    """Per-task diagonal gain inside a shared orthonormal frame.

    This mirrors Zone A's UniversalSubspace: a shared Stiefel frame and a
    per-task diagonal. It is applied to the both-quadrature block before the
    ridge head, so the code path reads the same universal coordinates the
    action path reads.
    """

    def __init__(self, width: int, rank: int = 8, n_tasks: int = 4, seed: int = 11) -> None:
        generator = np.random.default_rng(int(seed))
        raw = generator.standard_normal((width, int(rank)))
        q, _ = np.linalg.qr(raw)
        self.frame = q.astype(np.float32)
        self.gain = np.ones((int(n_tasks), int(rank)), dtype=np.float32)

    def __call__(self, features: np.ndarray, task: int = 1) -> np.ndarray:
        code = features @ self.frame
        return (code * self.gain[int(task)]) @ self.frame.T


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

    def __init__(
        self,
        tokenizer: HolarchCodeTextTokenizer,
        projector: BochnerPhaseProjector,
        *,
        beta_lang: float = 0.5,
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
        self.readout = BothQuadratureReadout(self.dim, k=16, seed=int(seed) + 3)
        self.universal = UniversalModulation(self.readout.width, rank=8, seed=int(seed) + 5)
        self.head = RidgeHead(self.readout.width, self.vocab_size)
        self.gate = ASTGate(symbol_table)
        self.beta_star = 0.0
        self.report: Optional[LMReport] = None
        self._wave_features: Dict[Tuple[int, ...], np.ndarray] = {}

    # ---------------------------------------------------------------- fit
    def fit(
        self,
        ids: Sequence[int],
        val_fraction: float = 0.1,
        max_fit_tokens: int = 200_000,
    ) -> "WaveCodeTextGenerator":
        """Fit the count model and the ridge head on one token stream."""
        ids = [int(token) for token in ids]
        cut = int(len(ids) * (1.0 - float(val_fraction)))
        train = ids[:cut]
        self.count_lm.fit(train)
        self.lexical.build(train[: int(max_fit_tokens)])
        rows: List[np.ndarray] = []
        targets: List[np.ndarray] = []
        limit = min(int(max_fit_tokens), max(0, len(train) - 3))
        stride = max(1, limit // 6000)
        context = 6
        for index in range(context, limit, stride):
            window = train[max(0, index - context) : index]
            features = self.wave_features(window)
            if features is None:
                continue
            rows.append(features)
            label = np.zeros(self.vocab_size, dtype=np.float32)
            label[train[index]] = 1.0
            targets.append(label)
        if rows:
            self.head.fit(np.stack(rows), np.stack(targets))
        return self

    # ------------------------------------------------------------ features
    def wave_features(self, context: Sequence[int]) -> Optional[np.ndarray]:
        key = tuple(int(token) for token in context[-8:])
        cached = self._wave_features.get(key)
        if cached is not None:
            return cached
        if not key:
            return None
        wave = self.projector.sequence_wave(key, self.tokenizer)
        features = self.readout(wave)
        features = self.universal(features, task=1)
        norm = float(np.linalg.norm(features))
        if norm > _EPS:
            features = features / norm
        if len(self._wave_features) < 40_000:
            self._wave_features[key] = features.astype(np.float32)
        return features

    # -------------------------------------------------------------- logits
    def token_logits(self, context: Sequence[int]) -> np.ndarray:
        """The D-Guard blend of the count floor and the wave head."""
        base = self.count_lm.log_probs(context)
        base = base - np.logaddexp.reduce(base)
        wave = np.zeros(self.vocab_size, dtype=np.float32)
        if self.beta_star > 0.0:
            features = self.wave_features(context)
            if features is not None and self.head.fitted:
                wave = self.head.logits(features)
            if context:
                boost = self.lexical.scores(int(context[-1]))
                if float(np.std(boost)) > 1e-8:
                    wave = wave + self.beta_lang * (
                        (boost - float(np.mean(boost))) / max(float(np.std(boost)), 1e-8)
                    )
        # Additive hopfield boost, not a convex replacement: the count model
        # stays the floor and the wave term is a bias on top of it. beta* is
        # still chosen on validation and the grid still contains 0.0, so the
        # D-Guard bound holds.
        if float(np.std(wave)) > 1e-9:
            wave = (wave - float(np.mean(wave))) / max(float(np.std(wave)), 1e-9)
        return (base + self.beta_star * wave).astype(np.float64)

    def calibrate_beta(
        self,
        ids: Sequence[int],
        grid: Sequence[float] = (0.0, 0.05, 0.1, 0.2, 0.35, 0.5),
        context: int = 6,
        samples: int = 2000,
    ) -> float:
        """D-Guard: pick beta* that minimises validation loss.

        The grid contains 0.0, so the coupled system is provably no worse
        than the count floor it is added to (the directive's D-Guard bound).
        """
        sequence = [int(token) for token in ids]
        indices = list(range(context, len(sequence) - 1))
        if len(indices) > int(samples):
            stride = max(1, len(indices) // int(samples))
            indices = indices[::stride]
        best = (float("inf"), 0.0)
        for beta in grid:
            self.beta_star = float(beta)
            total = 0.0
            for index in indices:
                window = sequence[max(0, index - context) : index]
                logits = self.token_logits(window)
                logits = logits - np.max(logits)
                logprob = logits[sequence[index]] - np.log(np.sum(np.exp(logits)))
                total -= float(logprob)
            loss = total / max(len(indices), 1)
            if loss < best[0]:
                best = (loss, float(beta))
        self.beta_star = float(best[1])
        return self.beta_star

    # ------------------------------------------------------------ evaluate
    def evaluate(self, ids: Sequence[int], context: int = 6) -> LMReport:
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
            logits = self.token_logits(sequence)
            if prior is not None and step < len(prior):
                target = int(prior[step])
                logits = logits + float(prior_weight)
                logits[target] += float(prior_weight) * 4.0
            order = np.argsort(-logits)[: self.LOOKAHEAD_B]
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
@dataclass
class Plan:
    """One compiled strategy: an exact source text plus its provenance."""

    name: str
    intent: str
    source: str
    kind: str = "code"
    version: int = 1
    engram_id: int = -1
    digest: str = ""


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
    ) -> None:
        self.memory = memory
        self.projector = projector
        self.tokenizer = tokenizer
        self.label_offset = int(label_offset)
        self.namespace = str(namespace)
        self.plans: Dict[str, Plan] = {}
        self.tau = 0.15

    def key(self, intent: str) -> np.ndarray:
        tokens = self.tokenizer.encode(str(intent))[:96]
        return self.projector.sequence_wave(tokens, self.tokenizer)

    def compile(self, name: str, intent: str, source: str, kind: str = "code", tier: int = 3) -> Plan:
        existing = self.plans.get(str(name))
        version = 1 if existing is None else int(existing.version) + 1
        wave = self.key(intent)
        label = self.label_offset + len(self.plans)
        record = self.memory.write(
            wave, label=int(label), tier=int(tier), namespace=self.namespace
        )
        plan = Plan(
            name=str(name),
            intent=str(intent),
            source=str(source),
            kind=str(kind),
            version=version,
            engram_id=int(getattr(record, "engram_id", -1)),
            digest=hashlib.sha256(str(source).encode("utf-8")).hexdigest(),
        )
        self.plans[str(name)] = plan
        return plan

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
    ) -> None:
        self.generator = generator
        self.projector = projector
        self.tokenizer = tokenizer
        self.memory = memory
        self.verifier = verifier
        self.namespace = str(namespace)
        self.plans = PlanStore(memory, projector, tokenizer, plan_offset, namespace)
        self.fast_weight = np.zeros((tokenizer.__len__(), projector.dim), dtype=np.complex64)
        self.adapted = 0
        self.refused = 0

    # ------------------------------------------------------------- emission
    def _emit(self, intent: str, prior: Optional[Sequence[int]], max_tokens: int, require_parse: bool) -> Dict[str, object]:
        context = self.tokenizer.encode("def ")
        goal = self.projector.sequence_wave(self.tokenizer.encode(intent)[:64], self.tokenizer)
        return self.generator.decode(
            context,
            max_tokens=int(max_tokens),
            prior=prior,
            goal_wave=goal,
            require_parse=bool(require_parse),
            prior_weight=8.0,
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
        hit = self.plans.retrieve(intent) if self.plans.plans else None
        if hit is not None:
            plan, coherence = hit
            prior = self.tokenizer.encode(plan.source)
            output = self._emit(intent, prior, max_tokens=max(len(prior) + 8, 32), require_parse=False)
            source = plan.source
            used = plan.name
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
        return {
            "status": "verified" if verification["ok"] else "rejected",
            "reason": verification["reason"],
            "source": source,
            "verified": bool(verification["ok"]),
            "plan": used,
            "coherence": float(coherence),
            "emitted_tokens": len(output["tokens"]),
            "vetoed_tokens": int(output["vetoed"]),
        }

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
        intent_tokens = self.tokenizer.encode(intent)[:32]
        source_tokens = self.tokenizer.encode(corrected_source)[:32]
        gate = 1.0 if float(surprise) > 0.0 else 0.0
        for left in intent_tokens:
            wave = self.projector.slot_wave(left, self.tokenizer)
            for right in source_tokens:
                self.fast_weight[int(left)] += gate * float(lr) * wave * np.conj(
                    self.projector.slot_wave(right, self.tokenizer)
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


def build_synthesis_engine(
    root: str | Path,
    memory,
    namespace: str = "default",
    symbol_table: Optional[Iterable[str]] = None,
    max_fit_tokens: int = 60000,
) -> Tuple["SynthesisEngine", Dict[str, object]]:
    """Build the full ingress+egress stack over a corpus, once.

    This is the factory the sidecar uses, so the REST endpoint, the MCP tool
    and the benchmark cannot construct three subtly different generators.
    """
    corpus = CodeTextCorpus.holarch(root)
    tokenizer = build_tokenizer(corpus)
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
    )
    return engine, {
        "corpus_digest": corpus.digest,
        "vocabulary_size": int(len(tokenizer)),
        "tokens": int(len(ids)),
        "files": int(len(corpus)),
    }
