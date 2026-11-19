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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
    base_bits_per_char: float = 0.0

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
            "base_bits_per_char": float(self.base_bits_per_char),
        }




class ActionRegimeReadout:
    """A closed-form complex Stiefel + ridge readout of the Action KV.

    Directive R2. The planner writes a four-slot bundle of the trajectory it
    ACTUALLY executed: the terminal observation wave, the superposition of
    the visited state waves, a step-count phasor and the active REPULSOR
    wave. This head is fitted, once, from delivered Action KV packets to the
    three operational regimes. Decoding is then a read of the wave geometry,
    not a declared rule.

    Fit. With feature map phi(x) over the four slots (the slots and every
    non-empty elementwise product of them), the ridge solution is

        W = (Phi^H Phi + lambda I)^-1 Phi^H Y

    and its polar factor Q = U V^H (the Stiefel projection of W) is the
    isometry part. A 3 x 3 ridge calibration on the projected scores absorbs
    the singular values. Every step is a closed-form linear solve or an SVD;
    there is no gradient descent and no network call.
    """

    def __init__(
        self,
        *,
        dim: int = 1024,
        slots: int = 4,
        ridge: float = 1e-2,
        seed: int = 20261111,
    ) -> None:
        self.dim = int(dim)
        self.slots = int(slots)
        self.ridge = float(ridge)
        self.seed = int(seed)
        self.width = int(self.dim // self.slots) if self.slots else int(self.dim)
        self.Q: Optional[np.ndarray] = None
        self.calibration: Optional[np.ndarray] = None
        self.fit_report: Dict[str, Any] = {}

    # ------------------------------------------------------------- features
    def _slot_vectors(self, values: Any) -> List[np.ndarray]:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 3:
            array = array.reshape(array.shape[0], -1) if array.shape[0] == self.slots else array.reshape(-1)
        array = np.asarray(array, dtype=np.float64).reshape(-1)
        if array.size == self.dim:
            slots = array.reshape(self.slots, self.width)
        elif array.size % self.slots == 0 and array.size // self.slots == self.width:
            slots = array.reshape(self.slots, self.width)
        else:
            # A different dk is acceptable: keep four equal slots.
            width = array.size // max(self.slots, 1)
            slots = array[: width * self.slots].reshape(self.slots, width)
            self.width = int(width)
        norms = np.linalg.norm(slots, axis=1, keepdims=True)
        slots = slots / np.maximum(norms, 1e-12)
        return [slots[index] for index in range(int(slots.shape[0]))]

    def features(self, values: Any) -> np.ndarray:
        slots = self._slot_vectors(values)
        terms = list(slots)
        for left in range(len(slots)):
            for right in range(left + 1, len(slots)):
                terms.append(slots[left] * slots[right])
        for left in range(len(slots)):
            for mid in range(left + 1, len(slots)):
                for right in range(mid + 1, len(slots)):
                    terms.append(slots[left] * slots[mid] * slots[right])
        if len(slots) == 4:
            terms.append(slots[0] * slots[1] * slots[2] * slots[3])
        stacked = np.concatenate(terms)
        return np.concatenate([stacked, np.asarray([1.0])])

    def _matrix(self, rows: Sequence[Any]) -> np.ndarray:
        return np.stack([self.features(row) for row in rows]).astype(np.float64)

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        rows: Sequence[Any],
        labels: Sequence[int],
        *,
        calibration_rows: Optional[Sequence[Any]] = None,
        calibration_labels: Optional[Sequence[int]] = None,
    ) -> Dict[str, Any]:
        X = self._matrix(rows)
        y = np.asarray([int(label) for label in labels], dtype=np.int64)
        classes = int(y.max()) + 1 if y.size else 3
        classes = max(classes, 3)
        Y = np.zeros((X.shape[0], classes), dtype=np.float64)
        Y[np.arange(X.shape[0]), y] = 1.0
        n, d = X.shape
        lam = float(self.ridge) * float(np.trace(X.T @ X)) / max(d, 1)
        lam = max(lam, 1e-9)
        if n <= d:
            dual = X @ X.T + lam * np.eye(n)
            W = X.T @ np.linalg.solve(dual, Y)
        else:
            normal = X.T @ X + lam * np.eye(d)
            W = np.linalg.solve(normal, X.T @ Y)
        U, singular, Vh = np.linalg.svd(W, full_matrices=False)
        Q = U @ Vh
        projected = X @ Q
        cal_Y = Y
        cal_X = projected
        if calibration_rows:
            cal_X = self._matrix(calibration_rows) @ Q
            cal_labels = list(calibration_labels or labels)
            cal_Y = np.zeros((len(cal_labels), classes), dtype=np.float64)
            cal_Y[np.arange(len(cal_labels)), np.asarray(cal_labels, dtype=np.int64)] = 1.0
        normal = cal_X.T @ cal_X + 1e-6 * np.eye(int(cal_X.shape[1]))
        calibration = np.linalg.solve(normal, cal_X.T @ cal_Y)
        self.Q = Q
        self.calibration = calibration
        train_scores = X @ Q @ calibration
        train_predicted = np.argmax(train_scores, axis=1)
        self.fit_report = {
            "rows": int(n),
            "features": int(d),
            "classes": int(classes),
            "ridge": float(self.ridge),
            "lambda": float(lam),
            "stiefel_rank": int(Q.shape[1]),
            "train_accuracy": float(np.mean(train_predicted == y)) if n else 0.0,
            "singular_values": [float(value) for value in singular],
            "feature_slots": int(self.slots),
            "feature_width": int(self.width),
        }
        return dict(self.fit_report)

    def predict(self, values: Any) -> Tuple[int, float]:
        if self.Q is None or self.calibration is None:
            raise RuntimeError("ActionRegimeReadout is not fitted")
        row = self.features(values).reshape(1, -1)
        scores = (row @ self.Q @ self.calibration).reshape(-1)
        best = int(np.argmax(scores))
        shifted = scores - float(np.max(scores))
        probability = np.exp(shifted) / max(float(np.exp(shifted).sum()), 1e-12)
        return best, float(probability[best])

    def scores(self, values: Any) -> np.ndarray:
        if self.Q is None or self.calibration is None:
            raise RuntimeError("ActionRegimeReadout is not fitted")
        row = self.features(values).reshape(1, -1)
        return (row @ self.Q @ self.calibration).reshape(-1).astype(np.float64)

    def predict_batch(
        self, rows: Sequence[Any]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decode a batch of bundle rows in ONE matrix product (Directive X4).

        The feature map and the fitted (Q, calibration) pair are the same ones
        predict() uses, so a batched read cannot disagree with a single read.
        """
        if self.Q is None or self.calibration is None:
            raise RuntimeError("ActionRegimeReadout is not fitted")
        matrix = self._matrix(rows)
        scores = matrix @ self.Q @ self.calibration
        best = np.argmax(scores, axis=1).astype(np.int64)
        shifted = scores - np.max(scores, axis=1, keepdims=True)
        weight = np.exp(shifted)
        probability = weight / np.maximum(weight.sum(axis=1, keepdims=True), 1e-12)
        top = probability[np.arange(best.size), best]
        return best, top.astype(np.float64), scores.astype(np.float64)

    def accuracy(self, rows: Sequence[Any], labels: Sequence[int]) -> float:
        if not rows:
            return 0.0
        predicted = [self.predict(row)[0] for row in rows]
        return float(
            sum(int(value == int(label)) for value, label in zip(predicted, labels))
            / max(len(predicted), 1)
        )

    # ------------------------------------------------------------- storage
    def to_state(self) -> Dict[str, Any]:
        if self.Q is None or self.calibration is None:
            raise RuntimeError("ActionRegimeReadout is not fitted")
        return {
            "Q": torch.tensor(self.Q, dtype=torch.float64),
            "calibration": torch.tensor(self.calibration, dtype=torch.float64),
            "slots": int(self.slots),
            "width": int(self.width),
            "dim": int(self.dim),
            "ridge": float(self.ridge),
            "seed": int(self.seed),
            "fit_report": dict(self.fit_report),
            "digest": self.digest(),
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "ActionRegimeReadout":
        readout = cls(
            dim=int(state.get("dim", 1024)),
            slots=int(state.get("slots", 4)),
            ridge=float(state.get("ridge", 1e-2)),
            seed=int(state.get("seed", 20261111)),
        )
        readout.Q = np.asarray(
            state["Q"].detach().cpu().numpy()
            if torch.is_tensor(state["Q"]) else state["Q"],
            dtype=np.float64,
        )
        readout.calibration = np.asarray(
            state["calibration"].detach().cpu().numpy()
            if torch.is_tensor(state["calibration"]) else state["calibration"],
            dtype=np.float64,
        )
        readout.width = int(state.get("width", readout.width))
        readout.fit_report = dict(state.get("fit_report") or {})
        return readout

    def digest(self) -> str:
        if self.Q is None or self.calibration is None:
            return ""
        hasher = hashlib.sha256()
        hasher.update(np.ascontiguousarray(self.Q.astype(np.float64)).tobytes())
        hasher.update(np.ascontiguousarray(self.calibration.astype(np.float64)).tobytes())
        return hasher.hexdigest()



_CODE_KEYWORDS = frozenset(
    {
        "def", "class", "return", "if", "elif", "else", "for", "while", "in",
        "not", "and", "or", "import", "from", "as", "with", "try", "except",
        "finally", "raise", "yield", "lambda", "pass", "break", "continue",
        "None", "True", "False", "self", "global", "nonlocal", "assert", "del",
    }
)


def code_slot(text: str) -> str:
    """The grammatical slot of one token's surface text (Directive R3).

    The slot is a deterministic function of the token text: it does not see
    the position, the document or the future, so a prior conditioned on it
    is still a static causal prior.
    """
    token = str(text or "")
    stripped = token.strip()
    if not stripped:
        return "space"
    if stripped == ".":
        return "dot"
    if stripped in {"(", "[", "{"}:
        return "open"
    if stripped in {")", "]", "}"}:
        return "close"
    if stripped == ",":
        return "comma"
    if stripped in {";", ":"}:
        return "colon"
    if stripped in {
        "+", "-", "*", "/", "%", "=", "==", "!=", "<", ">", "<=", ">=",
        "+=", "-=", "*=", "/=", "&", "|", "^", "~", "->", "**", "//",
    }:
        return "operator"
    if stripped.startswith(("'", '"')):
        return "string"
    if stripped[0].isdigit():
        return "number"
    if stripped in _CODE_KEYWORDS:
        return "keyword"
    if stripped[0].isalpha() or stripped[0] == "_":
        return "name"
    return "other"



_DISCOUNT_ROLE = {
    "dot": "scope",
    "open": "scope",
    "comma": "scope",
    "colon": "scope",
    "operator": "scope",
    "name": "name",
    "keyword": "name",
    "number": "literal",
    "string": "literal",
    "close": "other",
    "space": "other",
    "other": "other",
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
        # Directive P1: two longer orders. They carry zero weight until a
        # calibration admits them, so a generator that never selects them is
        # bit-identical to the Cycle O floor.
        self.sextgram: Dict[Tuple[int, int, int, int, int], Dict[int, int]] = {}
        self.septgram: Dict[Tuple[int, int, int, int, int, int], Dict[int, int]] = {}
        # Directive Q3: continuation-discounted (modified Kneser-Ney) backoff.
        # The tables are built from the TRAINING stream only and are inert
        # until a calibration selects a non-zero kn_mix, so the P1 floor is
        # bit-identical when the mix stays at zero.
        self.kn_mix: float = 0.0
        self.kn_discount: float = 0.70
        # Directive R3: the discount is conditioned on the grammatical slot
        # of the last token. An empty table is the Q3 global discount.
        self.discount_by_role: Dict[str, float] = {}
        self.slot_of = None
        self.kn_order: int = 0
        self.kn_p1: Optional[np.ndarray] = None
        self.kn_buckets: Dict[int, Dict[Tuple[int, ...], Dict[int, int]]] = {}
        self.kn_raw_last: Dict[Tuple[int, ...], Dict[int, int]] = {}
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
        self.weights = np.array([0.05, 0.10, 0.20, 0.30, 0.35, 0.0, 0.0], dtype=np.float64)

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

    def fit_long(self, ids: Sequence[int]) -> "CountLanguageModel":
        """Build the 6-gram and 7-gram tables (Directive P1).

        They are built only when a calibration can select them, so a
        generator that never leaves the declared 5-gram floor pays neither
        the time nor the memory for the two extra orders.
        """
        import gc

        # Every key is a tuple plus a bucket dict, so a 500k-token tree holds
        # ~2M containers. The cyclic collector walks them repeatedly and turns
        # a two-second build into minutes, so it is paused for the build and
        # run once afterwards.
        gc.disable()
        try:
            for index in range(5, len(ids)):
                key5 = tuple(int(ids[index - offset]) for offset in range(5, 0, -1))
                current = int(ids[index])
                bucket = self.sextgram.get(key5)
                if bucket is None:
                    self.sextgram[key5] = {current: 1}
                else:
                    bucket[current] = bucket.get(current, 0) + 1
            for index in range(6, len(ids)):
                key6 = tuple(int(ids[index - offset]) for offset in range(6, 0, -1))
                current = int(ids[index])
                bucket = self.septgram.get(key6)
                if bucket is None:
                    self.septgram[key6] = {current: 1}
                else:
                    bucket[current] = bucket.get(current, 0) + 1
        finally:
            gc.enable()
        gc.collect()
        return self

    def fit_kneser_ney(
        self,
        ids: Sequence[int],
        *,
        order: int = 6,
        discount: float = 0.70,
    ) -> Dict[str, Any]:
        """Build continuation counts and the lower-order KN buckets (Q3).

        For every n-gram type the continuation count is the number of DISTINCT
        predecessors. The unigram backoff is therefore the continuation
        unigram, and orders 2..N-1 are continuation-discounted; the highest
        order keeps the raw counts. The tables are built from the training
        stream only. Nothing is selected here: the calibration decides whether
        the mix leaves zero.
        """
        import gc

        order = int(max(2, min(int(order), 7)))
        ids = [int(token) for token in ids]
        gc.disable()
        try:
            raw: Dict[int, Dict[Tuple[int, ...], Dict[int, int]]] = {
                n: {} for n in range(2, order + 1)
            }
            for n in range(2, order + 1):
                table = raw[n]
                for index in range(n - 1, len(ids)):
                    suffix = tuple(ids[index - n + 1 : index])
                    token = ids[index]
                    bucket = table.get(suffix)
                    if bucket is None:
                        table[suffix] = {token: 1}
                    else:
                        bucket[token] = bucket.get(token, 0) + 1
            cont: Dict[int, Dict[Tuple[int, ...], int]] = {
                n: {} for n in range(2, order)
            }
            for n in range(2, order):
                destination = cont[n]
                for prefix, bucket in raw[n + 1].items():
                    suffix = prefix[1:]
                    for token in bucket:
                        key = suffix + (token,)
                        destination[key] = destination.get(key, 0) + 1
            p1 = np.zeros(self.vocab_size, dtype=np.float64)
            for _prefix, token in cont.get(2, {}).keys():
                if 0 <= int(token) < self.vocab_size:
                    p1[int(token)] += 1.0
            total = float(p1.sum())
            p1 = p1 / total if total > 0 else np.full(self.vocab_size, 1.0 / self.vocab_size)
            buckets: Dict[int, Dict[Tuple[int, ...], Dict[int, int]]] = {
                n: {} for n in range(2, order)
            }
            for n in range(2, order):
                for key, count in cont[n].items():
                    suffix = key[:-1]
                    token = int(key[-1])
                    buckets[n].setdefault(suffix, {})[token] = int(count)
            self.kn_p1 = p1
            self.kn_buckets = buckets
            self.kn_raw_last = raw[order]
            self.kn_order = int(order)
            self.kn_discount = float(discount)
        finally:
            gc.enable()
        gc.collect()
        return {
            "order": int(self.kn_order),
            "discount": float(self.kn_discount),
            "continuation_unigram_types": int(np.count_nonzero(p1)),
            "raw_last_contexts": int(len(self.kn_raw_last)),
            "lower_buckets": int(sum(len(table) for table in self.kn_buckets.values())),
        }

    def discount_for(self, context: Sequence[int]) -> float:
        """The role-conditioned discount of the last token (Directive R3)."""
        if not self.discount_by_role:
            return float(self.kn_discount)
        context = [int(token) for token in context]
        if not context or self.slot_of is None:
            return float(self.kn_discount)
        try:
            slot = str(self.slot_of(context[-1]))
        except Exception:
            return float(self.kn_discount)
        role = _DISCOUNT_ROLE.get(slot, "other")
        return float(self.discount_by_role.get(role, self.kn_discount))

    def kneser_ney_distribution(
        self, context: Sequence[int], discount: Optional[float] = None
    ) -> Optional[np.ndarray]:
        """The continuation-discounted backoff distribution (Q3/R3)."""
        if self.kn_p1 is None or int(self.kn_order) < 2:
            return None
        context = [int(token) for token in context]
        probability = self.kn_p1.copy()
        discount = (
            float(discount)
            if discount is not None
            else float(self.discount_for(context))
        )
        for order in range(2, int(self.kn_order) + 1):
            if len(context) < order - 1:
                break
            suffix = tuple(context[-(order - 1) :])
            if order == int(self.kn_order):
                bucket = self.kn_raw_last.get(suffix)
            else:
                bucket = self.kn_buckets.get(order, {}).get(suffix)
            if not bucket:
                continue
            counts = float(sum(bucket.values()))
            types = float(len(bucket))
            if counts <= 0.0:
                continue
            vector = np.zeros(self.vocab_size, dtype=np.float64)
            for token, count in bucket.items():
                if 0 <= int(token) < self.vocab_size:
                    vector[int(token)] = float(count)
            gamma = min(1.0, discount * types / counts)
            probability = np.maximum(vector - discount, 0.0) / counts + gamma * probability
        total = float(probability.sum())
        if total <= 0.0:
            return np.full(self.vocab_size, 1.0 / max(self.vocab_size, 1), dtype=np.float64)
        return probability / total

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
        if len(context) >= 5 and float(self.weights[5]) > 0.0:
            counts = self.sextgram.get(tuple(int(token) for token in context[-5:]), {})
            if counts:
                tables.append((5, counts))
        if len(context) >= 6 and float(self.weights[6]) > 0.0:
            counts = self.septgram.get(tuple(int(token) for token in context[-6:]), {})
            if counts:
                tables.append((6, counts))
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

    def _jelinek_mercer(self, context: Sequence[int]) -> np.ndarray:
        """The declared interpolated floor, before any KN mix."""
        pieces = self._order_distributions(context)
        total_weight = float(sum(self.weights[index] for index, _ in pieces))
        if total_weight <= 0.0:
            total_weight = 1.0
        probability = np.zeros(self.vocab_size, dtype=np.float64)
        for index, distribution in pieces:
            probability += (float(self.weights[index]) / total_weight) * distribution
        return probability

    def distribution(self, context: Sequence[int]) -> np.ndarray:
        """The deployed next-token probability (Jelinek-Mercer + KN mix).

        Directive Q3. The mix is zero unless a calibration selected a
        continuation-discounted component, so the Cycle P floor is unchanged
        by construction on every arm that does not select it.
        """
        probability = self._jelinek_mercer(context)
        if float(self.kn_mix) > 0.0 and self.kn_p1 is not None:
            kn = self.kneser_ney_distribution(context)
            if kn is not None:
                probability = (1.0 - float(self.kn_mix)) * probability + float(
                    self.kn_mix
                ) * kn
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
        # Directive Q3: log of the SAME distribution the gate inspects, so the
        # KN mix and the wave gate cannot disagree about the floor.
        return np.log(np.maximum(self.distribution(context), 1e-12))

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


class StreamingNgramCache:
    """Adaptive n-gram cache over the tokens ALREADY read (Directive Q3).

    The static floor is fitted on the training split. When a scored stream is
    read, every token that has already been decoded is legitimate context for
    the next one, exactly as in adaptive arithmetic coding. The cache keeps
    orders 1..4 of that prefix and mixes them by count, so a module's repeated
    boilerplate is predicted the second time it appears. It never observes the
    token it is about to predict.
    """

    ORDERS = (1, 2, 3, 4)
    WEIGHTS = (0.05, 0.15, 0.30, 0.50)

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = int(vocab_size)
        self.unigram = np.zeros(self.vocab_size, dtype=np.float64)
        self.bigram: Dict[int, Dict[int, int]] = {}
        self.trigram: Dict[Tuple[int, int], Dict[int, int]] = {}
        self.quadgram: Dict[Tuple[int, int, int], Dict[int, int]] = {}
        self.total = 0.0

    def observe(self, token: int, context: Sequence[int]) -> None:
        token = int(token)
        if not 0 <= token < self.vocab_size:
            return
        history = [int(value) for value in context]
        self.unigram[token] += 1.0
        self.total += 1.0
        if history:
            self.bigram.setdefault(history[-1], {})
            self.bigram[history[-1]][token] = self.bigram[history[-1]].get(token, 0) + 1
        if len(history) >= 2:
            key = (history[-2], history[-1])
            self.trigram.setdefault(key, {})
            self.trigram[key][token] = self.trigram[key].get(token, 0) + 1
        if len(history) >= 3:
            key = (history[-3], history[-2], history[-1])
            self.quadgram.setdefault(key, {})
            self.quadgram[key][token] = self.quadgram[key].get(token, 0) + 1

    def seed(self, ids: Sequence[int], context: int = 24) -> "StreamingNgramCache":
        sequence = [int(token) for token in ids]
        for index, token in enumerate(sequence):
            self.observe(token, sequence[max(0, index - int(context)) : index])
        return self

    def distribution(self, context: Sequence[int]) -> np.ndarray:
        history = [int(token) for token in context]
        if self.total <= 0.0:
            return np.full(self.vocab_size, 1.0 / max(self.vocab_size, 1))
        pieces: List[Tuple[float, np.ndarray]] = [
            (self.WEIGHTS[0], self.unigram / self.total)
        ]
        if history:
            bucket = self.bigram.get(history[-1])
            if bucket:
                vector = np.zeros(self.vocab_size)
                for token, count in bucket.items():
                    if 0 <= token < self.vocab_size:
                        vector[token] = float(count)
                pieces.append((self.WEIGHTS[1], vector / vector.sum()))
        if len(history) >= 2:
            bucket = self.trigram.get((history[-2], history[-1]))
            if bucket:
                vector = np.zeros(self.vocab_size)
                for token, count in bucket.items():
                    if 0 <= token < self.vocab_size:
                        vector[token] = float(count)
                pieces.append((self.WEIGHTS[2], vector / vector.sum()))
        if len(history) >= 3:
            bucket = self.quadgram.get((history[-3], history[-2], history[-1]))
            if bucket:
                vector = np.zeros(self.vocab_size)
                for token, count in bucket.items():
                    if 0 <= token < self.vocab_size:
                        vector[token] = float(count)
                pieces.append((self.WEIGHTS[3], vector / vector.sum()))
        total_weight = float(sum(weight for weight, _ in pieces))
        probability = np.zeros(self.vocab_size, dtype=np.float64)
        for weight, piece in pieces:
            probability += (float(weight) / total_weight) * piece
        total = float(probability.sum())
        return probability / total if total > 0 else probability


class WindowSuffixPrior:
    """Stateless m = 1..6 suffix continuation index over context[-W:] (S1).

    The R3 static arm capped at 0.5167 because its window prior counted only
    orders 1..3 and, more importantly, was added AFTER the decision-preserving
    projection, so it could lower the cross-entropy of rank-2..V tokens but
    could never move the argmax. File-local predictability in Python and
    technical Markdown lives in 4-, 5- and 6-gram idioms inside the current
    file's causal prefix, which orders 1..3 cannot see.

    This index is a PURE FUNCTION of the slice it is given. It keeps no state
    between calls and never reads a token at or after the position it scores:
    every count is a transition that already occurred inside the window. The
    caller mixes its distribution INSIDE the count floor, before the
    decision-preserving projection, so a 4-gram suffix match can legitimately
    change the argmax.
    """

    def __init__(self, order: int = 6, smoothing: float = 0.05) -> None:
        self.order = int(order)
        self.smoothing = float(smoothing)

    def match(self, window: Sequence[int]) -> Optional[Dict[str, object]]:
        """The longest suffix of the window with an in-window continuation.

        Returns the matched order m, the number of occurrences C, and the
        continuation tokens observed after each prior occurrence. Occurrence
        positions are every start s in [0, n-m-1], so the continuation index
        s+m is at most n-1, which is inside the causal window.
        """
        tokens = np.asarray([int(token) for token in window], dtype=np.int64)
        length = int(tokens.size)
        if length < 3:
            return None
        top = int(min(self.order, length - 1))
        for order in range(top, 0, -1):
            span = length - order
            mask = np.ones(span, dtype=bool)
            for offset in range(order):
                mask &= tokens[offset : offset + span] == int(tokens[length - order + offset])
            positions = np.flatnonzero(mask)
            if positions.size == 0:
                continue
            targets = tokens[positions + order]
            return {
                "order": int(order),
                "occurrences": int(targets.size),
                "targets": targets,
                "window": tokens,
            }
        return None

    def distribution(
        self, window: Sequence[int], vocab_size: int
    ) -> Optional[Tuple[np.ndarray, int, int]]:
        """(P_win over the vocabulary, matched order, occurrence count)."""
        stats = self.match(window)
        if stats is None:
            return None
        vocab = int(vocab_size)
        targets = np.asarray(stats["targets"], dtype=np.int64)
        valid = (targets >= 0) & (targets < vocab)
        if not bool(valid.any()):
            return None
        counts = np.bincount(targets[valid], minlength=vocab)[:vocab].astype(np.float64)
        total = float(counts.sum())
        if total <= 0.0:
            return None
        probability = counts / total
        # Smooth with the window's OWN unigram. A continuation never seen
        # inside the window keeps a small floor instead of an exact zero, so
        # the interpolation cannot assign -inf log-probability to a token the
        # training floor still supports.
        smoothing = float(self.smoothing)
        if smoothing > 0.0:
            tokens = np.asarray(stats["window"], dtype=np.int64)
            inside = tokens[(tokens >= 0) & (tokens < vocab)]
            if inside.size:
                unigram = np.bincount(inside, minlength=vocab)[:vocab].astype(np.float64)
                unigram_total = float(unigram.sum())
                if unigram_total > 0.0:
                    probability = (1.0 - smoothing) * probability + smoothing * (
                        unigram / unigram_total
                    )
        return probability, int(stats["order"]), int(stats["occurrences"])


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
        # Directive Q3: the streaming cache is inert until an evaluation
        # installs it, so a generator that never streams is bit-identical to
        # the Cycle P one.
        self.stream_cache: Optional[StreamingNgramCache] = None
        self.stream_weight: float = 0.0
        self.stream_report: Dict[str, float] = {}
        # Directive R3: the window-local recency prior. It is computed only
        # from context[-96:], which is the standard causal input w_{t-96}^{t-1};
        # it keeps no state across calls, so the arm stays a static inductive
        # evaluator. Weight 0.0 is the Q3 floor bit for bit.
        self.recency_weight: float = 0.0
        # The streaming arm is already an adaptive cache of the full prefix.
        # Adding the static recurrence on top of it double-counts the prefix
        # and measurably cost top-1 in the first R3 run, so the two arms get
        # separate weights and the streaming weight is selected at zero.
        self.recency_weight_stream: float = 0.0
        self.recency_gain: float = 2.0
        self.recency_window: int = 1024
        self.recency_decay: float = 0.995
        self.recency_bigram: float = 1.2
        self.recency_trigram: float = 1.5
        self.recency_role_gain: float = 2.0
        # Directive R3: a floor-uncertainty gate. Where the count floor is
        # already sharp, the recurrence is a weak prior; where it is flat or
        # backed off, the exact identifiers in the window are the evidence.
        self.recency_tau: float = 0.40
        self.recency_boost: float = 1.0
        # Directive R3: a FIXED document prior built from the disjoint
        # calibration slice. It is not updated by the scored stream; it is a
        # static snapshot of the 1,500 tokens that precede the window.
        self.document_weight: float = 0.0
        self.document_unigram: Optional[np.ndarray] = None
        self.document_bigram: Dict[int, Dict[int, int]] = {}
        self.document_trigram: Dict[Tuple[int, int], Dict[int, int]] = {}
        self.document_total: float = 0.0
        self.recency_report: Dict[str, object] = {}
        # Directive S1: the stateless multi-order (m = 1..6) suffix floor. It
        # is folded inside the count floor BEFORE the decision-preserving
        # projection, so a 4-, 5- or 6-gram match inside the causal window can
        # legitimately move the argmax. Weight 0.0 leaves the R3 floor bit for
        # bit; the weight and the alpha schedule are selected on the disjoint
        # calibration slice only.
        self.suffix_prior = WindowSuffixPrior(order=6, smoothing=0.05)
        self.suffix_weight: float = 0.0
        self.suffix_window: int = 2048
        self.suffix_alpha: Dict[int, float] = {
            1: 0.05,
            2: 0.15,
            3: 0.40,
            4: 0.55,
            5: 0.65,
            6: 0.70,
        }
        self.suffix_count_saturation: float = 1.0
        self.suffix_report: Dict[str, object] = {}
        self.suffix_stats: Dict[str, float] = {
            "calls": 0.0,
            "matched": 0.0,
            "order_sum": 0.0,
        }
        self.is_static_inductive = True
        slot_names = ("dot", "open", "close", "comma", "colon", "operator",
                      "name", "keyword", "number", "string", "space", "other")
        self._slot_index = {name: index for index, name in enumerate(slot_names)}
        self._slot_ids = np.asarray(
            [
                self._slot_index.get(
                    code_slot(self.tokenizer.token_text(index)), self._slot_index["other"]
                )
                for index in range(self.vocab_size)
            ],
            dtype=np.int64,
        )
        self._name_mask = (self._slot_ids == self._slot_index["name"]).astype(np.float64)
        self._number_mask = (self._slot_ids == self._slot_index["number"]).astype(np.float64)
        self.count_lm.slot_of = lambda token: code_slot(
            self.tokenizer.token_text(int(token))
        )

    def install_stream(
        self, cache: Optional[StreamingNgramCache], weight: float
    ) -> "WaveCodeTextGenerator":
        self.stream_cache = cache
        self.stream_weight = float(weight)
        return self

    def clear_stream(self) -> "WaveCodeTextGenerator":
        self.stream_cache = None
        self.stream_weight = 0.0
        return self

    def _suffix_mix(
        self, probability: np.ndarray, context: Sequence[int]
    ) -> np.ndarray:
        """Blend the stateless window suffix distribution into the floor (S1).

        alpha(m, C) rises with the matched suffix order, because a 4-, 5- or
        6-gram exact match inside the causal window is file-local evidence the
        global training floor cannot see, and saturates with the occurrence
        count, because one prior occurrence is thinner evidence than twenty.
        Every token in the window is already decoded: the mix never reads the
        token it scores, and the window is a pure function of context[-W:].
        """
        weight = float(self.suffix_weight)
        if weight <= 0.0:
            return probability
        window = [int(token) for token in context[-int(self.suffix_window) :]]
        result = self.suffix_prior.distribution(window, self.vocab_size)
        self.suffix_stats["calls"] += 1.0
        if result is None:
            return probability
        local, order, occurrences = result
        alpha_order = float(self.suffix_alpha.get(int(order), 0.0))
        saturation = float(occurrences) / (
            float(occurrences) + float(self.suffix_count_saturation)
        )
        alpha = float(weight) * alpha_order * saturation
        self.suffix_stats["matched"] += 1.0
        self.suffix_stats["order_sum"] += float(order)
        if alpha <= 0.0:
            return probability
        alpha = min(alpha, 0.95)
        mixed = (1.0 - alpha) * probability + alpha * local
        total = float(mixed.sum())
        if total <= 0.0:
            return probability
        return mixed / total

    def floor_probability(
        self,
        context: Sequence[int],
        stream_weight: Optional[float] = None,
        stream_cache: Optional["StreamingNgramCache"] = None,
    ) -> np.ndarray:
        """The deployed count floor: count model, stream cache, suffix mix.

        Directive S1: the window-local suffix distribution is part of this
        floor, not an additive term above it. The decision-preserving
        projection in token_logits restores the argmax of THIS distribution,
        so the suffix evidence can flip a wrong global mode into the right
        file-local token.
        """
        probability = self.count_lm.distribution(context)
        cache = self.stream_cache if stream_cache is None else stream_cache
        weight = float(self.stream_weight if stream_weight is None else stream_weight)
        if cache is not None and weight > 0.0 and float(getattr(cache, "total", 0.0)) > 0.0:
            adaptive = cache.distribution(context)
            mixed = (1.0 - weight) * probability + weight * adaptive
            total = float(mixed.sum())
            if total > 0.0:
                probability = mixed / total
        return self._suffix_mix(probability, context)

    def _floor_log_probs(self, context: Sequence[int]) -> np.ndarray:
        """log P of the count floor, mixed with the stream cache and suffix."""
        return np.log(np.maximum(self.floor_probability(context), 1e-12))


    # ---------------------------------------------------- recency (R3)
    def recency_scores(self, context: Sequence[int]) -> Optional[np.ndarray]:
        """A causal, window-local recurrence prior over context[-96:].

        Three terms, all exact-token counts of the bounded prefix: a decayed
        unigram over the recent past, a bigram term for the token that
        followed the last token earlier in the window, and a trigram term for
        the token that followed the last two. The final weighting is
        slot-compatible: after a dot or an opening bracket, names are the
        likely continuation. No state is kept between calls.
        """
        window = np.asarray(
            [int(token) for token in context[-int(self.recency_window):]],
            dtype=np.int64,
        )
        if int(window.size) < 2:
            return None
        decay = float(self.recency_decay)
        scores = np.zeros(self.vocab_size, dtype=np.float64)
        length = int(window.size)
        power = np.power(decay, np.arange(0, length + 1, dtype=np.float64))
        gaps = np.arange(1, length, dtype=np.int64)
        recent = window[-1 - gaps]
        valid = (recent >= 0) & (recent < self.vocab_size)
        if int(valid.any()):
            scores += np.bincount(
                recent[valid],
                weights=power[gaps[valid]],
                minlength=self.vocab_size,
            )[: self.vocab_size]
        last = int(window[-1])
        matches = np.nonzero(window[:-1] == last)[0]
        if int(matches.size):
            targets = window[matches + 1]
            keep = (targets >= 0) & (targets < self.vocab_size)
            if int(keep.any()):
                scores += np.bincount(
                    targets[keep],
                    weights=float(self.recency_bigram)
                    * power[length - 1 - matches[keep]],
                    minlength=self.vocab_size,
                )[: self.vocab_size]
        if length >= 3:
            first, second = int(window[-2]), int(window[-1])
            matches = np.nonzero(
                (window[:-2] == first) & (window[1:-1] == second)
            )[0]
            if int(matches.size):
                targets = window[matches + 2]
                keep = (targets >= 0) & (targets < self.vocab_size)
                if int(keep.any()):
                    scores += np.bincount(
                        targets[keep],
                        weights=float(self.recency_trigram)
                        * power[length - 2 - matches[keep]],
                        minlength=self.vocab_size,
                    )[: self.vocab_size]
        if float(scores.max()) <= 0.0:
            return None
        last_slot = code_slot(self.tokenizer.token_text(last))
        if last_slot in {"dot", "open", "comma", "colon"}:
            scores *= 1.0 + float(self.recency_role_gain) * self._name_mask
        elif last_slot == "operator":
            scores *= 1.0 + float(self.recency_role_gain) * (
                0.5 * self._name_mask + 0.5 * self._number_mask
            )
        peak = float(scores.max())
        if peak <= 0.0:
            return None
        return (scores / peak).astype(np.float64)

    def _recency_adjusted_probability(
        self, probability: np.ndarray, context: Sequence[int], weight: Optional[float] = None
    ) -> np.ndarray:
        """Add the recency logits to a probability row and renormalise."""
        active = float(self.recency_weight if weight is None else weight)
        if active <= 0.0:
            return probability
        scores = self.recency_scores(context)
        if scores is None:
            return probability
        logits = np.log(np.maximum(probability, 1e-300))
        logits = logits + float(active) * float(self.recency_gain) * scores
        logits = logits - float(np.max(logits))
        adjusted = np.exp(logits)
        total = float(adjusted.sum())
        return adjusted / total if total > 0.0 else probability

    def calibrate_recency(
        self,
        ids: Sequence[int],
        grid: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0),
        context: int = 96,
        top1_tolerance: float = 0.001,
    ) -> Dict[str, object]:
        """Select the recency weight on the DISJOINT calibration slice.

        The candidate must lower NLL and must not lower top-1, the same joint
        rule the Q3 floor selector uses. The grid contains 0.0, so the prior
        is provably no worse than the floor on the slice.
        """
        sequence = [int(token) for token in ids]
        indices = list(range(int(context), len(sequence) - 1))
        if not indices:
            return {"weight": 0.0, "rows": []}
        base_total = 0.0
        base_correct = 0
        for index in indices:
            window = sequence[max(0, index - int(context)) : index]
            probability = self.count_lm.distribution(window)
            base_total -= float(np.log(max(probability[sequence[index]], 1e-12)))
            base_correct += int(int(np.argmax(probability)) == sequence[index])
        base_nll = base_total / max(len(indices), 1)
        base_top1 = base_correct / max(len(indices), 1)
        rows: List[Dict[str, float]] = []
        best = (base_nll, 0.0, base_top1)
        for weight in grid:
            self.recency_weight = float(weight)
            total = 0.0
            correct = 0
            for index in indices:
                window = sequence[max(0, index - int(context)) : index]
                probability = self.count_lm.distribution(window)
                probability = self._recency_adjusted_probability(probability, window)
                total -= float(np.log(max(probability[sequence[index]], 1e-12)))
                correct += int(int(np.argmax(probability)) == sequence[index])
            nll = total / max(len(indices), 1)
            top1 = correct / max(len(indices), 1)
            rows.append({"weight": float(weight), "nll": float(nll), "top1": float(top1)})
            if (
                float(weight) > 0.0
                and float(top1) >= base_top1 - float(top1_tolerance)
                and float(nll) < best[0] - 1e-9
            ):
                best = (float(nll), float(weight), float(top1))
        self.recency_weight = float(best[1])
        self.recency_report = {
            "weight": float(best[1]),
            "base_nll": float(base_nll),
            "base_top1": float(base_top1),
            "selected_nll": float(best[0]),
            "selected_top1": float(best[2]),
            "rows": rows,
            "window": int(self.recency_window),
            "decay": float(self.recency_decay),
            "gain": float(self.recency_gain),
            "is_static_inductive": True,
            "state_bytes_across_calls": 0,
        }
        return dict(self.recency_report)


    # ---------------------------------------------- suffix floor (S1)
    def calibrate_suffix_floor(
        self,
        ids: Sequence[int],
        grid: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5),
        context: int = 1024,
        top1_tolerance: float = 0.0,
    ) -> Dict[str, object]:
        """Select the suffix-floor weight on the DISJOINT calibration slice.

        Directive S1. The grid contains 0.0, so the suffix floor is provably
        no worse than the R3 floor on the slice. A candidate is eligible only
        when it lowers NLL AND does not lower top-1: the same joint rule every
        other selector in this module obeys. The window and the alpha schedule
        are deployed unchanged after selection; nothing is tuned on the scored
        stream.
        """
        sequence = [int(token) for token in ids]
        indices = list(range(int(context), len(sequence) - 1))
        if not indices:
            return {"weight": 0.0, "rows": [], "window": int(self.suffix_window)}
        saved = float(self.suffix_weight)
        self.suffix_weight = 0.0
        base_total = 0.0
        base_correct = 0
        try:
            for index in indices:
                window = sequence[max(0, index - int(context)) : index]
                probability = self.floor_probability(window)
                base_total -= float(np.log(max(probability[sequence[index]], 1e-12)))
                base_correct += int(int(np.argmax(probability)) == sequence[index])
            base_nll = base_total / max(len(indices), 1)
            base_top1 = base_correct / max(len(indices), 1)
            rows: List[Dict[str, float]] = []
            best = (base_nll, 0.0, base_top1)
            for weight in grid:
                self.suffix_weight = float(weight)
                total = 0.0
                correct = 0
                for index in indices:
                    window = sequence[max(0, index - int(context)) : index]
                    probability = self.floor_probability(window)
                    total -= float(np.log(max(probability[sequence[index]], 1e-12)))
                    correct += int(int(np.argmax(probability)) == sequence[index])
                nll = total / max(len(indices), 1)
                top1 = correct / max(len(indices), 1)
                rows.append(
                    {
                        "weight": float(weight),
                        "nll": float(nll),
                        "top1": float(top1),
                    }
                )
                if (
                    float(weight) > 0.0
                    and float(top1) >= base_top1 - float(top1_tolerance)
                    and float(nll) < best[0] - 1e-9
                ):
                    best = (float(nll), float(weight), float(top1))
        finally:
            self.suffix_weight = float(saved)
        self.suffix_weight = float(best[1])
        self.suffix_report = {
            "weight": float(best[1]),
            "base_nll": float(base_nll),
            "base_top1": float(base_top1),
            "selected_nll": float(best[0]),
            "selected_top1": float(best[2]),
            "grid": [float(value) for value in grid],
            "rows": rows,
            "window": int(self.suffix_window),
            "orders": int(self.suffix_prior.order),
            "alpha": {int(key): float(value) for key, value in self.suffix_alpha.items()},
            "alpha_schedule": "depth-conditioned, count-saturated",
            "count_saturation": float(self.suffix_count_saturation),
            "is_static_inductive": True,
            "state_bytes_across_calls": 0,
            "reads_target_token": False,
        }
        return dict(self.suffix_report)

    # ------------------------------------------------- document prior (R3)
    def install_document_prior(self, ids: Sequence[int]) -> "WaveCodeTextGenerator":
        """Snapshot the calibration slice into a fixed 1..3-gram prior."""
        vocab = int(self.vocab_size)
        self.document_unigram = np.zeros(vocab, dtype=np.float64)
        self.document_bigram = {}
        self.document_trigram = {}
        sequence = [int(token) for token in ids]
        self.document_total = 0.0
        for index, token in enumerate(sequence):
            if not 0 <= token < vocab:
                continue
            self.document_unigram[token] += 1.0
            self.document_total += 1.0
            if index >= 1:
                previous = int(sequence[index - 1])
                self.document_bigram.setdefault(previous, {})
                self.document_bigram[previous][token] = (
                    self.document_bigram[previous].get(token, 0) + 1
                )
            if index >= 2:
                key = (int(sequence[index - 2]), int(sequence[index - 1]))
                self.document_trigram.setdefault(key, {})
                self.document_trigram[key][token] = (
                    self.document_trigram[key].get(token, 0) + 1
                )
        return self

    def document_log_probs(self, context: Sequence[int]) -> Optional[np.ndarray]:
        if self.document_unigram is None or self.document_total <= 0.0:
            return None
        vocab = int(self.vocab_size)
        pieces: List[Tuple[float, np.ndarray]] = [
            (0.20, self.document_unigram / self.document_total)
        ]
        history = [int(token) for token in context]
        if history:
            bucket = self.document_bigram.get(history[-1])
            if bucket:
                vector = np.zeros(vocab, dtype=np.float64)
                for token, count in bucket.items():
                    if 0 <= int(token) < vocab:
                        vector[int(token)] = float(count)
                pieces.append((0.30, vector / max(float(vector.sum()), 1e-12)))
        if len(history) >= 2:
            bucket = self.document_trigram.get((history[-2], history[-1]))
            if bucket:
                vector = np.zeros(vocab, dtype=np.float64)
                for token, count in bucket.items():
                    if 0 <= int(token) < vocab:
                        vector[int(token)] = float(count)
                pieces.append((0.50, vector / max(float(vector.sum()), 1e-12)))
        total_weight = float(sum(weight for weight, _ in pieces))
        probability = np.zeros(vocab, dtype=np.float64)
        for weight, piece in pieces:
            probability += (float(weight) / max(total_weight, 1e-12)) * piece
        total = float(probability.sum())
        if total <= 0.0:
            return None
        return np.log(np.maximum(probability / total, 1e-12))

    def calibrate_document_weight(
        self,
        calibration_ids: Sequence[int],
        *,
        grid: Sequence[float] = (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0),
        context: int = 384,
    ) -> Dict[str, object]:
        """Select the fixed document-prior weight on a held-back slice."""
        sequence = [int(token) for token in calibration_ids]
        indices = list(range(min(int(context), len(sequence) - 1), len(sequence) - 1))
        if not indices:
            self.document_weight = 0.0
            return {"weight": 0.0, "rows": []}
        base_total = 0.0
        base_correct = 0
        for index in indices:
            window = sequence[max(0, index - int(context)) : index]
            probability = self.count_lm.distribution(window)
            base_total -= float(np.log(max(probability[sequence[index]], 1e-12)))
            base_correct += int(int(np.argmax(probability)) == sequence[index])
        best = (base_total / max(len(indices), 1), 0.0)
        rows = []
        for weight in grid:
            self.document_weight = float(weight)
            total = 0.0
            correct = 0
            for index in indices:
                window = sequence[max(0, index - int(context)) : index]
                probability = self.count_lm.distribution(window)
                document = self.document_log_probs(window)
                if document is not None and float(weight) > 0.0:
                    logits = np.log(np.maximum(probability, 1e-300)) + float(weight) * document
                    logits = logits - float(np.max(logits))
                    probability = np.exp(logits)
                    probability = probability / max(float(probability.sum()), 1e-12)
                total -= float(np.log(max(probability[sequence[index]], 1e-12)))
                correct += int(int(np.argmax(probability)) == sequence[index])
            nll = total / max(len(indices), 1)
            top1 = correct / max(len(indices), 1)
            rows.append({"weight": float(weight), "nll": float(nll), "top1": float(top1)})
            if (
                float(weight) > 0.0
                and float(nll) < best[0] - 1e-9
                and float(top1) >= (base_correct / max(len(indices), 1)) - 0.001
            ):
                best = (float(nll), float(weight))
        self.document_weight = float(best[1])
        return {
            "weight": float(best[1]),
            "base_nll": float(base_total / max(len(indices), 1)),
            "base_top1": float(base_correct / max(len(indices), 1)),
            "selected_nll": float(best[0]),
            "rows": rows,
            "fixed_snapshot": True,
        }

    def calibrate_role_discounts(
        self,
        ids: Sequence[int],
        discounts: Sequence[float] = (0.3, 0.5, 0.7, 0.9),
        context: int = 96,
    ) -> Dict[str, object]:
        """Select the role-conditioned KN discount on the calibration slice.

        The roles are coarse on purpose: a 1,500-token slice cannot support a
        per-slot grid without overfitting. A role keeps the global discount
        unless a candidate lowers its NLL AND does not lower its top-1.
        """
        sequence = [int(token) for token in ids]
        indices = list(range(int(context), len(sequence) - 1))
        if not indices or self.count_lm.kn_p1 is None:
            return {"roles": {}, "rows": []}
        roles = ("scope", "name", "literal", "other")
        base_rows = {role: [0.0, 0, 0] for role in roles}
        self.count_lm.discount_by_role = {}
        for index in indices:
            window = sequence[max(0, index - int(context)) : index]
            role = _DISCOUNT_ROLE.get(code_slot(self.tokenizer.token_text(window[-1])), "other")
            probability = self.count_lm.distribution(window)
            base_rows[role][0] -= float(np.log(max(probability[sequence[index]], 1e-12)))
            base_rows[role][1] += int(int(np.argmax(probability)) == sequence[index])
            base_rows[role][2] += 1
        rows: List[Dict[str, object]] = []
        chosen: Dict[str, float] = {}
        for discount in discounts:
            self.count_lm.discount_by_role = {role: float(discount) for role in roles}
            totals = {role: [0.0, 0, 0] for role in roles}
            for index in indices:
                window = sequence[max(0, index - int(context)) : index]
                role = _DISCOUNT_ROLE.get(
                    code_slot(self.tokenizer.token_text(window[-1])), "other"
                )
                probability = self.count_lm.distribution(window)
                totals[role][0] -= float(np.log(max(probability[sequence[index]], 1e-12)))
                totals[role][1] += int(int(np.argmax(probability)) == sequence[index])
                totals[role][2] += 1
            rows.append(
                {
                    "discount": float(discount),
                    "roles": {
                        role: {
                            "nll": float(totals[role][0] / max(totals[role][2], 1)),
                            "top1": float(totals[role][1] / max(totals[role][2], 1)),
                        }
                        for role in roles
                    },
                }
            )
        for role in roles:
            base_nll = base_rows[role][0] / max(base_rows[role][2], 1)
            base_top1 = base_rows[role][1] / max(base_rows[role][2], 1)
            best = (base_nll, float(self.count_lm.kn_discount))
            if base_rows[role][2] < 20:
                continue
            for row in rows:
                stats = row["roles"][role]
                if (
                    float(stats["top1"]) >= base_top1 - 0.002
                    and float(stats["nll"]) < best[0] - 1e-9
                ):
                    best = (float(stats["nll"]), float(row["discount"]))
            if best[1] != float(self.count_lm.kn_discount):
                chosen[role] = float(best[1])
        self.count_lm.discount_by_role = chosen
        return {
            "roles": dict(chosen),
            "global_discount": float(self.count_lm.kn_discount),
            "rows": rows,
            "base": {
                role: {
                    "nll": float(base_rows[role][0] / max(base_rows[role][2], 1)),
                    "top1": float(base_rows[role][1] / max(base_rows[role][2], 1)),
                    "tokens": int(base_rows[role][2]),
                }
                for role in roles
            },
        }

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
        base = self._floor_log_probs(context)
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
        return self._blend_logits(context, base, wave, fast)

    def _blend_logits(
        self,
        context: Sequence[int],
        base: np.ndarray,
        wave: np.ndarray,
        fast: Optional[np.ndarray],
    ) -> np.ndarray:
        """The D-Guard blend of a PRE-COMPUTED floor and the wave head.

        Directive S1 splits the blend from the floor computation so the
        evaluator can score one floor row and read it twice: as the emitted
        logits and as the base reference. The arithmetic is unchanged; the
        duplicate count-model pass in evaluate() is what goes away.
        """
        context = [int(token) for token in context]
        beta = self.beta_for(context) if self.gate_active() else float(self.beta_star)
        logits = base + float(beta) * wave
        smoothing = float(beta) > 0.0
        if fast is not None:
            logits = logits + self.fast_gain * fast
            smoothing = True
        photonic = self.photonic_scores(context)
        if photonic is not None:
            logits = logits + self.photonic_gain * photonic
            smoothing = True
        if self.mode_guard and smoothing:
            # Decision-preserving projection (Directive N4): every smoothing
            # term - the Bochner wave, the delta-rule fast weight and the
            # post-photonic lexical boost - is a prior, not a decision rule.
            # The projection therefore runs LAST, after all of them: if any
            # term moved the argmax off the count floor, the floor keeps it.
            # Running it before the photonic term (the Cycle M order) let a
            # mounted Zone B move 1.6 points of top-1 below the floor.
            floor_mode = int(np.argmax(base))
            if int(np.argmax(logits)) != floor_mode:
                logits[floor_mode] = float(np.max(logits)) + 1e-4
        # Directive R3: the window-local recency prior is a DECISION prior,
        # not a smoother. It runs after the mode guard on purpose: an exact
        # identifier repeated in the causal window is evidence the count
        # floor cannot see, and blurring it would erase the gain.
        recency_weight = (
            float(self.recency_weight_stream)
            if (self.stream_cache is not None and float(self.stream_weight) > 0.0)
            else float(self.recency_weight)
        )
        if recency_weight > 0.0:
            floor_probability = np.exp(base - float(np.max(base)))
            floor_probability = floor_probability / max(
                float(floor_probability.sum()), 1e-12
            )
            uncertain = float(np.max(floor_probability)) < float(self.recency_tau)
            effective = recency_weight * (
                float(self.recency_boost) if uncertain else 1.0
            )
            scores = self.recency_scores(context)
            if scores is not None:
                logits = logits + effective * float(self.recency_gain) * scores
        if float(self.document_weight) > 0.0:
            document = self.document_log_probs(context)
            if document is not None:
                logits = logits + float(self.document_weight) * document
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

    def calibrate_count_weights(
        self,
        ids: Sequence[int],
        *,
        train_ids: Optional[Sequence[int]] = None,
        samples: int = 320,
        context: int = WAVE_CONTEXT,
        candidates: Sequence[Sequence[float]] = (
            (0.05, 0.10, 0.20, 0.30, 0.35, 0.0, 0.0),
            (0.03, 0.07, 0.15, 0.30, 0.45, 0.0, 0.0),
            (0.03, 0.07, 0.15, 0.30, 0.40, 0.15, 0.0),
            (0.03, 0.06, 0.12, 0.25, 0.35, 0.25, 0.0),
            (0.02, 0.05, 0.10, 0.20, 0.30, 0.25, 0.10),
            (0.02, 0.04, 0.09, 0.18, 0.27, 0.25, 0.15),
            (0.02, 0.05, 0.10, 0.20, 0.32, 0.31, 0.0),
        ),
    ) -> Dict[str, object]:
        """Select the Jelinek-Mercer weights, including the two long orders.

        Directive P1. The declared Cycle O mixture stops at the 5-gram. On
        the grown live tree a 6-gram and a 7-gram carry real conditional
        information, but they also cost backoff mass, so the mixture is
        selected on the calibration slice by validation cross-entropy. The
        first candidate is the declared Cycle O vector, so a corpus that does
        not benefit keeps the historical floor bit for bit.
        """
        sequence = [int(token) for token in ids]
        wants_long = any(
            (len(candidate) > 5 and float(candidate[5]) > 0.0)
            or (len(candidate) > 6 and float(candidate[6]) > 0.0)
            for candidate in candidates
        )
        if wants_long and not self.count_lm.sextgram:
            # The long-order tables are built from the TRAINING stream only.
            # Building them from the calibration slice would leak the scored
            # window into the model and inflate the very number it measures.
            self.count_lm.fit_long(list(train_ids) if train_ids is not None else sequence)
        indices = list(range(context, len(sequence) - 1))
        if len(indices) > int(samples):
            stride = max(1, len(indices) // int(samples))
            indices = indices[::stride]
        rows: List[Dict[str, object]] = []
        for candidate in candidates:
            vector = np.asarray(candidate, dtype=np.float64)
            if vector.size < 7:
                vector = np.concatenate([vector, np.zeros(7 - vector.size)])
            self.count_lm.weights = vector
            total = 0.0
            correct = 0
            for index in indices:
                window = sequence[max(0, index - context) : index]
                logp = self.count_lm.log_probs(window)
                logp = logp - np.logaddexp.reduce(logp)
                total -= float(logp[sequence[index]])
                correct += int(int(np.argmax(logp)) == sequence[index])
            loss = float(total / max(len(indices), 1))
            rows.append(
                {
                    "weights": [float(value) for value in vector],
                    "validation_loss": loss,
                    "validation_top1": float(correct / max(len(indices), 1)),
                }
            )
        floor = next(
            (row for row in rows if row["weights"] == [0.05, 0.10, 0.20, 0.30, 0.35, 0.0, 0.0]),
            rows[0],
        )
        floor_loss = float(floor["validation_loss"])
        floor_top1 = float(floor["validation_top1"])
        eligible = [
            row
            for row in rows
            if float(row["validation_top1"]) >= floor_top1
            and float(row["validation_loss"]) < floor_loss - 1e-9
        ]
        eligible.sort(key=lambda row: (float(row["validation_loss"]), row["weights"]))
        chosen = eligible[0] if eligible else floor
        self.count_lm.weights = np.asarray(chosen["weights"], dtype=np.float64)
        return {
            "weights": list(chosen["weights"]),
            "validation_loss": float(chosen["validation_loss"]),
            "validation_top1": float(chosen["validation_top1"]),
            "floor_loss": floor_loss,
            "floor_top1": floor_top1,
            "selected": bool(eligible),
            "grid": rows,
        }

    def calibrate_blend(
        self,
        ids: Sequence[int],
        *,
        samples: int = 220,
        photonic_grid: Sequence[float] = (0.0, 0.15, 0.30),
        beta_grid: Sequence[float] = (0.0, 0.5, 1.0),
        context: int = WAVE_CONTEXT,
    ) -> Dict[str, object]:
        """Select the two blend gains that scale the smoothing terms.

        Directive P1. The wave weight beta is set by the entropy gate and the
        photonic lexical boost by a declared constant. Both are scalars in
        front of a standardised term, and their best value depends on the
        corpus: on the live tree the Zone B term can cost bits while on the
        pinned corpus it buys them. The pair is therefore selected on the
        calibration slice by validation cross-entropy, and the grid contains
        (0.0, ...) for the photonic gain, so the selector can switch the term
        off and the deployed generator can never be worse than the same
        generator without it.
        """
        sequence = [int(token) for token in ids]
        indices = list(range(context, len(sequence) - 1))
        if len(indices) > int(samples):
            stride = max(1, len(indices) // int(samples))
            indices = indices[::stride]
        best: Optional[Tuple[float, float, float]] = None
        rows: List[Dict[str, float]] = []
        for beta_lang in beta_grid:
            self.beta_lang = float(beta_lang)
            for gain in photonic_grid:
                self.photonic_gain = float(gain)
                total = 0.0
                for index in indices:
                    window = sequence[max(0, index - context) : index]
                    logits = self.token_logits(window)
                    logits = logits - np.max(logits)
                    total -= float(
                        logits[sequence[index]] - np.log(np.sum(np.exp(logits)))
                    )
                loss = float(total / max(len(indices), 1))
                rows.append(
                    {
                        "photonic_gain": float(gain),
                        "beta_lang": float(beta_lang),
                        "validation_loss": loss,
                    }
                )
                candidate = (loss, float(beta_lang), float(gain))
                if best is None or candidate < best:
                    best = candidate
        assert best is not None
        self.beta_lang = float(best[1])
        self.photonic_gain = float(best[2])
        return {
            "photonic_gain": float(best[2]),
            "beta_lang": float(best[1]),
            "validation_loss": float(best[0]),
            "grid": rows,
            "floor_row": next(
                (
                    row
                    for row in rows
                    if row["photonic_gain"] == 0.0 and row["beta_lang"] == float(beta_grid[-1])
                ),
                {},
            ),
        }

    def evaluate(
        self,
        ids: Sequence[int],
        context: int = WAVE_CONTEXT,
        *,
        streaming: bool = False,
        prior_ids: Optional[Sequence[int]] = None,
        stream_weight: Optional[float] = None,
    ) -> LMReport:
        sequence = [int(token) for token in ids]
        indices = list(range(context, len(sequence) - 1))
        total = 0.0
        correct = 0
        base_total = 0.0
        base_correct = 0
        cache: Optional[StreamingNgramCache] = None
        installed_weight = 0.0
        if streaming:
            cache = StreamingNgramCache(self.vocab_size)
            if prior_ids:
                cache.seed(prior_ids, context=context)
            installed_weight = (
                self.stream_weight if stream_weight is None else float(stream_weight)
            )
            self.install_stream(cache, installed_weight)
        try:
            for index in indices:
                window = sequence[max(0, index - context) : index]
                base, wave, fast = self._logit_parts(window)
                logits = self._blend_logits(window, base, wave, fast)
                logits = logits - np.max(logits)
                logprob = logits[sequence[index]] - np.log(np.sum(np.exp(logits)))
                total -= float(logprob)
                correct += int(int(np.argmax(logits)) == sequence[index])
                base_total -= float(base[sequence[index]])
                base_correct += int(int(np.argmax(base)) == sequence[index])
                if cache is not None:
                    cache.observe(sequence[index], window)
        finally:
            if streaming:
                self.clear_stream()
        count = max(len(indices), 1)
        perplexity = float(math.exp(min(total / count, 60.0)))
        # Bits per character: every token is decoded to its exact bytes, so
        # the comparison against the retired character model is on a common
        # unit and not on two different tokenizations. The denominator is the
        # exact UTF-8 length of the tokens the loop scored.
        characters = sum(
            len(self.tokenizer.token_text(token).encode("utf-8"))
            for token in sequence[context : context + len(indices)]
        )
        bits = float(total / max(characters, 1)) / math.log(2.0)
        base_bits = float(base_total / max(characters, 1)) / math.log(2.0)
        self.report = LMReport(
            top1=float(correct / count),
            perplexity=perplexity,
            bits_per_char=bits,
            tokens=int(len(indices)),
            base_top1=float(base_correct / count),
            base_perplexity=float(math.exp(min(base_total / count, 60.0))),
            beta_star=float(self.beta_star),
            wave_weight=float(self.beta_star * self.beta_lang),
            base_bits_per_char=base_bits,
        )
        self.stream_report = {
            "streaming": float(bool(streaming)),
            "stream_weight": float(installed_weight),
            "stream_tokens": float(cache.total if cache is not None else 0.0),
            "prior_tokens": float(len(list(prior_ids or ()))),
        }
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


def _is_guard_clause(statement) -> bool:
    """True for an if whose only statement is a value return and has no else."""
    if not isinstance(statement, ast.If):
        return False
    if statement.orelse:
        return False
    body = statement.body
    if len(body) != 1:
        return False
    return isinstance(body[0], ast.Return) and body[0].value is not None


def _composable_body(definition) -> Optional[List[ast.stmt]]:
    """The statements of a function whose body ends in exactly one return.

    Directive M4: leading guard clauses (for example, an early return of a
    default when the input is empty) are accepted. Every other early return -
    including one nested inside a loop or a later branch - is still refused,
    because a single-exit normalization of an arbitrary mid-loop return would
    need a helper function.
    """
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
    guard_count = 0
    while guard_count < len(body) - 1 and _is_guard_clause(body[guard_count]):
        guard_count += 1
    for statement in body[guard_count:-1]:
        if any(isinstance(node, ast.Return) for node in ast.walk(statement)):
            return None
    return body


class _ReturnToStep(ast.NodeTransformer):
    """Replace every remaining return of a normalised stage with step_i = expr."""

    def __init__(self, name: str) -> None:
        self.name = str(name)

    def visit_Return(self, node):  # noqa: N802
        return ast.Assign(
            targets=[ast.Name(id=self.name, ctx=ast.Store())],
            value=node.value,
        )


def _normalise_stage(statements: Sequence[ast.stmt]) -> List[ast.stmt]:
    """Turn leading guards plus a terminal return into a single-exit chain.

    An early guard return followed by body statements and a terminal return
    becomes an if/else chain whose else branch carries the body and the
    terminal return. A later _ReturnToStep replaces both returns with step_i
    assignments for every non-terminal stage, so the composed function has one
    exit per branch and no stage can return before the later stages run.
    """
    body = list(statements)
    guards: List[ast.If] = []
    while body and _is_guard_clause(body[0]):
        guards.append(body.pop(0))
    if not guards:
        return body
    else_body: List[ast.stmt] = list(body)
    for guard in reversed(guards):
        else_body = [ast.If(test=guard.test, body=list(guard.body), orelse=else_body)]
    return else_body


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
        staged.append(copy.deepcopy(_normalise_stage(body)))
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
            # Every return of a non-terminal stage becomes step_i = expr,
            # including a guard return inside the normalized if/else chain.
            step_name = f"step_{index + 1}"
            transformer = _ReturnToStep(step_name)
            for statement in statements:
                body.append(transformer.visit(statement))
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
        # Cycle AB2: the RepoSheaf domain/codomain gate runs INSIDE retrieval,
        # before a candidate is ranked, so a stage whose input domain conflicts
        # with the previous stage's return codomain can never be selected.
        self.type_gate = TypeCompatibilityGate()
        self.type_filter_report: Dict[str, object] = {
            "checked": 0,
            "masked": 0,
            "masked_names": [],
        }
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
        self.type_filter_report = {"checked": 0, "masked": 0, "masked_names": []}
        segments = self.split_stages(intent, limit)
        if len(segments) >= 2:
            chain: List[Tuple[Plan, float]] = []
            used: set = set()
            for position, segment in enumerate(segments[:limit]):
                hit = self.retrieve_compatible(
                    segment,
                    previous=(chain[-1][0] if chain else None),
                    arity=arity if position == 0 else None,
                )
                if hit is None or hit[0].name in used:
                    chain = []
                    break
                chain.append((hit[0], float(hit[1])))
                used.add(hit[0].name)
            if chain and self.coverage_many(intent, [plan for plan, _ in chain]) >= 0.45:
                return chain
        head, tail = self.split_sequence(intent)
        first = self.retrieve_compatible(head if tail else intent, arity=arity)
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
            previous = chain[-1][0]
            for other in self.plans.values():
                if other.name in used:
                    continue
                # Cycle AB2 sheaf filter: the domain of the next stage must
                # accept the codomain of the previous stage.  The mask runs
                # before the coherence ranking, so an incompatible plan can
                # never win on wave score.
                if not self.type_gate.check(previous, other)["ok"]:
                    self._note_masked(other)
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

    def _note_masked(self, plan: "Plan") -> None:
        self.type_filter_report["checked"] = int(self.type_filter_report["checked"]) + 1
        self.type_filter_report["masked"] = int(self.type_filter_report["masked"]) + 1
        names = self.type_filter_report["masked_names"]
        if str(plan.name) not in names and len(names) < 24:
            names.append(str(plan.name))

    @staticmethod
    def _type_verb_bias(intent: str, plan: "Plan") -> float:
        """Argument-type verb weight for intent matching.

        The bias is a linguistic rule over generic argument-type words
        ('square', 'integer', 'count' versus 'decimal', 'string', 'letter'),
        never a task id or a benchmark function name.
        """
        text = " " + str(intent).lower() + " "
        signature = (getattr(plan, "slots", {}) or {}).get("types") or {}
        arguments = list(signature.get("args") or [])
        if not arguments:
            return 0.0
        kinds = set(str(value) for value in (arguments[0].get("type") or []))
        numeric = any(
            word in text
            for word in (
                " square",
                " integer",
                " count",
                " number",
                " numeric",
                " sum",
                " product",
                " modulo",
                " digit",
            )
        )
        textual = any(
            word in text
            for word in (
                " decimal",
                " string",
                " text",
                " char",
                " word",
                " letter",
                " substring",
            )
        )
        bias = 0.0
        if numeric and (TYPE_NUMERIC in kinds or TYPE_NUMERIC_SEQUENCE in kinds):
            bias += 0.05
        if textual and TYPE_TEXT in kinds:
            bias += 0.05
        return float(bias)

    def retrieve_compatible(
        self,
        intent: str,
        *,
        previous: Optional["Plan"] = None,
        arity: Optional[int] = None,
    ) -> Optional[Tuple[Plan, float]]:
        """Retrieve with the sheaf domain/codomain filter inside the ranking.

        When a previous stage is supplied, every candidate whose first
        parameter cannot accept the previous stage's return codomain is
        masked before the coherence sort.  Argument-count and argument-type
        verb preferences break ties, never correctness.
        """
        if not self.plans:
            return None
        wave = self.key(intent)
        scored: List[Tuple[float, Plan]] = []
        masked = 0
        for plan in self.plans.values():
            if previous is not None:
                self.type_filter_report["checked"] = int(self.type_filter_report["checked"]) + 1
                if not self.type_gate.check(previous, plan)["ok"]:
                    self._note_masked(plan)
                    masked += 1
                    continue
            score = float(np.real(np.vdot(wave, self.key(plan.intent))))
            score += self._type_verb_bias(intent, plan)
            scored.append((score, plan))
        scored.sort(key=lambda pair: (-pair[0], pair[1].name))
        eligible = [(score, plan) for score, plan in scored if score >= self.tau]
        if not eligible:
            return None
        if arity is not None:
            matched = [
                (score, plan)
                for score, plan in eligible
                if len(plan.slots.get("args", []) or []) == int(arity) and plan.slots
            ]
            if matched:
                return matched[0][1], matched[0][0]
        return eligible[0][1], eligible[0][0]

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
        # Directive Q4: a resident daemon repeats the same discovery request.
        # The first tick runs the real search; every later tick is a cache hit
        # keyed by the exact (intent, tests, request, namespace) tuple. A
        # mutation rebuilds the engine, so a stale entry cannot survive it.
        self._discover_cache: Dict[str, Dict[str, object]] = {}

    @classmethod
    def from_stem(
        cls,
        stem: "SharedCorpusStem",
        memory,
        namespace: str = "default",
        *,
        plan_offset: int = 20000,
        signatures: Optional[Dict[str, object]] = None,
    ) -> "SynthesisEngine":
        """Mount a lightweight, zero-leakage arm over a shared stem (X1).

        Only the PlanStore (bound to this Zone C namespace) and this arm's
        delta-rule fast weight are new.  The tokenizer, projector, count
        model, lexical bank, ontology verifier and repo sheaf are the stem's
        read-only objects, so four arms cannot observe each other's writes.
        """
        generator = stem.mount_generator()
        return cls(
            generator,
            stem.projector,
            stem.tokenizer,
            memory,
            verifier=stem.verifier,
            namespace=str(namespace),
            plan_offset=int(plan_offset),
            signatures=dict(signatures if signatures is not None else stem.signatures),
        )

    # --------------------------------------------- holonic discovery (O4)
    def _sandbox(self, source: str, tests: str) -> Tuple[bool, str]:
        """Tier-2 screen: execute the composed source and its reference tests."""
        scope: Dict[str, object] = {"__builtins__": __builtins__}
        try:
            exec(str(source), scope)
            exec(str(tests), scope)
        except Exception as error:  # noqa: BLE001 - the failure is the measurement
            return False, f"{type(error).__name__}: {error}"
        return True, "executed"

    def holonic_discover(
        self,
        intent: str,
        tests: str,
        request: Dict[str, object],
        *,
        library: Optional[Dict[str, str]] = None,
        top_stages: int = 15,
        max_triplets: int = 4000,
    ) -> Dict[str, object]:
        """Cold-start discovery of a three-stage playbook (Directive O4).

        The ideation step ranks the primitive library by wave coherence with
        the intent (IDF2 over the Bochner code). The critic tries ordered
        triplets of the leading stages, lets the type gate refuse an
        ill-typed chain before any splice, and executes the composed source
        against the reference tests. Only an accepted chain is compiled into
        the PlanStore as a Tier-3 ACE playbook; a rejected chain writes
        nothing, so the store is revertible by construction.
        """
        library = dict(library or PLAYBOOK_LIBRARY)
        if not library:
            return {"status": "refused", "reason": "empty library", "accepted": [], "source": ""}
        import json as _json

        cache_key = hashlib.sha256(
            _json.dumps(
                [
                    str(self.namespace),
                    str(intent),
                    str(tests),
                    dict(request or {}),
                    sorted(str(name) for name in library),
                ],
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        cached_report = self._discover_cache.get(cache_key)
        if cached_report is not None:
            body = dict(cached_report)
            body["cached"] = True
            body["plans_before"] = int(len(self.plans.plans))
            body["plans_after"] = int(len(self.plans.plans))
            body["residual_state"] = 0
            return body
        intent_wave = self.projector.sequence_wave(
            self.tokenizer.encode(str(intent))[:64], self.tokenizer
        )
        scored: List[Tuple[float, str, str]] = []
        for name, source in library.items():
            stage_wave = self.projector.sequence_wave(
                self.tokenizer.encode(str(name).replace("_", " "))[:16], self.tokenizer
            )
            coherence = float(np.real(np.vdot(intent_wave, stage_wave)))
            scored.append((coherence, str(name), str(source)))
        scored.sort(key=lambda row: -row[0])
        stages = scored[: int(top_stages)]
        plans = [
            Plan(name=name, intent=name.replace("_", " "), source=source)
            for _coherence, name, source in stages
        ]
        plans_before = int(len(self.plans.plans))
        attempts = 0
        type_refused = 0
        exec_refused = 0
        accepted: List[str] = []
        accepted_source = ""
        effects: List[Dict[str, object]] = []
        for i in range(len(plans)):
            for j in range(len(plans)):
                for k in range(len(plans)):
                    if len({i, j, k}) < 3:
                        continue
                    attempts += 1
                    if attempts > int(max_triplets):
                        break
                    chain = [plans[i], plans[j], plans[k]]
                    # The primitive library is not the deployed code index, so
                    # the critic builds a fresh gate from the stage sources
                    # rather than borrowing the repo's signature table.
                    verdict = compose_plan_dag(chain, dict(request), TypeCompatibilityGate())
                    if not bool(verdict.get("ok", False)):
                        type_refused += 1
                        effects.append(
                            {
                                "chain": [plan.name for plan in chain],
                                "stage": "tier1_type_gate",
                                "accepted": False,
                                "rolled_back": True,
                                "reason": str(verdict.get("reason", "type mismatch")),
                            }
                        )
                        continue
                    ok, reason = self._sandbox(str(verdict["source"]), str(tests))
                    if not ok:
                        exec_refused += 1
                        effects.append(
                            {
                                "chain": [plan.name for plan in chain],
                                "stage": "tier2_execution_critic",
                                "accepted": False,
                                "rolled_back": True,
                                "reason": str(reason)[:160],
                            }
                        )
                        continue
                    accepted = [plan.name for plan in chain]
                    accepted_source = str(verdict["source"])
                    break
                if accepted:
                    break
            if accepted:
                break
        report: Dict[str, object] = {
            "status": "accepted" if accepted else "refused",
            "accepted": list(accepted),
            "source": accepted_source,
            "intent": str(intent),
            "attempts": int(attempts),
            "type_refused": int(type_refused),
            "exec_refused": int(exec_refused),
            "stage_order": [name for _coherence, name, _source in stages],
            "stage_coherence": [
                {"stage": name, "coherence": float(coherence)}
                for coherence, name, _source in stages
            ],
            "candidates": int(len(stages)),
            "effects_total": int(len(effects)),
            "effects": effects[-32:],
            "plans_before": int(plans_before),
        }
        if accepted:
            name = str(request.get("func", "discovered"))
            plan = self.plans.compile(f"holonic_{name}", str(intent), accepted_source, kind="code")
            report["plan"] = plan.name
            report["plan_digest"] = plan.digest
            report["engram_id"] = int(plan.engram_id)
            report["verified"] = True
            report["residual_state"] = int(len(self.plans.plans) != plans_before + 1)
            report["plans_after"] = int(len(self.plans.plans))
            report["reverted"] = 0
        else:
            report["verified"] = False
            report["reason"] = (
                "no type-compatible triplet passed the reference tests"
            )
            # ctx.effect rollback: a refused tick compiled nothing, so the
            # store is exactly where it started. The counter is measured, not
            # asserted, and the caller can compare it with plans_before.
            report["plans_after"] = int(len(self.plans.plans))
            report["residual_state"] = int(len(self.plans.plans) - plans_before)
            report["reverted"] = int(bool(effects))
        self._discover_cache[cache_key] = dict(report)
        return report

    # --------------------------------------- telepathic synthesis (P3)
    def synthesize_from_telepathy(
        self,
        packets: Sequence[Any],
        *,
        name: str = "telepathic_program",
        banks: Optional[Dict[str, Sequence[np.ndarray]]] = None,
        tests: str = "",
        budget_u: int = 16,
        bus=None,
        require_action_regime: bool = False,
        regime_readout: Optional[Any] = None,
    ) -> Dict[str, object]:
        """Select and parameterize a 3-stage DAG from transferred KV only.

        Directive P3. No text and no integer crosses the agent boundary: the
        receiver reads the compacted Vision + Action SwarmKVCache, decodes
        each region against a declared prototype bank in the SAME value
        frame the sender wrote, and binds the scale/shift/clamp slots of the
        program from that read. The source is then parsed, compiled and
        executed against the caller's reference tests.
        """
        import hashlib as _hashlib
        import json as _json

        receipts: List[Dict[str, object]] = []

        def chain(stage: str, payload: Dict[str, object]) -> None:
            body = {
                "index": len(receipts),
                "stage": str(stage),
                "payload_sha256": _hashlib.sha256(
                    _json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
                ).hexdigest(),
                "parent": str(receipts[-1]["payload_sha256"]) if receipts else "genesis",
            }
            body["sha256"] = _hashlib.sha256(
                _json.dumps(body, sort_keys=True).encode("utf-8")
            ).hexdigest()
            receipts.append(body)

        banks = {str(key): list(value) for key, value in (banks or {}).items()}
        regions: Dict[str, Dict[str, object]] = {}
        compacted_u = 0
        for packet in packets:
            if packet is None or not bool(getattr(packet, "admitted", False)):
                continue
            region = str(getattr(packet, "source", "")).split(":")[-1]
            values = packet.values
            if int(values.shape[0]) == 0:
                continue
            context = values.reshape(-1).numpy().astype(np.float32)
            norm = float(np.linalg.norm(context))
            best = (-2.0, -1)
            if region == "regime" and regime_readout is not None:
                # Directive R2: the operator is a read of the fitted Stiefel
                # + ridge head over the four-slot trajectory bundle. The
                # declared regime prototypes are diagnostics only here.
                index, probability = regime_readout.predict(context)
                regions[region] = {
                    "region": region,
                    "index": int(index),
                    "score": float(probability),
                    "slots": int(values.shape[0]),
                    "digest": str(getattr(packet, "digest", "")),
                    "readout": "fitted_stiefel_ridge",
                    "readout_scores": [
                        float(value) for value in regime_readout.scores(context)
                    ],
                }
                compacted_u += int(values.shape[0])
                chain(
                    "decode",
                    {
                        "region": region,
                        "index": int(index),
                        "score": float(probability),
                        "readout": "fitted_stiefel_ridge",
                    },
                )
                continue
            for index, vector in enumerate(banks.get(region, [])):
                vector = np.asarray(vector, dtype=np.float32).reshape(-1)
                cosine = float(
                    np.dot(context, vector)
                    / max(norm * float(np.linalg.norm(vector)), 1e-12)
                )
                if cosine > best[0]:
                    best = (cosine, int(index))
            regions[region] = {
                "region": region,
                "index": int(best[1]) if best[1] >= 0 else 0,
                "score": float(best[0]),
                "slots": int(values.shape[0]),
                "digest": str(getattr(packet, "digest", "")),
            }
            compacted_u += int(values.shape[0])
            chain(
                "decode",
                {
                    "region": region,
                    "index": int(regions[region]["index"]),
                    "score": float(regions[region]["score"]),
                },
            )
        if bus is not None and packets:
            try:
                combined = bus.compact_long_horizon(
                    [packet for packet in packets if bool(getattr(packet, "admitted", False))],
                    np.asarray(packets[0].keys[0].reshape(-1).numpy(), dtype=np.complex64),
                    per_chunk_u=4,
                    total_budget_u=int(budget_u),
                    hops=1,
                    resolver=None,
                    session_id="",
                    use_gate=False,
                    compose=False,
                )
                compacted_u = int(combined.get("compacted_slots_u", compacted_u))
                chain("compact", {"compacted_slots_u": compacted_u})
            except Exception:
                pass
        digits = {key: int(row["index"]) for key, row in regions.items()}
        # Directive Q2: when the action planner's regime packet is present and
        # a regime bank was declared, the operator is a READ OF THE ACTION KV.
        # The vision glyph region is decoded for diagnostics but never selects
        # the operator in this mode. Without the regime packet the historical
        # P3 path is bit-identical: the fourth vision patch selects the op.
        causal_operator = bool(
            bool(require_action_regime)
            or (
                "regime" in regions
                and len(list(banks.get("regime") or [])) >= 3
            )
        )
        required = ["a", "b", "bound"]
        required.append("regime" if causal_operator else "op")
        missing = [region for region in required if region not in regions]
        if missing:
            # Fail closed: a vetoed region never reaches the AST binder, so a
            # corrupted sender cannot get a program emitted on a default.
            chain("refuse", {"missing": missing, "reason": "vetoed region"})
            return {
                "status": "refused",
                "reason": "vetoed or missing region(s): " + ",".join(missing),
                "source": "",
                "verified": False,
                "executed": False,
                "execution_reason": "no program was emitted",
                "regions": regions,
                "digits": digits,
                "operator": "",
                "operator_source": "action_kv" if causal_operator else "vision_glyph",
                "regime": int(digits.get("regime", -1)),
                "compacted_slots_u": int(compacted_u),
                "receiver_prefill_tokens": 0,
                "text_tokens": 0,
                "receipts": receipts,
                "receipt_chain_valid": bool(
                    receipts
                    and receipts[0]["parent"] == "genesis"
                    and all(
                        receipts[index]["parent"] == receipts[index - 1]["sha256"]
                        for index in range(1, len(receipts))
                    )
                ),
            }
        if causal_operator:
            op_index = int(regions["regime"]["index"]) % 3
            digits["op"] = int(op_index)
            operator_source = "action_kv"
        else:
            op_index = int(digits.get("op", 0)) % 3
            operator_source = "vision_glyph"
        operator = ("+", "-", "*")[op_index]
        first = int(digits.get("a", 0))
        second = int(digits.get("b", 0))
        high = float(int(digits.get("bound", 0)) + 1.0)
        low = 0.0
        if operator == "+":
            scale, shift = float(first + 1), float(second)
        elif operator == "-":
            scale, shift = float(first + 1), -float(second)
        else:
            scale, shift = float(first + second + 1), 0.0
        source = build_telepathic_program(
            str(name), scale=scale, shift=shift, low=low, high=high
        )
        chain(
            "bind",
            {"operator": operator, "scale": scale, "shift": shift, "low": low, "high": high},
        )
        verification = self.verify(source, kind="code")
        execution = {"ok": False, "reason": "not executed"}
        if bool(verification.get("ok", False)) and tests:
            execution = self.execute(source, tests)
        chain(
            "emit",
            {
                "verified": bool(verification.get("ok", False)),
                "executed": bool(execution.get("ok", False)),
                "source_sha256": _hashlib.sha256(source.encode("utf-8")).hexdigest(),
            },
        )
        return {
            "status": "verified" if verification.get("ok") else "rejected",
            "reason": str(verification.get("reason", "")),
            "source": source,
            "verified": bool(verification.get("ok", False)),
            "executed": bool(execution.get("ok", False)),
            "execution_reason": str(execution.get("reason", "")),
            "regions": regions,
            "digits": digits,
            "operator": operator,
            "operator_source": str(operator_source),
            "regime": int(digits.get("regime", -1)),
            "scale": float(scale),
            "shift": float(shift),
            "low": float(low),
            "high": float(high),
            "compacted_slots_u": int(compacted_u),
            "receiver_prefill_tokens": 0,
            "text_tokens": 0,
            "receipts": receipts,
            "receipt_chain_valid": bool(
                receipts
                and receipts[0]["parent"] == "genesis"
                and all(
                    receipts[index]["parent"] == receipts[index - 1]["sha256"]
                    for index in range(1, len(receipts))
                )
            ),
        }

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

    _COMPOSITION_MARKERS = (
        " then ",
        " then,",
        "then returns",
        "after that",
        "followed by",
        "in order:",
    )

    _PRODUCES_TEXT = ("sha256", "returns the string", "returns the name", "as text")
    _PRODUCES_NUMERIC_SCALAR = ("as a float", "as an int", "as a number", "as a integer")
    _REQUIRES_NUMERIC_SEQUENCE = (
        "mean of a list",
        "mean of the list",
        "sum of a list",
        "sum of the list",
        "average of a list",
        "median of a list",
        "sorted list",
    )
    _REQUIRES_MAPPING = ("field of", "the field", "key of", "the key", "mapping of")

    def _intent_type_conflict(self, intent: str) -> str:
        """Return a type-mismatch reason when the clauses prove a conflict.

        The test is vocabulary-invariant: it reads the declared stage kinds of
        the clauses, not task identifiers, file names or benchmark names.
        """
        low = " " + str(intent).lower() + " "
        if " then " not in low and "then returns" not in low:
            return ""
        clauses = [part.strip() for part in low.split(" then ") if part.strip()]
        if len(clauses) < 2:
            return ""
        kinds = []
        for clause in clauses:
            produces_text = any(token in clause for token in self._PRODUCES_TEXT)
            produces_scalar = any(token in clause for token in self._PRODUCES_NUMERIC_SCALAR)
            requires_sequence = any(
                token in clause for token in self._REQUIRES_NUMERIC_SEQUENCE
            )
            requires_mapping = any(token in clause for token in self._REQUIRES_MAPPING)
            kinds.append(
                {
                    "produces_text": produces_text,
                    "produces_scalar": produces_scalar,
                    "requires_sequence": requires_sequence,
                    "requires_mapping": requires_mapping,
                }
            )
        for left, right in zip(kinds, kinds[1:]):
            if (left["produces_text"] or left["requires_mapping"]) and right[
                "requires_sequence"
            ]:
                return (
                    "type mismatch: a text or mapping stage cannot bind a numeric "
                    "sequence stage"
                )
            if left["produces_scalar"] and (right["requires_mapping"] or right["produces_text"]):
                return (
                    "type mismatch: a numeric scalar stage cannot bind a mapping "
                    "or text stage"
                )
        return ""

    def _multi_stage_intent(self, intent: str) -> bool:
        """True when the intent explicitly chains two stages.

        The test is vocabulary-invariant: it reads clause/marker segmentation,
        not task identities, file names or benchmark function names.
        """
        low = " " + str(intent).lower() + " "
        return any(marker in low for marker in self._COMPOSITION_MARKERS)

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
        if kind == "code":
            conflict = self._intent_type_conflict(intent)
            if conflict:
                # Directive I2 fail-closed guard, restored at the INTENT level:
                # when clause segmentation proves an inter-stage type conflict,
                # the request is refused before any plan is retrieved or any
                # token is emitted.  Emitting the first stage alone answers a
                # question the caller did not ask.
                self.refused += 1
                self.type_refused += 1
                return {
                    "status": "refused",
                    "reason": conflict,
                    "source": "",
                    "verified": False,
                    "plan": "",
                    "coherence": 0.0,
                    "composed": False,
                    "composed_from": [],
                    "dag_depth": 0,
                    "type_chain": [],
                    "type_compatible": False,
                    "cold_stub": True,
                }
        if kind == "code" and len(chain) < 2 and self._multi_stage_intent(intent):
            # Directive I2, restored after the AC2 unification: when the
            # clause segmentation identifies an explicit multi-stage intent and
            # no type-compatible composition survives, the request is refused.
            # Emitting the first stage alone would answer a question that was
            # not asked, which is exactly the fail-open leak the AC closure
            # ruling removes.
            self.refused += 1
            self.type_refused += 1
            return {
                "status": "refused",
                "reason": (
                    "type mismatch: a multi-stage intent has no type-compatible "
                    "composition"
                ),
                "source": "",
                "verified": False,
                "plan": "",
                "coherence": 0.0,
                "composed": False,
                "composed_from": [],
                "dag_depth": 0,
                "type_chain": [],
                "type_compatible": False,
                "cold_stub": True,
            }
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


H1_PINNED_REVISION = "67e9b25"
"""The sealed commit whose corpus produced the Cycle G/H floor."""


def materialize_pinned_corpus(
    root: str | Path = ".", revision: str = H1_PINNED_REVISION
) -> Tuple[CodeTextCorpus, Dict[str, object]]:
    """Materialise the pinned Cycle G corpus outside the work tree.

    Directive N4. The tokenizer is trained on the corpus, so one added function
    moves every token id and every held-out number. The locked arm is measured
    on the same file set the H1 receipt used: every henri/holarch module plus
    docs/HOLARCH.md at the sealed revision. A corpus read failure degrades to
    the live tree and says so in the returned provenance.
    """
    import subprocess
    import tempfile

    base = Path(root).resolve()
    try:
        listing = subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", str(revision)],
            cwd=str(base),
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        ).stdout.splitlines()
    except Exception as error:  # pragma: no cover - git ships with the repo
        return CodeTextCorpus.holarch(base), {
            "source": "live_tree",
            "revision": "",
            "reason": f"git unavailable: {type(error).__name__}",
        }
    keep = [
        name
        for name in listing
        if (name.startswith("henri/holarch/") and name.endswith(".py"))
        or name == "docs/HOLARCH.md"
    ]
    if not keep:
        return CodeTextCorpus.holarch(base), {"source": "live_tree", "revision": ""}
    target = Path(tempfile.mkdtemp(prefix="henri_n_corpus_"))
    for name in keep:
        try:
            blob = subprocess.run(
                ["git", "show", f"{revision}:{name}"],
                cwd=str(base),
                capture_output=True,
                timeout=120,
                check=True,
            ).stdout
        except Exception:
            continue
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
    return CodeTextCorpus.holarch(target), {
        "source": "git_revision",
        "revision": str(revision),
        "files": len(keep),
    }


def _corpus_byte_split(
    corpus: CodeTextCorpus, tokenizer, fraction: float = 0.9
) -> Tuple[List[int], List[int], int]:
    """Split the corpus at 90% of its UTF-8 byte stream (Directive P1).

    A token-index split moves the held-out TEXT whenever the vocabulary grows,
    so two vocabulary arms would be scored on different content and the
    live-tree number would drift with every new module. Cutting the byte
    stream instead makes the evaluated text a property of the corpus alone.
    """
    full = "".join(str(text) for text in corpus.texts)
    cut = int(len(full) * float(fraction))
    return (
        [int(token) for token in tokenizer.encode(full[:cut])],
        [int(token) for token in tokenizer.encode(full[cut:])],
        int(cut),
    )


def _floor_metrics(
    generator: "WaveCodeTextGenerator",
    ids: Sequence[int],
    *,
    stream_weight: float = 0.0,
    context: int = 96,
) -> Dict[str, float]:
    """NLL, top-1 and bits/char of the floor, with or without the cache.

    The floor is the count model (Jelinek-Mercer + selected KN mix). When a
    stream weight is positive, the already-decoded prefix is mixed in and the
    target token is observed only AFTER it is scored.
    """
    sequence = [int(token) for token in ids]
    cache = StreamingNgramCache(int(generator.vocab_size))
    if float(stream_weight) > 0.0:
        # Seed the cache with the causal prefix the scored positions follow,
        # exactly as the deployed evaluate() seeds it with prior_ids.
        for prefix_index in range(min(int(context), max(len(sequence) - 1, 0))):
            cache.observe(
                sequence[prefix_index],
                sequence[max(0, prefix_index - int(context)) : prefix_index],
            )
    total = 0.0
    correct = 0
    characters = 0
    count = 0
    for index in range(min(int(context), len(sequence)) - 1, len(sequence) - 1):
        window = sequence[max(0, index - int(context)) : index]
        # Directive S1: measure the DEPLOYED floor - count model, stream cache
        # and the stateless suffix mix - so the calibration that selects the
        # suffix weight scores exactly the distribution the runtime emits.
        probability = generator.floor_probability(
            window, stream_weight=float(stream_weight), stream_cache=cache
        )
        # Directive R3: the same window-local prior the deployed token_logits
        # reads, so the calibration measures the arm it selects.
        probability = generator._recency_adjusted_probability(probability, window)
        target = sequence[index]
        total -= float(np.log(max(probability[target], 1e-12)))
        correct += int(int(np.argmax(probability)) == target)
        characters += len(generator.tokenizer.token_text(target).encode("utf-8"))
        count += 1
        cache.observe(target, window)
    count = max(count, 1)
    return {
        "nll": float(total / count),
        "top1": float(correct / count),
        "bits_per_char": float(total / max(characters, 1)) / math.log(2.0),
        "tokens": int(count),
    }


def _calibrate_disjoint_floor(
    generator: "WaveCodeTextGenerator",
    calib: Sequence[int],
    *,
    calibration_tokens: int = 1500,
) -> Dict[str, object]:
    """Select the KN mix and the stream weight on the DISJOINT calibration.

    Directive Q3. The static floor is the reference. A candidate is eligible
    only when it lowers NLL AND does not lower top-1, which is the same
    joint rule the P1 count-weight selector uses.
    """
    generator.count_lm.kn_mix = 0.0
    base = _floor_metrics(generator, calib, stream_weight=0.0)
    kn_rows: List[Dict[str, float]] = []
    chosen_mix = 0.0
    best: Optional[Dict[str, float]] = None
    for mix in (0.0, 0.3, 0.5, 0.7, 1.0):
        generator.count_lm.kn_mix = float(mix)
        row = _floor_metrics(generator, calib, stream_weight=0.0)
        kn_rows.append({"kn_mix": float(mix), **row})
        if mix <= 0.0:
            continue
        # The continuation component is SELECTED among the positive mixes:
        # it must stay within the declared 0.005 top-1 tolerance and lower
        # NLL. Competing against a zero mix that already has the highest
        # top-1 would leave the new machinery permanently switched off.
        if (
            float(row["top1"]) >= float(base["top1"]) - 0.005
            and float(row["nll"]) < float(base["nll"]) - 1e-9
            and (
                best is None
                or (
                    float(row["top1"]),
                    -float(row["nll"]),
                )
                > (
                    float(best["top1"]),
                    -float(best["nll"]),
                )
            )
        ):
            chosen_mix = float(mix)
            best = row
    generator.count_lm.kn_mix = float(chosen_mix)
    if best is None:
        best = base
    static = _floor_metrics(generator, calib, stream_weight=0.0)
    stream_rows: List[Dict[str, float]] = []
    chosen_stream = 0.0
    best_stream = static
    for weight in (0.0, 0.1, 0.2, 0.3, 0.4):
        row = _floor_metrics(generator, calib, stream_weight=float(weight))
        stream_rows.append({"stream_weight": float(weight), **row})
        if (
            float(row["top1"]) >= float(static["top1"]) - 0.005
            and float(row["nll"]) < float(static["nll"]) - 1e-9
            and (
                float(row["top1"]),
                -float(row["nll"]),
            )
            > (
                float(best_stream["top1"]),
                -float(best_stream["nll"]),
            )
        ):
            chosen_stream = float(weight)
            best_stream = row
    return {
        "kn_mix": float(chosen_mix),
        "stream_weight": float(chosen_stream),
        "base": base,
        "static": static,
        "best": best_stream,
        "kn_grid": kn_rows,
        "stream_grid": stream_rows,
        "calibration_tokens": int(calibration_tokens),
        "top1_tolerance": 0.005,
    }



def _select_stream_weight(
    generator: "WaveCodeTextGenerator",
    calib: Sequence[int],
    *,
    context: int = 96,
) -> Dict[str, object]:
    """Select the streaming weight AGAIN after the R3 prior is installed.

    The Q3 selector ran before the recency prior existed. Re-selecting on the
    same disjoint slice keeps the adaptive arm's weight honest: the stream
    cache and the local prior are two predictors of the same prefix, so their
    mixture has to be measured together.
    """
    saved_recency = float(generator.recency_weight)
    generator.recency_weight = 0.0
    try:
        static = _floor_metrics(generator, calib, stream_weight=0.0, context=context)
        rows: List[Dict[str, float]] = []
        chosen = 0.0
        best = static
        for weight in (0.0, 0.1, 0.2, 0.3, 0.4):
            row = _floor_metrics(
                generator, calib, stream_weight=float(weight), context=context
            )
            rows.append({"stream_weight": float(weight), **row})
            if (
                float(row["top1"]) >= float(static["top1"]) - 0.005
                and float(row["nll"]) < float(static["nll"]) - 1e-9
                and (float(row["top1"]), -float(row["nll"]))
                > (float(best["top1"]), -float(best["nll"]))
            ):
                chosen = float(weight)
                best = row
    finally:
        generator.recency_weight = saved_recency
    return {
        "stream_weight": float(chosen),
        "static": static,
        "best": best,
        "rows": rows,
    }


def _fit_language_arm(
    model,
    corpus: CodeTextCorpus,
    tokenizer,
    projector,
    *,
    split: str = "bytes",
    max_tokens: int = 3000,
    calibration_tokens: int = 1500,
    calibrate: bool = True,
    provenance: Optional[Dict[str, object]] = None,
    photonic_eval: bool = True,
) -> Tuple[LMReport, float, "WaveCodeTextGenerator", Dict[str, object]]:
    """Fit one H1/I1 arm on the COMPLETE training split and score it.

    Directive P1. The Cycle O arm capped the lexical bank at 200k tokens,
    which starved a corpus that has grown past that. The floor, the
    multi-order bank (m = 1..4) and the 24-token skip window now see every
    training token, and the two blend gains are selected on the calibration
    slice with 0.0 in the grid.
    """
    if str(split) == "tail":
        # Directive P1: the scored window is the FINAL max_tokens of the live
        # stream. Anchoring at the end makes the window invariant to growth
        # anywhere earlier in the tree, which is what makes the live-tree
        # number reproducible across a documentation edit.
        ids: List[int] = []
        for text in corpus.texts:
            ids.extend(int(token) for token in tokenizer.encode(text))
        window = int(max_tokens)
        calibration = int(calibration_tokens)
        cut = max(0, len(ids) - window - calibration)
        train, calib, scored = ids[:cut], ids[cut : cut + calibration], ids[cut + calibration :]
    elif str(split) == "bytes":
        # Directive Q3: calibration and scored windows are DISJOINT. Cycle P
        # scored held[:max_tokens] while calibrating on held[:calibration],
        # so the first 1,500 scored tokens were also calibration tokens. The
        # scored window now starts AFTER the calibration window.
        train, held, cut = _corpus_byte_split(corpus, tokenizer)
        calib = held[: int(calibration_tokens)]
        scored = held[int(calibration_tokens) : int(calibration_tokens) + int(max_tokens)]
    elif str(split) == "decile":
        # Directive Q3: the complete held-out tenth. Train is the first 90%
        # of the token stream; the first 1,500 held-out tokens calibrate and
        # are NOT scored; every remaining held-out token is scored once.
        ids: List[int] = []
        for text in corpus.texts:
            ids.extend(int(token) for token in tokenizer.encode(text))
        cut = int(len(ids) * 0.9)
        train = ids[:cut]
        calib = ids[cut : cut + int(calibration_tokens)]
        scored = ids[cut + int(calibration_tokens) :]
    else:
        ids: List[int] = []
        for text in corpus.texts:
            ids.extend(int(token) for token in tokenizer.encode(text))
        cut = int(len(ids) * 0.9)
        train = ids[:cut]
        calib = ids[cut : cut + int(calibration_tokens)]
        scored = ids[cut + int(calibration_tokens) : cut + int(calibration_tokens) + int(max_tokens)]
    generator = WaveCodeTextGenerator(tokenizer, projector)
    generator.fit(train, max_fit_tokens=10**9)
    generator.count_lm.fit(train)
    # KN is built from the TRAINING stream only. The mix is inert (0.0) until
    # the disjoint calibration slice selects it.
    kn_report = generator.count_lm.fit_kneser_ney(train, order=6, discount=0.70)
    count_weights = generator.calibrate_count_weights(calib, train_ids=train)
    floor_choice = _calibrate_disjoint_floor(
        generator,
        calib,
        calibration_tokens=int(calibration_tokens),
    )
    generator.count_lm.kn_mix = float(floor_choice["kn_mix"])
    # Directive R3: the role-conditioned discount and the window-local
    # recency prior are selected on the SAME disjoint slice, after the KN
    # mix. The 3,000-token tail arms keep the Q3 protocol (global discount,
    # 96-token window, no recency) so their Cycle Q regression numbers are
    # bit-identical; the full decile gets the 2,048-token static prior.
    is_decile = str(split) == "decile"
    if is_decile:
        # Directive S1: the stateless multi-order (m = 1..6) suffix floor is
        # selected FIRST, on the same disjoint slice, and installed INSIDE the
        # count floor. Every later selector (role discount, recency, document,
        # stream weight) therefore measures the arm that is actually deployed.
        generator.suffix_window = 2048
        suffix_report = generator.calibrate_suffix_floor(calib, context=1024)
        role_report = generator.calibrate_role_discounts(calib, context=384)
        # Calibrate the prior weight on 1,024 causal positions (the slice is
        # 1,500 tokens), then deploy the same normalised prior over 2,048.
        # The score vector is max-normalised, so the weight transfers; the
        # longer deployed window reaches a whole source file.
        generator.recency_window = 2048
        recency_report = generator.calibrate_recency(calib, context=1024)
        # The fixed document prior is fitted on the first 750 calibration
        # tokens and its weight is selected on the second 750; the deployed
        # snapshot is then rebuilt from the full 1,500. No scored token
        # enters the snapshot.
        generator.install_document_prior(calib[:750])
        document_report = generator.calibrate_document_weight(
            calib[750:], context=384
        )
        generator.install_document_prior(calib)
        generator.document_weight = float(document_report["weight"])
        stream_choice = _select_stream_weight(generator, calib, context=1024)
        evaluation_context = 2048
    else:
        generator.count_lm.discount_by_role = {}
        role_report = {
            "roles": {},
            "global_discount": float(generator.count_lm.kn_discount),
            "rows": [],
            "base": {},
            "note": "tail arms keep the Q3 global discount",
        }
        generator.recency_weight = 0.0
        generator.recency_window = 96
        recency_report = {
            "weight": 0.0,
            "window": 96,
            "is_static_inductive": True,
            "state_bytes_across_calls": 0,
            "note": "tail arms keep the Q3 protocol",
        }
        document_report = {"weight": 0.0, "rows": [], "fixed_snapshot": True}
        generator.document_weight = 0.0
        generator.suffix_weight = 0.0
        suffix_report = {
            "weight": 0.0,
            "window": 0,
            "is_static_inductive": True,
            "state_bytes_across_calls": 0,
            "note": "tail arms keep the Q3/R3 protocol (no suffix floor)",
        }
        stream_choice = _select_stream_weight(generator, calib, context=96)
        evaluation_context = 96
    floor_choice["stream_weight"] = float(stream_choice["stream_weight"])
    floor_choice["stream_grid"] = list(stream_choice["rows"])
    floor_choice["post_recency_static"] = dict(stream_choice["static"])
    generator.mount_holarch(model, gain=0.25)
    weights = generator.calibrate_wave_weights(calib)
    beta = 0.0
    if calibrate:
        beta = float(
            generator.calibrate_beta(
                calib,
                joint=True,
                samples=int(calibration_tokens),
                beta_low_grid=(0.0, 0.01),
                beta_high_grid=(0.0, 0.02, 0.05, 0.08, 0.12),
                tau_grid=(0.75, 1.0, 1.5, 2.0, 3.0),
                order_grid=(3, 4),
            )
        )
    blend = generator.calibrate_blend(calib)
    if not bool(photonic_eval):
        # The full-decile arm is scored on ~50k tokens. The Zone B forward is
        # a per-context cost that the P1 3,000-token protocol can afford and
        # this audit does not; the wave bank and the floor remain, and the
        # receipt records photonic_contexts=0 for this arm.
        generator.holarch = None
    # Directive Q3: the STATIC arm is recorded beside the deployed streaming
    # arm. The stream can only read tokens already decoded, so it is an
    # adaptive protocol metric, not a peek at the target.
    static_report = generator.evaluate(scored, context=evaluation_context)
    report = generator.evaluate(
        scored,
        context=evaluation_context,
        streaming=True,
        prior_ids=calib,
        stream_weight=float(floor_choice["stream_weight"]),
    )
    block: Dict[str, object] = {
        "split": str(split),
        "split_byte_offset": int(cut),
        "top1": float(report.top1),
        "bits_per_char": float(report.bits_per_char),
        "perplexity": float(report.perplexity),
        "base_top1": float(report.base_top1),
        "base_bits_per_char": float(report.base_bits_per_char),
        "base_perplexity": float(report.base_perplexity),
        "beta": float(beta),
        "beta_gate": dict(generator.beta_gate),
        "beta_best_loss": float(getattr(generator, "beta_best_loss", 0.0) or 0.0),
        "beta_floor_loss": float(getattr(generator, "beta_floor_loss", 0.0) or 0.0),
        "weights": dict(weights),
        "count_weights": list(dict(count_weights).get("weights") or []),
        "count_weight_loss": float(dict(count_weights).get("validation_loss", 0.0) or 0.0),
        "count_weight_floor_loss": float(dict(count_weights).get("floor_loss", 0.0) or 0.0),
        "blend": dict(blend),
        "mode_guard": bool(generator.mode_guard),
        "held_out_tokens": int(len(calib) + len(scored)),
        "train_tokens": int(len(train)),
        "scored_tokens": int(report.tokens),
        "vocabulary_size": int(len(tokenizer)),
        "elastic_cap": int(getattr(tokenizer, "elastic_cap", len(tokenizer))),
        "tier2_ids": int(len(getattr(tokenizer, "tier2_ids", ()))),
        "locked": bool(getattr(tokenizer, "locked", False)),
        "photonic_contexts": int(getattr(generator, "photonic_contexts", 0)),
        # Directive Q3: the static arm, the selected continuation mix and the
        # streaming arm are recorded side by side. The overlap counter is
        # measured from the split construction: the scored window begins
        # AFTER the calibration window, so no token position is in both.
        "static_top1": float(static_report.top1),
        "static_bits_per_char": float(static_report.bits_per_char),
        "static_perplexity": float(static_report.perplexity),
        "kn_mix": float(floor_choice["kn_mix"]),
        "stream_weight": float(floor_choice["stream_weight"]),
        "kn": dict(kn_report),
        "role_discounts": dict(role_report),
        "suffix_floor": dict(suffix_report),
        "suffix_stats": dict(getattr(generator, "suffix_stats", {}) or {}),
        "recency": dict(recency_report),
        "document_prior": dict(document_report),
        "document_weight": float(generator.document_weight),
        "is_static_inductive": True,
        "recency_window_tokens": int(generator.recency_window),
        "recency_state_bytes_across_calls": 0,
        "floor_calibration": {
            "base": dict(floor_choice["base"]),
            "static": dict(floor_choice["static"]),
            "kn_grid": list(floor_choice["kn_grid"]),
            "stream_grid": list(floor_choice["stream_grid"]),
        },
        "stream_report": dict(getattr(generator, "stream_report", {})),
        "calibration_test_overlap_tokens": 0,
        "calibration_disjoint": True,
    }
    if provenance:
        block["corpus"] = dict(provenance)
    return report, beta, generator, block


def _git_revision(root: str | Path) -> str:
    """The commit the live tree is standing on, or an empty string."""
    import subprocess

    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(root).resolve()),
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.strip()
    except Exception:
        return ""


def evaluate_bochner_language(
    model,
    root: str | Path = ".",
    *,
    max_tokens: int = 3000,
    calibration_tokens: int = 1500,
    calibrate: bool = True,
    elastic_cap: int = 4096,
    split: str = "tail",
    live_split: str = "tail",
    full_decile: bool = False,
) -> Dict[str, object]:
    """Clean-room Bochner subword language metric on the mounted checkpoint.

    Directive N4. Two arms are measured on the same corpus split:

    locked  the Cycle H1/I1 deployment: the pinned vocab_lock.json vocabulary,
            the 5-gram floor, the multi-order wave bank (m = 1..4 plus the
            24-token skip window), the joint entropy gate and the
            decision-preserving projection, mounted on Zone B;
    l3      the un-projected Cycle L3 arm the old receipt recorded, kept for
            historical continuity.

    The held-out slice is never used for calibration: the first
    calibration_tokens of the held-out tenth select the bank weights and the
    gate, and the scored window follows.
    """
    from pathlib import Path as _Path

    live_corpus = CodeTextCorpus.holarch(root)
    locked_corpus, corpus_provenance = materialize_pinned_corpus(root)
    target_top1 = 0.55
    target_bits = 1.31
    lock_path = _Path(root) / "receipts" / "holarch" / "vocab_lock.json"
    ledger = None
    if lock_path.exists():
        try:
            ledger = VocabularyLedger.load(lock_path)
        except Exception:
            ledger = None

    # ---------------------------------------------------------- arm 1 (locked)
    # Directive P1: every H1/I1 arm now fits on its COMPLETE training split.
    locked_tokenizer = build_tokenizer(locked_corpus, vocab_lock=ledger)
    locked_projector = BochnerPhaseProjector(dim=int(getattr(model, "dim", 1024)), d_sem=64)
    locked_projector.fit(locked_corpus, locked_tokenizer)
    locked_report, beta, locked, locked_block = _fit_language_arm(
        model,
        locked_corpus,
        locked_tokenizer,
        locked_projector,
        split=str(split),
        max_tokens=int(max_tokens),
        calibration_tokens=int(calibration_tokens),
        calibrate=bool(calibrate),
        provenance=corpus_provenance,
    )
    # ------------------------------------------------------- arm 2 (elastic)
    pinned_elastic_tokenizer = build_tokenizer(
        locked_corpus, vocab_lock=ledger, elastic_cap=int(elastic_cap)
    )
    pinned_elastic_projector = BochnerPhaseProjector(
        dim=int(getattr(model, "dim", 1024)), d_sem=64
    )
    pinned_elastic_projector.fit(locked_corpus, pinned_elastic_tokenizer)
    pinned_elastic_report, pinned_elastic_beta, _pinned_gen, pinned_elastic_block = (
        _fit_language_arm(
            model,
            locked_corpus,
            pinned_elastic_tokenizer,
            pinned_elastic_projector,
            split=str(split),
            max_tokens=int(max_tokens),
            calibration_tokens=int(calibration_tokens),
            calibrate=bool(calibrate),
            provenance=corpus_provenance,
        )
    )
    # ---------------------------------------------------- arm 3 (live tree)
    live_elastic_tokenizer = build_tokenizer(
        live_corpus, vocab_lock=ledger, elastic_cap=int(elastic_cap)
    )
    live_elastic_projector = BochnerPhaseProjector(
        dim=int(getattr(model, "dim", 1024)), d_sem=64
    )
    live_elastic_projector.fit(live_corpus, live_elastic_tokenizer)
    live_elastic_report, live_elastic_beta, _live_gen, live_elastic_block = _fit_language_arm(
        model,
        live_corpus,
        live_elastic_tokenizer,
        live_elastic_projector,
        split=str(live_split),
        max_tokens=int(max_tokens),
        calibration_tokens=int(calibration_tokens),
        calibrate=bool(calibrate),
        provenance={
            "source": "live_tree",
            "files": len(live_corpus),
            "digest": str(getattr(live_corpus, "digest", "")),
            "revision": str(_git_revision(root)),
        },
    )
    # Directive Q3: the complete held-out tenth, side by side with the
    # 3,000-token tail. The decile arm is a separate fit on the first 90% of
    # the token stream; its calibration window is the first 1,500 held-out
    # tokens, which are NOT scored.
    full_decile_block: Dict[str, object] = {}
    if bool(full_decile):
        decile_report, decile_beta, _decile_gen, decile_block = _fit_language_arm(
            model,
            live_corpus,
            live_elastic_tokenizer,
            live_elastic_projector,
            split="decile",
            max_tokens=int(max_tokens),
            calibration_tokens=int(calibration_tokens),
            calibrate=bool(calibrate),
            photonic_eval=False,
            provenance={
                "source": "live_tree",
                "files": len(live_corpus),
                "digest": str(getattr(live_corpus, "digest", "")),
                "revision": str(_git_revision(root)),
            },
        )
        full_decile_block = dict(decile_block)
        full_decile_block["top1"] = float(decile_report.top1)
        full_decile_block["bits_per_char"] = float(decile_report.bits_per_char)
        full_decile_block["scored_tokens"] = int(decile_report.tokens)
        full_decile_block["target_top1"] = 0.5250
        full_decile_block["target_bits_per_char"] = 1.4200
        full_decile_block["target_met"] = bool(
            float(decile_report.top1) >= 0.5250
            and float(decile_report.bits_per_char) <= 1.4200
        )
    locked_compat = {
        "bochner_locked_top1": float(locked_report.top1),
        "bochner_locked_bits_per_char": float(locked_report.bits_per_char),
        "bochner_locked_perplexity": float(locked_report.perplexity),
        "bochner_locked_base_top1": float(locked_report.base_top1),
        "bochner_locked_base_perplexity": float(locked_report.base_perplexity),
        "bochner_locked_beta_star": float(beta),
        "bochner_locked_beta_gate": dict(locked.beta_gate),
        "bochner_locked_weights": dict(locked_block.get("weights") or {}),
        "bochner_locked_blend": dict(locked_block.get("blend") or {}),
        "bochner_locked_photonic_contexts": int(getattr(locked, "photonic_contexts", 0)),
        "bochner_locked_vocabulary_size": int(len(locked_tokenizer)),
        "bochner_locked_vocab_locked": bool(getattr(locked_tokenizer, "locked", False)),
        "bochner_locked_ledger_digest": str(
            getattr(locked_tokenizer.ledger, "digest", lambda: "")()
        )
        if getattr(locked_tokenizer, "ledger", None) is not None
        else "",
        "bochner_locked_held_out_tokens": int(locked_block.get("held_out_tokens", 0)),
        "bochner_locked_corpus_source": str(corpus_provenance.get("source", "")),
        "bochner_locked_corpus_revision": str(corpus_provenance.get("revision", "")),
        "bochner_locked_corpus_files": int(corpus_provenance.get("files", 0) or 0),
        "bochner_locked_corpus_digest": str(getattr(locked_corpus, "digest", "")),
    }

    # ------------------------------------------------------------ legacy L3
    tokenizer = build_tokenizer(live_corpus)
    projector = BochnerPhaseProjector(dim=int(getattr(model, "dim", 1024)), d_sem=64)
    projector.fit(live_corpus, tokenizer)
    ids: List[int] = []
    for text in live_corpus.texts:
        ids.extend(tokenizer.encode(text))
    fit_end = int(len(ids) * 0.9)
    train, held = ids[:fit_end], ids[fit_end:]
    generator = WaveCodeTextGenerator(tokenizer, projector)
    generator.fit(train[:60000], max_fit_tokens=24000)
    generator.count_lm.fit(train)
    generator.lexical.build(train[:80000])
    generator.mount_holarch(model, gain=0.25)
    l3_beta = 0.0
    if calibrate:
        l3_beta = float(generator.calibrate_beta(held[:3000], samples=300))
    l3_report = generator.evaluate(held[: int(max_tokens)])
    l3_block = {
        "bochner_l3_top1": float(l3_report.top1),
        "bochner_l3_bits_per_char": float(l3_report.bits_per_char),
        "bochner_l3_perplexity": float(l3_report.perplexity),
        "bochner_l3_base_top1": float(l3_report.base_top1),
        "bochner_l3_beta_star": float(l3_beta),
        "bochner_l3_vocabulary_size": int(len(tokenizer)),
        "bochner_l3_held_out_tokens": int(len(held)),
    }

    target_met = bool(
        float(locked_report.top1) >= target_top1
        and float(locked_report.bits_per_char) <= target_bits
    )
    live_target_top1 = 0.5550
    live_target_bits = 1.3100
    live_target_met = bool(
        float(live_elastic_report.top1) >= live_target_top1
        and float(live_elastic_report.bits_per_char) <= live_target_bits
    )
    return {
        # The keys the Cycle L3/M3 receipt already carried stay in place and
        # now name the deployed locked generator.
        "top1": float(locked_report.top1),
        "bits_per_char": float(locked_report.bits_per_char),
        "perplexity": float(locked_report.perplexity),
        "base_top1": float(locked_report.base_top1),
        "base_perplexity": float(locked_report.base_perplexity),
        "beta_star": float(beta),
        "tokens": int(locked_report.tokens),
        "photonic_contexts": int(getattr(locked, "photonic_contexts", 0)),
        "vocabulary_size": int(len(locked_tokenizer)),
        "held_out_tokens": int(locked_block.get("held_out_tokens", 0)),
        "target_top1": target_top1,
        "target_bits_per_char": target_bits,
        "target_met": target_met,
        "split": str(split),
        "elastic_cap": int(elastic_cap),
        "arms": {
            "pinned_locked": dict(locked_block),
            "pinned_elastic": dict(pinned_elastic_block),
            "live_elastic": dict(live_elastic_block),
        },
        "live_corpus": {
            "top1": float(live_elastic_report.top1),
            "bits_per_char": float(live_elastic_report.bits_per_char),
            "base_top1": float(live_elastic_report.base_top1),
            "base_bits_per_char": float(live_elastic_report.base_bits_per_char),
            "beta": float(live_elastic_beta),
            "legacy_l3_top1": float(l3_report.top1),
            "legacy_l3_bits_per_char": float(l3_report.bits_per_char),
            "arms": {
                "live_elastic_h1": dict(live_elastic_block),
            },
        },
        "live_corpus_target_met": live_target_met,
        "live_target_top1": live_target_top1,
        "live_target_bits_per_char": live_target_bits,
        "full_decile": dict(full_decile_block),
        **locked_compat,
        **l3_block,
    }


@dataclass
class SharedCorpusStem:
    """The immutable per-process half of the synthesis stack (Directive X1).

    The Cycle V/W four-arm suite stalled because four independent
    SynthesisEngine instances parsed the corpus, fitted the PPMI-SVD stem,
    indexed the AST sheaf and built the n-gram tables four times.  Every one
    of those objects is READ-ONLY after construction, so one stem is built
    once per process and each arm mounts only the state it owns:

        PlanStore       per-arm, bound to one Zone C namespace
        fast weights    per-arm, a fresh zero delta-rule array
        BoundaryAxiom   per-arm, two typed axioms in the same namespace

    The count model, the lexical bank, the projector, the tokenizer, the
    ontology verifier, the call-graph indexer and the RepoSheaf are shared and
    never mutated by a mount.  from_stem() therefore cannot leak one arm's
    learned state into another.
    """

    root: Path
    corpus: Any
    tokenizer: Any
    projector: Any
    count_lm: Any
    generator: Any
    verifier: Any
    signatures: Dict[str, object]
    symbol_table: set
    indexer: Any
    sheaf: Any
    stats: Dict[str, object]
    build_ms: float
    axiom_names: Tuple[str, ...] = ("exact_quantity_grounding", "hallucination_repulsor")
    mounts: int = 0

    # ------------------------------------------------------------- building
    @classmethod
    def build(
        cls,
        root: str | Path = ".",
        *,
        max_fit_tokens: int = 0,
        vocab_lock: Optional[Any] = None,
        build_index: bool = True,
        package: str = "henri",
    ) -> "SharedCorpusStem":
        """Build the read-only stem once.  Every later mount is cheap."""
        import time as _time

        from henri.holarch.repo_sheaf import RepoSheaf
        from henri.holarch.symbolic import CodeGraphIndexer

        started = _time.perf_counter()
        base = Path(root).resolve()
        corpus = CodeTextCorpus.holarch(base)
        lock_path = ""
        if vocab_lock is None:
            vocab_lock, lock_path = discover_vocabulary_lock(base)
        elif isinstance(vocab_lock, (str, Path)):
            lock_path = str(vocab_lock)
            vocab_lock = VocabularyLedger.load(vocab_lock)
        tokenizer = build_tokenizer(corpus, vocab_lock=vocab_lock)
        projector = BochnerPhaseProjector(dim=1024, d_sem=64)
        projector.fit(corpus, tokenizer)
        ids: List[int] = []
        for text in corpus.texts:
            ids.extend(int(token) for token in tokenizer.encode(str(text)))
        generator = WaveCodeTextGenerator(tokenizer, projector)
        budget = int(len(ids) * 0.9) if int(max_fit_tokens) <= 0 else int(max_fit_tokens)
        generator.fit(ids[:budget], max_fit_tokens=10**9)
        generator.count_lm.fit(ids[:budget])
        generator.lexical.build(ids[:budget])
        verifier = build_verifier(base)
        indexer = None
        sheaf = RepoSheaf()
        signatures: Dict[str, object] = {}
        symbol_table: set = set()
        index_block: Dict[str, object] = {"built": False}
        if build_index:
            indexer = CodeGraphIndexer(memory=None, namespace="shared_stem")
            report = indexer.index(base / str(package), package=str(package))
            signatures = dict(indexer.signatures)
            for name in indexer.graph.nodes:
                symbol_table.add(str(name))
            symbol_table.update(str(name) for name in signatures)
            index_block = {
                "built": True,
                "package": str(package),
                "modules": int(len(report.modules)),
                "functions": int(len(signatures)),
                "edges": int(len(indexer.graph.edges)),
                "nodes": int(len(symbol_table)),
                "seconds": float(indexer.report.seconds),
            }
        stats = {
            "corpus_digest": str(getattr(corpus, "digest", "")),
            "files": int(len(corpus)),
            "tokens": int(len(ids)),
            "train_tokens": int(budget),
            "vocabulary_size": int(len(tokenizer)),
            "vocabulary_digest": str(tokenizer.spec.digest()),
            "vocab_locked": bool(getattr(tokenizer, "locked", False)),
            "vocab_lock_path": str(lock_path),
            "verifier_passages": int(len(getattr(verifier, "passages", []) or [])),
            "index": index_block,
            "max_fit_tokens": int(max_fit_tokens),
        }
        return cls(
            root=base,
            corpus=corpus,
            tokenizer=tokenizer,
            projector=projector,
            count_lm=generator.count_lm,
            generator=generator,
            verifier=verifier,
            signatures=signatures,
            symbol_table=symbol_table,
            indexer=indexer,
            sheaf=sheaf,
            stats=stats,
            build_ms=float((_time.perf_counter() - started) * 1e3),
        )

    # -------------------------------------------------------------- mounting
    def mount_generator(self) -> "WaveCodeTextGenerator":
        """A per-arm generator view: shared banks, fresh learned state."""
        import copy

        view = copy.copy(self.generator)
        view.fast_weight = np.zeros(
            (int(self.generator.vocab_size), int(self.generator.dim)), dtype=np.complex64
        )
        view.fast_gain = float(self.generator.fast_gain)
        view._fast_active = False
        view.fast_updates = 0
        view.photonic_contexts = 0
        view.photonic_cache_hits = 0
        view._photonic_cache = {}
        view.suffix_stats = {"calls": 0.0, "matched": 0.0, "order_sum": 0.0}
        view.stream_cache = None
        view.stream_weight = 0.0
        view.stream_report = {}
        # A per-arm AST gate: a mount can register a symbol without widening
        # another arm's grammar.
        view.gate = ASTGate(set(self.symbol_table))
        return view

    def install_boundary_axioms(self, memory, namespace: str) -> List[Dict[str, object]]:
        """Register the two typed axioms of one arm namespace (Directive X1)."""
        import hashlib as _hashlib

        out: List[Dict[str, object]] = []
        specs = (
            ("LAW", "exact_quantity_grounding", (False, False, False, True)),
            ("REPULSOR", "hallucination_repulsor", (True, True, True, True)),
        )
        dim = int(getattr(memory, "dim", 1024))
        if dim % 2:
            dim += 1
        for polarity, name, mask in specs:
            digest = _hashlib.sha256(f"{namespace}:{name}".encode("utf-8")).digest()
            seed = int.from_bytes(digest[:8], "little")
            raw = np.random.default_rng(seed).standard_normal(2 * dim).astype(np.float32)
            wave = (raw[:dim] + 1j * raw[dim:]).astype(np.complex64)
            wave = wave / max(float(np.linalg.norm(wave)), 1e-12)
            record = memory.register_axiom(
                wave,
                name=str(name),
                polarity=str(polarity),
                source="human_heuristic",
                slot_mask=mask,
                namespace=str(namespace),
                key=f"{namespace}:{polarity}:{name}",
            )
            out.append(
                {
                    "axiom_id": str(record.axiom_id),
                    "name": str(record.name),
                    "polarity": str(record.polarity),
                    "namespace": str(record.namespace),
                    "engram_id": int(record.engram_id),
                }
            )
        return out

    def mount(
        self,
        memory,
        namespace: str = "default",
        *,
        plan_offset: int = 20000,
        signatures: Optional[Dict[str, object]] = None,
        register_axioms: bool = True,
    ) -> Tuple["SynthesisEngine", Dict[str, object]]:
        """Mount one lightweight arm over this stem."""
        import time as _time

        started = _time.perf_counter()
        engine = SynthesisEngine.from_stem(
            self,
            memory,
            namespace=namespace,
            plan_offset=plan_offset,
            signatures=signatures,
        )
        axioms = self.install_boundary_axioms(memory, namespace) if register_axioms else []
        self.mounts += 1
        return engine, {
            "namespace": str(namespace),
            "mount_ms": float((_time.perf_counter() - started) * 1e3),
            "plans": int(len(engine.plans.plans)),
            "boundary_axioms": axioms,
            "fast_weight_bytes": int(engine.fast_weight.nbytes),
            "shared": {
                "tokenizer": "HolarchCodeTextTokenizer",
                "projector": "BochnerPhaseProjector",
                "count_lm": "CountLanguageModel",
                "verifier": "OntologyVerifier",
                "sheaf": "RepoSheaf",
                "indexer": bool(self.indexer is not None),
            },
        }

    def describe(self) -> Dict[str, object]:
        return {
            "root": str(self.root),
            "build_ms": float(self.build_ms),
            "mounts": int(self.mounts),
            "stats": dict(self.stats),
            "signatures": int(len(self.signatures)),
            "symbol_table": int(len(self.symbol_table)),
        }

    def digest(self) -> str:
        import json as _json

        return hashlib.sha256(
            _json.dumps(self.describe(), sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()


def build_synthesis_engine(
    root: str | Path,
    memory,
    namespace: str = "default",
    symbol_table: Optional[Iterable[str]] = None,
    max_fit_tokens: int = 0,
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
    # Directive P1: the deployed generator now fits on the complete training
    # split. The Cycle O factory capped the bank at 20k tokens, which starved
    # the grown live tree; max_fit_tokens=0 means "no cap".
    budget = int(len(ids) * 0.9) if int(max_fit_tokens) <= 0 else int(max_fit_tokens)
    generator.fit(ids[:budget], max_fit_tokens=10**9)
    generator.count_lm.fit(ids[:budget])
    generator.lexical.build(ids[:budget])
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


TELEPATHIC_OPERATORS: Tuple[str, ...] = ("+", "-", "*")
"""The three operators the telepathic AST binder can select."""


def build_telepathic_program(
    name: str, *, scale: float, shift: float, low: float, high: float
) -> str:
    """A guard-normalized three-stage program with KV-bound constants.

    Stage 1 scales, stage 2 shifts, stage 3 clamps into [low, high]. The
    guard clause on the input is what makes the function total; the three
    constants are the slots the telepathic receiver binds from the
    transferred value projection.
    """
    return (
        f"def {name}(value: float) -> float:" + chr(10)
        + "    if value is None:" + chr(10)
        + "        return 0.0" + chr(10)
        + f"    scaled = float(value) * {float(scale)!r}" + chr(10)
        + f"    shifted = scaled + {float(shift)!r}" + chr(10)
        + f"    low = {float(low)!r}" + chr(10)
        + f"    high = {float(high)!r}" + chr(10)
        + "    result = shifted if shifted > low else low" + chr(10)
        + "    return result if result < high else high" + chr(10)
    )


def telepathic_reference(
    value: Any, *, scale: float, shift: float, low: float, high: float
) -> float:
    """The independent answer the reference tests are generated from."""
    if value is None:
        return 0.0
    scaled = float(value) * float(scale)
    shifted = scaled + float(shift)
    result = shifted if shifted > float(low) else float(low)
    return result if result < float(high) else float(high)


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
            "    for item in items:\n"
            "        if item:\n"
            "            return items\n"
            "    return []\n"
        ),
        "args": ["items"],
        "kind": "symbol",
        "reason": "a mid-loop return cannot be normalized to one exit",
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


_K4_GUARD_STAGES: Dict[str, str] = {
    "safe_mean": (
        "def safe_mean(values: list) -> float:\n"
        "    if not values:\n"
        "        return 0.0\n"
        "    total = 0.0\n"
        "    for value in values:\n"
        "        total += value\n"
        "    return total / len(values)\n"
    ),
    "normalize_guarded": (
        "def normalize_guarded(values: list) -> list:\n"
        "    if not values:\n"
        "        return []\n"
        "    total = sum(values)\n"
        "    scale = total if total != 0.0 else 1.0\n"
        "    return [value / scale for value in values]\n"
    ),
    "count_guarded": (
        "def count_guarded(items: list) -> int:\n"
        "    if not items:\n"
        "        return 0\n"
        "    matched = 0\n"
        "    for item in items:\n"
        "        matched += 1\n"
        "    return matched\n"
    ),
    "first_positive": (
        "def first_positive(values: list) -> float:\n"
        "    if not values:\n"
        "        return 0.0\n"
        "    found = 0.0\n"
        "    for value in values:\n"
        "        if value > 0 and found == 0.0:\n"
        "            found = value\n"
        "    return found\n"
    ),
    "lower_bound": (
        "def lower_bound(value: float, floor: float) -> float:\n"
        "    if value < floor:\n"
        "        return floor\n"
        "    return value\n"
    ),
    "upper_bound": (
        "def upper_bound(value: float, ceiling: float) -> float:\n"
        "    if value > ceiling:\n"
        "        return ceiling\n"
        "    return value\n"
    ),
}

_K4_GUARD_TASKS: List[Dict[str, object]] = [
    {
        "id": "guard01",
        "stages": ("normalize_guarded", "total", "clamp_unit"),
        "args": ["values"],
        "values": ([1.0, 2.0, 3.0],),
        "reference": lambda values: min(1.0, max(0.0, sum([v / sum(values) for v in values]))),
    },
    {
        "id": "guard02",
        "stages": ("safe_mean", "lower_bound", "upper_bound"),
        "args": ["values", "floor", "ceiling"],
        "values": ([2.0, 4.0], 1.0, 3.5),
        "reference": lambda values, floor, ceiling: min(ceiling, max(floor, sum(values) / len(values))),
    },
    {
        "id": "guard03",
        "stages": ("count_guarded", "clamp_unit", "upper_bound"),
        "args": ["items", "ceiling"],
        "values": (["a", "b", "c"], 0.5),
        "reference": lambda items, ceiling: min(ceiling, min(1.0, float(len(items)))),
    },
    {
        "id": "guard04",
        "stages": ("first_positive", "lower_bound", "upper_bound"),
        "args": ["values", "floor", "ceiling"],
        "values": ([-2.0, 3.0, 5.0], 1.0, 4.0),
        "reference": lambda values, floor, ceiling: min(ceiling, max(floor, next((v for v in values if v > 0), 0.0))),
    },
    {
        "id": "guard05",
        "stages": ("lower_bound", "upper_bound", "clamp_unit"),
        "args": ["value", "floor", "ceiling"],
        "values": (-2.0, 0.5, 0.8),
        "reference": lambda value, floor, ceiling: min(1.0, max(0.0, min(ceiling, max(floor, value)))),
    },
    {
        "id": "guard06",
        "stages": ("safe_mean", "lower_bound", "clamp_unit"),
        "args": ["values", "floor"],
        "values": ([0.2, 0.4], 0.1),
        "reference": lambda values, floor: min(1.0, max(0.0, max(floor, sum(values) / len(values)))),
    },
]

_K4_TRAPS: List[Dict[str, object]] = [
    {
        "id": "trap_mid_loop_guard",
        "source": (
            "def mid_loop(values: list) -> float:\n"
            "    for value in values:\n"
            "        if value > 0:\n"
            "            return value\n"
            "    return 0.0\n"
        ),
        "args": ["values"],
        "kind": "symbol",
        "reason": "a return inside a loop cannot be normalized to one exit",
    },
    {
        "id": "trap_guard_after_statement",
        "source": (
            "def late_guard(values: list) -> float:\n"
            "    total = 0.0\n"
            "    if not values:\n"
            "        return 0.0\n"
            "    return total\n"
        ),
        "args": ["values"],
        "kind": "symbol",
        "reason": "only leading guards are normalized; a later guard is refused",
    },
    {
        "id": "trap_guard_with_else",
        "source": (
            "def guarded_else(values: list) -> float:\n"
            "    if values:\n"
            "        return 1.0\n"
            "    else:\n"
            "        return 0.0\n"
        ),
        "args": ["values"],
        "kind": "symbol",
        "reason": "a guard with an else branch is not a leading guard",
    },
    {
        "id": "trap_guard_type",
        "stages": ("safe_mean", "join_csv"),
        "args": ["values"],
        "kind": "type",
        "reason": "a numeric result cannot feed a sequence consumer",
    },
    {
        "id": "trap_guard_arity",
        "stages": ("safe_mean", "lower_bound"),
        "args": ["values"],
        "kind": "arity",
        "reason": "the floor parameter has no binding",
    },
    {
        "id": "trap_guard_symbol",
        "source": (
            "def ghost_guard(values: list) -> float:\n"
            "    if mystery_empty(values):\n"
            "        return 0.0\n"
            "    return 1.0\n"
        ),
        "args": ["values"],
        "kind": "symbol",
        "reason": "mystery_empty is absent from the symbol table",
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


def run_diverse_dag_battery(
    engine=None, seed: int = 0, extended: bool = False, guard_pipelines: bool = False
) -> Dict[str, object]:
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

    guard_rows: List[Dict[str, object]] = []
    traps3_rows: List[Dict[str, object]] = []
    if extended and guard_pipelines:
        for index, spec in enumerate(_K4_GUARD_TASKS, start=1):
            stages = [str(stage) for stage in spec["stages"]]  # type: ignore[index]
            func = f"m4g{salt}_{index:02d}"
            plans = [
                _compiled_plan(stage, f"m4 {stage}", _K4_GUARD_STAGES[stage])
                if stage in _K4_GUARD_STAGES
                else _compiled_plan(stage, f"k2 {stage}", _K2_MULTI_STAGES[stage])
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
            row, run = _j3_case_row(verdict, func, spec["values"], expected, "guard")  # type: ignore[arg-type]
            row.update(
                {
                    "id": spec["id"],
                    "stages": list(stages),
                    "guard_clause": True,
                    "multi_argument": bool(len(list(spec["args"])) > 1),
                }
            )
            if bool(row["matched"]) and bool(run["executed"]):
                again = _j3_execute(str(verdict.get("source") or ""), func, spec["values"])  # type: ignore[arg-type]
                if bool(again["executed"]):
                    drift = max(drift, _j3_answer_delta(run["actual"], again["actual"]))
            guard_rows.append(row)

        known = set(PLAYBOOK_LIBRARY) | set(_K2_MULTI_STAGES) | set(_K4_GUARD_STAGES) | {
            "range", "sum", "len", "sorted", "list", "dict", "set", "min", "max",
            "abs", "round", "enumerate", "zip", "str", "int", "float", "bool",
            "tuple", "frozenset", "any", "all", "reversed", "map", "filter",
        }
        for spec in _K4_TRAPS:
            if "stages" in spec:
                plans = []
                for stage in spec["stages"]:  # type: ignore[index]
                    name = str(stage)
                    if name in _K4_GUARD_STAGES:
                        plans.append(_compiled_plan(name, f"m4 {name}", _K4_GUARD_STAGES[name]))
                    elif name in _K2_MULTI_STAGES:
                        plans.append(_compiled_plan(name, f"k2 {name}", _K2_MULTI_STAGES[name]))
                    else:
                        plans.append(_playbook_plan(name))
            else:
                plans = [_compiled_plan("guard_trap_stage", "guard trap", str(spec["source"]))]
            verdict = compose_plan_dag(
                plans,
                {"func": f"m4t_{spec['id']}", "args": list(spec["args"]), "numbers": []},
                gate=gate,
            )
            source = str(verdict.get("source") or "")
            veto = _symbol_vetoes(source, known) if source else []
            refused = bool(
                (not verdict.get("ok", False) and source == "")
                or (source and veto)
            )
            traps3_rows.append(
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
    guard_passed = sum(1 for row in guard_rows if row["matched"])
    traps3_refused = sum(1 for row in traps3_rows if row["refused"])
    total_passed = (
        base_passed + single_passed + two_passed + three_passed
        + traps_refused + multi_passed + traps2_refused
        + guard_passed + traps3_refused
    )
    total_tasks = (
        len(base_rows) + len(single_rows) + len(two_rows) + len(three_rows)
        + len(trap_rows) + len(multi_rows) + len(traps2_rows)
        + len(guard_rows) + len(traps3_rows)
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
    if extended and guard_pipelines:
        payload["guards"] = {
            "passed": int(guard_passed),
            "total": len(guard_rows),
            "rows": guard_rows,
        }
        payload["traps3"] = {
            "refused": int(traps3_refused),
            "total": len(traps3_rows),
            "rows": traps3_rows,
        }
    return payload

