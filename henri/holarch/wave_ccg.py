
"""Non-commutative Combinatory Categorial Grammar rotor parser (Sovereign T4a).

Each atomic syntactic type is a 4x4 block rotor on the four semantic slots
(Entity, Morphism, Object, Law).  A slash category X/Y carries the rotor
U_X U_Y^H, so forward application is an exact algebraic contraction:

    R(X/Y) R(Y) = U_X U_Y^H U_Y = U_X

The chart parser is a real CKY reduction over those categories.  A sentence
becomes a canonical four-slot Engram DAG.  The parser never splits on a
literal sentence template.  Two clauses that contradict each other are
refused fail-closed.
"""

from __future__ import annotations

import hashlib
import itertools
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

D_SLOT = 64
BLOCK = 16
SLOTS = ("entity", "morphism", "object", "law")
EPS = 1e-12


def _phases(text: str, count: int, salt: str = "") -> np.ndarray:
    digest = hashlib.sha256((salt + "|" + str(text)).encode("utf-8")).digest()
    raw = np.frombuffer(digest * 4, dtype=np.uint8)[: count * 2].astype(np.float64)
    angles = (raw / 255.0) * 2.0 * math.pi
    return np.exp(1j * angles[:count])


def slot_wave(slot: str, text: str) -> np.ndarray:
    block = SLOTS.index(slot)
    wave = np.zeros(D_SLOT, dtype=np.complex128)
    phase = _phases(str(text).lower().strip(), BLOCK, salt=slot)
    wave[block * BLOCK : (block + 1) * BLOCK] = phase
    return wave / math.sqrt(BLOCK)


def frame_wave(frame: "Frame") -> np.ndarray:
    wave = np.zeros(D_SLOT, dtype=np.complex128)
    for name in sorted(frame.entity):
        wave += slot_wave("entity", name)
    for name in sorted(frame.object):
        wave += slot_wave("object", name)
    if frame.morphism:
        wave += slot_wave("morphism", frame.morphism[0])
    for law in sorted(frame.law):
        wave += slot_wave("law", law)
    norm = float(np.linalg.norm(wave))
    return wave / max(norm, EPS)


def coherence(left: "Frame", right: "Frame") -> float:
    return float(min(1.0, max(0.0, np.real(np.vdot(frame_wave(left), frame_wave(right))))))


def _unitary(name: str) -> np.ndarray:
    angles = [float(np.angle(_phases(name, 1, salt=f"u{index}")[0])) for index in range(3)]
    a, b, c = angles
    g1 = np.array([[math.cos(a), -math.sin(a), 0, 0],
                   [math.sin(a), math.cos(a), 0, 0],
                   [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.complex128)
    g2 = np.array([[1, 0, 0, 0], [0, math.cos(b), 0, -math.sin(b)],
                   [0, 0, 1, 0], [0, math.sin(b), 0, math.cos(b)]], dtype=np.complex128)
    g3 = np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                   [0, 0, math.cos(c), -math.sin(c)],
                   [0, 0, math.sin(c), math.cos(c)]], dtype=np.complex128)
    return g1 @ g2 @ g3


SWAP_0_2 = np.eye(4, dtype=np.complex128)[[2, 1, 0, 3]]


@dataclass(frozen=True)
class Cat:
    atom: Optional[str] = None
    direction: Optional[str] = None
    result: Optional["Cat"] = None
    arg: Optional["Cat"] = None
    passive: bool = False

    @staticmethod
    def of(name: str, passive: bool = False) -> "Cat":
        return Cat(atom=str(name), passive=bool(passive))

    @staticmethod
    def slash(direction: str, result: "Cat", arg: "Cat") -> "Cat":
        return Cat(direction=str(direction), result=result, arg=arg)

    def text(self) -> str:
        if self.atom is not None:
            return self.atom + ("_pass" if self.passive else "")
        left = self.result.text() if self.result else "?"
        right = self.arg.text() if self.arg else "?"
        return f"({left}{self.direction}{right})"

    def rotor(self) -> np.ndarray:
        if self.atom is not None:
            return _unitary(self.atom)
        return self.result.rotor() @ self.arg.rotor().conj().T

    def is_slash(self) -> bool:
        return self.direction is not None


S = Cat.of("S")
NP = Cat.of("NP")
N = Cat.of("N")
PP = Cat.of("PP")
VP = Cat.slash("\\", S, NP)
TV = Cat.slash("/", VP, NP)
DET = Cat.slash("/", NP, N)
MOD = Cat.slash("/", S, S)


@dataclass
class Frame:
    entity: List[str] = field(default_factory=list)
    morphism: List[str] = field(default_factory=list)
    object: List[str] = field(default_factory=list)
    law: List[str] = field(default_factory=list)
    head: Optional[str] = None
    passive: bool = False
    negated: bool = False
    props: List[str] = field(default_factory=list)

    def copy(self) -> "Frame":
        return Frame(list(self.entity), list(self.morphism), list(self.object),
                     list(self.law), self.head, self.passive, self.negated,
                     list(self.props))

    def merge(self, other: "Frame") -> "Frame":
        out = self.copy()
        out.entity.extend(other.entity)
        out.morphism.extend(other.morphism)
        out.object.extend(other.object)
        out.law.extend(other.law)
        out.head = self.head or other.head
        out.passive = bool(self.passive or other.passive)
        out.negated = bool(self.negated or other.negated)
        out.props = list(self.props) + list(other.props)
        return out

    def canonical(self) -> Dict[str, List[str]]:
        return {
            "entity": sorted(set(str(v) for v in self.entity)),
            "morphism": sorted(set(str(v) for v in self.morphism)),
            "object": sorted(set(str(v) for v in self.object)),
            "law": sorted(set(str(v) for v in self.law)),
        }


@dataclass
class Item:
    cat: Cat
    sem: Frame
    span: Tuple[int, int]
    derivation: Tuple[str, ...] = ()


def _span(a: Item, b: Item) -> Tuple[int, int]:
    return (min(a.span[0], b.span[0]), max(a.span[1], b.span[1]))


def _forward(function: Item, argument: Item) -> Optional[Item]:
    cat = function.cat
    if not cat.is_slash() or cat.arg is None or cat.result is None:
        return None
    if cat.direction != "/" or argument.cat != cat.arg:
        return None
    kind = cat.text()
    sem = Frame()
    if kind == "(NP/N)":
        sem = Frame(head=argument.sem.head)
    elif kind == "((S\\NP)/NP)":
        # Transitive verb: the object NP fills the object slot.
        sem = Frame(morphism=list(function.sem.morphism),
                    object=[argument.sem.head] if argument.sem.head else list(argument.sem.entity),
                    law=[v for v in function.sem.law if v != "passive"],
                    passive=function.sem.passive)
        sem.negated = bool(function.sem.negated)
    elif kind == "((S\\NP)/(S\\NP))":
        sem = function.sem.merge(argument.sem)
        sem.negated = bool(function.sem.negated) != bool(argument.sem.negated)
        sem.law = [value for value in sem.law if value != "negated"]
        sem.props = []
    elif kind == "(S/S)":
        sem = function.sem.merge(argument.sem)
        sem.law.append("conditional")
    elif kind == "((S\\NP)/PP)":
        sem = function.sem.merge(argument.sem)
    else:
        sem = function.sem.merge(argument.sem)
    return Item(cat.result, sem, _span(function, argument),
                function.derivation + argument.derivation + (">",))


def _backward(left: Item, right: Item) -> Optional[Item]:
    cat = right.cat
    if not cat.is_slash() or cat.arg is None or cat.result is None:
        return None
    if cat.direction != "\\" or left.cat != cat.arg:
        return None
    kind = cat.text()
    sem = Frame()
    if kind == "(S\\NP)":
        subject = left.sem.head or (left.sem.entity[0] if left.sem.entity else "")
        sem = Frame(morphism=list(right.sem.morphism), law=list(right.sem.law))
        if right.sem.passive:
            sem.entity = list(right.sem.object)
            sem.object = [subject] if subject else []
        else:
            sem.entity = [subject] if subject else []
            sem.object = list(right.sem.object)
        sem.passive = bool(right.sem.passive)
        sem.negated = bool(right.sem.negated)
        morphism = right.sem.morphism[0] if right.sem.morphism else ""
        if morphism:
            if sem.object:
                prop = f"{morphism}:{sem.entity[0] if sem.entity else ''}:{sem.object[0]}"
            else:
                prop = f"{morphism}:{sem.entity[0] if sem.entity else ''}"
            sem.props = [("not(" + prop + ")") if sem.negated else prop]
    elif kind == "(S/S)":
        sem = right.sem.merge(left.sem)
    else:
        sem = left.sem.merge(right.sem)
    return Item(cat.result, sem, _span(left, right),
                left.derivation + right.derivation + ("<",))


def _compose(function: Item, argument: Item) -> Optional[Item]:
    left, right = function.cat, argument.cat
    if not left.is_slash() or not right.is_slash():
        return None
    if left.direction != "/" or right.direction != "/":
        return None
    if left.arg != right.result:
        return None
    sem = function.sem.merge(argument.sem)
    sem.negated = bool(function.sem.negated) != bool(argument.sem.negated)
    sem.props = []
    return Item(Cat.slash("/", left.result, right.arg), sem,
                _span(function, argument), function.derivation + argument.derivation + (">B",))


# ----------------------------------------------------------------- lexicon
LEXICON: Dict[str, List[Tuple[Cat, Frame]]] = {}


def _add(words: Sequence[str], cat: Cat, frame: Frame) -> None:
    for word in words:
        LEXICON.setdefault(str(word).lower(), []).append((cat, frame.copy()))


NOUNS = (
    "parser", "schema", "record", "cache", "threshold", "table", "writer",
    "reader", "service", "module", "field", "list", "number", "buffer",
    "index", "queue", "worker", "report", "value", "key", "sum", "average",
    "product", "quotient", "remainder", "entity", "morphism", "object", "law",
    "matrix", "vector", "function", "graph", "node", "edge", "counter",
    "result", "input", "output", "state", "event", "request", "response",
    "invoice", "ledger", "customer", "dataset", "pipeline", "loader",
)
for _word in NOUNS:
    _add([_word], N, Frame(head=_word))
for _word in ("alice", "bob", "alpha", "beta", "gamma", "delta", "henri"):
    _add([_word], NP, Frame(head=_word, entity=[_word]))
for _word in ("0", "1", "2", "3", "4", "5", "8", "10", "12", "16", "32", "64"):
    _add([_word], NP, Frame(head=_word, entity=[_word]))
for _word in ("the", "a", "an", "every", "each", "all", "some", "any", "this", "that"):
    _add([_word], DET, Frame())

VERBS = {
    "validate": ("validates", "validated"), "read": ("reads", "read"),
    "write": ("writes", "written"), "store": ("stores", "stored"),
    "load": ("loads", "loaded"), "parse": ("parses", "parsed"),
    "compute": ("computes", "computed"), "check": ("checks", "checked"),
    "contain": ("contains", "contained"), "divide": ("divides", "divided"),
    "equal": ("equals", "equalled"), "exceed": ("exceeds", "exceeded"),
    "return": ("returns", "returned"), "accept": ("accepts", "accepted"),
    "reject": ("rejects", "rejected"), "sort": ("sorts", "sorted"),
    "copy": ("copies", "copied"), "increment": ("increments", "incremented"),
    "decrement": ("decrements", "decremented"), "verify": ("verifies", "verified"),
    "preserve": ("preserves", "preserved"), "append": ("appends", "appended"),
    "emit": ("emits", "emitted"), "flush": ("flushes", "flushed"),
}
for _lemma, (_third, _part) in VERBS.items():
    _add([_third], TV, Frame(morphism=[_lemma]))
    _add([_part], TV, Frame(morphism=[_lemma]))
    _add([_lemma], TV, Frame(morphism=[_lemma]))
    _add([_third], VP, Frame(morphism=[_lemma]))
    _add([_lemma], VP, Frame(morphism=[_lemma]))
    for _aux in ("is", "are", "was", "were", "be"):
        _add([f"{_aux} {_part} by", f"{_aux} {_part}"],
             Cat.slash("/", VP, NP), Frame(morphism=[_lemma], passive=True))

# Intransitive relation between two entities, e.g. "12 equals 12".
for _lemma, _third in (("equal", "equals"), ("exceed", "exceeds")):
    _add([_third], Cat.slash(chr(92), S, NP), Frame(morphism=[_lemma]))

_add(["never", "not"], Cat.slash("/", VP, VP), Frame(law=["negated"], negated=True))
for _word in ("empty", "full", "valid", "invalid", "present"):
    _add([f"is {_word}", f"are {_word}"], VP, Frame(morphism=[_word]))
BACK = chr(92)
for _word in ("empty", "full", "valid", "invalid"):
    _add([_word], Cat.slash(BACK, S, NP), Frame(morphism=[_word]))
_add(["always"], Cat.slash("/", VP, VP), Frame())
_add(["must", "should"], Cat.slash("/", VP, VP), Frame(law=["required"]))
_add(["can", "may"], Cat.slash("/", VP, VP), Frame())
for _word in ("when", "if", "because"):
    _add([_word], MOD, Frame(law=["conditional"]))
_add(["then"], MOD, Frame())
CONJUNCTION_WORDS = {"and", "then", "or", "but"}


# ------------------------------------------------------------------ parser
@dataclass
class ParseResult:
    status: str
    frame: Frame
    coherence: float
    derivation: Tuple[str, ...]
    residual: float
    reason: str = ""
    sagnac_delta: float = 0.0

    def as_dict(self) -> Dict[str, object]:
        return {
            "status": self.status,
            "dag": self.frame.canonical(),
            "coherence": float(self.coherence),
            "sagnac_delta": float(self.sagnac_delta),
            "derivation": list(self.derivation),
            "type_residual": float(self.residual),
            "reason": self.reason,
        }


class WaveCCGParser:
    def __init__(self, lexicon: Optional[Dict[str, List[Tuple[Cat, Frame]]]] = None) -> None:
        self.lexicon = dict(lexicon or LEXICON)

    @staticmethod
    def tokens(text: str) -> List[str]:
        return [token.strip().lower() for token in str(text).replace(",", " ").replace(";", " ").split()]

    def _lexical(self, tokens: Sequence[str]) -> Dict[Tuple[int, int], List[Item]]:
        chart: Dict[Tuple[int, int], List[Item]] = {}
        count = len(tokens)
        for start in range(count):
            for length in (4, 3, 2, 1):
                end = start + length
                if end > count:
                    continue
                phrase = " ".join(tokens[start:end])
                for cat, frame in self.lexicon.get(phrase, []):
                    self._push(chart.setdefault((start, end), []),
                               Item(cat, frame.copy(), (start, end), (f"lex:{phrase}",)))
        return chart

    def parse(self, text: str) -> ParseResult:
        tokens = self.tokens(text)
        if not tokens:
            return ParseResult("ABSTAIN: ccg type residual", Frame(), 0.0, (), 1.0,
                               "empty specification")
        chart = self._lexical(tokens)
        count = len(tokens)
        for width in range(1, count + 1):
            for start in range(0, count - width + 1):
                end = start + width
                cell = chart.setdefault((start, end), [])
                for split in range(start + 1, end):
                    left = chart.get((start, split), [])
                    right = chart.get((split, end), [])
                    for a in left:
                        for b in right:
                            for item in (_forward(a, b), _backward(a, b), _compose(a, b)):
                                if item is not None:
                                    self._push(cell, item)
                for split in range(start + 1, end - 1):
                    if tokens[split] not in CONJUNCTION_WORDS:
                        continue
                    left = chart.get((start, split), [])
                    right = chart.get((split + 1, end), [])
                    for a in left:
                        for b in right:
                            if a.cat == b.cat:
                                self._push(cell, Item(a.cat, a.sem.merge(b.sem), (start, end),
                                                      a.derivation + b.derivation + ("conj",)))
        roots = [item for item in chart.get((0, count), []) if item.cat == S]
        if not roots:
            covered = 0
            for (s, e), items in chart.items():
                if s == 0 and items:
                    covered = max(covered, e)
            residual = 1.0 - covered / float(count)
            return ParseResult("ABSTAIN: ccg type residual", Frame(), 0.0, (),
                               float(max(residual, EPS)), "no S root spans the clause")
        best = max(roots, key=lambda item: (len(item.derivation), len(item.sem.law)))
        veto = self._contradiction(best.sem)
        delta = 0.0
        if best.sem.negated:
            delta += 0.35
        if veto:
            delta += 0.45
        return ParseResult(
            "ABSTAIN: ccg type residual" if veto else "PARSED",
            best.sem,
            0.0,
            best.derivation,
            float(0.0 if not veto else 1.0),
            veto,
            float(min(1.0, delta)),
        )

    @staticmethod
    def _contradiction(frame: Frame) -> str:
        laws = sorted(frame.law)
        if "negated" in laws and "required" in laws:
            return "negated and required in the same clause"
        if frame.entity and frame.object and frame.entity == frame.object:
            return "a clause with identical subject and object"
        props = sorted(set(frame.props))
        pool = set(props)
        for prop in props:
            if prop.startswith("not(") and prop[4:-1] in pool:
                return f"asserted and denied: {prop[4:-1]}"
        equal: Dict[str, set] = {}
        for prop in props:
            parts = prop.split(":")
            if len(parts) == 3 and parts[0] == "equal":
                equal.setdefault(parts[1], set()).add(parts[2])
        for head, values in equal.items():
            if len(values) > 1:
                return f"two exact values for {head}"
        for prop in props:
            parts = prop.split(":")
            if len(parts) == 3 and parts[0] in {"exceed", "less"}:
                if f"{parts[0]}:{parts[2]}:{parts[1]}" in pool:
                    return f"order cycle on {parts[1]} and {parts[2]}"
        return ""

    @staticmethod
    def _push(cell: List[Item], item: Item) -> None:
        for existing in cell:
            if existing.cat == item.cat and existing.sem.canonical() == item.sem.canonical():
                return
        cell.append(item)


def active_passive_pair(parser: WaveCCGParser, subject: str, verb_third: str,
                        verb_part: str, obj: str) -> Tuple[ParseResult, ParseResult, float]:
    active = parser.parse(f"the {subject} {verb_third} the {obj}")
    passive = parser.parse(f"the {obj} is {verb_part} by the {subject}")
    return active, passive, coherence(active.frame, passive.frame)


# ------------------------------------------------------------------ corpus
@dataclass
class GoldSpec:
    text: str
    dag: Dict[str, List[str]]
    kind: str


def _dag(entity=(), morphism=(), obj=(), law=()) -> Dict[str, List[str]]:
    return {
        "entity": sorted(set(entity)),
        "morphism": sorted(set(morphism)),
        "object": sorted(set(obj)),
        "law": sorted(set(law)),
    }


def build_corpus() -> List[GoldSpec]:
    """32 compositional prose specifications with independent gold DAGs.

    The gold DAG is built from the schema, not from the parser output.  The
    renderer below only chooses the surface form.
    """
    specs: List[GoldSpec] = []
    pairs = [
        ("parser", "validates", "validated", "validate", "schema"),
        ("writer", "stores", "stored", "store", "record"),
        ("reader", "loads", "loaded", "load", "table"),
        ("service", "verifies", "verified", "verify", "request"),
        ("worker", "parses", "parsed", "parse", "response"),
        ("module", "checks", "checked", "check", "field"),
        ("loader", "copies", "copied", "copy", "buffer"),
        ("index", "sorts", "sorted", "sort", "list"),
        ("queue", "emits", "emitted", "emit", "event"),
        ("counter", "increments", "incremented", "increment", "value"),
    ]
    for subject, third, part, lemma, obj in pairs:
        specs.append(GoldSpec(f"the {subject} {third} the {obj}",
                              _dag([subject], [lemma], [obj]), "active"))
        specs.append(GoldSpec(f"the {obj} is {part} by the {subject}",
                              _dag([subject], [lemma], [obj]), "passive"))
    conditionals = [
        ("cache", "reads", "read", "table", "writer", "stores", "stored", "record"),
        ("parser", "validates", "validated", "schema", "service", "verifies", "verified", "request"),
        ("worker", "parses", "parsed", "response", "index", "sorts", "sorted", "list"),
        ("reader", "loads", "loaded", "buffer", "module", "checks", "checked", "field"),
        ("loader", "copies", "copied", "buffer", "queue", "emits", "emitted", "event"),
    ]
    LEMMA = {third: base for base, (third, _part) in VERBS.items()}
    LEMMA.update({third: lemma for _subject, third, _part, lemma, _obj in pairs})
    for s1, v1, p1, o1, s2, v2, p2, o2 in conditionals:
        specs.append(GoldSpec(
            f"if the {s1} {v1} the {o1} then the {s2} {v2} the {o2}",
            _dag([s1, s2], [LEMMA[v1], LEMMA[v2]], [o1, o2], ["conditional"]), "conditional"))
    coordinations = [
        ("parser", "validates", "validate", "schema", "record"),
        ("writer", "stores", "store", "record", "buffer"),
        ("index", "sorts", "sort", "list", "table"),
        ("service", "checks", "check", "request", "response"),
        ("worker", "parses", "parse", "input", "output"),
        ("reader", "loads", "load", "table", "index"),
        ("loader", "copies", "copy", "buffer", "record"),
    ]
    for subject, third, lemma, o1, o2 in coordinations:
        specs.append(GoldSpec(f"the {subject} {third} the {o1} and {third} the {o2}",
                              _dag([subject], [lemma], [o1, o2]), "coordination"))
    return specs


def build_coherence_pairs() -> List[Tuple[str, str, str]]:
    """Eight active/passive pairs used by the wave coherence gate."""
    return [
        ("the parser validates the schema", "the schema is validated by the parser", "validate"),
        ("the writer stores the record", "the record is stored by the writer", "store"),
        ("the reader loads the table", "the table is loaded by the reader", "load"),
        ("the service verifies the request", "the request is verified by the service", "verify"),
        ("the worker parses the response", "the response is parsed by the worker", "parse"),
        ("the module checks the field", "the field is checked by the module", "check"),
        ("the loader copies the buffer", "the buffer is copied by the loader", "copy"),
        ("the index sorts the list", "the list is sorted by the index", "sort"),
    ]


def build_reversals() -> List[Tuple[str, str]]:
    """Three non-commutative subject/object reversals."""
    return [
        ("alice divides bob", "bob divides alice"),
        ("the parser validates the schema", "the schema validates the parser"),
        ("the index sorts the list", "the list sorts the index"),
    ]


def build_contradictions() -> List[GoldSpec]:
    """Six logically contradictory specifications; all must be vetoed."""
    return [
        GoldSpec("the parser validates the schema and the parser never validates the schema",
                 _dag(), "asserted-denied"),
        GoldSpec("the threshold equals 12 and the threshold equals 16",
                 _dag(), "two-values"),
        GoldSpec("the cache never reads the table and the cache reads the table",
                 _dag(), "asserted-denied"),
        GoldSpec("the writer must store the record and the writer never stores the record",
                 _dag(), "modal-denied"),
        GoldSpec("the node exceeds the edge and the edge exceeds the node",
                 _dag(), "order-cycle"),
        GoldSpec("the cache reads the cache", _dag(), "self-reference"),
    ]
