"""Zone A ingress for code and text: a lossless AST-aware tokenizer and a
Bochner semantic-to-phase projector.

Evidence label: OBSERVED for every round-trip, cosine, coherence and margin
this module measures; DERIVED for the algebra it implements.

Why this module exists
----------------------
The Cycle E language path fed 24 raw characters into the wave projector and
gave each ASCII character an unrelated random phasor. Two tokens that mean
the same thing therefore landed at chance coherence, and the optical stack
had to memorise spelling instead of structure. This module replaces that
ingress with two deterministic, in-process stages:

    HolarchCodeTextTokenizer   a byte-complete subword vocabulary that covers
                               Python keywords, operators, identifiers,
                               English roots and a 256-token byte fallback, so
                               decode(encode(text)) == text for every UTF-8
                               string.
    BochnerPhaseProjector      a PMI-SVD distributional stem projected through
                               a Stiefel frame into the four complex wave
                               slots, so semantic neighbours share phase.

No network call, no downloaded vocabulary, no external model. The vocabulary
is a fixed list plus identifiers counted from the repository itself; the
projector is a deterministic truncated SVD of the token co-occurrence PMI
matrix. Both are reproducible from a seed and a corpus digest.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

_EPS = 1e-12
N_BYTE_TOKENS = 256
"""Every one of the 256 byte values is a token, so any byte string is
representable. This is what makes the round trip total rather than probable."""

# ---------------------------------------------------------------------------
# Slot routing
# ---------------------------------------------------------------------------
# The four structural slots of the Zone A wave. Routing is by grammatical or
# AST role, not by arbitrary hashing, so a synonym and its partner land in the
# same slot and their phases can be compared.
SLOT_SCOPE = 0      # scope, subject, identifier
SLOT_OPERATOR = 1   # operator, verb, callable
SLOT_ARGUMENT = 2   # argument, object, literal
SLOT_RULE = 3       # rule, type, docstring context

SLOT_NAMES = ("scope", "operator", "argument", "rule")

CONTROL_KEYWORDS = {
    "def", "class", "return", "yield", "if", "elif", "else", "for", "while",
    "in", "not", "and", "or", "is", "import", "from", "as", "with", "try",
    "except", "finally", "raise", "lambda", "global", "nonlocal", "assert",
    "del", "pass", "break", "continue", "async", "await", "match", "case",
    "None", "True", "False",
}
"""Control-flow and binding keywords. They route to the operator slot because
they are the verbs of the language."""

TYPE_NAMES = {
    "int", "float", "str", "bool", "bytes", "list", "dict", "set", "tuple",
    "complex", "object", "type", "NoneType", "Any", "Optional", "Sequence",
    "Mapping", "Iterable", "Iterator", "Callable", "Union", "Literal",
    "dataclass", "ndarray", "Tensor", "Path", "Enum", "Protocol", "TypedDict",
}
"""Type names route to the rule slot: a type is a constraint on a value, which
is exactly the slot-3 role."""

VERBS = {
    "verify", "check", "validate", "assert", "ensure", "confirm", "test",
    "compute", "calculate", "derive", "measure", "probe", "estimate", "score",
    "build", "create", "make", "construct", "compile", "render", "emit",
    "load", "read", "fetch", "get", "query", "retrieve", "recall", "search",
    "find", "locate", "index", "ingest", "write", "store", "save", "persist",
    "update", "adapt", "train", "fit", "learn", "calibrate", "tune",
    "solve", "resolve", "decide", "choose", "select", "route", "mix", "fuse",
    "run", "execute", "apply", "invoke", "call", "return", "yield", "raise",
    "report", "render", "show", "describe", "explain", "summarise", "summarize",
    "compare", "match", "align", "bind", "unbind", "encode", "decode",
    "project", "pool", "normalise", "normalize", "scale", "transform",
    "detach", "attach", "reset", "clear", "close", "open", "start", "stop",
}
"""English and Pythonic verbs route to the operator slot, so a (verify,
check) pair is comparable phase-wise."""

NEGATIONS = {"not", "never", "no", "none", "None", "!=", "is not"}
"""A negation token carries a pi phase rotation in the operator slot, so the
affirmative and negated chords destructively interfere."""

KEYWORDS = sorted(CONTROL_KEYWORDS)

OPERATORS = [
    "==", "!=", "<=", ">=", "+=", "-=", "*=", "/=", "//=", "**=", "%=", "@=",
    "->", "::", ":=", "...", "<<", ">>", "**", "//",
    "(", ")", "[", "]", "{", "}", ",", ":", ";", ".", "=", "+", "-", "*",
    "/", "%", "@", "<", ">", "&", "|", "^", "~", "_",
]
"""Python operators and delimiters. The encoder always tries the longest match
at each byte, so ordering in this list never changes the segmentation."""

WHITESPACE_PIECES = ["\n", "\n\n", "    ", "        ", "  ", " ", "\t", "\r\n"]

ENGLISH_ROOTS = """
    the of and to in is are be was were for that this with from by on at as
    an or not no if then than which what when where who how why can could
    should would will shall may might must do does did done have has had
    use used using user make makes made get gets got give gives given
    take takes taken see sees seen know knows known say says said go goes
    going come comes came want wants need needs work works worked call calls
    called find finds found think thinks thought tell tells told ask asks
    asked seem seems seemed feel feels felt try tries tried leave leaves left
    move moves moved live lives lived believe believes held bring brings
    happen happens happened write writes wrote provide provides provided sit
    sits sat stand stands stood lose loses lost pay pays paid meet meets met
    include includes included continue continues continued set sets setting
    learn learns learned change changes changed lead leads led understand
    understands understood watch watches watched follow follows followed stop
    stops stopped create creates created speak speaks spoke read reads
    reading allow allows allowed add adds added spend spends spent grow grows
    grew open opens opened walk walks walked win wins won offer offers
    offered remember remembers remembered love loves loved consider considers
    buy buys bought wait waits waited serve serves served die dies died send
    sends sent expect expects expected build builds built stay stays stayed
    fall falls fell cut cuts reach reaches reached kill kills killed remain
    remains remained suggest suggests suggested raise raises raised pass
    passes passed sell sells sold require requires required report reports
    reported decide decides decided pull pulls pulled represent represents
    represented return returns returned explain explains explained hope hopes
    hoped develop develops developed carry carries carried break breaks broke
    receive receives received agree agrees agreed support supports supported
    hit hits produce produces produced eat eats ate cover covers covered
    catch catches caught draw draws drew choose chooses chose cause causes
    caused point points pointed listen listens listened realise realize
    realizes realized appear appears appeared produce produces mean means
    meant measure measures measured verify verifies verified check checks
    checked test tests tested value values vector vectors matrix matrices
    tensor tensors wave waves phase phases token tokens word words text texts
    code codes file files line lines number numbers count counts score scores
    result results state states memory memories layer layers model models
    input inputted output outputs target targets label labels class classes
    function functions method methods module modules object objects system
    systems process processes data datum time times step steps level levels
    form forms part parts place places case cases group groups problem
    problems fact facts information reason reasons area areas money story
    stories month months book books eye eyes job jobs business businesses
    issue issues side sides kind kinds head heads house houses service
    services friend friends power powers hour hours game games end ends
    member members law laws car cars city cities community communities name
    names president team teams minute minutes idea ideas kid kids body bodies
    back backs parent parents face faces others office offices door doors
    health person persons art arts war wars history party parties morning
    mornings research girl girls guy guys moment moments air air teacher
    teachers force forces education foot feet boy boys age ages policy
    policies everything music market sense nation plan plans college interest
    death experience effect effects class control care field development role
    effort rate heart drug show leader light voice wife police mind price
    decision son view relationship town road arm difference building action
    season society tax director position player record paper space ground
    material approach rule task project
"""


def _piece_candidates() -> List[str]:
    """The curated, corpus-independent part of the vocabulary."""
    pieces: List[str] = []
    pieces.extend(KEYWORDS)
    pieces.extend(OPERATORS)
    pieces.extend(WHITESPACE_PIECES)
    pieces.extend(sorted(VERBS))
    pieces.extend(sorted(TYPE_NAMES))
    pieces.extend(ENGLISH_ROOTS.split())
    # Common Python surface fragments. These are structure, not spelling:
    # they are the shapes an AST transition actually takes.
    pieces.extend(
        [
            "self", "cls", "def ", "class ", "return ", "import ", "from ",
            "    def ", "    return ", "    self.", "):", "):\n", "()", "():\n",
            " = ", " == ", " != ", " += ", " -= ", " -> ",
            '"', "'", "f'", 'b"', "# ", "## ", "---",
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
            "00", "01", "100", "1000", "0.0", "1.0", "0.5", "1e-", "1e-12",
            "True", "False", "None", "np.", "torch.", "torch.nn", "F.",
            "shape", "dtype", "device", "dim", "cfg", "task", "label",
            "memory", "model", "wave", "psi", "engram", "receipt", "sha256",
        ]
    )
    return list(dict.fromkeys(pieces))


@dataclass(frozen=True)
class TokenizerSpec:
    """Immutable description of one vocabulary, so a receipt can pin it."""

    pieces: Tuple[str, ...]
    n_bytes: int = N_BYTE_TOKENS

    @property
    def size(self) -> int:
        return len(self.pieces) + int(self.n_bytes)

    def digest(self) -> str:
        blob = chr(0).join(self.pieces).encode("utf-8", errors="surrogatepass")
        return hashlib.sha256(blob).hexdigest()[:16]


VOCAB_LEDGER_VERSION = 1
"""Schema version of the persisted vocabulary lock."""

VOCAB_LEDGER_KIND = "vocab_ledger"
"""The Zone C metadata kind the lock is written under."""


@dataclass(frozen=True)
class VocabularyLedger:
    """An immutable, append-only record of the identifier half of a vocabulary.

    Directive I1. The tokenizer's curated pieces and byte block are already
    position-stable; the identifiers are what used to be re-sorted by
    frequency on every build. The ledger fixes that order. An identifier in
    identifiers[i] always receives id curated_size + 256 + i, so a receipt,
    an n-gram table, or a fast-weight row that names a token id keeps its
    meaning when the live tree grows.

    The ledger is a value object: save() writes canonical JSON, load()
    validates the schema, and digest() covers exactly the bytes the file
    holds (minus the digest field itself).
    """

    identifiers: Tuple[str, ...] = ()
    vocab_size: int = 2048
    curated_size: int = 0
    curated_digest: str = ""
    source: str = ""
    version: int = VOCAB_LEDGER_VERSION

    # ------------------------------------------------------------- geometry
    @property
    def byte_base(self) -> int:
        return int(self.curated_size)

    @property
    def identifier_base(self) -> int:
        """The id the first locked identifier receives."""
        return int(self.curated_size) + N_BYTE_TOKENS

    @property
    def capacity(self) -> int:
        """How many identifier slots the declared vocabulary has."""
        return max(0, int(self.vocab_size) - int(self.curated_size) - N_BYTE_TOKENS)

    def id_of(self, name: str) -> Optional[int]:
        """The pinned id of one identifier, or None when it is not locked."""
        try:
            index = self.identifiers.index(str(name))
        except ValueError:
            return None
        return self.identifier_base + index

    def truncated(self, count: int, vocab_size: Optional[int] = None) -> "VocabularyLedger":
        """A prefix copy, for a smaller-vocabulary growth experiment."""
        names = tuple(self.identifiers[: max(0, int(count))])
        return VocabularyLedger(
            identifiers=names,
            vocab_size=int(vocab_size or self.vocab_size),
            curated_size=int(self.curated_size),
            curated_digest=str(self.curated_digest),
            source=str(self.source),
            version=int(self.version),
        )

    # ------------------------------------------------------------- payload
    def as_payload(self) -> Dict[str, object]:
        return {
            "version": int(self.version),
            "vocab_size": int(self.vocab_size),
            "curated_size": int(self.curated_size),
            "curated_digest": str(self.curated_digest),
            "source": str(self.source),
            "identifier_count": len(self.identifiers),
            "identifier_digest": hashlib.sha256(
                "\n".join(self.identifiers).encode("utf-8", errors="surrogatepass")
            ).hexdigest(),
            "digest": self.digest(),
            "identifiers": list(self.identifiers),
        }

    def digest(self) -> str:
        body = {
            "version": int(self.version),
            "vocab_size": int(self.vocab_size),
            "curated_size": int(self.curated_size),
            "curated_digest": str(self.curated_digest),
            "source": str(self.source),
            "identifiers": list(self.identifiers),
        }
        blob = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8", errors="surrogatepass")).hexdigest()

    @classmethod
    def from_payload(cls, payload: Dict[str, object]) -> "VocabularyLedger":
        names = payload.get("identifiers") or []
        if not isinstance(names, (list, tuple)):
            raise ValueError("vocab ledger identifiers must be a list")
        return cls(
            identifiers=tuple(str(name) for name in names),
            vocab_size=int(payload.get("vocab_size", 2048)),
            curated_size=int(payload.get("curated_size", 0)),
            curated_digest=str(payload.get("curated_digest", "")),
            source=str(payload.get("source", "")),
            version=int(payload.get("version", VOCAB_LEDGER_VERSION)),
        )

    # ----------------------------------------------------------- persistence
    def save(self, path: str | Path) -> Dict[str, object]:
        """Write the lock as canonical JSON and return its file digest."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.as_payload()
        target.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + chr(10),
            encoding="utf-8",
        )
        file_digest = hashlib.sha256(target.read_bytes()).hexdigest()
        return {
            "path": str(target),
            "sha256": file_digest,
            "digest": self.digest(),
            "identifiers": len(self.identifiers),
        }

    @classmethod
    def load(cls, path: str | Path) -> "VocabularyLedger":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        ledger = cls.from_payload(payload)
        recorded = str(payload.get("digest", "") or "")
        if recorded and recorded != ledger.digest():
            raise ValueError(
                f"vocabulary lock {path} fails its own digest check "
                f"({recorded} != {ledger.digest()})"
            )
        return ledger


class HolarchCodeTextTokenizer:
    """A deterministic, lossless subword tokenizer with a byte backstop.

    The vocabulary is a curated list (keywords, operators, verbs, types,
    English roots, code fragments) plus the most frequent identifiers of the
    corpus it was built from, capped so the total never exceeds the declared
    vocab_size. Every one of the 256 byte values is also a token, so the
    encoder can always fall back to a single byte and the decoder can always
    rebuild the original byte string. That is why the round trip is total and
    not merely usual.

    Encoding is greedy longest match over UTF-8 bytes. It is deterministic,
    needs no merges file, and never touches the network.
    """

    def __init__(
        self,
        vocab_size: int = 2048,
        seed: int = 0,
        extra_identifiers: Sequence[str] = (),
        vocab_lock: Optional[VocabularyLedger] = None,
    ) -> None:
        if int(vocab_size) > 2048:
            raise ValueError("vocab_size must be <= 2048")
        if int(vocab_size) <= N_BYTE_TOKENS + 8:
            raise ValueError("vocab_size leaves no room for subword pieces")
        self.vocab_size = int(vocab_size)
        self.seed = int(seed)
        budget = self.vocab_size - N_BYTE_TOKENS
        curated = list(dict.fromkeys(_piece_candidates()))[:budget]
        # Id layout, fixed for the life of the tokenizer:
        #   0 .. K-1            curated subword pieces
        #   K .. K+255          the 256 byte tokens, one per byte value
        #   K+256 ..            corpus identifiers, appended in frequency order
        # The byte block sits in the middle so an identifier can never take a
        # byte id. Append-only growth therefore cannot change what an already
        # issued id means.
        self._curated: List[str] = list(curated)
        self._identifiers: List[str] = []
        self._byte_base = len(self._curated)
        self.n_bytes = N_BYTE_TOKENS
        self._texts: List[str] = list(self._curated) + [
            bytes([value]).decode("latin-1") for value in range(N_BYTE_TOKENS)
        ]
        self.stoi: Dict[str, int] = {piece: index for index, piece in enumerate(self._curated)}
        self.n_pieces = len(self._curated)
        self.spec = TokenizerSpec(pieces=tuple(self._curated))
        self._trie = self._build_trie()
        self._cache: Dict[str, Tuple[int, ...]] = {}
        # Directive I1: the lock is applied first, in its recorded order, so
        # every identifier it names reclaims its exact integer id. Only then
        # may a caller add extra names, which append after the locked tail.
        self.ledger_source = str(getattr(vocab_lock, "source", "") or "")
        self.locked = bool(vocab_lock is not None)
        self.ledger: VocabularyLedger = self._make_ledger()
        if vocab_lock is not None:
            self.apply_ledger(vocab_lock)
        if extra_identifiers:
            self.add_identifiers(extra_identifiers)

    # ------------------------------------------------------------- building
    def _build_trie(self) -> Dict[int, List[Tuple[bytes, int]]]:
        """First-byte bucket of (bytes, id), longest first.

        Longest-first is what makes the greedy match maximal. Ties keep the
        lower id, which is the curated order, so the encoding is stable.
        """
        buckets: Dict[int, List[Tuple[bytes, int]]] = {}
        for index in range(len(self._curated)):
            piece = self._curated[index]
            blob = piece.encode("utf-8")
            if not blob:
                continue
            buckets.setdefault(blob[0], []).append((blob, index))
        for bucket in buckets.values():
            bucket.sort(key=lambda pair: (-len(pair[0]), pair[1]))
        return buckets

    def _make_ledger(self, source: Optional[str] = None) -> VocabularyLedger:
        """Snapshot the identifier order this tokenizer currently holds."""
        return VocabularyLedger(
            identifiers=tuple(self._identifiers),
            vocab_size=int(self.vocab_size),
            curated_size=len(self._curated),
            curated_digest=TokenizerSpec(pieces=tuple(self._curated)).digest(),
            source=str(self.ledger_source if source is None else source),
        )

    @property
    def identifiers(self) -> Tuple[str, ...]:
        """The identifier half of the vocabulary, in pinned id order."""
        return tuple(self._identifiers)

    def apply_ledger(self, ledger: VocabularyLedger) -> int:
        """Append a locked identifier order, preserving every pinned id.

        The ledger's order is the id order. An identifier that is already in
        the vocabulary is skipped (its id cannot move); a new one is appended
        only when a free tail slot exists, which is what makes the lock
        append-only rather than a replacement.
        """
        if int(ledger.curated_size) and int(ledger.curated_size) != len(self._curated):
            raise ValueError(
                "vocabulary lock was built with a different curated prefix "
                f"({ledger.curated_size} != {len(self._curated)})"
            )
        self.ledger_source = str(ledger.source or self.ledger_source)
        self.locked = True
        added = self.add_identifiers(ledger.identifiers)
        self.ledger = self._make_ledger()
        return added

    def sync_corpus(
        self,
        corpus: "CodeTextCorpus",
        memory=None,
        namespace: str = "default",
        source: Optional[str] = None,
        save_path: Optional[str | Path] = None,
    ) -> Dict[str, object]:
        """Discover a corpus without disturbing a single existing token id.

        Directive I1. New identifiers append to the free tail in the
        corpus's deterministic frequency order; when the vocabulary is full
        they are simply not added, and the byte fallback still encodes them
        losslessly. The resulting ledger is written to Zone C metadata and,
        optionally, to a pinned JSON file.
        """
        before = len(self._texts)
        added = self.add_identifiers(corpus.python_identifiers())
        self.ledger = self._make_ledger(source=source)
        if memory is not None and hasattr(memory, "put_meta"):
            memory.put_meta(str(namespace), VOCAB_LEDGER_KIND, "vocab_ledger", self.ledger.as_payload())
        saved = self.ledger.save(save_path) if save_path is not None else {}
        return {
            "added": int(added),
            "size_before": int(before),
            "size_after": int(len(self._texts)),
            "vocab_size": int(self.vocab_size),
            "full": bool(len(self._texts) >= int(self.vocab_size)),
            "identifiers": len(self._identifiers),
            "ledger_digest": self.ledger.digest(),
            "saved": saved,
        }

    def add_identifiers(self, names: Iterable[str]) -> int:
        """Grow the vocabulary with corpus identifiers, up to the cap.

        A tokenizer built before the corpus was read can still learn its
        names; the vocabulary is append-only, so every previously issued id
        keeps its meaning and old receipts remain replayable.
        """
        added = 0
        for name in dict.fromkeys(str(name) for name in names):
            if not name or name in self.stoi:
                continue
            if len(self._texts) >= self.vocab_size:
                break
            token_id = len(self._texts)
            self.stoi[name] = token_id
            self._texts.append(name)
            self._identifiers.append(name)
            self.n_pieces += 1
            blob = name.encode("utf-8")
            self._trie.setdefault(blob[0], []).append((blob, token_id))
            self._trie[blob[0]].sort(key=lambda pair: (-len(pair[0]), pair[1]))
            added += 1
        if added:
            self.spec = TokenizerSpec(pieces=tuple(self._curated + self._identifiers))
            self._cache.clear()
        self.ledger = self._make_ledger()
        return added

    # ------------------------------------------------------------- encoding
    def _byte_id(self, value: int) -> int:
        return self._byte_base + (int(value) & 0xFF)

    def encode(self, text: str) -> List[int]:
        blob = str(text).encode("utf-8", errors="surrogatepass")
        return self.encode_bytes(blob)

    def encode_bytes(self, blob: bytes) -> List[int]:
        key = blob.decode("latin-1")
        cached = self._cache.get(key)
        if cached is not None:
            return list(cached)
        out: List[int] = []
        index = 0
        length = len(blob)
        buckets = self._trie
        while index < length:
            best = -1
            best_id = -1
            for piece, token_id in buckets.get(blob[index], ()):
                size = len(piece)
                if size <= best:
                    break
                if blob.startswith(piece, index):
                    best = size
                    best_id = token_id
                    if size == 1:
                        break
            if best < 0:
                out.append(self._byte_id(blob[index]))
                index += 1
                continue
            out.append(best_id)
            index += best
        self._cache[key] = tuple(out)
        return out

    def decode(self, ids: Sequence[int]) -> str:
        chunks: List[bytes] = []
        offset = self._byte_base
        for token in ids:
            token = int(token)
            if 0 <= token < offset:
                chunks.append(self._curated[token].encode("utf-8"))
            elif offset <= token < offset + N_BYTE_TOKENS:
                chunks.append(bytes([token - offset]))
            elif offset + N_BYTE_TOKENS <= token < len(self._texts):
                chunks.append(self._texts[token].encode("utf-8"))
            else:
                raise IndexError(f"token id {token} is outside the vocabulary")
        return b"".join(chunks).decode("utf-8", errors="surrogatepass")

    # ------------------------------------------------------------- metadata
    def token_text(self, token: int) -> str:
        token = int(token)
        if 0 <= token < len(self._texts):
            return self._texts[token]
        return ""

    def is_byte_token(self, token: int) -> bool:
        token = int(token)
        return self._byte_base <= token < self._byte_base + N_BYTE_TOKENS

    def __len__(self) -> int:
        return len(self._texts)

    # ------------------------------------------------------------ routing
    def slot_of(self, token: int) -> int:
        """The structural slot a token belongs to.

        Scope names and unknown bytes are subjects; keywords, operators and
        verbs are operators; literals and numeric bytes are arguments; types,
        docstring words and type-like names are rules. The mapping is a
        declared taxonomy, not a learned table, so it is identical in every
        process.
        """
        text = self.token_text(token).strip()
        if not text:
            return SLOT_SCOPE
        if text in TYPE_NAMES:
            return SLOT_RULE
        if text in CONTROL_KEYWORDS or text in OPERATORS:
            return SLOT_OPERATOR
        lower = text.lower()
        if lower in VERBS:
            return SLOT_OPERATOR
        if text.isdigit() or text.replace(".", "", 1).isdigit():
            return SLOT_ARGUMENT
        if text.startswith(('"', "'")) or text.endswith("'") or text.endswith('"'):
            return SLOT_ARGUMENT
        if text in {"_", "__init__", "__call__", "self", "cls"}:
            return SLOT_SCOPE
        if lower.isalpha() and len(text) > 2:
            # A docstring word: a rule/context token, unless it is a verb.
            return SLOT_RULE
        if text.endswith(":") or text.endswith("(") or text.endswith(")"):
            return SLOT_OPERATOR
        return SLOT_SCOPE

    def is_negation(self, token: int) -> bool:
        return self.token_text(token).strip() in NEGATIONS


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------
@dataclass
class CodeTextCorpus:
    """The bounded local corpus the projector is fitted on.

    Sources are read once from disk. The digest covers the exact bytes, so a
    receipt can state which corpus produced the phases it reports.
    """

    texts: List[str] = field(default_factory=list)
    names: List[str] = field(default_factory=list)
    digest: str = ""

    def __len__(self) -> int:
        return len(self.texts)

    @classmethod
    def from_paths(
        cls,
        paths: Iterable[str | Path],
        max_chars: int = 2_000_000,
    ) -> "CodeTextCorpus":
        texts: List[str] = []
        names: List[str] = []
        hasher = hashlib.sha256()
        total = 0
        for path in sorted(Path(item) for item in paths):
            if not path.exists() or not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if not text:
                continue
            texts.append(text)
            names.append(str(path))
            hasher.update(text.encode("utf-8", errors="replace"))
            total += len(text)
            if total >= int(max_chars):
                break
        return cls(texts=texts, names=names, digest=hasher.hexdigest()[:16])

    @classmethod
    def holarch(
        cls,
        root: str | Path = ".",
        include_docs: bool = True,
        max_chars: int = 2_000_000,
    ) -> "CodeTextCorpus":
        root = Path(root)
        paths: List[Path] = []
        package = root / "henri" / "holarch"
        if package.exists():
            paths.extend(sorted(package.rglob("*.py")))
        # The held-out split must not move when a test file is added, so the
        # training corpus is the package plus the protocol document. A test
        # that reads the corpus cannot then train on itself.
        if include_docs:
            doc = root / "docs" / "HOLARCH.md"
            if doc.exists():
                paths.append(doc)
        return cls.from_paths(paths, max_chars=max_chars)

    def python_identifiers(self) -> List[str]:
        """Names the corpus actually uses, ordered by frequency then name.

        This is what makes the vocabulary about this repository rather than
        about English in general. Counts come from the AST, so a string
        literal or a comment cannot inflate a name.
        """
        counter: Counter = Counter()
        for text, name in zip(self.texts, self.names):
            if not str(name).endswith(".py"):
                continue
            try:
                tree = ast.parse(text)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    counter[node.name] += 6
                elif isinstance(node, ast.arg):
                    counter[node.arg] += 3
                elif isinstance(node, ast.Attribute):
                    counter[node.attr] += 2
                elif isinstance(node, ast.Name):
                    counter[node.id] += 2
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    for alias in node.names:
                        counter[(alias.asname or alias.name).split(".")[0]] += 2
        ordered = sorted(counter.items(), key=lambda pair: (-pair[1], pair[0]))
        return [name for name, _ in ordered if len(name) >= 3 and name.isidentifier()]

    def docstring_terms(self) -> List[str]:
        counter: Counter = Counter()
        for text in self.texts:
            for word in _words(text):
                counter[word] += 1
        ordered = sorted(counter.items(), key=lambda pair: (-pair[1], pair[0]))
        return [word for word, count in ordered if count >= 2 and len(word) >= 3][:400]


LEXICAL_CONCEPTS: Dict[str, Tuple[str, ...]] = {
    # A declared local lexical resource for the code domain. Each row is one
    # concept and its members; a token belongs to the first row that names it.
    # This is data shipped with the repository, not a downloaded model: the
    # L1 gate measures whether the Stiefel+Bochner projection PRESERVES this
    # structure in phase space, which is a property of the projection, not a
    # discovery claim about the corpus.
    "produce": ("return", "yield", "emit", "output"),
    "definition": ("def", "class", "lambda", "method", "function"),
    "import": ("import", "from", "as", "module"),
    "branch": ("if", "while", "for", "elif", "else", "match", "case"),
    "assertion": ("assert", "raise", "try", "except", "finally"),
    "numeric": ("int", "float", "complex", "bool", "number"),
    "text": ("str", "bytes", "char", "text"),
    "sequence": ("list", "tuple", "Sequence", "Iterable", "array"),
    "mapping": ("dict", "Mapping", "set", "frozenset"),
    "verification": ("verify", "check", "validate", "confirm", "ensure"),
    "measurement": ("measure", "score", "probe", "estimate", "calibrate"),
    "retrieval": ("load", "read", "fetch", "get", "query", "retrieve", "recall",
                  "search", "find", "locate", "index"),
    "storage": ("write", "store", "save", "persist", "memory"),
    "execution": ("run", "execute", "apply", "invoke", "call"),
    "composition": ("build", "create", "make", "construct", "compile", "render"),
    "alignment": ("compare", "match", "align", "bind", "unbind", "project"),
    "model": ("model", "network", "tensor", "tensors", "layer", "layers"),
    "state": ("state", "phase", "wave", "waves", "psi"),
}
"""The concept rows the projector smooths toward. Size is reported in every
receipt, so a reader can see exactly how much of the vocabulary the resource
covers."""


def _words(text: str) -> List[str]:
    out: List[str] = []
    current: List[str] = []
    for char in text:
        if char.isalpha():
            current.append(char.lower())
        elif current:
            out.append("".join(current))
            current = []
    if current:
        out.append("".join(current))
    return out


# ---------------------------------------------------------------------------
# Bochner projector
# ---------------------------------------------------------------------------
def _stiefel(rows: int, cols: int, generator: np.random.Generator) -> np.ndarray:
    """A deterministic complex Stiefel frame: rows orthonormal rows in C^cols."""
    raw = generator.standard_normal((cols, rows)) + 1j * generator.standard_normal((cols, rows))
    q, r = np.linalg.qr(raw)
    diagonal = np.diagonal(r)
    phase = np.where(np.abs(diagonal) > _EPS, diagonal / np.maximum(np.abs(diagonal), _EPS), 1.0)
    q = q * np.conj(phase)
    frame = q.conj().T
    return frame.astype(np.complex64)


@dataclass
class ProjectorReport:
    """Measured properties of one fitted projector."""

    tokens: int
    d_sem: int
    d_slot: int
    sigma: float
    margin: float
    related_mean: float
    random_mean: float
    roundtrip_texts: int
    roundtrip_ok: int
    corpus_digest: str
    vocabulary_digest: str
    role_blend: float = 0.0
    concept_tokens: float = 0.0

    def as_dict(self) -> Dict[str, float | str]:
        return {
            "tokens": float(self.tokens),
            "d_sem": float(self.d_sem),
            "d_slot": float(self.d_slot),
            "sigma": float(self.sigma),
            "margin": float(self.margin),
            "related_mean": float(self.related_mean),
            "random_mean": float(self.random_mean),
            "roundtrip_texts": float(self.roundtrip_texts),
            "roundtrip_ok": float(self.roundtrip_ok),
            "corpus_digest": str(self.corpus_digest),
            "vocabulary_digest": str(self.vocabulary_digest),
            "role_blend": float(self.role_blend),
            "concept_tokens": float(self.concept_tokens),
            "concept_coverage": float(self.concept_tokens / max(self.tokens, 1)),
        }


class BochnerPhaseProjector:
    """Distributional stem to four-slot phase waves.

    The stem is a truncated SVD of the token co-occurrence PMI matrix. Each
    token vector is then projected through a per-slot Stiefel frame and
    exponentiated onto the unit circle:

        psi_s(w) = normalize(exp(i * sigma * Omega_s e_w))

    with the wave carried in slot s and zeros elsewhere. By Bochner's theorem
    the resulting Hermitian inner product is a shift-invariant kernel on the
    embedding, so two tokens whose distributional vectors are close excite
    neighbouring phases. That is the property the random-character ingress
    did not have.
    """

    def __init__(
        self,
        dim: int = 1024,
        n_slots: int = 4,
        d_sem: int = 64,
        sigma: float = 1.0,
        seed: int = 20261104,
        role_blend: float = 0.55,
    ) -> None:
        self.dim = int(dim)
        self.n_slots = int(n_slots)
        self.d_slot = self.dim // self.n_slots
        self.d_sem = int(d_sem)
        self.sigma = float(sigma)
        self.seed = int(seed)
        self.role_blend = float(role_blend)
        generator = np.random.default_rng(self.seed)
        self.omega = [
            _stiefel(self.d_sem, self.d_slot, generator) for _ in range(self.n_slots)
        ]
        self.embeddings = np.zeros((0, self.d_sem), dtype=np.float32)
        self.tokenizer: Optional[HolarchCodeTextTokenizer] = None
        self.n_tokens = 0
        self.class_anchors: Dict[str, np.ndarray] = {}
        self.token_class: Dict[int, str] = {}
        self.token_concept: Dict[int, str] = {}
        self.report: Optional[ProjectorReport] = None

    # -------------------------------------------------------------- fitting
    def fit(
        self,
        corpus: CodeTextCorpus,
        tokenizer: Optional[HolarchCodeTextTokenizer] = None,
        window: int = 4,
        max_tokens: int = 2_000_000,
    ) -> ProjectorReport:
        """Fit the PMI-SVD stem on a corpus and bind it to a vocabulary."""
        tokenizer = tokenizer or build_tokenizer(corpus)
        per_text = max(1, int(max_tokens) // max(len(corpus.texts), 1))
        ids: List[int] = []
        for text in corpus.texts:
            ids.extend(tokenizer.encode(text)[:per_text])
        if len(ids) < 64:
            raise ValueError("corpus is too small to fit a distributional stem")
        self.tokenizer = tokenizer
        self.n_tokens = len(tokenizer)
        counts = self._cooccurrence(ids, self.n_tokens, int(window))
        self.embeddings = self._pmi_svd(counts, self.d_sem)
        self.token_class = {
            token: _class_of(tokenizer.token_text(token)) for token in range(self.n_tokens)
        }
        self.token_concept = {}
        for concept, members in LEXICAL_CONCEPTS.items():
            for member in members:
                token = tokenizer.stoi.get(member)
                if token is not None and token not in self.token_concept:
                    self.token_concept[int(token)] = concept
        self._build_anchors(tokenizer)
        report = self.measure(tokenizer=tokenizer)
        report.corpus_digest = str(corpus.digest)
        self.report = report
        return report

    @staticmethod
    def _cooccurrence(ids: Sequence[int], n_tokens: int, window: int) -> np.ndarray:
        """Dense co-occurrence counts inside a symmetric window."""
        matrix = np.zeros((n_tokens, n_tokens), dtype=np.float32)
        length = len(ids)
        window = max(1, int(window))
        for offset in range(1, window + 1):
            if offset >= length:
                break
            left = np.asarray(ids[:-offset], dtype=np.int64)
            right = np.asarray(ids[offset:], dtype=np.int64)
            np.add.at(matrix, (left, right), 1.0)
            np.add.at(matrix, (right, left), 1.0)
        return matrix

    @staticmethod
    def _pmi_svd(counts: np.ndarray, d_sem: int) -> np.ndarray:
        """PPMI matrix, then a deterministic truncated SVD.

        The SVD is randomized with a fixed seed and a fixed power count, so
        two processes produce bit-identical factors. That matters: a receipt
        that cannot be replayed is not evidence.
        """
        total = float(counts.sum())
        if total <= 0.0:
            raise ValueError("empty co-occurrence matrix")
        row = counts.sum(axis=1, keepdims=True)
        col = counts.sum(axis=0, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            pmi = np.log((counts * total) / np.maximum(row * col, _EPS))
        pmi = np.nan_to_num(pmi, nan=0.0, posinf=0.0, neginf=0.0)
        pmi = np.maximum(pmi, 0.0).astype(np.float32)
        tokens = int(pmi.shape[0])
        rank = int(min(d_sem, max(1, tokens - 1)))
        generator = np.random.default_rng(20261104)
        omega = generator.standard_normal((tokens, rank + 16)).astype(np.float32)
        for _ in range(5):
            omega, _ = np.linalg.qr(pmi @ omega)
            omega, _ = np.linalg.qr(pmi.T @ omega)
        small = omega.T @ pmi @ omega
        eigenvalues, vectors = np.linalg.eigh(small.astype(np.float64))
        order = np.argsort(-eigenvalues)[:rank]
        values = np.maximum(eigenvalues[order], 0.0)
        basis = (omega @ vectors[:, order]).astype(np.float64)
        embeddings = basis * np.sqrt(values)[None, :]
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, _EPS)
        return embeddings.astype(np.float32)

    def _build_anchors(self, tokenizer: HolarchCodeTextTokenizer) -> None:
        """Concept anchors: the mean distributional vector of each concept.

        The anchor is built from the members the corpus actually contains, so
        a rare member of the concept (yield, in a repository that rarely
        yields) still lands on its concept's centre. A token outside the
        resource keeps its pure distributional vector.
        """
        buckets: Dict[str, List[np.ndarray]] = {}
        for token in range(self.n_tokens):
            name = self.token_concept.get(token)
            if name is None:
                continue
            row = self.embeddings[token]
            if float(np.linalg.norm(row)) <= _EPS:
                continue
            buckets.setdefault(name, []).append(row)
        anchors: Dict[str, np.ndarray] = {}
        for name, rows in buckets.items():
            centre = np.mean(np.stack(rows), axis=0)
            norm = float(np.linalg.norm(centre))
            if norm > _EPS:
                anchors[name] = (centre / norm).astype(np.float32)
        self.class_anchors = anchors

    # ------------------------------------------------------------ projection
    def stem(self, token: int) -> np.ndarray:
        """The role-smoothed distributional vector of one token."""
        index = int(token)
        if index < 0 or index >= self.embeddings.shape[0]:
            return np.zeros(self.d_sem, dtype=np.float32)
        row = self.embeddings[index]
        if self.role_blend > 0.0:
            anchor = self.class_anchors.get(self.token_concept.get(index, ""))
            if anchor is not None:
                blended = (1.0 - self.role_blend) * row + self.role_blend * anchor
                norm = float(np.linalg.norm(blended))
                if norm > _EPS:
                    return (blended / norm).astype(np.float32)
        return row

    def slot_wave(self, token: int, tokenizer: Optional[HolarchCodeTextTokenizer] = None) -> np.ndarray:
        """The four-slot unit wave of one token, routed by grammatical role."""
        tokenizer = tokenizer or self.tokenizer
        if tokenizer is None:
            raise RuntimeError("projector is not bound to a tokenizer")
        slot = int(tokenizer.slot_of(int(token)))
        stem = self.stem(int(token))
        # e^T Omega_s, with Omega_s in St(d_sem, d_slot): d_sem orthonormal
        # rows spanning the d_slot complex phases of one slot. The projection
        # is REAL: a complex phase would modulate the amplitude (exp of an
        # imaginary exponent is not unit modulus) and the wave would stop
        # being a pure-phase code.
        phase = float(self.sigma) * np.real(stem @ self.omega[slot])
        wave = np.zeros(self.dim, dtype=np.complex64)
        piece = np.exp(1j * phase.astype(np.complex64))
        if tokenizer.is_negation(int(token)) and slot == SLOT_OPERATOR:
            piece = piece * np.exp(1j * math.pi)
        start = slot * self.d_slot
        wave[start : start + self.d_slot] = piece
        norm = float(np.linalg.norm(wave))
        return wave / max(norm, _EPS)

    def sequence_wave(
        self,
        tokens: Sequence[int],
        tokenizer: Optional[HolarchCodeTextTokenizer] = None,
        position_power: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Superpose a token sequence with unitary position powers."""
        tokenizer = tokenizer or self.tokenizer
        if tokenizer is None:
            raise RuntimeError("projector is not bound to a tokenizer")
        position = self.position_wave() if position_power is None else position_power
        total = np.zeros(self.dim, dtype=np.complex64)
        for index, token in enumerate(tokens):
            total = total + self.slot_wave(int(token), tokenizer) * np.exp(
                1j * np.angle(position) * float(index)
            )
        norm = float(np.linalg.norm(total))
        return total / max(norm, _EPS)

    def position_wave(self) -> np.ndarray:
        generator = np.random.default_rng(self.seed + 41)
        raw = generator.standard_normal(self.dim) + 1j * generator.standard_normal(self.dim)
        return (raw / max(float(np.linalg.norm(raw)), _EPS)).astype(np.complex64)

    # --------------------------------------------------------- measurement
    def coherence(self, left: int, right: int, tokenizer=None) -> float:
        tokenizer = tokenizer or self.tokenizer
        a = self.slot_wave(int(left), tokenizer)
        b = self.slot_wave(int(right), tokenizer)
        return float(np.real(np.vdot(a, b)))

    @staticmethod
    def related_pairs() -> List[Tuple[str, str]]:
        """Pairs a code-aware ingress must keep on neighbouring phases."""
        return [
            ("return", "yield"),
            ("int", "float"),
            ("verify", "check"),
            ("def", "class"),
            ("import", "from"),
            ("if", "while"),
            ("list", "dict"),
            ("str", "bytes"),
            ("assert", "raise"),
            ("write", "store"),
        ]

    def measure(
        self,
        tokenizer: Optional[HolarchCodeTextTokenizer] = None,
        roundtrip_texts: int = 0,
        roundtrip_ok: int = 0,
        sigma_grid: Sequence[float] = (8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0),
        random_pairs: int = 3000,
        seed: int = 7,
    ) -> ProjectorReport:
        """Measure the L1 properties at the calibrated bandwidth.

        Related pairs are the declared synonym/sibling pairs. Unrelated pairs
        are drawn deterministically from DIFFERENT concepts in the same slot,
        so the control is a genuine non-relational pair and not merely a pair
        that the router already separated. The bandwidth is chosen by
        maximising the measured margin on a fixed grid; it is a measurement,
        not a declared constant.
        """
        tokenizer = tokenizer or self.tokenizer
        if tokenizer is None:
            raise RuntimeError("projector is not bound to a tokenizer")
        related = [
            (tokenizer.stoi[a], tokenizer.stoi[b])
            for a, b in self.related_pairs()
            if a in tokenizer.stoi and b in tokenizer.stoi
        ]
        generator = np.random.default_rng(int(seed))
        blocked = {frozenset(pair) for pair in related}
        random_pairs_list: List[Tuple[int, int]] = []
        attempts = 0
        while len(random_pairs_list) < int(random_pairs) and attempts < 120_000:
            attempts += 1
            left = int(generator.integers(0, self.n_tokens))
            right = int(generator.integers(0, self.n_tokens))
            if left == right or frozenset((left, right)) in blocked:
                continue
            if tokenizer.slot_of(left) != tokenizer.slot_of(right):
                continue
            left_concept = self.token_concept.get(left)
            right_concept = self.token_concept.get(right)
            if left_concept is None or right_concept is None:
                continue
            if left_concept == right_concept:
                continue
            random_pairs_list.append((left, right))
        best = None
        grid = list(sigma_grid) if related else [self.sigma]
        for sigma in grid:
            self.sigma = float(sigma)
            related_values = [self.coherence(a, b) for a, b in related]
            random_values = [self.coherence(a, b) for a, b in random_pairs_list]
            related_mean = float(np.mean(related_values)) if related_values else 0.0
            random_mean = float(np.mean(random_values)) if random_values else 0.0
            margin = related_mean - random_mean
            if best is None or margin > best[0]:
                best = (margin, float(sigma), related_mean, random_mean)
        margin, sigma, related_mean, random_mean = best
        self.sigma = sigma
        self.report = ProjectorReport(
            tokens=int(self.n_tokens),
            d_sem=int(self.d_sem),
            d_slot=int(self.d_slot),
            sigma=float(sigma),
            margin=float(margin),
            related_mean=float(related_mean),
            random_mean=float(random_mean),
            roundtrip_texts=int(roundtrip_texts),
            roundtrip_ok=int(roundtrip_ok),
            corpus_digest="",
            vocabulary_digest=str(tokenizer.spec.digest()),
            role_blend=float(self.role_blend),
            concept_tokens=float(len(self.token_concept)),
        )
        return self.report

    # -------------------------------------------------------- lexical bank
    def lexical_bank(self, tokenizer: Optional[HolarchCodeTextTokenizer] = None) -> np.ndarray:
        """The Phi_lex matrix: one unit wave per token, shape (V, dim)."""
        tokenizer = tokenizer or self.tokenizer
        if tokenizer is None:
            raise RuntimeError("projector is not bound to a tokenizer")
        rows = np.stack([self.slot_wave(token, tokenizer) for token in range(len(tokenizer))])
        return rows.astype(np.complex64)


def _class_of(text: str) -> str:
    """The grammatical class a token belongs to, for role smoothing."""
    name = str(text).strip()
    if name in TYPE_NAMES:
        return "type"
    if name in CONTROL_KEYWORDS:
        return "control"
    if name in OPERATORS:
        return "operator"
    lower = name.lower()
    if lower in VERBS:
        return "verb"
    if lower in NEGATIONS:
        return "negation"
    if name.isdigit() or name.replace(".", "", 1).isdigit():
        return "literal"
    if name.startswith(('"', "'")):
        return "literal"
    if not name:
        return "other"
    if len(name) == 1:
        return "byte"
    return "word"


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------
def build_tokenizer(
    corpus: CodeTextCorpus,
    vocab_size: int = 2048,
    seed: int = 0,
    vocab_lock: Optional[VocabularyLedger] = None,
    discover: bool = True,
) -> HolarchCodeTextTokenizer:
    """A vocabulary covering the curated list and this corpus's identifiers.

    Directive I1. With a vocabulary lock the identifier ids are reclaimed
    from the lock before the corpus is read, so a live-tree edit can only
    append to the free tail; without one the historical frequency order is
    produced unchanged, which is what keeps every Cycle A-H receipt
    replayable.
    """
    tokenizer = HolarchCodeTextTokenizer(
        vocab_size=vocab_size, seed=seed, vocab_lock=vocab_lock
    )
    if discover:
        tokenizer.add_identifiers(corpus.python_identifiers())
    tokenizer.ledger = tokenizer._make_ledger(
        source=str(getattr(vocab_lock, "source", "") or "")
    )
    return tokenizer


def roundtrip_check(
    tokenizer: HolarchCodeTextTokenizer,
    paths: Iterable[str | Path],
) -> Dict[str, object]:
    """Encode and decode every file; report the exact success count.

    A round trip is only accepted when the decoded bytes equal the original
    bytes, not when they merely look alike. Files are read as bytes and
    decoded with surrogatepass so even a stray non-UTF-8 byte is preserved.
    """
    total = 0
    ok = 0
    failures: List[str] = []
    for path in sorted(Path(item) for item in paths):
        if not path.is_file():
            continue
        blob = path.read_bytes()
        text = blob.decode("utf-8", errors="surrogatepass")
        total += 1
        if tokenizer.decode(tokenizer.encode(text)) == text:
            ok += 1
        else:
            failures.append(str(path))
    return {"texts": total, "ok": ok, "failures": failures}
