"""Deterministic ontology and quantity verifier inside henri.holarch.

Evidence label: OBSERVED for every acceptance and rejection rate this module
reports; DERIVED for the two rules it implements.

What it is
----------
A local, content-addressed claim checker. It indexes the technical corpus
(docs/HOLARCH.md and the docstrings of henri/holarch/*.py) into passages,
each identified by the SHA-256 of its exact text, and answers one question:

    given this sentence, does the corpus support it?

Three answers are possible and all three are used by the sidecar:

    supported      every rare term and every number in the claim is present
                   in the retrieved evidence, and no signature rule fails.
    contradicted   the claim names a supported subject but states a number
                   that the evidence states differently.
    unsupported    something in the claim has no evidence at all: an invented
                   constant, a fabricated identifier, or a missing term.

Two rules are mandatory and neither is statistical:

1. IDF support. Every rare term of the claim (high inverse document
   frequency, longer than two characters) must appear in the retrieved
   evidence. A fabricated identifier is therefore rejected before any
   similarity score is consulted.
2. Exact quantity check. Every number in the claim must appear in the
   evidence that supports the claim's terms. The comparison is on the
   normalised literal, so 1,020,380 and 1020380 are the same quantity and
   1,020,381 is not.

BM25 ranks the passages. BM25 is used only to choose WHICH passages are
shown to the two rules; it can never accept a claim on its own. That
separation is what makes the verdict fail-closed.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
import time

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from henri.holarch.code_lang import ENGLISH_ROOTS, VERBS

_EPS = 1e-12

GENERIC_WORDS = set(ENGLISH_ROOTS.split()) | set(VERBS) | {
    "raised", "reached", "reaches", "measured", "shows", "showed", "using",
    "used", "given", "gives", "makes", "made", "takes", "taken", "holds",
    "held", "keeps", "kept", "comes", "came", "goes", "went", "seen",
    "says", "said", "found", "finds", "runs", "ran", "works", "worked",
    "needs", "needed", "allows", "allowed", "requires", "required",
    "contains", "contained", "includes", "included", "reports", "reported",
    "states", "stated", "records", "recorded", "emits", "emitted",
    "returns", "returned", "yields", "yielded", "produces", "produced",
    "many", "much", "valid", "hold", "holds", "record", "records",
    "resolve", "resolved", "resolves", "indicate", "indicates", "contain",
    "contains", "exist", "exists", "occur", "occurs", "appears", "appear",
    "achieve", "achieves", "reached", "reach", "reaches", "exceed",
    "exceeds", "remain", "remains", "carry", "carries", "give", "gives",
}
"""Names, verbs and quantifiers that carry no domain content. A word in this
set is never a mandatory support term."""
"""Plain English, not domain vocabulary. A rare English word is not evidence
of a technical claim, so it is never a mandatory support term."""

STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "to", "in", "is", "are", "was",
    "were", "be", "been", "being", "for", "on", "at", "by", "with", "from",
    "as", "that", "this", "these", "those", "it", "its", "has", "have",
    "had", "does", "do", "did", "can", "could", "should", "would", "will",
    "may", "might", "must", "not", "no", "than", "then", "which", "what",
    "when", "where", "who", "how", "why", "into", "over", "under", "per",
    "each", "all", "any", "both", "more", "most", "less", "least", "also",
    "such", "only", "very", "same", "other", "there", "their", "them",
    "they", "we", "our", "you", "your", "his", "her", "one", "two",
}

NUMBER_PATTERN = re.compile(r"[+-]?\d[\d,]*(?:\.\d+)?(?:[eE][+-]?\d+)?")
TERM_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_\.\-]{1,}")


def normalise_number(text: str) -> str:
    """Canonical form of a numeric literal: 1,020,380 -> 1020380."""
    value = str(text).strip().replace(",", "").replace("+", "")
    if value.endswith("%"):
        value = value[:-1]
    try:
        number = float(value)
    except ValueError:
        return value
    if not math.isfinite(number):
        return value
    if number == int(number) and abs(number) < 1e18:
        return str(int(number))
    return repr(number)


def extract_numbers(text: str) -> List[str]:
    """Every numeric literal in a text, normalised and de-duplicated."""
    out: List[str] = []
    for match in NUMBER_PATTERN.findall(str(text)):
        value = normalise_number(match)
        if value not in out:
            out.append(value)
    return out


def extract_terms(text: str) -> List[str]:
    """Lower-case content terms, dotted identifiers kept whole."""
    out: List[str] = []
    for match in TERM_PATTERN.findall(str(text)):
        term = match.lower().strip(".-")
        if len(term) < 2:
            continue
        out.append(term)
    return out


@dataclass(frozen=True)
class Passage:
    """One content-addressed chunk of the corpus."""

    sha256: str
    source: str
    text: str
    order: int

    @classmethod
    def build(cls, source: str, text: str, order: int) -> "Passage":
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return cls(sha256=digest, source=str(source), text=str(text), order=int(order))


@dataclass
class ClaimVerdict:
    """The outcome of one claim check, with the evidence that produced it."""

    status: str
    claim: str
    reason: str = ""
    confidence: float = 0.0
    passages: List[str] = field(default_factory=list)
    evidence_text: str = ""
    missing_terms: List[str] = field(default_factory=list)
    missing_numbers: List[str] = field(default_factory=list)
    found_numbers: List[str] = field(default_factory=list)
    signature_rule: str = ""

    @property
    def supported(self) -> bool:
        return self.status == "supported"

    def as_dict(self) -> Dict[str, object]:
        return {
            "status": self.status,
            "reason": self.reason,
            "confidence": float(self.confidence),
            "passages": list(self.passages),
            "missing_terms": list(self.missing_terms),
            "missing_numbers": list(self.missing_numbers),
            "found_numbers": list(self.found_numbers),
            "signature_rule": self.signature_rule,
            "evidence_chars": len(self.evidence_text),
        }


def _chunks(text: str, target: int = 900, overlap: int = 160) -> List[str]:
    """Split on blank lines, then window anything longer than target."""
    blocks = [block.strip() for block in re.split(r"\n\s*\n", str(text)) if block.strip()]
    out: List[str] = []
    for block in blocks:
        if len(block) <= target:
            out.append(block)
            continue
        step = max(1, int(target) - int(overlap))
        start = 0
        while start < len(block):
            out.append(block[start : start + int(target)].strip())
            if start + int(target) >= len(block):
                break
            start += step
    return [chunk for chunk in out if len(chunk) >= 40]


def _docstring_passages(path: Path) -> List[Tuple[str, str]]:
    """(label, text) for each module, class and function docstring."""
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source)
    except (OSError, SyntaxError):
        return []
    out: List[Tuple[str, str]] = []
    module_doc = ast.get_docstring(tree)
    if module_doc:
        out.append((f"{path.name}::<module>", module_doc))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            doc = ast.get_docstring(node)
            if doc:
                out.append((f"{path.name}::{node.name}", doc))
    return out


@dataclass
class Signature:
    """One callable's declared parameter surface, read from the AST."""

    name: str
    parameters: Tuple[str, ...]
    accepts_kwargs: bool

    def accepts(self, keyword: str) -> bool:
        return str(keyword) in self.parameters or bool(self.accepts_kwargs)


class OntologyVerifier:
    """Content-addressed passages, BM25 ranking, and two hard rules.

    Construction indexes the corpus once. After that, verify_claim touches
    no file: it scores the indexed passages and applies the IDF and quantity
    rules. The index is deterministic, so the same corpus always yields the
    same passage digests and the same verdicts.
    """

    def __init__(
        self,
        root: str | Path = ".",
        docs: Sequence[str] = ("docs/HOLARCH.md",),
        package: str = "henri/holarch",
        chunk_chars: int = 900,
        rare_quantile: float = 0.55,
    ) -> None:
        self.root = Path(root)
        self.docs = tuple(docs)
        self.package = str(package)
        self.chunk_chars = int(chunk_chars)
        self.rare_quantile = float(rare_quantile)
        self.passages: List[Passage] = []
        self.terms: List[List[str]] = []
        self.numbers: List[List[str]] = []
        self.df: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_freq_total = 0
        self.signatures: Dict[str, Signature] = {}
        self.corpus_digest = ""
        self.rare_threshold = 0.0
        self.mean_length = 1.0

    # ---------------------------------------------------------------- build
    def build(self) -> Dict[str, float]:
        hasher = hashlib.sha256()
        self.passages = []
        order = 0
        for name in self.docs:
            path = self.root / name
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            hasher.update(text.encode("utf-8"))
            for chunk in _chunks(text, target=self.chunk_chars):
                self.passages.append(Passage.build(str(name), chunk, order))
                order += 1
        package = self.root / self.package
        for path in sorted(package.glob("*.py")) if package.exists() else []:
            hasher.update(path.name.encode("utf-8"))
            for label, doc in _docstring_passages(path):
                for chunk in _chunks(doc, target=self.chunk_chars):
                    self.passages.append(Passage.build(f"{self.package}/{label}", chunk, order))
                    order += 1
            self._index_signatures(path)
        # Content addressing means identity: the same chunk twice is one
        # passage, not two votes. Keep the first occurrence.
        unique: List[Passage] = []
        seen: set = set()
        for passage in self.passages:
            if passage.sha256 in seen:
                continue
            seen.add(passage.sha256)
            unique.append(passage)
        self.passages = [
            Passage(passage.sha256, passage.source, passage.text, index)
            for index, passage in enumerate(unique)
        ]
        self.corpus_digest = hasher.hexdigest()[:16]
        self._index_terms()
        return {
            "passages": float(len(self.passages)),
            "signatures": float(len(self.signatures)),
            "vocabulary": float(len(self.df)),
            "mean_passage_terms": float(self.mean_length),
        }

    def _index_signatures(self, path: Path) -> None:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, SyntaxError):
            return
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        self._register_signature(f"{node.name}.{child.name}", child)
                        self._register_signature(child.name, child)
                self._register_class(node.name)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._register_signature(node.name, node)

    def _register_class(self, name: str) -> None:
        self.signatures.setdefault(
            str(name), Signature(name=str(name), parameters=("self",), accepts_kwargs=True)
        )

    def _register_signature(self, name: str, node) -> None:
        args = list(getattr(node.args, "posonlyargs", [])) + list(node.args.args)
        parameters = [str(arg.arg) for arg in args]
        parameters += [str(arg.arg) for arg in node.args.kwonlyargs]
        accepts_kwargs = node.args.kwarg is not None
        key = str(name)
        existing = self.signatures.get(key)
        if existing is None:
            self.signatures[key] = Signature(
                name=key, parameters=tuple(dict.fromkeys(parameters)), accepts_kwargs=accepts_kwargs
            )
        else:
            merged = tuple(dict.fromkeys(list(existing.parameters) + parameters))
            self.signatures[key] = Signature(
                name=key, parameters=merged, accepts_kwargs=existing.accepts_kwargs or accepts_kwargs
            )

    def _index_terms(self) -> None:
        self.terms = [extract_terms(passage.text) for passage in self.passages]
        self.numbers = [extract_numbers(passage.text) for passage in self.passages]
        self.doc_freq_total = len(self.passages)
        df: Dict[str, int] = {}
        for terms in self.terms:
            for term in set(terms):
                df[term] = df.get(term, 0) + 1
        self.df = df
        total = max(self.doc_freq_total, 1)
        self.idf = {
            term: math.log((total - count + 0.5) / (count + 0.5) + 1.0)
            for term, count in df.items()
        }
        values = sorted(self.idf.values()) if self.idf else [0.0]
        index = min(len(values) - 1, max(0, int(self.rare_quantile * len(values))))
        self.rare_threshold = float(values[index])
        self.mean_length = float(
            sum(len(terms) for terms in self.terms) / max(len(self.terms), 1)
        )
        self._build_bm25_matrices()

    # --------------------------------------------------------------- search
    def _build_bm25_matrices(self) -> None:
        """Precompute the BM25 scoring matrix once per corpus (Directive K4).

        A row is one passage, a column is one term, and the entry is

            A[p, t] = idf_t * f_pt (k1 + 1) / (f_pt + k1 (1 - b + b L_p / avg))

        so the score of a query is A @ q for the 0/1 query vector q. That is
        one float32 matmul, and numpy releases the CPython GIL inside it.
        """
        k1, b = 1.5, 0.75
        matrix_rows = len(self.terms)
        vocab: Dict[str, int] = {}
        for terms in self.terms:
            for term in terms:
                if term not in vocab:
                    vocab[term] = len(vocab)
        width = len(vocab)
        self._bm25_vocab = vocab
        self._bm25_k1 = float(k1)
        self._bm25_b = float(b)
        if matrix_rows == 0 or width == 0:
            self._bm25_matrix = np.zeros((matrix_rows, 0), dtype=np.float32)
            return
        matrix = np.zeros((matrix_rows, width), dtype=np.float32)
        average = max(self.mean_length, _EPS)
        lengths = np.asarray([len(terms) for terms in self.terms], dtype=np.float64)
        norm = float(k1) * (1.0 - float(b) + float(b) * lengths / average)
        for index, terms in enumerate(self.terms):
            if not terms:
                continue
            counts: Dict[str, float] = {}
            for term in terms:
                counts[term] = counts.get(term, 0.0) + 1.0
            row = norm[index]
            for term, frequency in counts.items():
                column = vocab.get(term)
                if column is None:
                    continue
                weight = float(self.idf.get(term, 0.0))
                matrix[index, column] = weight * (frequency * (k1 + 1.0)) / (
                    frequency + row
                )
        self._bm25_matrix = matrix

    def bm25(self, claim: str, top_k: int = 4) -> List[Tuple[float, int]]:
        query = [term for term in extract_terms(claim) if term not in STOPWORDS]
        if not query or not self.passages:
            return []
        matrix = getattr(self, "_bm25_matrix", None)
        vocab = getattr(self, "_bm25_vocab", None)
        if matrix is None or vocab is None:
            return self._bm25_python(claim, top_k)
        vector = np.zeros((matrix.shape[1],), dtype=np.float32)
        for term in set(query):
            column = vocab.get(term)
            if column is not None:
                vector[column] = 1.0
        if not bool(vector.any()):
            return []
        scores = matrix @ vector
        finite = np.flatnonzero(scores > 0.0)
        if finite.size == 0:
            return []
        order = sorted(finite.tolist(), key=lambda index: (-float(scores[index]), int(index)))
        return [(float(scores[index]), int(index)) for index in order[: max(1, int(top_k))]]

    def _bm25_python(self, claim: str, top_k: int = 4) -> List[Tuple[float, int]]:
        """The Cycle J per-passage loop, retained as the parity reference."""
        query = [term for term in extract_terms(claim) if term not in STOPWORDS]
        if not query or not self.passages:
            return []
        average = max(self.mean_length, _EPS)
        scores: List[Tuple[float, int]] = []
        k1, b = 1.5, 0.75
        for index, terms in enumerate(self.terms):
            if not terms:
                continue
            counts: Dict[str, int] = {}
            for term in terms:
                counts[term] = counts.get(term, 0) + 1
            length = len(terms)
            score = 0.0
            for term in query:
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                weight = self.idf.get(term, 0.0)
                score += weight * (frequency * (k1 + 1.0)) / (
                    frequency + k1 * (1.0 - b + b * length / average)
                )
            if score > 0.0:
                scores.append((float(score), index))
        scores.sort(key=lambda pair: (-pair[0], pair[1]))
        return scores[: max(1, int(top_k))]

    @staticmethod
    def _is_generic(term: str) -> bool:
        """True for a plain English word, including its common inflections."""
        word = str(term).lower()
        if word in GENERIC_WORDS:
            return True
        for suffix in ("ing", "ed", "es", "s", "ly", "d"):
            if word.endswith(suffix) and word[: -len(suffix)] in GENERIC_WORDS:
                return True
        return False

    def rare_terms(self, claim: str) -> List[str]:
        """Claim terms whose absence from evidence would matter.

        A term is mandatory only when it is RARE in this corpus and NOT plain
        English. That distinction is what lets a claim say "raised" without
        demanding that the corpus contain that word, while still demanding the
        corpus contain "covariance" or "ZoneCMemory".
        """
        out: List[str] = []
        for term in extract_terms(claim):
            if term in STOPWORDS or len(term) < 3:
                continue
            if term.replace(".", "").isdigit():
                continue
            if self._is_generic(term):
                continue
            if self.idf.get(term, self.rare_threshold + 1.0) >= self.rare_threshold:
                out.append(term)
        return list(dict.fromkeys(out))

    # ---------------------------------------------------- signature checking
    def check_signature_claim(self, claim: str) -> Optional[str]:
        """Return a failure reason when a callable claim contradicts the AST.

        The rule is mechanical: if the claim names a callable the index knows
        and attaches a keyword the AST does not declare, the claim is
        unsupported. Nothing here guesses about semantics.
        """
        text = str(claim)
        # Only a dotted callable is matched, and only when the claim also
        # names a keyword or an argument. A bare shared method name (query,
        # solve, write) would otherwise fire on ordinary prose.
        referenced = [
            token
            for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*", text)
            if token in self.signatures
        ]
        keywords = re.findall(
            r"(?:keyword(?:\s+argument)?|argument|parameter|kwarg)\s+['\"]?([A-Za-z_][A-Za-z0-9_]*)",
            text,
            flags=re.IGNORECASE,
        )
        keywords += re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)=(?!=)", text)
        targets = referenced
        if not targets:
            return None
        for target in targets:
            signature = self.signatures[target]
            for keyword in dict.fromkeys(keywords):
                if keyword.lower() in {"self", "cls"}:
                    continue
                if not signature.accepts(keyword):
                    return (
                        f"signature: {target} does not accept keyword {keyword!r}; "
                        f"declared parameters are {list(signature.parameters)[:12]}"
                    )
        return None

    # ---------------------------------------------------------------- verify
    def verify_claim(self, claim: str, top_k: int = 4) -> ClaimVerdict:
        """Decide supported, contradicted, or unsupported for one claim."""
        text = str(claim).strip()
        if not text:
            return ClaimVerdict("unsupported", text, "empty claim", 0.0)
        signature_failure = self.check_signature_claim(text)
        if signature_failure:
            return ClaimVerdict(
                "unsupported", text, signature_failure, 0.0, signature_rule=signature_failure
            )
        ranked = self.bm25(text, top_k=max(4, int(top_k)))
        if not ranked:
            return ClaimVerdict("unsupported", text, "no passage matched the claim", 0.0)
        rare = self.rare_terms(text)
        content = [
            term
            for term in extract_terms(text)
            if term not in STOPWORDS
            and len(term) >= 3
            and not term.replace(".", "").isdigit()
        ]
        claim_numbers = extract_numbers(text)
        # The evidence set is the top_k term-ranked passages, widened by any
        # deeper passage that both states one of the claim's quantities AND
        # shares at least two of its content terms. That widening is what
        # lets a composite sentence cite a metric table and a parameter count
        # from two adjacent paragraphs, while a number in an unrelated table
        # still cannot vote: it has no shared subject.
        selected = [index for _score, index in ranked]
        for _score, index in self.bm25(text, top_k=24):
            if index in selected:
                continue
            shared = sum(1 for term in set(content) if term in self.terms[index])
            states = any(number in self.numbers[index] for number in claim_numbers)
            if states and shared >= 2:
                selected.append(index)
        best_score = float(ranked[0][0])
        missing_terms: List[str] = []
        for term in rare:
            if not any(term in self.terms[index] for index in selected):
                missing_terms.append(term)
        if missing_terms:
            evidence = self._evidence_text(selected or [ranked[0][1]])
            return ClaimVerdict(
                "unsupported",
                text,
                f"rare term(s) absent from evidence: {missing_terms}",
                0.0,
                [self.passages[index].sha256 for index in (selected or [ranked[0][1]])],
                evidence,
                missing_terms=missing_terms,
            )
        # Rule 2: every number in the claim must appear in the evidence that
        # supports the claim's terms.
        evidence_indices = selected or [ranked[0][1]]
        missing_numbers: List[str] = [];
        found_numbers: List[str] = []
        for number in claim_numbers:
            if any(number in self.numbers[index] for index in evidence_indices):
                found_numbers.append(number)
            else:
                missing_numbers.append(number)
        evidence = self._evidence_text(evidence_indices)
        passages = [self.passages[index].sha256 for index in evidence_indices]
        if missing_numbers:
            contradicted = self._find_contradiction(text, evidence_indices, missing_numbers)
            status = "contradicted" if contradicted else "unsupported"
            reason = (
                f"quantity {contradicted} is stated differently in the evidence"
                if contradicted
                else f"number(s) absent from evidence: {missing_numbers}"
            )
            return ClaimVerdict(
                status,
                text,
                reason,
                0.0,
                passages,
                evidence,
                missing_numbers=missing_numbers,
                found_numbers=found_numbers,
            )
        coverage = float(len(found_numbers)) / max(len(claim_numbers), 1)
        confidence = min(1.0, 0.5 + 0.5 * coverage) * min(1.0, best_score / 12.0 + 0.4)
        return ClaimVerdict(
            "supported",
            text,
            "all rare terms and all quantities present in evidence",
            float(confidence),
            passages,
            evidence,
            found_numbers=found_numbers,
        )

    def _evidence_text(self, indices: Sequence[int], limit: int = 2400) -> str:
        parts = [self.passages[int(index)].text for index in indices[:4]]
        return chr(10).join(parts)[: int(limit)]

    def _find_contradiction(
        self,
        claim: str,
        indices: Sequence[int],
        missing: Sequence[str],
    ) -> str:
        """A number the evidence states differently for the same subject."""
        terms = [term for term in extract_terms(claim) if term not in STOPWORDS]
        for index in indices:
            passage = self.passages[int(index)]
            for sentence in re.split(r"(?<=[.!?])\s+", passage.text):
                lower = sentence.lower()
                hits = sum(1 for term in terms if term in lower)
                if hits < max(2, len(terms) // 2):
                    continue
                for number in extract_numbers(sentence):
                    for wanted in missing:
                        if wanted not in sentence and number != wanted:
                            return str(wanted)
        return ""

    # ------------------------------------------------------------- reporting
    def report(self) -> Dict[str, object]:
        return {
            "passages": len(self.passages),
            "signatures": len(self.signatures),
            "vocabulary": len(self.df),
            "rare_threshold": float(self.rare_threshold),
            "mean_passage_terms": float(self.mean_length),
            "corpus_digest": str(self.corpus_digest),
            "passage_digests": [passage.sha256[:16] for passage in self.passages[:8]],
        }


def build_verifier(root: str | Path = ".", **kwargs) -> OntologyVerifier:
    """Construct and build a verifier in one call."""
    verifier = OntologyVerifier(root=root, **kwargs)
    verifier.build()
    return verifier


# =========================================================================
# Directive S3: the TBAC-sandboxed exteroceptive corpus browser
# =========================================================================
class SandboxViolation(PermissionError):
    """Raised when a browser request leaves the allow-listed envelope."""


class SandboxedCorpusBrowser:
    """Read-only local corpus access under an explicit directory allow-list.

    The browser is the RUNTIME half of the two-tier grounding pipeline: when a
    Zone C query returns the C1 empty-retrieval veto (or an ungrounded topic),
    the caller asks the browser to read the allow-listed corpus, extract
    verified triples, crystallise them into Zone C as Tier-2 engrams, and
    answer from the passage that grounded them.

    The envelope is fail-closed and checked BEFORE any file is opened: a URL
    or scheme (http://, https://, file:, s3://) is refused, any '..' component
    is refused (traversal), a resolved path outside every allowed root is
    refused, and a non-file, a binary suffix or an oversized file is refused.

    Every accepted call returns a chained SHA-256 receipt over the roots, the
    query, the passages read and the engram ids written.
    """

    ALLOWED_SUFFIXES = (".md", ".txt", ".rst", ".json", ".csv", ".tsv", ".py", ".yaml", ".yml")
    FORBIDDEN_SCHEMES = ("http://", "https://", "ftp://", "file:", "s3://", "gs://", "//")

    def __init__(
        self,
        memory,
        roots,
        *,
        namespace: str = "corpus",
        max_file_bytes: int = 2_000_000,
        max_files: int = 512,
        extractor=None,
    ) -> None:
        resolved = []
        for root in roots:
            path = Path(root).expanduser().resolve()
            if path.is_dir():
                resolved.append(path)
        if not resolved:
            raise SandboxViolation("browser needs at least one existing allow-listed root")
        self.memory = memory
        self.roots = resolved
        self.namespace = str(namespace)
        self.max_file_bytes = int(max_file_bytes)
        self.max_files = int(max_files)
        self.extractor = extractor
        self.history = []
        self._previous = ""
        self.reads = 0
        self.writes = 0
        self.refusals = 0

    # ------------------------------------------------------------- envelope
    def resolve(self, candidate, *, must_exist: bool = True) -> Path:
        """Resolve one path inside the envelope, or raise SandboxViolation."""
        text = str(candidate).strip()
        lowered = text.lower()
        for scheme in self.FORBIDDEN_SCHEMES:
            if lowered.startswith(scheme):
                self.refusals += 1
                raise SandboxViolation(f"refused scheme: {scheme}")
        raw = Path(text)
        if ".." in raw.parts:
            self.refusals += 1
            raise SandboxViolation("refused path traversal: '..' component")
        if raw.is_absolute():
            resolved = raw.resolve()
        else:
            resolved = (self.roots[0] / raw).resolve()
        if not any(self._inside(resolved, root) for root in self.roots):
            self.refusals += 1
            raise SandboxViolation(f"refused path outside the allow-list: {resolved}")
        if must_exist and not resolved.exists():
            self.refusals += 1
            raise SandboxViolation(f"refused missing path: {resolved}")
        if must_exist and not resolved.is_file():
            self.refusals += 1
            raise SandboxViolation(f"refused non-file path: {resolved}")
        if must_exist and resolved.suffix.lower() not in self.ALLOWED_SUFFIXES:
            self.refusals += 1
            raise SandboxViolation(f"refused suffix: {resolved.suffix}")
        if must_exist and resolved.stat().st_size > self.max_file_bytes:
            self.refusals += 1
            raise SandboxViolation(f"refused oversized file: {resolved.stat().st_size} bytes")
        return resolved

    @staticmethod
    def _inside(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
        except ValueError:
            return False
        return True

    def files(self):
        found = []
        for root in self.roots:
            for path in sorted(root.rglob("*")):
                if len(found) >= self.max_files:
                    return found
                if not path.is_file():
                    continue
                if path.suffix.lower() not in self.ALLOWED_SUFFIXES:
                    continue
                try:
                    if path.stat().st_size > self.max_file_bytes:
                        continue
                except OSError:
                    continue
                found.append(path)
        return found

    # ------------------------------------------------------------- retrieval
    @staticmethod
    def _terms(text: str):
        return [
            token
            for token in re.findall(r"[A-Za-z0-9_]+", str(text).lower())
            if len(token) > 2
        ]

    def scan(self, query: str, *, limit: int = 6):
        """Rank allow-listed passages against the query by term coverage."""
        wanted = set(self._terms(query))
        if not wanted:
            return []
        scored = []
        for path in self.files():
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            self.reads += 1
            offset = 0
            for line in text.splitlines():
                content = line.strip()
                offset += 1
                if not content:
                    continue
                terms = set(self._terms(content))
                if not terms:
                    continue
                overlap = len(wanted & terms)
                if overlap == 0:
                    continue
                scored.append(
                    {
                        "source": str(path),
                        "line": int(offset),
                        "text": content,
                        "score": float(overlap) / float(len(wanted)),
                        "overlap": int(overlap),
                    }
                )
        scored.sort(key=lambda row: (-float(row["score"]), str(row["source"]), int(row["line"])))
        # Keep at most two passages per source so one verbose file cannot fill
        # the whole evidence set.
        seen = {}
        kept = []
        for row in scored:
            count = seen.get(str(row["source"]), 0)
            if count >= 2:
                continue
            seen[str(row["source"])] = count + 1
            kept.append(row)
            if len(kept) >= int(limit):
                break
        return kept

    # ---------------------------------------------------------- crystallise
    def browse_and_crystallize(
        self,
        query: str,
        namespace=None,
        *,
        path=None,
        limit: int = 4,
        min_score: float = 0.34,
    ):
        """Ground a query in the allow-listed corpus and crystallise it.

        A query with no passage at or above min_score is NOT answered: it
        returns grounded=False and writes nothing, which is the fail-closed
        behaviour the C1 rule requires.
        """
        namespace = str(namespace or self.namespace)
        started = time.perf_counter()
        try:
            if path is not None:
                resolved = self.resolve(path)
                passages = []
                text = resolved.read_text(encoding="utf-8", errors="replace")
                wanted = set(self._terms(query))
                for offset, line in enumerate(text.splitlines(), start=1):
                    terms = set(self._terms(line))
                    if not terms or not wanted:
                        continue
                    overlap = len(wanted & terms)
                    if overlap:
                        passages.append(
                            {
                                "source": str(resolved),
                                "line": int(offset),
                                "text": line.strip(),
                                "score": float(overlap) / float(len(wanted)),
                                "overlap": int(overlap),
                            }
                        )
                passages.sort(key=lambda row: -float(row["score"]))
                passages = passages[: max(int(limit), 1)]
            else:
                passages = self.scan(query, limit=limit)
        except SandboxViolation as error:
            return {
                "grounded": False,
                "answered": False,
                "refused": True,
                "reason": str(error),
                "query": str(query),
                "written": 0,
                "receipt": "",
                "elapsed_ms": float((time.perf_counter() - started) * 1e3),
            }
        accepted = [row for row in passages if float(row["score"]) >= float(min_score)]
        if not accepted:
            self.history.append(
                {"query": str(query), "grounded": False, "written": 0}
            )
            return {
                "grounded": False,
                "answered": False,
                "refused": False,
                "reason": "no-grounded-passage",
                "query": str(query),
                "written": 0,
                "passages": passages,
                "receipt": "",
                "elapsed_ms": float((time.perf_counter() - started) * 1e3),
            }
        written_ids = []
        triples = []
        for row in accepted:
            text = str(row["text"])
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if self.memory is not None:
                generator = np.random.default_rng(int(digest[:16], 16) % (2**63 - 1))
                dim = int(getattr(self.memory, "dim", 1024))
                raw = generator.standard_normal(dim) + 1j * generator.standard_normal(dim)
                wave = (raw / max(float(np.linalg.norm(raw)), 1e-12)).astype(np.complex64)
                record = self.memory.write(
                    wave,
                    label=int(digest[:6], 16) % 4096,
                    tier=2,
                    namespace=namespace,
                )
                written_ids.append(int(record.engram_id))
                self.writes += 1
            triples.extend(self._triples(str(row["source"]), int(row["line"]), text))
        answer = str(accepted[0]["text"])
        payload = {
            "query": str(query),
            "namespace": namespace,
            "sources": [str(row["source"]) for row in accepted],
            "lines": [int(row["line"]) for row in accepted],
            "passage_sha256": [
                hashlib.sha256(str(row["text"]).encode("utf-8")).hexdigest()
                for row in accepted
            ],
            "written": len(written_ids),
            "previous": self._previous,
        }
        receipt = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        self._previous = receipt
        self.history.append({"grounded": True, "receipt": receipt, **payload})
        return {
            "grounded": True,
            "answered": True,
            "refused": False,
            "reason": "",
            "query": str(query),
            "answer": answer,
            "passages": accepted,
            "triples": triples[:32],
            "engram_ids": written_ids,
            "written": len(written_ids),
            "receipt": receipt,
            "chain_length": len(self.history),
            "elapsed_ms": float((time.perf_counter() - started) * 1e3),
        }

    @staticmethod
    def _triples(source: str, line: int, text: str):
        """Deterministic (subject, relation, object) extraction from one line."""
        out = []
        separator = "=" if "=" in text and ":" not in text.split("=")[0] else ":"
        if separator in text:
            subject = text.split(separator, 1)[0].strip()
            rest = text.split(separator, 1)[1].strip()
            value = rest.split(",")[0].strip()
            if subject and value:
                out.append(
                    {
                        "subject": subject,
                        "relation": "has_value",
                        "object": value,
                        "source": str(source),
                        "line": str(int(line)),
                    }
                )
        return out

    def report(self):
        return {
            "roots": [str(root) for root in self.roots],
            "files_visible": int(len(self.files())),
            "reads": int(self.reads),
            "writes": int(self.writes),
            "refusals": int(self.refusals),
            "grounded_calls": int(sum(1 for row in self.history if row.get("grounded"))),
            "chain_length": int(len(self.history)),
            "head": str(self._previous),
        }