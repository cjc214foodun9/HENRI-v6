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
import math
import re
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

    # --------------------------------------------------------------- search
    def bm25(self, claim: str, top_k: int = 4) -> List[Tuple[float, int]]:
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
