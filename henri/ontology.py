"""The immutable ontology layer.

Evidence label: DERIVED for the chain, OBSERVED for every entry a run adds.

The ontology is an append-only, hash-chained list of grounded assertions.
One entry records one fact about the system: a measurement, a gate verdict,
or a promotion decision. An entry can never be edited. A new fact is a new
entry, and every entry carries the hash of the one before it, so a tampered
history fails verification at the first altered row.

This is the layer the orchestrator updates as it works. It is deliberately
small: a fact is only admissible when it names the evidence that produced
it, and the evidence is a file path plus a SHA-256.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

GENESIS = "0" * 64


def _canonical(payload: Dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


@dataclass(frozen=True)
class Entry:
    """One immutable assertion.

    Attributes:
        seq: position in the chain, starting at 0.
        ts: unix seconds when the entry was appended.
        kind: measurement, gate, decision, or frame.
        subject: the thing the assertion is about.
        predicate: the relation, for example "measured" or "passed".
        object: the value, serialised as text.
        evidence: file path and SHA-256 of the artefact that supports it.
        prev: hash of the previous entry, or GENESIS.
        digest: hash of this entry including prev.
    """

    seq: int
    ts: float
    kind: str
    subject: str
    predicate: str
    object: str
    evidence: Dict[str, str] = field(default_factory=dict)
    prev: str = GENESIS
    digest: str = ""

    def body(self) -> Dict:
        payload = asdict(self)
        payload.pop("digest", None)
        return payload


class OntologyLedger:
    """Append-only hash-chained store on disk."""

    def __init__(self, path: str | Path = "data/holarch/ontology.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ read
    def entries(self) -> List[Entry]:
        if not self.path.exists():
            return []
        rows: List[Entry] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append(Entry(**json.loads(line)))
        return rows

    def head(self) -> str:
        rows = self.entries()
        return rows[-1].digest if rows else GENESIS

    # ----------------------------------------------------------------- write
    def append(
        self,
        kind: str,
        subject: str,
        predicate: str,
        obj: str,
        evidence: Optional[Dict[str, str]] = None,
    ) -> Entry:
        rows = self.entries()
        prev = rows[-1].digest if rows else GENESIS
        entry = Entry(
            seq=len(rows),
            ts=time.time(),
            kind=str(kind),
            subject=str(subject),
            predicate=str(predicate),
            object=str(obj),
            evidence=dict(evidence or {}),
            prev=prev,
        )
        payload = entry.body()
        payload["prev"] = prev
        digest = hashlib.sha256(_canonical(payload)).hexdigest()
        entry = Entry(**{**entry.body(), "digest": digest})
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(entry), sort_keys=True) + "\n")
        return entry

    # ----------------------------------------------------------------- verify
    def verify(self) -> Dict[str, object]:
        """Recompute the chain. Any edited row fails at its own index."""
        rows = self.entries()
        prev = GENESIS
        for index, entry in enumerate(rows):
            payload = entry.body()
            payload["prev"] = prev
            expected = hashlib.sha256(_canonical(payload)).hexdigest()
            if entry.prev != prev:
                return {"ok": False, "broken_at": index, "reason": "prev mismatch", "length": len(rows)}
            if entry.digest != expected:
                return {"ok": False, "broken_at": index, "reason": "digest mismatch", "length": len(rows)}
            prev = entry.digest
        return {"ok": True, "length": len(rows), "head": prev}

    # ------------------------------------------------------------------ query
    def query(self, subject: Optional[str] = None, kind: Optional[str] = None) -> List[Entry]:
        rows = self.entries()
        if subject is not None:
            rows = [row for row in rows if row.subject == subject]
        if kind is not None:
            rows = [row for row in rows if row.kind == kind]
        return rows


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
