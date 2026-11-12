"""Pinned language-evaluation split (Directive Z1).

The corpus is this repository, so appending a documentation section used to
move the held-out token stream.  This module pins the split to the byte
digest of every package source file and EXCLUDES docs/HOLARCH.md, so a
documentation edit cannot move the evaluation.  The pinned split digest is a
sha256 over the sorted (name, sha256) pairs and the declared rule; the live
corpus digest is recorded next to it so the receipt shows that the documents
moved while the pinned split did not.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from henri.holarch.code_lang import CodeTextCorpus, build_tokenizer

RULE = {
    "source": "henri/holarch/**/*.py",
    "excludes": ["docs/HOLARCH.md"],
    "val_fraction": 0.1,
    "boundary_gap": 6,
    "split": "per-document contiguous tail with a 6-token boundary gap",
}
"""The declared split rule.  It is part of the pinned digest."""


def pinned_paths(root: str | Path) -> List[Path]:
    base = Path(root) / "henri" / "holarch"
    return sorted(path for path in base.rglob("*.py") if path.is_file())


def file_records(paths: Sequence[Path], root: str | Path) -> List[Dict[str, Any]]:
    base = Path(root)
    records: List[Dict[str, Any]] = []
    for path in sorted(paths):
        data = path.read_bytes()
        records.append(
            {
                "name": path.relative_to(base).as_posix(),
                "bytes": int(len(data)),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    return records


def split_digest(records: Sequence[Dict[str, Any]], rule: Dict[str, Any] = RULE) -> str:
    payload = json.dumps(
        {"files": [dict(row) for row in records], "rule": dict(rule)},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class PinnedSplit:
    """The pinned corpus, its tokenizer, and the pinned train/val split."""

    records: List[Dict[str, Any]]
    digest: str
    live_corpus_digest: str
    train_tokens: int
    val_tokens: int
    segments: List[Tuple[str, List[int], List[int]]]
    train_ids: List[int] = field(default_factory=list)
    val_ids: List[int] = field(default_factory=list)
    val_fingerprint: List[int] = field(default_factory=list)

    def payload(self) -> Dict[str, Any]:
        return {
            "stage": "Z1-pinned-split",
            "rule": dict(RULE),
            "files": [dict(row) for row in self.records],
            "file_count": int(len(self.records)),
            "digest": self.digest,
            "live_corpus_digest": self.live_corpus_digest,
            "train_tokens": int(self.train_tokens),
            "val_tokens": int(self.val_tokens),
            "documents": int(len(self.segments)),
            "val_fingerprint": [int(value) for value in self.val_fingerprint],
            "note": (
                "docs/HOLARCH.md is excluded, so appending a section cannot move "
                "this split; the live corpus digest is recorded to show the difference"
            ),
        }


def build_pinned_split(
    root: str | Path,
    *,
    val_fraction: float = 0.1,
    boundary_gap: int = 6,
    fingerprint: int = 16,
) -> PinnedSplit:
    base = Path(root)
    paths = pinned_paths(base)
    records = file_records(paths, base)
    corpus = CodeTextCorpus.from_paths(paths)
    tokenizer = build_tokenizer(corpus)
    segments = corpus.stratified_segments(
        tokenizer, val_fraction=val_fraction, boundary_gap=boundary_gap
    )
    train_ids: List[int] = []
    val_ids: List[int] = []
    for _name, train, val in segments:
        train_ids.extend(int(token) for token in train)
        val_ids.extend(int(token) for token in val)
    live = CodeTextCorpus.holarch(base)
    return PinnedSplit(
        records=records,
        digest=split_digest(records),
        live_corpus_digest=str(getattr(live, "digest", "")),
        train_tokens=int(len(train_ids)),
        val_tokens=int(len(val_ids)),
        segments=[(str(name), list(train), list(val)) for name, train, val in segments],
        train_ids=train_ids,
        val_ids=val_ids,
        val_fingerprint=val_ids[: int(fingerprint)],
    )


def verify_pinned(root: str | Path, recorded: Dict[str, Any]) -> Dict[str, Any]:
    """Recompute the pinned digests and compare them with the recorded ones."""
    base = Path(root)
    records = file_records(pinned_paths(base), base)
    current = split_digest(records)
    expected = str(recorded.get("digest", ""))
    recorded_files = {
        str(row.get("name")): str(row.get("sha256", ""))
        for row in (recorded.get("files") or [])
    }
    mismatched = sorted(
        row["name"] for row in records
        if recorded_files.get(row["name"], "") != row["sha256"]
    )
    missing = sorted(set(recorded_files) - {row["name"] for row in records})
    added = sorted({row["name"] for row in records} - set(recorded_files))
    live = CodeTextCorpus.holarch(base)
    return {
        "digest_recorded": expected,
        "digest_current": current,
        "invariant": bool(expected and expected == current),
        "mismatched_files": mismatched[:16],
        "missing_files": missing[:16],
        "added_files": added[:16],
        "live_corpus_digest_recorded": str(recorded.get("live_corpus_digest", "")),
        "live_corpus_digest_current": str(getattr(live, "digest", "")),
        "live_corpus_moved": bool(
            str(recorded.get("live_corpus_digest", "")) != str(getattr(live, "digest", ""))
        ),
    }
