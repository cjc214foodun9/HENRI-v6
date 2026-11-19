"""Freeze the pinned evaluation corpus into an immutable snapshot (Cycle AA).

    python scripts/snapshot_pinned_corpus.py

The Z1 pin hashed the LIVE tree, so any edit to a package file moved the
evaluation.  Cycle AA freezes the exact bytes the digest 8f270f34... was
computed over, plus the token ids the split produced.  Every later cycle loads
this snapshot, so a code edit cannot move the held-out token stream.

Nothing is written unless the recomputed digest equals the recorded one.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from henri.holarch.code_lang import CodeTextCorpus, build_tokenizer
from henri.holarch.pinned_split import RULE, build_pinned_split, file_records, pinned_paths

EXPECTED = "8f270f34e8f35ef84e415d076bea20a7bc15edff337f89ed72ad607447ba8554"
OUT = ROOT / "receipts" / "holarch" / "pinned_corpus_snapshot.json"

def main() -> int:
    split = build_pinned_split(ROOT)
    if split.digest != EXPECTED:
        print("[snapshot] REFUSED digest=%s expected=%s" % (split.digest, EXPECTED))
        return 1
    paths = pinned_paths(ROOT)
    texts = [path.read_text(encoding="utf-8") for path in paths]
    payload = {
        "stage": "AA-pinned-corpus-snapshot",
        "note": (
            "Frozen bytes of the 38 package files the pinned digest was computed "
            "over. docs/HOLARCH.md is excluded. Evaluation loads THIS snapshot, so "
            "code edits cannot move the held-out stream."
        ),
        "rule": dict(RULE),
        "digest": split.digest,
        "live_corpus_digest": split.live_corpus_digest,
        "train_tokens": int(split.train_tokens),
        "val_tokens": int(split.val_tokens),
        "files": [
            {"name": record["name"], "bytes": record["bytes"],
             "sha256": record["sha256"], "text": text}
            for record, text in zip(split.records, texts)
        ],
        "segments": [
            {"name": name, "train": [int(t) for t in train], "val": [int(t) for t in val]}
            for name, train, val in split.segments
        ],
    }
    body = json.dumps(payload, separators=(",", ":"))
    OUT.write_text(body, encoding="utf-8")
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    OUT.with_name(OUT.name + ".sha256").write_text(digest + chr(10), encoding="utf-8")
    print("[snapshot] digest=%s files=%d train=%d val=%d bytes=%d sha256=%s" % (
        split.digest, len(payload["files"]), split.train_tokens, split.val_tokens,
        len(body), digest[:16]))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
