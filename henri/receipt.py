"""One receipt writer, one identity rule. Phase 10.

WHY THIS FILE EXISTS
--------------------
Phase 9 lost two rounds to a label defect. `scripts/probe_phase9_*.py`
computed a digest of the receipt body BEFORE the self-field was inserted,
stored that digest IN the field, then rewrote the file. The print
statement labelled the stored value `receipt sha256`. A reader running
`sha256sum` on the file got a DIFFERENT number, because the file had
changed at the rewrite. Four receipts carried the wrong label, and the
value was copied into the register and two commit messages.

The arithmetic is not a defect. Two identities are both real:

    body_sha256_self_field  = SHA256(canonical body WITHOUT the field)
    file_sha256_on_disk     = SHA256(the bytes on disk, WITH the field)

They MUST differ, because SHA256 is collision-resistant and the field
adds bytes. Recording both is correct. Labelling one as the other is not.
Four hand-rolled copies of a hash-then-rewrite sequence is how the label
defect spread, so Phase 10 has exactly one implementation and every probe
calls it.

WHAT THIS DOES NOT PROVE
------------------------
A self-digest proves payload-to-field linkage. It does not prove the field
is unaltered: an editor can rewrite body and field together. It is not a
signature and not a MAC. Stated here so a reader does not over-read it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

BODY_FIELD = "receipt_sha256_of_body_before_self_field"
FILE_FIELD = "file_sha256_on_disk"


def _canonical(body: dict) -> bytes:
    """Byte form used for BOTH hashes. Sort keys, no space drift."""
    return json.dumps(body, indent=2, sort_keys=True,
                      default=float).encode("utf-8")


def write_receipt(path: str | Path, body: dict[str, Any]) -> dict[str, str]:
    """Write `body` with both identities, then read the file back.

    Returns `{BODY_FIELD, FILE_FIELD, "path"}`. The readback is not
    decoration: a receipt that was written but not re-read cannot support
    a claim about what is on disk.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    payload = {k: v for k, v in body.items() if k != BODY_FIELD}
    body_sha = hashlib.sha256(_canonical(payload)).hexdigest()
    payload[BODY_FIELD] = body_sha

    out.write_bytes(_canonical(payload))
    file_sha = hashlib.sha256(out.read_bytes()).hexdigest()

    # Second write: the file hash now describes a file that CONTAINS the
    # file hash. Re-hashing after this write is what makes the printed
    # value reproducible by `sha256sum`.
    payload[FILE_FIELD] = file_sha
    out.write_bytes(_canonical(payload))
    final = hashlib.sha256(out.read_bytes()).hexdigest()

    return {BODY_FIELD: body_sha, FILE_FIELD: final, "path": str(out)}


def verify_receipt(path: str | Path) -> dict:
    """Recompute both identities from the file on disk.

    `body_match` strips the self-field and re-canonicalises, exactly the
    operation the field name declares. `file_match` hashes the bytes.
    """
    p = Path(path)
    raw = p.read_bytes()
    doc = json.loads(raw.decode("utf-8"))
    stored_body = doc.pop(BODY_FIELD, None)
    stored_file = doc.pop(FILE_FIELD, None)
    body_sha = hashlib.sha256(_canonical(doc)).hexdigest()
    file_sha = hashlib.sha256(raw).hexdigest()
    # NOTE: `file_sha` cannot equal `stored_file` here, because popping the
    # two fields changes the bytes. It is reported for the record only.
    return {
        "path": str(p),
        "body_sha256_self_field": stored_body,
        "body_recomputed": body_sha,
        "body_match": bool(stored_body == body_sha),
        "file_sha256_on_disk": stored_file,
        "file_recomputed_after_pop": file_sha,
        "file_note": ("recomputed value differs by construction: this "
                      "function removed the two identity fields before "
                      "hashing. Use `sha256sum` on the file for the "
                      "external identity."),
    }


def print_identities(ids: dict[str, str]) -> None:
    """Print both labels on separate lines, spelled out in full."""
    print("  %-24s %s" % (BODY_FIELD, ids[BODY_FIELD]))
    print("  %-24s %s" % (FILE_FIELD, ids[FILE_FIELD]))


__all__ = ["write_receipt", "verify_receipt", "print_identities",
           "BODY_FIELD", "FILE_FIELD"]
