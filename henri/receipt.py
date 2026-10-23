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
FILE_FIELD_LABEL = "file_sha256_on_disk_note"
SIDECAR_SUFFIX = ".sha256"

# -- MEASURED DEFECT IN THE FIRST VERSION OF THIS FILE ------------------
#
# The first version wrote `FILE_FIELD`, then hashed the file and RETURNED
# that hash. The returned value was correct. The EMBEDDED value was the
# hash of the serialization BEFORE the field was inserted, because the
# field's own bytes could not be part of it.
#
# MEASURED, before the fix, on all four Phase 10 receipts: embedded
# `file_sha256_on_disk` did not equal `sha256sum` of the file. That is the
# exact Phase 9 label defect, reproduced inside the module written to
# prevent it.
#
# The arithmetic is not repairable by iteration in general: a file cannot
# contain its own SHA256. So the reproducible identity moves to a SIDECAR,
# and the embedded field carries a label saying what it actually is.
FILE_FIELD_NOTE = (
    "hash of the serialization BEFORE this field was inserted. A file "
    "cannot contain its own SHA256. The reproducible external identity "
    "is in the sibling '<name>.sha256' sidecar, verifiable with "
    "`sha256sum -c`.")


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
    payload[FILE_FIELD_LABEL] = FILE_FIELD_NOTE
    out.write_bytes(_canonical(payload))
    final = hashlib.sha256(out.read_bytes()).hexdigest()

    # THIRD ARTIFACT: the only value a reader can reproduce. `sha256sum -c`
    # on this sidecar either passes or fails, with no interpretation.
    sidecar = out.with_suffix(out.suffix + SIDECAR_SUFFIX)
    # MEASURED DEFECT: `write_text("%s  %s\n")` becomes CRLF on Windows, so
    # the sidecar line reads `abc...  name.json\r`. `sha256sum -c` then fails
    # with `FAILED open or read` because the filename carries a stray CR.
    # A sidecar no reader can check is not a reproducible identity, so the
    # newline is pinned to LF and the bytes are written explicitly.
    sidecar.write_bytes(("%s  %s\n" % (final, out.name)).encode("utf-8"))

    return {BODY_FIELD: body_sha, FILE_FIELD: final, "path": str(out),
            "sidecar": str(sidecar)}


def verify_receipt(path: str | Path) -> dict:
    """Recompute both identities from the file on disk.

    `body_match` strips the self-field and re-canonicalises, exactly the
    operation the field name declares. `file_match` checks the SIDECAR,
    because that is the only identity a reader can reproduce: a file
    cannot contain its own SHA256.
    """
    p = Path(path)
    raw = p.read_bytes()
    doc = json.loads(raw.decode("utf-8"))
    stored_body = doc.pop(BODY_FIELD, None)
    stored_file = doc.pop(FILE_FIELD, None)
    doc.pop(FILE_FIELD_LABEL, None)
    body_sha = hashlib.sha256(_canonical(doc)).hexdigest()
    actual = hashlib.sha256(raw).hexdigest()

    sidecar = p.with_suffix(p.suffix + SIDECAR_SUFFIX)
    claimed = None
    if sidecar.exists():
        text = sidecar.read_text(encoding="utf-8").split()
        claimed = text[0] if text else None

    return {
        "path": str(p),
        "body_sha256_self_field": stored_body,
        "body_recomputed": body_sha,
        "body_match": bool(stored_body == body_sha),
        "file_sha256_actual": actual,
        "file_sha256_sidecar": claimed,
        "file_match": bool(claimed == actual),
        "sidecar": str(sidecar) if sidecar.exists() else None,
        "embedded_file_field_note": (
            "the embedded `file_sha256_on_disk` is the PRE-INSERTION hash "
            "and is NOT reproducible. Trust the sidecar."),
    }


def print_identities(ids: dict[str, str]) -> None:
    """Print both labels on separate lines, spelled out in full."""
    print("  %-24s %s" % (BODY_FIELD, ids[BODY_FIELD]))
    print("  %-24s %s" % (FILE_FIELD, ids[FILE_FIELD]))


__all__ = ["write_receipt", "verify_receipt", "print_identities",
           "BODY_FIELD", "FILE_FIELD"]
