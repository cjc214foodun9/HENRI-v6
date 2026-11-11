"""Verify every holarch receipt sidecar and the documentation index.

Exit 0 only when every JSON receipt has a matching .sha256 sidecar, every
receipt filename appears in docs/HOLARCH.md, and Sections 1..35 are monotonic
and unique. This is the fourteenth closure command.

    python scripts/verify_holarch_consistency.py
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECEIPTS = ROOT / "receipts" / "holarch"
DOCS = ROOT / "docs" / "HOLARCH.md"


def main() -> int:
    receipts = sorted(RECEIPTS.glob("*.json"))
    invalid = []
    missing = []
    for path in receipts:
        sidecar = path.with_name(path.name + ".sha256")
        if not sidecar.exists():
            missing.append(str(path.relative_to(ROOT)))
            continue
        expected = sidecar.read_text(encoding="utf-8").strip()
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected != actual:
            invalid.append(str(path.relative_to(ROOT)))
    text = DOCS.read_text(encoding="utf-8")
    index_path = RECEIPTS / "RECEIPT_INDEX.md"
    index_text = index_path.read_text(encoding="utf-8") if index_path.exists() else ""
    undocumented = [
        path.name for path in receipts if path.name not in text and path.name not in index_text
    ]
    numbers = [int(value) for value in re.findall(r"^## (\d+)\.", text, flags=re.M)]
    monotonic = all(numbers[index] < numbers[index + 1] for index in range(len(numbers) - 1))
    unique = len(numbers) == len(set(numbers))
    sections = bool(numbers == list(range(1, 39)) and monotonic and unique)
    # Monotonic truncation guard: the section list must be contiguous from 1,
    # and the language metrics must still hold their sealed bars.
    r3 = json.loads((RECEIPTS / "r3_static_and_streaming_full_decile_language.json")
                    .read_text(encoding="utf-8")) if (
        RECEIPTS / "r3_static_and_streaming_full_decile_language.json").exists() else {}
    decile = dict(r3.get("full_decile") or {})
    streaming_bpc = float(decile.get("bits_per_char", 9.9))
    static_bpc = float(decile.get("static_bits_per_char", 9.9))
    language_guard = bool(streaming_bpc <= 1.2200 and static_bpc <= 1.4200)
    o_t_archive = (ROOT / "docs" / "HOLARCH_CYCLES_O_TO_T.md").exists()
    payload = {
        "receipts": int(len(receipts)),
        "sidecars_present": int(len(receipts) - len(missing)),
        "sidecars_valid": int(len(receipts) - len(missing) - len(invalid)),
        "missing_sidecars": missing,
        "invalid_sidecars": invalid,
        "undocumented_receipts": undocumented,
        "doc_sections": numbers,
        "sections_1_to_38": sections,
        "r3_streaming_bits_per_char": streaming_bpc,
        "r3_static_bits_per_char": static_bpc,
        "language_guard": language_guard,
        "o_to_t_archive_present": bool(o_t_archive),
        "valid": bool(not missing and not invalid and not undocumented
                      and sections and language_guard and o_t_archive),
    }
    print(json.dumps(payload, indent=2))
    return 0 if payload["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
