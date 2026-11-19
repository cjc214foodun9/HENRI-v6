"""Vendor a deterministic slice of the official public ARC-AGI dataset.

    python scripts/vendor_arc_agi_public.py --source arc_tmp

Rule (declared before any task is inspected): take the union of the official
training/ and evaluation/ JSON files, sort by filename, and keep every
16th file starting at index 0 until 48 tasks are held.  No task is filtered
by size, colour count or difficulty.  Every file's SHA-256 and the upstream
commit are recorded beside the copy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "receipts" / "holarch" / "external" / "arc_agi_public"
COUNT = 48
STRIDE = 16


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Vendor the ARC-AGI public slice")
    parser.add_argument("--source", default="arc_tmp")
    args = parser.parse_args(argv)
    source = Path(args.source)
    if not (source / "data" / "training").exists():
        print("[arc] source not found:", source)
        return 1
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
        remote = subprocess.check_output(
            ["git", "-C", str(source), "remote", "get-url", "origin"], text=True
        ).strip()
    except Exception:  # noqa: BLE001 - provenance is recorded as unknown, not invented
        commit, remote = "", ""
    files = sorted(
        (source / "data" / "training").glob("*.json")
    ) + sorted((source / "data" / "evaluation").glob("*.json"))
    names = sorted(path.name for path in files)
    by_name = {path.name: path for path in files}
    selected = [names[index] for index in range(0, len(names), STRIDE)][:COUNT]
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for name in selected:
        path = by_name[name]
        data = path.read_bytes()
        payload = json.loads(data.decode("utf-8"))
        target = OUT / name
        target.write_bytes(data)
        records.append(
            {
                "file": name,
                "split": "training" if path.parent.name == "training" else "evaluation",
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "train_pairs": len(payload.get("train") or []),
                "test_pairs": len(payload.get("test") or []),
            }
        )
    manifest = {
        "stage": "AA3-external-arc-agi-slice",
        "upstream": remote,
        "commit": commit,
        "dataset": "ARC-AGI (Chollet) public training/evaluation JSON",
        "rule": (
            "union of data/training and data/evaluation, sorted by filename, "
            "every 16th file from index 0, 48 tasks"
        ),
        "stride": STRIDE,
        "count": len(records),
        "files": records,
        "digest": hashlib.sha256(
            json.dumps(
                [(row["file"], row["sha256"]) for row in records],
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    }
    body = json.dumps(manifest, indent=2)
    (OUT / "MANIFEST.json").write_text(body, encoding="utf-8")
    (OUT / "MANIFEST.json.sha256").write_text(
        hashlib.sha256(body.encode("utf-8")).hexdigest() + chr(10), encoding="utf-8"
    )
    print(
        "[arc] vendored %d tasks from %s (%s) digest=%s"
        % (len(records), remote, commit[:12], manifest["digest"][:16])
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
