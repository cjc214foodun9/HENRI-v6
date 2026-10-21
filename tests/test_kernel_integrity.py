"""Tests for the immutable-physics gate.

The gate exists to stop Boss Fight 2: an agent editing a sealed file until
a probe passes. A gate that cannot detect drift is worthless, so every
positive test is paired with a tamper that must be caught.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import verify_kernel_integrity as vk


def test_all_six_sealed_files_are_pinned():
    assert len(vk.PINNED) == 6
    for path in ("henri/qfhrr.py", "henri/readout.py", "henri/gram.py",
                 "henri/sagnac_homodyne_gate.py", "henri/memory.py",
                 "henri/form_pin.py"):
        assert path in vk.PINNED
        assert len(vk.PINNED[path]) == 64          # full sha256 hex


def test_repository_is_currently_sealed():
    """The real check on the real tree. Fails if a sealed file changed."""
    rec = vk.check()
    assert rec["status"] == "SEALED", rec["drift"] + rec["invariant_failures"]
    assert rec["files_checked"] == 6
    assert rec["drift"] == []
    assert rec["missing"] == []
    assert rec["invariant_failures"] == []


def test_negative_control_tampered_digest_is_caught(monkeypatch):
    """A wrong pinned digest MUST produce DRIFT, not a silent pass.

    Without this, a typo in PINNED would leave the gate open forever and
    every test would still be green.
    """
    bad = dict(vk.PINNED)
    bad["henri/readout.py"] = "0" * 64
    monkeypatch.setattr(vk, "PINNED", bad)
    rec = vk.check()
    assert rec["status"] == "DRIFT"
    assert len(rec["drift"]) == 1
    assert rec["drift"][0]["file"] == "henri/readout.py"


def test_negative_control_missing_file_is_caught(monkeypatch):
    bad = dict(vk.PINNED)
    bad["henri/does_not_exist.py"] = "a" * 64
    monkeypatch.setattr(vk, "PINNED", bad)
    rec = vk.check()
    assert rec["status"] == "DRIFT"
    assert "henri/does_not_exist.py" in rec["missing"]


def test_negative_control_absent_invariant_is_caught(monkeypatch):
    """A digest cannot express intent. Deleting `beta*` while keeping the
    bytes-derived digest consistent must still fail."""
    bad = ({"file": "henri/readout.py",
            "pattern": r"beta:\s*float\s*=\s*99\.9\b",
            "label": "deliberately_absent"},)
    monkeypatch.setattr(vk, "INVARIANTS", bad)
    rec = vk.check()
    assert rec["status"] == "DRIFT"
    assert rec["invariant_failures"][0]["reason"] == "PATTERN_NOT_FOUND"


def test_known_invariants_hold():
    """beta*, tau_veto, and the margin definition are read, not assumed."""
    readout = vk.blob_bytes("henri/readout.py").decode()
    assert "beta: float = 26.1" in readout
    assert "margin = top - second" in readout
    sagnac = vk.blob_bytes("henri/sagnac_homodyne_gate.py").decode()
    assert "TAU_VETO_CEILING = 0.0431" in sagnac


def test_blob_bytes_is_line_ending_independent():
    """The digest is over the stored blob, so CRLF cannot fake drift."""
    raw = vk.blob_bytes("henri/readout.py")
    assert raw is not None
    assert b"\r\n" not in raw
