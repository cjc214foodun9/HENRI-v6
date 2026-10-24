"""G-SHAPE1 unit layer: argument-shape adaptation.

WHY THIS EXISTS
Register 24 measured the `exception` bucket as 87/105 `TypeError`, with
observed messages `'int' object is not subscriptable` and `'int' object is
not iterable`. That is argument SHAPE. `adapt_arity` resizes the parameter
LIST, which cannot fix a body that subscripts a scalar it was handed.

DESIGN CONSTRAINT PINNED HERE: ADDITIVE. `adapt_shape` returns a NEW source
and never edits the template. The caller keeps the original candidate and
offers the guarded form as an extra. A pass must not be creditable to a
mechanism that deleted the working candidate.

WHAT THIS FILE DELIBERATELY DOES NOT CLAIM
A guard makes the CALL legal. It does not make the function CORRECT. The
`test_guard_legalises_but_does_not_repair_arithmetic` case pins that
distinction so a later reader cannot read a PASS as semantic repair.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from henri.code_synth import (          # noqa: E402
    SHAPE_ADAPT_ENV,
    adapt_shape,
    shape_adapt_enabled,
    shape_use_sites,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(SHAPE_ADAPT_ENV, raising=False)


def run_source(src: str, calls: str):
    """Execute `calls` against `src` in a fresh interpreter. Returns (rc, out)."""
    program = src + "\n\n" + textwrap.dedent(calls) + "\n"
    r = subprocess.run([sys.executable, "-c", program],
                       capture_output=True, text=True, timeout=60)
    return r.returncode, (r.stdout + r.stderr).strip()


# -- detection -------------------------------------------------------------

def test_default_off():
    """A new mechanism ships OFF. A default-ON mechanism would silently
    change every existing synthesis receipt."""
    assert shape_adapt_enabled() is False


def test_switch_reads_live_at_dispatch():
    """MEASURED DEFECT from item 1, pinned here.

    An import-cached switch makes both arms of an ablation execute identical
    code. A zero delta then cannot be read as "the mechanism adds nothing"
    rather than "the ablation never fired".
    """
    seen = (shape_adapt_enabled(),)
    os.environ[SHAPE_ADAPT_ENV] = "1"
    seen += (shape_adapt_enabled(),)
    os.environ[SHAPE_ADAPT_ENV] = "0"
    seen += (shape_adapt_enabled(),)
    del os.environ[SHAPE_ADAPT_ENV]
    try:
        assert seen == (False, True, False)
    finally:
        os.environ.pop(SHAPE_ADAPT_ENV, None)


@pytest.mark.parametrize("value,expected", [
    ("1", True), ("true", True), ("YES", True),
    ("0", False), ("false", False), ("off", False), ("", False),
])
def test_switch_parses_values(monkeypatch, value, expected):
    monkeypatch.setenv(SHAPE_ADAPT_ENV, value)
    assert shape_adapt_enabled() is expected


def test_detects_subscript():
    assert shape_use_sites("def f(xs):\n    return xs[0]") == [("xs", "subscript")]


def test_detects_iteration():
    assert shape_use_sites(
        "def f(xs):\n    t = 0\n    for v in xs:\n        t += v\n    return t"
    ) == [("xs", "iterate")]


def test_reports_no_site_for_scalar_body():
    """A pure arithmetic body must yield NO candidate, so provenance never
    records a rewrite that did nothing."""
    assert shape_use_sites("def add(a, b):\n    return a + b") == []


def test_ignores_loop_binding_names():
    """A `for` header TARGET is written, not read. Guarding it would inject a
    store the author did not write."""
    src = "def f(xs):\n    for v in xs:\n        pass\n    return v"
    assert ("v", "iterate") not in shape_use_sites(src)


def test_ignores_locals_and_globals():
    src = "def f(x):\n    y = [1, 2]\n    return y[0] + x"
    assert shape_use_sites(src) == []


# -- rewriting -------------------------------------------------------------

def test_returns_none_when_no_site():
    assert adapt_shape("def add(a, b):\n    return a + b") is None


def test_guard_precedes_use_and_keeps_docstring_first():
    out = adapt_shape('def f(xs):\n    """doc."""\n    return xs[0]')
    assert out is not None
    body = out.split("\n")
    assert '"""doc."""' in body[1] or "doc." in "\n".join(body[:3])
    guard = next(i for i, l in enumerate(body) if "isinstance" in l)
    use = next(i for i, l in enumerate(body) if "xs[0]" in l)
    assert guard < use


def test_renames_entry_when_name_given():
    out = adapt_shape("def f(xs):\n    return xs[0]", "target_name")
    assert out is not None and "def target_name(" in out


# -- behaviour, executed ---------------------------------------------------

def test_guard_makes_scalar_call_legal():
    """The measured failure class: the test passes a scalar, the body
    subscripts it. The guarded form must survive the call."""
    adapted = adapt_shape("def f(xs):\n    return xs[0]")
    rc, out = run_source(adapted, "print(f(5))")
    assert rc == 0, out
    assert out.strip() == "5"


def test_guard_preserves_sequence_semantics():
    """Equivalence on the input the template was written for. A guard that
    changed sequence results would trade one defect for another."""
    original = "def f(xs):\n    return xs[0]"
    adapted = adapt_shape(original)
    rc_o, out_o = run_source(original, "print(f([7, 8]))")
    rc_a, out_a = run_source(adapted, "print(f([7, 8]))")
    assert (rc_o, out_o) == (rc_a, out_a) == (0, "7")


def test_guard_legalises_but_does_not_repair_arithmetic():
    """KNOWN LIMIT, pinned so a PASS cannot be read as semantic repair.

    `def combine(a): return a` resized to two parameters legalises the call
    while the body still returns one argument. The guard has the same
    boundary: it fixes the call, not the arithmetic.
    """
    adapted = adapt_shape("def f(xs):\n    return xs[0] + 1")
    rc, out = run_source(adapted, "print(f(5))")
    assert rc == 0
    assert out.strip() == "6"          # legal call, NOT the caller's intent


def test_original_source_is_not_mutated():
    src = "def f(xs):\n    return xs[0]"
    before = src
    adapt_shape(src)
    assert src == before


# -- wiring ---------------------------------------------------------------

def test_call_site_is_additive():
    """A definition is not a measurement. Read the CALL SITE and require the
    additive branch, since a replacing branch would delete the candidate the
    guard is meant to supplement."""
    text = (ROOT / "henri" / "code_synth.py").read_text(encoding="utf-8")
    assert "if shape_adapt_enabled():" in text
    assert "shape_adapted=True" in text
    assert "base = shaped" not in text
    assert "base, adapted = shaped" not in text
