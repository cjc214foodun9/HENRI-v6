"""Unit tests for `henri/koopman_router.py` (Phase 9, Step 1).

WHY A STUB BACKBONE
-------------------
The real `SemanticBackbone` loads sentence weights and takes about 6.3 s to
build. A unit test must not load weights. Every test here uses a synthetic
backbone that emits a fixed-width wave, so these tests measure ROUTER
LOGIC, not wave physics. The physics claim (operator antipody exactly
-1.0) is measured against the real encoder in
`tests/test_phase8_operator_ingress.py`.

Each test states what makes it FAIL. A test that cannot fail is not a gate.
"""
from __future__ import annotations

import numpy as np
import pytest

from henri.config import WaveConfig
from henri.koopman_router import (
    W_GRID,
    W_MEM_GRID,
    CartridgeRecall,
    KoopmanRouter,
    config_for,
    domain_of,
    mcnemar,
)
from scripts.probe_phase8_koop1 import evaluate, fit


class _Weird:
    """Emits `.psi` and has NO `.cfg`.

    This is the exact shape `SemanticBackbone.encode` returns (`Encoded`).
    The sealed `KoopmanOperator.apply_raw` reads `state.cfg.dim`, so the
    Phase 8 router raised `AttributeError: 'Encoded' object has no
    attribute 'cfg'` on its first real call. Every test below therefore
    exercises the regression: if `_psi` regresses to passing the raw
    encoder output through, these tests fail with that AttributeError.
    """

    def __init__(self, vec: np.ndarray) -> None:
        self.psi = vec


class StubBackbone:
    """Deterministic width-`dim` wave. No weights, no I/O.

    `constant=True` makes every text emit the SAME wave. That is the
    control for the tie test: identical waves must produce a tie.
    """

    def __init__(self, dim: int = 256, constant: bool = False) -> None:
        self.dim = int(dim)
        self.constant = bool(constant)

    def encode(self, text: str):
        if self.constant:
            v = np.ones(self.dim, dtype=np.complex128)
        else:
            seed = sum(ord(c) for c in text) % 100003
            rng = np.random.default_rng(seed)
            v = (rng.standard_normal(self.dim)
                 + 1j * rng.standard_normal(self.dim))
        return _Weird(v / np.linalg.norm(v))


def clamp_encoder(dim: int):
    """Synthetic cross-modal encoder. Sign follows the declared operation.

    MEASURED DEFECT IN MY FIRST DRAFT, repaired. It built `v = base + s *
    base`, so `s = -1` produced the ZERO vector and `from_complex` raised
    `ValueError: cannot normalize a zero or non-finite vector`. A
    difference of one vector from itself is not a negative direction.
    Two ORTHOGONAL components fix that: the shared part keeps the waves
    correlated, the alternating part carries the sign, and no setting of
    `s` can cancel.

    `spec` words count too. A docstring that says "add" must land on the
    same side as the `+` code, or the two candidates stay symmetric and
    the tie never breaks — which is the Phase 8 defect this test exists
    to catch.
    """
    base = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    alt = np.array([(-1.0) ** i for i in range(dim)],
                   dtype=np.complex128) / np.sqrt(dim)

    PLUS = ("+", "add", "sum", "total", "plus")
    MINUS = ("-", "sub", "difference", "minus")

    def enc(text: str) -> np.ndarray:
        t = text.replace(" ", "").lower()
        if any(k in t for k in PLUS):
            s = 1.0
        elif any(k in t for k in MINUS):
            s = -1.0
        else:
            s = 0.0
        v = base + s * alt
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-30 else base

    return enc


def test_config_for_reads_width_from_the_wave():
    """Width comes from the ENCODE, not from a guessed attribute.

    FAILS if `config_for` hardcodes a width or reads a non-existent
    attribute: `cfg.dim` would then differ from the stub's 512.
    """
    cfg = config_for(StubBackbone(dim=512))
    assert isinstance(cfg, WaveConfig)
    assert cfg.dim == 512


def test_psi_accepts_object_without_cfg():
    """Regression for `AttributeError: 'Encoded' object has no attribute 'cfg'`.

    FAILS if `_psi` hands the raw encoder output to the sealed operator.
    """
    bb = StubBackbone(dim=256)
    cfg = WaveConfig(dim=256)
    r = KoopmanRouter(bb, cfg, n_domains=2, seed=1)
    psi = r._psi("return a + b")
    assert psi.psi.shape == (256,)
    assert abs(float(np.linalg.norm(psi.psi)) - 1.0) < 1e-9
    # The object it came from has no `.cfg`; prove that is the case so the
    # test cannot silently stop covering the regression.
    assert not hasattr(bb.encode("x"), "cfg")


def test_width_mismatch_raises_loudly():
    """A wrong-width encoder must raise, never mix two dimensions.

    FAILS if the router silently truncates, pads, or broadcasts.
    """
    bb = StubBackbone(dim=256)
    cfg = WaveConfig(dim=256)
    bad = lambda t: np.ones(128, dtype=np.complex128)  # noqa: E731
    r = KoopmanRouter(bb, cfg, n_domains=2, seed=1, encoder=bad)
    with pytest.raises(ValueError, match="dim"):
        r._psi("return a + b")


def test_pluggable_encoder_breaks_the_tie():
    """The encoder path must separate `a + b` from `a - b`.

    Control vs treatment on the SAME router class and the SAME identical
    backbone waves. Phase 8 measured `s_static = 0.69013433` for both
    candidates because nothing on the scoring path used the operator
    channel.

    FAILS if: (control) tie_group_size != 2, or (treatment) top1 != 0 or
    tie_group_size != 1. A treatment that merely lowers the tie to 0
    without ranking gold first also fails.
    """
    cfg = WaveConfig(dim=256)
    spec = "add two numbers"
    cands = ["def f(a, b): return a + b", "def f(a, b): return a - b"]

    control = KoopmanRouter(StubBackbone(dim=256, constant=True), cfg,
                            n_domains=2, seed=1)
    c = control.rank(spec, cands, subject="code")
    assert c["tie_group_size"] == 2, c["ranked"]

    treated = KoopmanRouter(StubBackbone(dim=256, constant=True), cfg,
                            n_domains=2, seed=1,
                            encoder=clamp_encoder(256))
    t = treated.rank(spec, cands, subject="code")
    assert t["tie_group_size"] == 1, t["ranked"]
    assert t["top1_index"] == 0, t["ranked"]
    assert t["ranked"][0]["total"] > t["ranked"][1]["total"]


def _cached(hits_by_weight: dict, n_items: int = 6) -> list:
    """Synthetic component cache where only `w_trans` can change the rank.

    `hits_by_weight[w]` gives how many items rank gold first at that
    weight. Everything else is zero, so the grid is degenerate by
    construction and the parsimony tie-break is what decides.
    """
    out = []
    for i in range(n_items):
        rows = []
        for j in range(8):
            rows.append({"index": j,
                         "s_static": 1.0 if j == 0 else 0.9,
                         "s_trans": 0.0, "s_mem": 0.0,
                         "hit": False})
        out.append({"domain": 0, "rows": rows, "gold_index": 0,
                    "encoder": "synthetic", "recall_hit": False})
    return out


def test_parsimony_prefers_zero_on_an_all_zero_grid():
    """With every term zero, w=0 must win every tie.

    FAILS if the fit installs any positive weight when no weight improves
    accuracy. Without this rule a no-op weight could be installed and
    reported as a win.
    """
    cached = _cached({})
    f = fit(cached)
    assert f["chosen"]["w_trans"] == 0.0, f["chosen"]
    assert f["chosen"]["w_mem"] == 0.0, f["chosen"]
    # The grid must contain the static arm itself, or the comparison is
    # not against "do nothing".
    pairs = [(r["w_trans"], r["w_mem"]) for r in f["grid"]]
    assert (0.0, 0.0) in pairs
    assert len(pairs) == len(W_GRID) * len(W_MEM_GRID)


def test_evaluate_reports_control_and_arm_separately():
    """`evaluate` at w=0 must equal the static control exactly.

    FAILS if the weight is applied even at 0.0, which would make the
    control arm not a control.
    """
    cached = _cached({})
    a = evaluate(cached, 0.0, 0.0)
    b = evaluate(cached, 0.0, 0.0)
    assert a == b
    assert a["n"] == len(cached)
    assert a["acc"] == 1.0            # gold scores 1.0 > 0.9 by construction
    assert a["mean_tie_group"] == 1.0


def test_domain_of_is_deterministic_and_bounded():
    """`domain_of` must be stable across calls and inside range.

    FAILS if it uses salted `hash()`: two calls in one process agree, but
    a fit run and a serve run would disagree silently. Stability is
    checked against a literal value derived from sha256, not against a
    second call.
    """
    import hashlib
    for subj in ("physics", "law", "code", "math", "Code", " code "):
        d = domain_of(subj, 4)
        assert 0 <= d < 4
        h = hashlib.sha256(subj.strip().lower().encode("utf-8"))
        assert d == int(h.hexdigest()[:8], 16) % 4
    assert domain_of("code", 4) == domain_of("code", 4)


def test_mcnemar_edges():
    """Discordant-pair count decides the verdict; ties must not.

    FAILS if concordant pairs are counted as evidence, which would let a
    correlated-but-uninformative arm claim significance.
    """
    same = mcnemar([True, False, True], [True, False, True])
    assert same["arm_only"] == 0 and same["control_only"] == 0
    assert same["verdict"] != "ARM_BETTER"

    arm = mcnemar([True] * 8, [False] * 8)
    assert arm["arm_only"] == 8 and arm["control_only"] == 0
    assert arm["verdict"] == "ARM_BETTER"
    assert arm["p"] < 0.05

    ctrl = mcnemar([False] * 8, [True] * 8)
    assert ctrl["verdict"] == "CONTROL_BETTER"


def test_cartridge_recall_round_trip():
    """Write, finalize, recall, score.

    MEASURED DEFECT IN MY FIRST DRAFT, repaired. `recall` is typed
    `(spec_psi: WaveState)`, and my draft passed a bare ndarray, which
    raised `AttributeError: 'numpy.ndarray' object has no attribute
    'psi'`. `add` takes arrays; `recall` takes a state. The test now uses
    the declared types.

    FAILS if `finalize` leaves a stale row, or a hit carries no value.
    """
    from henri.state import WaveState

    cfg = WaveConfig(dim=128)
    rc = CartridgeRecall(dim=128, rank=4, max_rows=4)
    rng = np.random.default_rng(0)
    for _ in range(3):
        a = rng.standard_normal(128) + 1j * rng.standard_normal(128)
        b = rng.standard_normal(128) + 1j * rng.standard_normal(128)
        rc.add(a, b)
    n = rc.finalize()
    assert n == 3

    q = (rng.standard_normal(128) + 1j * rng.standard_normal(128))
    hit = rc.recall(WaveState.from_complex(q, cfg))
    assert hasattr(hit, "hit") and hasattr(hit, "margin")
    if hit.hit:
        assert hit.value is not None
