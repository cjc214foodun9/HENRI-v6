"""Phase 5 gates. Every test can fail; two are negative controls.

The critical ones are the match/read split and the weight-invariant antonym
result. Both were discovered by measurement, not by reading the spec.
"""

from __future__ import annotations

import numpy as np
import pytest

from henri import cartridge_bank as cb
from henri.cartridge_bank import CartridgeBank, build_bank, cartridge_bytes


def _unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def _book(m, D, seed=0, share=0.0):
    rng = np.random.default_rng(seed)
    base = _unit(rng.standard_normal(D) + 0j)
    k = _unit(rng.standard_normal((m, D)) + 1j * rng.standard_normal((m, D))
              + share * 90 * base)
    return k


# -- Action B: the match/read split --------------------------------------

def test_raw_query_against_ortho_keys_mixes_argmax():
    """NEGATIVE CONTROL. Matching ortho keys reproduces the G-SEM2 defect.

    If this test ever finds zero flips, the guard has gone vacuous and the
    Action B trap has been silently reintroduced.
    """
    from henri.gram import stiefel_retract
    D, m = 2048, 64
    k = _book(m, D, seed=5)
    k_ortho = stiefel_retract(k.astype(np.complex128))
    rng = np.random.default_rng(9)
    flips = 0
    for _ in range(12):
        q = _unit(rng.standard_normal(D) + 1j * rng.standard_normal(D))
        flips += int(np.argmax((k @ q.conj()).real)
                     != np.argmax((k_ortho @ q.conj()).real))
    assert flips > 0, "mixing must be observable or the control is vacuous"


def test_delta_rule_orthogonalizes_internally():
    """MEASURED FACT. Write-time Stiefel is redundant, not sufficient-but-needed.

    Writing 64 strongly correlated keys still leaves a stored-key Gram with
    off-diagonal at machine precision, so external key orthogonalisation
    adds cost without benefit.
    """
    D = 2048
    k = _book(64, D, seed=7, share=4.0)
    raw_cmax = float(np.abs(np.real(k @ k.conj().T) - np.eye(64)).max())
    assert raw_cmax > 0.5, "fixture must be strongly correlated"
    bank = CartridgeBank(dim=D, rank=64)
    c = bank.add_cartridge(k, _book(64, D, seed=8))
    assert c._stored_offdiag() < 1e-12
    assert bank.stats()["stored_offdiag_max"] < 1e-12


def test_cartridge_exact_recall_and_router():
    D = 1024
    k = _book(200, D, seed=11)
    v = _book(200, D, seed=12)
    subj = [f"t{i // 64}" for i in range(200)]
    bank = build_bank(k, v, dim=D, rank=64, subjects=subj)
    assert len(bank.cartridges) == 4
    i = 137
    res = bank.query(k[i], k_route=1)
    assert res.hits
    assert res.cartridge == i // 64
    j = res.cartridge * 64 + res.row
    assert float(np.linalg.norm(res.value - v[j])) < 1e-9


def test_rank_wall_is_enforced_not_degraded():
    bank = CartridgeBank(dim=512, rank=64)
    k = _book(65, 512, seed=3)
    with pytest.raises(ValueError, match="rank"):
        bank.add_cartridge(k, k)


def test_footprint_is_reported():
    assert cartridge_bytes(65536, 64, np.complex128) == 134217728
    assert cartridge_bytes(65536, 64, np.complex64) == 67108864
    bank = build_bank(_book(64, 512, seed=1), _book(64, 512, seed=2),
                      dim=512, rank=64)
    st = bank.stats()
    assert st["bytes_per_cartridge"] == 2 * 512 * 64 * 16
    assert st["gib_total"] >= 0.0


# -- Action A: weight-invariant antonym coherence ------------------------

def test_antonym_coherence_is_minus_one_at_every_weight():
    """DERIVED REPAIR. Flipping BOTH active slots gives -1.0 exactly.

    The spec's claim that slot-1 alone reaches -1.0 is false for a composed
    sentence, where coherence is `2 w0 - 1`. Flipping both slots removes the
    energy tradeoff entirely.
    """
    from henri.semantic_backbone import SemanticBackbone
    pairs = [("the pressure increases", "the pressure decreases"),
             ("the value is positive", "the value is negative"),
             ("gases are hot", "gases are cold")]
    for w in (0.5, 0.3, 0.2):
        bb = SemanticBackbone(w_content=w)
        got = float(np.mean([
            np.real(np.vdot(bb.encode(a).psi, bb.encode(b).psi))
            for a, b in pairs]))
        assert got < -0.95, f"w0={w} gave {got}"


def test_canonicalize_maps_both_members_to_one_root():
    from henri.semantic_backbone import canonicalize, polarity
    assert canonicalize("the pressure decreases") == \
        canonicalize("the pressure increases")
    assert polarity("the pressure decreases") == -1
    assert polarity("the pressure increases") == +1
    # Distinct groups must not collapse onto one root.
    assert canonicalize("hot") != canonicalize("increase")


def test_strip_removes_boilerplate_and_keeps_span():
    from henri.semantic_backbone import strip_common_option_tokens
    q = "the reaction rate because activation energy"
    opts = ["the reaction rate increases because activation energy drops",
            "the reaction rate decreases because activation energy rises",
            "the reaction rate increases because activation energy is equal",
            "the reaction rate is unchanged because activation energy is equal"]
    out = strip_common_option_tokens(q, opts)
    assert all("reaction" not in o for o in out)
    assert any("drops" in o for o in out)
    # Never emit an empty span.
    assert all(o.strip() for o in out)


# -- Action E: receipt honesty -------------------------------------------

def test_receipt_is_a_self_digest_not_a_signature():
    from henri.api import Receipt
    r = Receipt({"a": 1}).to_dict()
    assert r["digest_kind"] == "sha256_self_digest_not_signature"
    assert len(r["self_digest"]) == 64
    # Changing the payload changes the digest.
    assert Receipt({"a": 2}).to_dict()["self_digest"] != r["self_digest"]


def test_api_rotation_invariance(tmp_path):
    from henri.api import DigitalTwinAPI
    api = DigitalTwinAPI(dim=4096, rank=64)
    q = "which law gives the radius of a charge in a magnetic field"
    opts = ["r = mv/(qB)", "r = qB/(mv)", "r = mvqB", "r = qB/(m v^2)"]
    seen = set()
    for i in range(4):
        o = opts[i:] + opts[:i]
        seen.add(api.rank_options(q, o).to_dict()["selected_digest"])
    assert len(seen) == 1, "option order must not change the answer"


def test_api_ingest_rejects_above_rank():
    from henri.api import DigitalTwinAPI
    api = DigitalTwinAPI(dim=1024, rank=8)
    with pytest.raises(ValueError, match="rank"):
        api.ingest_cartridge([f"k{i}" for i in range(9)],
                            [f"v{i}" for i in range(9)])
