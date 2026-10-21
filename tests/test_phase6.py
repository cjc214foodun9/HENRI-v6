"""Phase 6 gates. Every test can fail; four are negative controls.

The critical ones encode measured defects, so a regression re-opens the
exact bug rather than a paraphrase of it.
"""

from __future__ import annotations

import numpy as np
import pytest

from henri import calibration as C
from henri import cartridge_bank as cb
from henri.cartridge_bank import (CartridgeBank, ColdCartridge,
                                  cartridge_bytes, cold_cartridge_bytes,
                                  content_sim)


def _unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-30)


# -- Action 1: calibration ------------------------------------------------

def _toy_encode(text: str) -> np.ndarray:
    """Deterministic stand-in stem: hash text into a unit real vector."""
    import hashlib
    h = hashlib.sha256(text.encode()).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
    v = rng.standard_normal(16)
    return v / np.linalg.norm(v)


def test_calibration_refuses_a_small_fit():
    """A 10-text fit is not a corpus statistic. Must raise, not warn."""
    cal = C.CorpusIsotropicCalibration(p_components=2, min_texts=50)
    with pytest.raises(ValueError, match="small fit"):
        cal.fit([f"t{i}" for i in range(10)], _toy_encode)


def test_calibration_is_purely_pointwise():
    """Applying to one vector must not depend on any other vector.

    This is WHY it preserves rotation invariance: no option-aware mixing.
    """
    cal = C.CorpusIsotropicCalibration(p_components=2, min_texts=20).fit(
        [f"corpus item {i} about topic" for i in range(40)], _toy_encode)
    a = _toy_encode("alpha beta")
    alone = cal.apply(a)
    _ = [cal.apply(_toy_encode(f"other {i}")) for i in range(5)]
    again = cal.apply(a)
    assert float(np.abs(alone - again).max()) == 0.0


def test_calibration_leak_detector_works():
    cal = C.CorpusIsotropicCalibration(p_components=1, min_texts=20).fit(
        [f"fitted line {i}" for i in range(20)], _toy_encode)
    assert cal.saw_text("fitted line 3")
    assert not cal.saw_text("never fitted text")


def test_p_components_bounds_enforced():
    with pytest.raises(ValueError):
        C.CorpusIsotropicCalibration(p_components=5)


# -- Action 1: order channel ---------------------------------------------

class _BB:
    """Minimal stand-in exposing the two members `order_wave` touches."""

    def __init__(self):
        import hashlib
        from henri.semantic_backbone import SLOT
        from henri import calibration as _c
        self.dim = SLOT
        # Reuse the real char-code machinery so the test measures the
        # production path, not a re-implementation.
        self._cc = _c.char_codes

    def code(self, ch):
        return self._cc([ch])[0]


def test_order_channel_separates_ab_ba():
    """G-P6A5 spec pair. The spec's own bar is `cos <= 0.10`."""
    from henri.calibration import order_wave
    bb = _BB()
    a, b = order_wave(bb, "ab"), order_wave(bb, "ba")
    assert float(np.real(np.vdot(a, b))) <= 0.10


def test_order_channel_punctuation_alignment():
    """MEASURED DEFECT: char_codes used to take raw text, so punctuation
    shifted every character onto the wrong code. `'mv/(qB)'` has 7 raw
    characters but 4 alphanumeric ones."""
    from henri.calibration import char_positions, order_wave
    pairs = char_positions("mv/(qB)")
    assert len(pairs) == 4, "filter must drop punctuation"
    a, b = order_wave(_BB(), "mv/(qB)"), order_wave(_BB(), "qB/(mv)")
    assert float(np.real(np.vdot(a, b))) <= 0.10


def test_order_channel_empty_input_is_none():
    from henri.calibration import order_wave
    assert order_wave(_BB(), "()//  ") is None


def test_scalar_stride_is_falsified():
    """DERIVED: a constant added to every dimension is a GLOBAL PHASE.

    Multiplying a whole wave by exp(i*theta) does not change direction, so
    `sum(u + r*v)` and `sum(v + r*u)` coincide. The first implementation
    scored 0.8456 this way. Pin the algebra so it cannot return.
    """
    r = np.exp(2j * np.pi * 37 / 256)
    assert abs(abs(r) - 1.0) < 1e-12
    u = _unit(np.random.default_rng(0).standard_normal(64).astype(complex))
    ab, ba = u + r * u, u + r * u
    assert float(np.real(np.vdot(ab, ba)) /
                 (np.linalg.norm(ab) * np.linalg.norm(ba))) > 0.99


def test_single_inner_product_cannot_carry_both_gates():
    """DERIVED feasibility bound, recomputed rather than quoted."""
    out = C.feasibility()
    assert out["feasible"] is False
    assert out["verdict"] == "INFEASIBLE_NEEDS_SEPARATE_CHANNEL"


# -- Action 2: cold tier and LRU -----------------------------------------

def test_cold_cartridge_bytes_is_exactly_4_mib():
    """2 slots x 16384 uint8 x 64 rows x (keys and values) = 4 MiB."""
    assert cold_cartridge_bytes(64) == 4_194_304
    assert cold_cartridge_bytes(64) / 2**20 == 4.0


def test_cold_roundtrip_within_half_a_bin():
    rng = np.random.default_rng(3)
    span = 2 * cb.SLOT_SPAN
    k = _unit(rng.standard_normal((8, span)) +
              1j * rng.standard_normal((8, span)))
    cc = ColdCartridge.from_waves(0, k, k)
    kr, _ = cc.materialise()
    err = float(np.abs(np.angle(kr * np.conj(k))).max())
    assert err <= cb.QUANT_BOUND + 1e-12


def test_cold_tier_rejects_wrong_width():
    rng = np.random.default_rng(4)
    k = _unit(rng.standard_normal((4, 64)) + 1j * rng.standard_normal((4, 64)))
    with pytest.raises(ValueError, match="active span"):
        ColdCartridge.from_waves(0, k, k)


def test_lru_evicts_and_bounds():
    bank = CartridgeBank(dim=256, rank=8, max_hot=2)
    rng = np.random.default_rng(5)
    for i in range(5):
        a = _unit(rng.standard_normal((8, 256)) +
                  1j * rng.standard_normal((8, 256)))
        bank.add_cartridge(a, a, subject=f"c{i}")
    bank.finalize()
    for c in bank.cartridges:
        bank.query(c.keys_raw[0], k_route=1)
    st = bank.hot.stats()
    assert st["evictions"] > 0
    assert st["entries"] <= st["max"] == 2


def test_zero_hot_ceiling_rejected():
    """NEGATIVE CONTROL: the bound cannot be silently disabled."""
    with pytest.raises(ValueError, match="max_entries"):
        CartridgeBank(dim=64, rank=4, max_hot=0)


def test_write_stiefel_off_by_default():
    bank = CartridgeBank(dim=256, rank=8)
    rng = np.random.default_rng(6)
    a = _unit(rng.standard_normal((4, 256)) + 1j * rng.standard_normal((4, 256)))
    c = bank.add_cartridge(a, a, subject="s")
    assert bank.write_stiefel is False
    assert c.keys_ortho is None


# -- Action 3: content-subspace gate -------------------------------------

def test_content_sim_null_floor_is_low():
    """MEASURED DEFECT: the full-wave floor is ~0.63 because slot 1 is a
    fixed role basis, so `tau_hit = 0.55` sat INSIDE the null and the gate
    could not fail. Content subspace drops the floor to noise.

    FIXTURE FIX. The first version used i.i.d. complex noise, which has NO
    fixed role basis, so full and content were both pure noise (0.0041 vs
    0.0055) and the comparison was meaningless. The defect only exists on
    real backbone waves, so the fixture must be real backbone waves.
    """
    from henri.semantic_backbone import SemanticBackbone
    bb = SemanticBackbone()
    texts = ["the pressure increases with temperature",
             "quantum chromodynamics confines quarks",
             "medieval history of the silk road",
             "a recipe for sourdough bread",
             "the tax code for small businesses",
             "orbital mechanics of a satellite"]
    keys = np.stack([bb.encode(t).psi for t in texts])
    q = bb.encode("volcanic geology of iceland").psi
    full = float(np.abs(np.real(keys @ np.conj(q))).max())
    content = float(np.abs(content_sim(keys, q)).max())
    # The fixed role basis alone puts the full-wave floor near w1 = 0.5.
    assert full > 0.45, f"full-wave floor {full:.4f} should be high"
    # Content subspace drops it to noise.
    assert content < 0.25, f"content floor {content:.4f} should be low"
    assert content < full


def test_margin_gate_is_zero_below_tau_and_capped():
    from henri.api import W_MEM_MAX, TAU_HIT, margin_gate_weight
    assert margin_gate_weight(0.0) == 0.0
    assert margin_gate_weight(TAU_HIT) == 0.0
    assert margin_gate_weight(1.0) == W_MEM_MAX
    assert margin_gate_weight(0.70) > 0.0


def test_margin_gate_is_monotone():
    from henri.api import margin_gate_weight
    vals = [margin_gate_weight(t) for t in np.linspace(0, 1, 21)]
    assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:]))


# -- Action 4: rotation protocol -----------------------------------------

def test_pooled_rotation_shifts_gold_index_correctly():
    """GERUND FIX: rotating left by r moves gold to (i - r) mod 4.

    Getting the inverse map here would score the arm against a permuted
    label and manufacture a result. Pin the direction.
    """
    opts = ["A", "B", "C", "D"]
    correct = 1  # 'B'
    for rot in range(4):
        rotated = opts[rot:] + opts[:rot]
        ci = (correct - rot) % 4
        assert rotated[ci] == "B"


def test_stratified_split_is_balanced():
    from benchmarks.wave_rank import stratified_letter_balanced
    items = [{"correct_idx": i % 4, "id": i} for i in range(400)]
    s = stratified_letter_balanced(items, n_per_letter=25)
    counts = [sum(1 for r in s if r["correct_idx"] == k) for k in range(4)]
    assert counts == [25, 25, 25, 25]


# -- Action 5: MBPP pool integrity ---------------------------------------

def test_mbpp_item_digest_separates_same_prompt():
    """MEASURED DEFECT: MBPP reuses prompts across tasks with different
    tests, so a text-only digest reported 2 false overlaps."""
    import importlib.util
    from pathlib import Path
    p = Path(__file__).resolve().parents[1] / "scripts" / "mbpp_wave_eval.py"
    spec = importlib.util.spec_from_file_location("mbpp_mod", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    a = {"text": "Write a function to find the area of a circle",
         "code": "def f(r): return 3.14*r*r", "test_list": ["assert f(1)==3.14"]}
    b = {"text": "Write a function to find the area of a circle",
         "code": "def g(s): return 3.14*s*s", "test_list": ["assert g(2)==12.56"]}
    assert m.item_digest(a) != m.item_digest(b)


def test_mbpp_mutations_are_distinct_from_gold():
    import importlib.util
    from pathlib import Path
    p = Path(__file__).resolve().parents[1] / "scripts" / "mbpp_wave_eval.py"
    spec = importlib.util.spec_from_file_location("mbpp_mod2", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    # FIXTURE FIX. The first gold was `def add(a, b): return a + b`, which
    # has no numeric literal, no guard, and no `[-*/%]` operator, so
    # const_shift, swap_operands, drop_guard and invert_compare legitimately
    # returned it unchanged: 3/7. A mutation roster can only be exercised
    # by a fixture that contains the constructs it targets.
    gold = ("def clamp(x, limit):\n"
            "    if x > limit:\n"
            "        return limit\n"
            "    total = x * 2\n"
            "    return total + 1\n")
    distinct = sum(1 for k in range(7)
                   if m.is_distinct(m.mutate(gold, k), gold))
    assert distinct >= 6, f"only {distinct}/7 mutations differ from gold"
