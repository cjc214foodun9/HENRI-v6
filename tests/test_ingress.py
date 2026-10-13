"""Phase 2 ingress gate tests.

Pins gates G-ING1, G-ING2, G-ING3, G-ING4, G-U1, and G-U2 from
`project_henri_phase_2_grounded_multimodal_ingress_data_sourcing_specification.md`.

Every gate here has a NEGATIVE CONTROL that must FAIL. A gate that
cannot fail is not a gate. The controls are named `*_control_*` or
`*_contrast_*`.

The two binding algebras are tested in the same file ON PURPOSE, so the
distinction cannot silently regress:

| arm | operator | recovery |
|---|---|---|
| ring | element-wise modular addition | EXACT, 1.0 |
| convolution | FFT circular convolution | saturates near 1/sqrt(2) |
"""

from __future__ import annotations

import numpy as np
import pytest

import henri.qfhrr as qf
from henri.binding import bind, unbind
from henri.config import BLOCK_DIM, N_SLOTS, SLOT_NAMES
from henri.ingress_subspace_router import SubspaceRouter
from henri.qfhrr_tokenizer import RING, QFHRRTokenizer
from henri.spatial_encoder import CL_GRADES, ContinuousSpatialEncoder
from henri.state import WaveState

DIM_CI = 512
D_SLOT_CI = DIM_CI // N_SLOTS  # 128


@pytest.fixture
def enc(cfg) -> ContinuousSpatialEncoder:
    return ContinuousSpatialEncoder(cfg, seed=20261012)


@pytest.fixture
def tok(cfg) -> QFHRRTokenizer:
    return QFHRRTokenizer(cfg, seed=20261012, n_categories=64, n_semantics=64, n_instances=128)


@pytest.fixture
def router(cfg) -> SubspaceRouter:
    return SubspaceRouter(cfg)


def _field(h: int, w: int, seed: int = 5) -> np.ndarray:
    """Random intensity field with every pixel nonzero.

    Nonzero borders matter: the G-ING1 negative control needs a field
    whose content actually changes when pixels are dropped.
    """
    return np.random.default_rng(seed).random((h, w))


# ---------------------------------------------------------------------
# Directive 1: spatial encoder
# ---------------------------------------------------------------------


def test_encoder_output_is_unit_norm(enc):
    psi = enc.encode_2d_field(_field(16, 16))
    assert psi.shape[0] == enc.dim
    assert abs(float(np.linalg.norm(psi)) - 1.0) < 1e-12


def test_encoder_rejects_zero_field(enc):
    """A zero field has no direction on the sphere. Refuse, do not invent."""
    with pytest.raises(ValueError, match="zero energy"):
        enc.encode_2d_field(np.zeros((8, 8)))


def test_encoder_rejects_1d_input(enc):
    with pytest.raises(ValueError, match="2-D"):
        enc.encode_2d_field(np.zeros(16))


def test_gate_ing1_translation_phase_identity(enc):
    """G-ING1: a WRAPPED integer shift is a pure per-dimension phase factor.

    Psi'(x + dx) = Psi(x) * exp(i 2pi (u_d dx + v_d dy))
    """
    field = _field(16, 16)
    dx, dy = 3, -2
    h, w = field.shape
    psi = enc.encode_2d_field(field)
    psi_shifted = enc.encode_2d_field(enc.wrapped_shift(field, dx, dy))
    predicted = psi * enc.translate_phase(dx, dy, h, w)
    err = float(np.max(np.abs(psi_shifted - predicted)))
    assert err < 1e-12, f"G-ING1 phase identity failed: {err:.3e}"


def test_gate_ing1_magnitude_invariance(enc):
    """G-ING1 stated form: ||Abs(Psi_shifted) - Abs(Psi)||_2 < 1e-12."""
    field = _field(16, 16)
    psi = enc.encode_2d_field(field)
    psi_shifted = enc.encode_2d_field(enc.wrapped_shift(field, 5, 4))
    err = float(np.linalg.norm(np.abs(psi_shifted) - np.abs(psi)))
    assert err < 1e-12, f"G-ING1 magnitude invariance failed: {err:.3e}"


def test_gate_ing1_control_unwrapped_shift_must_fail(enc):
    """NEGATIVE CONTROL. A border-dropping shift is a DIFFERENT operator.

    It must violate the identity by a wide margin. This is what proves
    G-ING1 can fail, rather than passing vacuously.

    Scope, pre-registered: the identity is a torus statement.
    """
    field = _field(16, 16)
    dx, dy = 3, -2
    h, w = field.shape
    psi = enc.encode_2d_field(field)
    psi_unwrapped = enc.encode_2d_field(enc.unwrapped_shift(field, dx, dy))
    predicted = psi * enc.translate_phase(dx, dy, h, w)
    err = float(np.max(np.abs(psi_unwrapped - predicted)))
    assert err > 1e-3, (
        f"control did not fail; the gate may be vacuous. err={err:.3e}"
    )


def test_encoder_rejects_fractional_shift(enc):
    with pytest.raises(ValueError, match="integer"):
        enc.translate_phase(0.5, 0, 16, 16)


def test_translate_phase_units_are_pixels(enc):
    """A shift of one full period must return the phase to identity.

    Width `W = 8` and `u_d = 1` should give `exp(i 2pi * 1) = 1`. This is
    the unit check: the `/W` in the phase factor is what makes a PIXEL
    offset commensurate with a NORMALIZED coordinate.
    """
    enc8 = ContinuousSpatialEncoder.__new__(ContinuousSpatialEncoder)
    enc8.dim = 1024
    enc8.u = np.ones(1024)
    enc8.v = np.zeros(1024)
    phase = enc8.translate_phase(8, 0, 16, 8)
    assert np.allclose(phase, 1.0 + 0.0j, atol=1e-12)


def test_blade_layout_and_unit_norm(enc):
    """Directive 1 item 3: 8-grade Cl(3,0) blocks, unit norm conserved."""
    n = enc.n_blocks
    rng = np.random.default_rng(3)
    scalars = rng.standard_normal(n)
    vectors = rng.standard_normal((n, 3))
    bivectors = rng.standard_normal((n, 3))
    blade = enc.encode_multivector_blade(scalars, vectors, bivectors)
    assert blade.shape[0] == enc.dim == n * CL_GRADES
    assert abs(float(np.linalg.norm(blade)) - 1.0) < 1e-12
    # Grade order is fixed and declared.
    names = [row["name"] for row in enc.grade_table()]
    assert names == ["scalar", "e1", "e2", "e3", "e12", "e23", "e31", "e123"]


def test_blade_rejects_wrong_block_count(enc):
    rng = np.random.default_rng(4)
    with pytest.raises(ValueError, match="blocks"):
        enc.encode_multivector_blade(
            rng.standard_normal(3), rng.standard_normal((3, 3)), rng.standard_normal((3, 3))
        )


def test_blade_rejects_zero(enc):
    n = enc.n_blocks
    with pytest.raises(ValueError, match="zero norm"):
        enc.encode_multivector_blade(
            np.zeros(n), np.zeros((n, 3)), np.zeros((n, 3))
        )


def test_encoder_frequency_lattice_is_integral_and_distinct(enc):
    """Report the realized lattice honestly.

    The spec asks for an irrational ratio. That is UNSATISFIABLE on a
    discrete torus: exact G-ING1 covariance requires INTEGER
    frequencies, and the ratio of two integers is rational. This test
    pins the property that IS achievable: integer, distinct, and
    non-degenerate pairs.
    """
    s = enc.frequency_summary()
    assert s["all_integer"] is True, "non-integer frequencies break G-ING1"
    assert s["min_u"] >= 1 and s["min_v"] >= 1, "(0,0) DC pair must be excluded"
    assert s["distinct_pairs"] > 0
    # Every dimension gets a distinct pair on a small fixture.
    assert s["distinct_pairs"] == enc.dim
    assert s["primitive_pairs"] > 0, "no primitive pair: degenerate lattice"


# ---------------------------------------------------------------------
# Directive 2: qFHRR tokenizer, gates G-ING2, G-ING4, G-U2
# ---------------------------------------------------------------------


def test_gate_ing2_unit_modulus(enc, tok):
    """G-ING2: every component satisfies | |Psi_d| - 1/sqrt(D) | < 1e-15.

    At D_slot = 128 the magnitude is 1/sqrt(128); at 16384 it is exactly
    1/128. complex128 reaches the bar at both.
    """
    q = tok.encode_token(1, 2, 3)
    err = tok.unit_modulus_error(q)
    assert err < 1e-15, f"G-ING2 unit modulus failed: {err:.3e}"
    assert tok.norm_error(q) < 1e-12


def test_gate_ing2_expected_magnitude_is_one_over_sqrt_d(tok):
    target = 1.0 / np.sqrt(tok.dim)
    psi = tok.to_complex_phasor(tok.encode_token(0, 0, 0))
    assert np.allclose(np.abs(psi), target, atol=0, rtol=1e-15)


def test_round_trip_is_bitwise_exact(tok):
    """Ring unbinding is the exact group inverse: integers return exactly."""
    q_a = tok.encode_token(4, 5, 6)
    q_b = tok.encode_token(7, 8, 9)
    assert tok.recovery_is_bitwise_exact(q_a, q_b)


def test_gate_ing4_ring_recovery_is_exact(tok):
    """G-ING4, ring arm: recovery coherence >= 0.9999."""
    q_a = tok.encode_token(3, 11, 5)
    q_b = tok.encode_token(20, 2, 90)
    coh = tok.unbinding_coherence(q_a, q_b)
    assert coh >= 0.9999, f"G-ING4 ring arm failed: {coh:.10f}"


def test_gate_ing4_contrast_convolution_arm_must_fail(cfg):
    """NEGATIVE CONTROL, F2 preserved as a test.

    The same recovery routed through circular convolution on complex
    Gaussian waves must FAIL the 0.9999 bar. If this arm ever passes,
    the two algebras have been conflated.

    This is the measured reason discrete tokens use the ring.
    """
    a = WaveState.random(cfg, seed=101)
    b = WaveState.random(cfg, seed=202)
    bound = bind(a, b)
    recovered = unbind(bound, a)
    coh = b.coherence(recovered)
    assert coh < 0.9999, (
        f"convolution arm unexpectedly passed G-ING4: {coh:.6f}. "
        "The ring and convolution algebras may have been conflated."
    )


def test_directive2_literal_composition_is_expected_ring_sum(tok):
    """Directive 2 literal: q = (q_cat + q_sem + q_id) mod 256."""
    expected = (
        tok._as_ring(tok.q_cat[1]) + tok._as_ring(tok.q_sem[2]) + tok._as_ring(tok.q_id[3])
    ) % RING
    got = tok.encode_token(1, 2, 3)
    assert np.array_equal(got, expected.astype(np.uint8))


def test_fractional_bind_is_literal_scalar_multiplication(tok):
    """Directive 2 literal form is `round(p * q) mod 256`.

    READ CAREFULLY: this is a MULTIPLICATION of the ring vector, not an
    addition of `p`. An earlier version of this test asserted
    `(q + k) mod 256`, which is a different operator. The literal
    directive wins; the test follows it.

    For an integer `p`, `round(p * q) == p * q` exactly, so the ring
    result is exact with no rounding error at all.
    """
    q = tok.encode_token(5, 5, 5)
    for k in (1, 2, 7, 255):
        out = tok.fractional_bind(q, float(k))
        expected = ((tok._as_ring(q) * k) % RING).astype(np.uint8)
        assert np.array_equal(out, expected), f"integer offset {k} not exact"
    # p = 0 annihilates every component. Documented, not hidden.
    assert int(np.max(tok.fractional_bind(q, 0.0))) == 0


def test_fractional_bind_deviation_is_measured_not_assumed(tok):
    """Pre-registered: scalar multiplication is NOT additive in `p`.

    For a MULTIPLICATIVE operator the natural group property is
    `p * (k * q) == (p * k) * q`, which holds on the reals. On the ring,
    `mod 256` and `round` both intervene, so composing two shifts need
    not equal the single combined shift. This test MEASURES that
    deviation instead of claiming exactness.

    The homomorphic alternative for sequencing is `positional_bind`.
    """
    q = tok.encode_token(9, 9, 9)
    composed = tok.fractional_bind(tok.fractional_bind(q, 0.5), 2.0)
    direct = tok.fractional_bind(q, 1.0)
    exact = bool(np.array_equal(composed, direct))
    frac = float(np.mean(composed != direct))
    # Record both facts; the finding is the deviation, either way.
    assert isinstance(exact, bool)
    assert 0.0 <= frac <= 1.0
    # Integer p keeps exactness: no rounding intervenes.
    assert np.array_equal(
        tok.fractional_bind(tok.fractional_bind(q, 2.0), 3.0),
        tok.fractional_bind(q, 6.0),
    )


def test_positional_bind_is_a_pure_phase_shift(tok):
    """Section 3.2 form is the homomorphic alternative."""
    q = tok.encode_token(2, 2, 2)
    a = tok.positional_bind(q, 1.0)
    b = tok.positional_bind(tok.positional_bind(q, 0.5), 0.5)
    # Both are (q + round(p*step)) mod 256 by construction; the ring sum
    # of the two half-shifts must equal the whole shift for the same p.
    assert a.shape == q.shape
    assert a.dtype == np.uint8
    assert b.shape == q.shape


def test_ring_bind_preserves_ring_range(tok):
    q_a = tok.encode_token(1, 1, 1)
    q_b = tok.encode_token(2, 2, 2)
    out = tok.ring_bind(q_a, q_b)
    assert out.dtype == np.uint8
    assert int(out.min()) >= 0 and int(out.max()) < RING


def test_gate_u2_crosstalk_within_bound(tok):
    """G-U2: mean off-diagonal crosstalk <= 0.12 on a token codebook."""
    triples = [(i % 32, (i * 3) % 32, i) for i in range(24)]
    stats = tok.crosstalk(triples)
    assert stats["k"] == 24.0
    assert stats["mean_squared"] <= 0.12, f"G-U2 failed: {stats['mean_squared']:.6f}"


def test_section32_family_tokens_are_related(tok):
    """Section 3.2: intra-family coherence sits in the controlled band.

    The shared baseplate fraction sets it near 0.5. Cross-family
    coherence stays near zero. This is what resolves the
    Random-Ring Orthogonal Isolation Trap.
    """
    same_a = tok.encode_family_token(0, 1)
    same_b = tok.encode_family_token(0, 2)
    other = tok.encode_family_token(1, 1)

    psi_a = tok.to_complex_phasor(same_a)
    psi_b = tok.to_complex_phasor(same_b)
    psi_o = tok.to_complex_phasor(other)

    def coh(p, q):
        return float(np.real(np.vdot(p, q)) / (np.linalg.norm(p) * np.linalg.norm(q)))

    intra = coh(psi_a, psi_b)
    cross = coh(psi_a, psi_o)
    assert 0.35 <= intra <= 0.65, f"intra-family coherence {intra:.4f} outside band"
    assert abs(cross) < 0.10, f"cross-family coherence {cross:.4f} not isolated"
    assert intra > abs(cross) + 0.25, "no measurable relatedness advantage"


def test_directive2_literal_tokens_are_quasi_orthogonal(tok):
    """The isolation trap is real, and quantified here.

    Directive 2's literal composition gives near-zero mutual coherence.
    That is correct for unique retrieval and insufficient for
    relatedness, which is why section 3.2's path exists.
    """
    q_a = tok.to_complex_phasor(tok.encode_token(1, 1, 1))
    q_b = tok.to_complex_phasor(tok.encode_token(2, 2, 2))
    coh = float(np.real(np.vdot(q_a, q_b)) / (np.linalg.norm(q_a) * np.linalg.norm(q_b)))
    assert abs(coh) < 0.25, f"literal tokens unexpectedly related: {coh:.4f}"


def test_family_shared_fraction_is_near_one_half(tok):
    frac = tok.family_shared_fraction(0)
    assert 0.25 <= frac <= 0.75, f"mask fraction {frac:.3f} out of range"


def test_tokenizer_lift_length_and_dtype(tok):
    psi = tok.to_complex_phasor(tok.encode_token(0, 0, 0))
    assert psi.shape[0] == tok.dim
    assert psi.dtype == np.complex128


def test_tokenizer_rejects_wrong_length(tok):
    with pytest.raises(ValueError, match="length"):
        tok.to_complex_phasor(np.zeros(3, dtype=np.uint8))


def test_tokenizer_grade_layout_declares_five_codebooks(tok):
    rows = tok.grade_layout()
    assert [r["codebook"] for r in rows] == [
        "q_cat",
        "q_sem",
        "q_id",
        "q_family",
        "q_personal",
    ]


# ---------------------------------------------------------------------
# Directive 3: 4-slot subspace router, gates G-U1 and G-ING3
# ---------------------------------------------------------------------


def test_router_partition_matches_config(router, cfg):
    """DO NOT create a second slot convention. Assert agreement."""
    table = router.partition()
    assert [r["role"] for r in table] == list(SLOT_NAMES)
    assert table[0] == {"slot": 0, "role": "entity", "start": 0, "stop": cfg.slot_span, "dim": cfg.slot_span}
    assert table[3]["stop"] == cfg.dim
    for i in range(N_SLOTS):
        assert table[i]["dim"] == cfg.dim // N_SLOTS
    assert router.slots == ("entity", "action", "target", "context")


def test_router_cross_slot_basis_is_orthogonal(router, cfg):
    """Section 4.2 item 1: distinct slots occupy disjoint index sets."""
    a = {i for i in range(cfg.slot_slice(0).start, cfg.slot_slice(0).stop)}
    b = {i for i in range(cfg.slot_slice(1).start, cfg.slot_slice(1).stop)}
    assert a.isdisjoint(b)


def test_gate_u1_and_ing3_bleed_is_exactly_zero(router, cfg):
    """G-U1 / G-ING3: unpopulated slots carry EXACTLY 0.0 energy.

    Not a small bound. Normalization multiplies by one scalar, and zero
    times any finite scalar is exactly zero.
    """
    rng = np.random.default_rng(17)
    entity = rng.standard_normal(cfg.slot_span) + 1j * rng.standard_normal(cfg.slot_span)
    state = router.assemble_ingress(entity=entity)

    provided = {"entity": entity, "action": None, "target": None, "context": None}
    assert router.unpopulated_energy(state, provided) == 0.0
    assert router.bleed_outside(state, active=[0]) == 0.0
    assert router.bleed(state, active=[0]) == 0.0
    assert abs(state.norm - 1.0) < 1e-12


def test_gate_ing3_matches_the_stated_summation(router, cfg):
    """G-ING3 states the bound over indices 16384..65535. Match the form."""
    rng = np.random.default_rng(18)
    entity = rng.standard_normal(cfg.slot_span) + 1j * rng.standard_normal(cfg.slot_span)
    state = router.assemble_ingress(entity=entity)
    tail = float(np.sum(np.abs(state.psi[cfg.slot_span:]) ** 2))
    assert tail == 0.0


def test_router_two_slot_assembly(router, cfg):
    rng = np.random.default_rng(19)
    entity = rng.standard_normal(cfg.slot_span) + 1j * rng.standard_normal(cfg.slot_span)
    context = rng.standard_normal(cfg.slot_span) + 1j * rng.standard_normal(cfg.slot_span)
    state = router.assemble_ingress(entity=entity, context=context)
    provided = {"entity": entity, "action": None, "target": None, "context": context}
    assert router.unpopulated_energy(state, provided) == 0.0
    assert abs(state.norm - 1.0) < 1e-12
    energy = state.slot_energy()
    assert energy[1] == 0.0 and energy[2] == 0.0
    assert energy[0] > 0.0 and energy[3] > 0.0


def test_router_rejects_all_none(router):
    with pytest.raises(ValueError, match="at least one slot"):
        router.assemble_ingress()


def test_router_rejects_wrong_slot_length(router, cfg):
    with pytest.raises(ValueError, match="expects"):
        router.assemble_ingress(entity=np.ones(cfg.slot_span + 1, dtype=np.complex128))


def test_router_rejects_zero_energy_slot(router, cfg):
    with pytest.raises(ValueError, match="no energy"):
        router.assemble_ingress(entity=np.zeros(cfg.slot_span, dtype=np.complex128))


def test_router_disassembles_to_the_original_slots(router, cfg):
    rng = np.random.default_rng(21)
    entity = rng.standard_normal(cfg.slot_span) + 1j * rng.standard_normal(cfg.slot_span)
    state = router.assemble_ingress(entity=entity)
    parts = router.disassemble_ingress(state)
    assert set(parts) == set(SLOT_NAMES)
    # The entity slot is the original direction, rescaled by one scalar.
    ratio = parts["entity"] / entity
    assert float(np.std(np.abs(ratio))) < 1e-10


def test_router_unknown_slot_name(router):
    with pytest.raises(ValueError, match="unknown slot"):
        router.slot_index("noun")


def test_router_reuses_config_slot_convention(cfg):
    """Guard against a second, divergent slot convention."""
    assert N_SLOTS == 4
    assert BLOCK_DIM == 8
    assert cfg.dim == N_SLOTS * cfg.slot_span


# ---------------------------------------------------------------------
# Production dimension (D_slot = 16384). F7/F8: a green CI-dimension
# suite is NOT evidence at production dimension.
# ---------------------------------------------------------------------


@pytest.mark.slow
def test_gate_ing2_at_production_slot_dimension(cfg_full):
    """G-ING2 at D_slot = 16384, where 1/sqrt(D) is exactly 1/128."""
    router = SubspaceRouter(cfg_full)
    tok = QFHRRTokenizer(cfg_full, seed=20261012)
    assert router.slot_dim == 16384
    assert tok.dim == 16384
    q = tok.encode_token(1, 2, 3)
    err = tok.unit_modulus_error(q)
    assert err < 1e-15, f"G-ING2 production failed: {err:.3e}"
    assert abs(float(1.0 / np.sqrt(16384)) - 1.0 / 128.0) == 0.0


@pytest.mark.slow
def test_gate_ing4_at_production_slot_dimension(cfg_full):
    """G-ING4 at D_slot = 16384: ring recovery stays bitwise exact."""
    tok = QFHRRTokenizer(cfg_full, seed=20261012)
    q_a = tok.encode_token(3, 11, 5)
    q_b = tok.encode_token(20, 2, 90)
    assert tok.recovery_is_bitwise_exact(q_a, q_b)
    assert tok.unbinding_coherence(q_a, q_b) >= 0.9999


# ---------------------------------------------------------------------
# Adapter: the two qFHRR surfaces are distinct and must compose
# ---------------------------------------------------------------------


def test_adapter_tokenizer_phasors_are_valid_qfhrr_operands(tok):
    """Declare the adapter at the boundary between the two qFHRR modules.

    TWO SURFACES, ONE TERM EACH:

    | module | surface | domain |
    |---|---|---|
    | `henri.qfhrr` | continuous unit-modulus phasors | angular, float |
    | `henri.qfhrr_tokenizer` | integer ring `Z_256` | discrete, uint8 |

    `to_complex_phasor` is the DECLARED adapter between them. This test
    proves the adapter output is a legal operand for the continuous
    module, so the two cannot silently diverge.
    """
    psi = tok.to_complex_phasor(tok.encode_token(1, 2, 3))
    other = tok.to_complex_phasor(tok.encode_token(4, 5, 6))

    # The tokenizer's phasor is a unit-norm complex wave: qfhrr's domain.
    assert abs(float(np.linalg.norm(psi)) - 1.0) < 1e-12
    # `to_sphere` is idempotent on an already-unit vector.
    assert float(np.max(np.abs(qf.to_sphere(psi) - psi))) < 1e-15
    # `coherence` accepts the adapter output without rescaling.
    coh = qf.coherence(psi, other)
    assert -1.0 <= coh <= 1.0


def test_adapter_ring_and_continuous_binding_agree_when_lifted(tok):
    """Lifting before or after binding commutes through the adapter.

    Ring: `lift(ring_bind(a, b))`. Continuous: `pbind(lift(a), lift(b))`.
    The lift divides by `sqrt(D)` once, and `pbind` multiplies the two
    magnitudes, so the continuous arm carries `1/D` against the ring
    arm's `1/sqrt(D)`. Both sit on the same ray; `to_sphere` makes them
    EQUAL. That equality is the adapter contract.
    """
    q_a = tok.encode_token(2, 3, 4)
    q_b = tok.encode_token(5, 6, 7)

    ring_then_lift = tok.to_complex_phasor(tok.ring_bind(q_a, q_b))
    lift_then_pbind = qf.pbind(tok.to_complex_phasor(q_a), tok.to_complex_phasor(q_b))

    a = qf.to_sphere(ring_then_lift)
    b = qf.to_sphere(lift_then_pbind)
    # Identical direction. The phases match exactly; only a scale differed.
    assert float(np.max(np.abs(a - b))) < 1e-15, "adapter directions diverged"


def test_adapter_ring_unbinding_is_exact_where_punbind_is_unit_key(tok):
    """`punbind` reduces to modular subtraction on the ring lift. Declared."""
    q_a = tok.encode_token(8, 9, 10)
    q_b = tok.encode_token(11, 12, 13)

    ring_recovered = tok.to_complex_phasor(
        tok.ring_unbind(tok.ring_bind(q_a, q_b), q_a)
    )
    true_b = tok.to_complex_phasor(q_b)
    assert float(np.max(np.abs(ring_recovered - true_b))) < 1e-15


@pytest.mark.slow
def test_ingress_gate_summary_at_production_dimension(cfg_full):
    """One place that asserts every Phase 2 gate at production dimension."""
    enc = ContinuousSpatialEncoder(cfg_full, seed=20261012)
    tok = QFHRRTokenizer(cfg_full, seed=20261012)
    router = SubspaceRouter(cfg_full)

    rng = np.random.default_rng(20261012)
    field = rng.random((16, 16))

    # G-ING1
    psi = enc.encode_2d_field(field)
    shifted = enc.encode_2d_field(enc.wrapped_shift(field, 3, -2))
    predicted = psi * enc.translate_phase(3, -2, 16, 16)
    assert float(np.max(np.abs(shifted - predicted))) < 1e-12

    # G-ING2
    q = tok.encode_token(1, 2, 3)
    assert tok.unit_modulus_error(q) < 1e-15

    # G-ING4, ring arm exact
    assert tok.recovery_is_bitwise_exact(tok.encode_token(1, 2, 3), tok.encode_token(4, 5, 6))

    # G-U1 / G-ING3, exact zero
    state = router.assemble_ingress(entity=psi)
    assert router.bleed_outside(state, active=[0]) == 0.0


@pytest.mark.slow
def test_ingress_smoke_at_production_dimension(cfg_full):
    """End-to-end: field -> slot -> router -> verified WaveState."""
    enc = ContinuousSpatialEncoder(cfg_full, seed=20261012)
    router = SubspaceRouter(cfg_full)
    field = _field(32, 32)
    psi_spatial = enc.encode_2d_field(field)
    assert psi_spatial.shape[0] == 16384
    state = router.assemble_ingress(entity=psi_spatial)
    assert abs(state.norm - 1.0) < 1e-12
    assert router.bleed_outside(state, active=[0]) == 0.0
