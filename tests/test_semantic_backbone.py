"""Tests for the semantic backbone. Every gate here can fail.

The two anti-vacuity controls matter most. A coherence gate that reads
0.99 can be produced by a map that collapsed every input to one point, so
the shuffled control must read far lower than the synonym mean. A
whitening gate that reads c_max = 0 can be produced by deleting the signal,
so the gamma=0 control must leave coherence untouched.
"""

from __future__ import annotations

import numpy as np
import pytest

from henri import semantic_backbone as sb


@pytest.fixture(scope="module")
def bb():
    return sb.SemanticBackbone()


def test_unit_modulus_is_exact(bb):
    """G-ING2 applies to the SLOT PHASOR: every component is 1/sqrt(D)."""
    u = bb.stem.encode("the pressure increases with temperature")
    psi = bb.proj.phasor(u)
    assert bb.proj.unit_modulus_error(psi) < 1e-15
    # The composed 2-slot state spreads energy, so its per-component
    # magnitude is sqrt(2)/sqrt(SLOT). The gate is a per-slot property.
    e = bb.encode("the pressure increases with temperature")
    assert abs(float(np.linalg.norm(e.psi)) - 1.0) < 1e-12


def test_inactive_slots_are_exact_zero(bb):
    """G-U1: slots 2 and 3 carry no energy. Exact, not small."""
    e = bb.encode("entropy always increases")
    tail = e.psi[2 * sb.SLOT:]
    assert float(np.max(np.abs(tail))) == 0.0
    assert e.active_slots == (0, 1)


def test_psi_is_on_the_unit_sphere(bb):
    e = bb.encode("a wave on the sphere")
    assert abs(float(np.linalg.norm(e.psi)) - 1.0) < 1e-12


def test_polarity_flip_is_exactly_pi_in_the_ring(bb):
    """+pi is exactly +128 in Z_256, so negation is an exact group op."""
    a = bb.encode("the reaction proceeds")
    b = bb.encode("the reaction does not proceed")
    assert a.pol == 1 and b.pol == -1
    delta = (b.q1.astype(np.int64) - a.q1.astype(np.int64)) % 256
    assert np.all(delta == 128)


def test_polarity_parity_cancels_double_negation():
    assert sb.polarity("the value is not unstable") == -1
    assert sb.polarity("not not true") == 1
    assert sb.polarity("temperature increases") == 1


def test_bochner_kernel_matches_closed_form():
    """The phase lift is an honest kernel, not an unknown approximation."""
    gen = np.random.default_rng(3)
    p = sb.BochnerPhaseProjector(64, 4096, sigma=float(np.sqrt(2.0)), seed=5)
    u = gen.standard_normal(64)
    u /= np.linalg.norm(u)
    rho = 0.7
    w = gen.standard_normal(64)
    w -= (w @ u) * u
    w /= np.linalg.norm(w)
    v = rho * u + np.sqrt(1 - rho ** 2) * w
    measured = float(np.mean(np.cos(p.angles(u) - p.angles(v))))
    predicted = float(np.exp(-p.sigma ** 2 * (1.0 - float(u @ v))))
    assert abs(measured - predicted) / predicted < 0.15


def test_scaling_factor_is_present():
    """Unit-norm rows collapse the kernel to 1. The sqrt(d) factor is the fix."""
    p = sb.BochnerPhaseProjector(64, 4096, sigma=1.0, seed=7)
    row_norm = float(np.linalg.norm(p.omega[0]))
    assert abs(row_norm - np.sqrt(64.0)) < 1e-9


def test_negative_control_shuffled_pairs_read_lower(bb):
    """Anti-vacuity. If a collapsed map were used, this would read ~equal."""
    syn = [("car", "automobile"), ("doctor", "physician"), ("big", "large")]
    paired = [float(np.real(np.vdot(bb.encode(a).psi, bb.encode(b).psi)))
              for a, b in syn]
    shuffled = [float(np.real(np.vdot(bb.encode(a).psi, bb.encode(c).psi)))
                for (a, _), (_, c) in zip(syn, syn[1:] + syn[:1])]
    assert np.mean(paired) > np.mean(shuffled) + 0.10


def test_common_mode_reject_reduces_coherence(bb):
    """CMR removes the shared carrier but does NOT orthogonalise.

    Measured fact. After projecting `K` rows orthogonal to the centroid
    they span only `K-1` dimensions, so they are all orthogonal to the
    common mode yet still mutually correlated. Measured residual here is
    ~0.24, not zero. Exact orthogonality is the job of the Stiefel step,
    and `test_whiten_drives_option_coherence_to_zero` pins that.
    A test asserting `c_max < 0.05` here would be asserting something CMR
    cannot deliver.
    """
    texts = ["the rate increases", "the rate decreases",
             "the rate is constant", "the rate is undefined"]
    book = np.stack([bb.encode(t).psi for t in texts], axis=0)
    out, _, rep = sb.common_mode_reject(book, book[0], gamma=1.0)
    assert rep["c_max_after"] < rep["c_max_before"]
    # Real reduction, but not orthogonality. Recorded as measured.
    assert rep["c_max_after"] < 0.5
    # Every row is orthogonal to the removed common mode.
    centroid = book.mean(axis=0)
    m = centroid / np.linalg.norm(centroid)
    for row in out:
        assert abs(np.real(np.vdot(m, row))) < 1e-9


def test_negative_control_gamma_zero_does_nothing(bb):
    """A control that cannot fail is not a control."""
    texts = ["the rate increases", "the rate decreases"]
    book = np.stack([bb.encode(t).psi for t in texts], axis=0)
    _, _, rep = sb.common_mode_reject(book, None, gamma=0.0)
    assert rep["c_max_after"] == rep["c_max_before"]


def test_whiten_drives_option_coherence_to_zero(bb):
    texts = ["alpha increases", "alpha decreases", "beta increases",
             "beta decreases"]
    book = np.stack([bb.encode(t).psi for t in texts], axis=0)
    out, _, rep = sb.whiten_options(book)
    assert rep["whitened"] is True
    assert rep["c_max_after"] < 1e-6
    assert rep["gram_side"] == 4


def test_delta_memory_exact_capacity_is_rank():
    """The derived bound, not a remembered number."""
    for rank in (8, 32, 64):
        rec = sb.delta_memory_capacity(rank=rank, dim=4096, m=rank)
        assert rec["exact_capacity"] == rank
        assert rec["verdict"] == "WITHIN_EXACT"
    over = sb.delta_memory_capacity(rank=64, dim=4096, m=65)
    assert over["verdict"] == "EXCEEDS_EXACT"


def test_delta_memory_measured_recall_stops_at_rank():
    """Measured confirmation of the rank wall at small D."""
    from henri.memory import DeltaMemory

    dim, rank, m = 512, 16, 32
    gen = np.random.default_rng(13)
    mem = DeltaMemory.build(dim=dim, rank=rank)
    keys, vals = [], []
    for _ in range(m):
        k = gen.standard_normal(dim) + 1j * gen.standard_normal(dim)
        k /= np.linalg.norm(k)
        v = gen.standard_normal(dim) + 1j * gen.standard_normal(dim)
        v /= np.linalg.norm(v)
        keys.append(k); vals.append(v)
        mem.write(k, v)
    errs = []
    for k, v in zip(keys, vals):
        p = mem.read(k, iterations=1)
        n = np.linalg.norm(p)
        p = p / n if n > 0 else p
        errs.append(float(np.linalg.norm(p - v)))
    # Beyond rank the residual is large for at least some associations.
    assert max(errs) > 0.3
