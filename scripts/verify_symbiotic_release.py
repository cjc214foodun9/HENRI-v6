#!/usr/bin/env python3
"""3-zone symbiotic release verification (HENRI-v7-BTO-SYMBIONT).

Every counted gate states the observation that makes it FAIL. Measurements no
realistic sample can decide are REPORTED, not counted.

Deviations from the source document, each deliberate:
  - tau_veto = 0.0431, the constant this tree ships. The document's 0.00811 has
    zero instantiations here.
  - Theorem 9.1 is tested as CONTROLLABILITY (authority grows with T), not as
    universality. Phase range is not the limiting factor; parameter count is.
  - Zone C is an in-memory ledger. TimescaleDB is BLOCKED and says so.
"""
from __future__ import annotations

import hashlib
import math
import sys
import time

import numpy as np

from henri.bto_pockels_d2nn import (
    BTODeepDiffractiveNetwork,
    BTOPockelsDiffractiveLayer,
    curie_weiss_gain,
)
from henri.holographic_ledger import (
    HolographicAgenticLedger,
    InMemoryLedgerStore,
    LedgerRecord,
)
from henri.symbiotic_cavity import HenriSymbioticCavity
from henri.transducer_swarm import ZoneATransducerSwarm

RESULTS = []
TAU = 0.0431


def record(gate: str, ok: bool, detail: str) -> None:
    RESULTS.append((gate, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {gate}: {detail}")


def unit(d: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal(d) + 1j * rng.standard_normal(d)
    return v / np.linalg.norm(v)


# ------------------------------------------------------------------------ ZONE A
def gate_zone_a() -> None:
    rng = np.random.default_rng(2026)
    swarm = ZoneATransducerSwarm(d_model=16384, seed=11)

    e = swarm.encode_text(["photon"])
    o = swarm.encode_operator(["expands"])
    a = swarm.route(slot0_entity=e)
    b = swarm.route(slot1_action=o)
    ip = abs(complex(np.vdot(a, b)))
    record("ZONE-A slot disjointness", ip <= 1e-15, f"|<slot0,slot1>|={ip:.2e} (bar 1e-15)")

    grid = rng.integers(0, 4, (8, 8))
    w = swarm.encode_grid(grid)
    dec = swarm.decode_grid(w, grid.shape)
    acc = float((dec == grid).mean())
    saved = swarm.cell_base.copy()
    swarm.cell_base = np.roll(saved, 7, axis=0)
    dec_bad = swarm.decode_grid(w, grid.shape)
    acc_bad = float((dec_bad == grid).mean())
    swarm.cell_base = saved
    ok = acc == 1.0 and acc_bad <= 0.35
    record(
        "ZONE-A grid round-trip", ok,
        f"cells={int(acc * grid.size)}/{grid.size} (bar 64/64), scrambled-code control={acc_bad:.3f} (bar <=0.35, chance 0.25)",
    )

    p0 = swarm.project_wave_to_slm_prefix(a)
    p1 = swarm.project_wave_to_slm_prefix(b)
    shape_ok = p0.shape == (16, 64) and p1.shape == (16, 64)
    distinct = not np.allclose(p0, p1)
    record(
        "ZONE-A SLM prefix", shape_ok and distinct,
        f"shape={p0.shape} (bar (16,64)), distinct={distinct}",
    )

    lex = swarm.lexicon
    i1 = lex.register_token("photon")
    i2 = lex.register_token("light")
    lex.register_token("granite")
    lex.diffuse_ppmi_graph(np.array([i1, i2]), np.array([i2, i1]), np.array([3.0, 3.0]), steps=35)
    syn = lex.similarity("photon", "light")
    unrel = abs(lex.similarity("photon", "granite"))
    record(
        "ZONE-A text grounding", syn >= 0.70 and unrel < 0.08,
        f"synonym={syn:.6f} (bar 0.70), unrelated={unrel:.6f} (bar 0.08)",
    )


# ------------------------------------------------------------------------ ZONE B
def gate_zone_b() -> None:
    rng = np.random.default_rng(7)
    d = 16384
    psi = unit(d, rng)

    thin = BTOPockelsDiffractiveLayer(d_model=d, pockels_gain=0.02, kerr=0.0, fresnel_zeta=0.0)
    out, _, _ = thin.step(psi)
    ident = float(np.linalg.norm(out - psi))
    record("ZONE-B identity at V=0,T=0", ident <= 1e-12, f"||out-in||={ident:.2e} (bar 1e-12)")

    full = BTOPockelsDiffractiveLayer(d_model=d, pockels_gain=0.02, kerr=0.05, fresnel_zeta=0.5)
    volt = np.full(d, 0.3)
    out2, gain2, corr2 = full.step(psi, (0.7, 0.7, 0.7, 0.7), volt)
    drift = abs(float(np.linalg.norm(out2)) - 1.0)
    record(
        "ZONE-B norm preservation", drift <= 1e-12 and corr2 <= 1e-12,
        f"norm_drift={drift:.2e} retraction_correction={corr2:.2e} max_gain={gain2:.3f} (bar 1e-12)",
    )

    ts = [0.0, 0.1, 0.3, 0.6, 0.9, 0.98]
    gains = [curie_weiss_gain(t) for t in ts]
    mono = all(b > a for a, b in zip(gains, gains[1:]))
    record(
        "ZONE-B Curie-Weiss gain", gains[0] == 1.0 and mono,
        f"Lambda(0)={gains[0]:.12f} (bar 1.0), range {gains[0]:.3f}->{gains[-1]:.3f}, monotone={mono}",
    )

    v = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    lam = curie_weiss_gain(0.6)
    got = full.pockels_phase(v, 0.6)
    want = -0.02 * lam * v
    err = float(np.max(np.abs(got - want)))
    record(
        "ZONE-B Pockels linearity", err <= 1e-15,
        f"max|phase-(-g*Lambda*V)|={err:.2e} at Lambda={lam:.4f} (bar 1e-15)",
    )

    # `reachable_shift` is a network-level diagnostic; a 1-layer network is the
    # exact baseline (no cascading), so the ratio isolates the Curie gain.
    pure = BTODeepDiffractiveNetwork(
        d_model=d, num_layers=1, pockels_gain=0.02, kerr=0.0, fresnel_zeta=0.0
    )
    shift_cold = pure.reachable_shift(psi, 1e-3, 0.0)
    shift_hot = pure.reachable_shift(psi, 1e-3, 0.95)
    ratio = shift_hot / max(shift_cold, 1e-18)
    kill = BTODeepDiffractiveNetwork(
        d_model=d, num_layers=1, pockels_gain=0.02, kerr=0.0, fresnel_zeta=0.0, kappa=0.0
    )
    k_ratio = kill.reachable_shift(psi, 1e-3, 0.95) / max(kill.reachable_shift(psi, 1e-3, 0.0), 1e-18)
    record(
        "ZONE-B reachable authority", ratio >= 5.0 and abs(k_ratio - 1.0) < 0.01,
        f"hot/cold shift ratio={ratio:.3f} (bar >=5), kappa=0 control={k_ratio:.6f} (bar ~1.0)",
    )

    fwd, _, _ = thin.step(psi, (0.5,) * 4, np.full(d, 0.7))
    back, _, _ = thin.conjugate_step(fwd, (0.5,) * 4, np.full(d, 0.7))
    inv = float(np.linalg.norm(back - psi))
    record(
        "ZONE-B phase+diffraction inverse", inv <= 1e-12,
        f"||U^-1 U psi - psi||={inv:.2e} (bar 1e-12; Kerr and post-phase excluded, documented)",
    )


# ------------------------------------------------------------------------ ZONE C
class _FailingStore(InMemoryLedgerStore):
    """Succeeds for `fail_after` appends, then raises. A store that always
    raises cannot even seed the ledger, so the atomicity gate would never run."""

    def __init__(self, fail_after: int = 1) -> None:
        super().__init__()
        self.fail_after = int(fail_after)

    def append(self, record: LedgerRecord) -> None:
        if self.count() >= self.fail_after:
            raise RuntimeError("injected store failure")
        super().append(record)


def gate_zone_c() -> None:
    rng = np.random.default_rng(17)
    d = 16384
    led = HolographicAgenticLedger(d_model=d, d_tiny=64, prefix_tokens=16)
    for m in range(8):
        k = unit(d, rng)
        led.commit(k, k, label=f"engram_{m}")

    cmax = led.max_coherence()
    record("ZONE-C Stiefel coherence", cmax < 1e-6, f"c_max={cmax:.2e} (bar 1e-6), engrams={led.size}")

    w = unit(d, rng)
    bins = led.quantize(w)
    back = led.dequantize(bins)
    phase_err = float(np.max(np.abs(np.angle(w * np.conj(back)))))
    # Phase-only fidelity. A raw inner product is ||w||_1 (~113), and dividing
    # by D over-corrects to ~0.007. The quantity that actually measures the
    # quantiser is mean cos(angle(w) - angle(back)), which is 1.0 at zero error.
    cos_norm = float(np.mean(np.real(np.exp(1j * (np.angle(w) - np.angle(back))))))
    bound = math.pi / 256.0
    # Kill control: the same wave through a 16-bin quantiser must fall BELOW the
    # 256-bin bar, otherwise the gate cannot distinguish resolutions.
    two_pi = 2.0 * math.pi
    coarse = np.rint((np.angle(w) % two_pi) * (16 / two_pi)).astype(np.int64) % 16
    coarse_back = np.exp(1j * coarse * (two_pi / 16))
    cos_coarse = float(np.mean(np.real(np.exp(1j * (np.angle(w) - np.angle(coarse_back))))))
    record(
        "ZONE-C uint8 round-trip",
        phase_err <= bound + 1e-12 and cos_norm >= math.cos(bound) - 1e-12 and cos_coarse < math.cos(bound),
        f"max_phase_err={phase_err:.2e} (bar {bound:.2e}), cosine={cos_norm:.8f} (bar {math.cos(bound):.8f}), "
        f"16-bin control={cos_coarse:.6f} (must be below the bar)",
    )

    v1, p1, ms1, hops1 = led.jit_page(w)
    for m in range(56):
        k = unit(d, rng)
        led.commit(k, k, label=f"bulk_{m}")
    v2, p2, ms2, hops2 = led.jit_page(w)
    shape_ok = p1.shape == p2.shape == (16, 64)
    bounded = float(np.max(np.abs(v2))) <= 1.5 + 1e-12
    record(
        "ZONE-C JIT paging", shape_ok and bounded,
        f"prefix shape {p1.shape}->{p2.shape} (bar invariant), |V|max={float(np.max(np.abs(v2))):.3f} (bar <=1.5), "
        f"latency={ms1:.2f}->{ms2:.2f} ms on CPU, charts={hops1}->{hops2} (depth, not SNR)",
    )

    broken = HolographicAgenticLedger(d_model=d, store=_FailingStore())
    broken.commit(unit(d, rng), unit(d, rng), label="seed")
    before_size = broken.size
    before_hash = hashlib.sha256(np.round(broken.boundary.real, 12).tobytes()).hexdigest()
    raised = False
    try:
        broken.commit(unit(d, rng), unit(d, rng), label="must_fail")
    except RuntimeError:
        raised = True
    after_size = broken.size
    after_hash = hashlib.sha256(np.round(broken.boundary.real, 12).tobytes()).hexdigest()
    record(
        "ZONE-C commit atomicity", raised and before_size == after_size and before_hash == after_hash,
        f"raised={raised}, size {before_size}->{after_size}, boundary_hash_equal={before_hash == after_hash}",
    )


# ---------------------------------------------------------------------- CAVITY
def gate_cavity() -> None:
    cav = HenriSymbioticCavity(d_model=16384, tau_veto=TAU, max_passes=8, seed=5)
    cav.teach_law(["photon"], ["expands"], ["wavepacket"])

    t1 = cav.step(["photon"], ["expands"])
    record(
        "CAVITY taught law (System 1)",
        t1.passes == 1 and t1.heated_passes == 0 and t1.converged
        and t1.label == "expands → wavepacket" and t1.margin >= 0.65,
        f"passes={t1.passes} heated={t1.heated_passes} converged={t1.converged} "
        f"label={t1.label!r} margin={t1.margin:.4f} cos={[round(c, 4) for c in t1.cos_trace]}",
    )

    t2 = cav.step(["photon"], ["expands"], phase_error=1.2)
    record(
        "CAVITY Curie repair (System 2)",
        t2.heated_passes >= 1 and t2.converged and t2.cos_trace[-1] >= 0.99,
        f"heated={t2.heated_passes} converged={t2.converged} final_cos={t2.cos_trace[-1]:.6f} "
        f"trace={[round(c, 4) for c in t2.cos_trace]}",
    )

    kill = HenriSymbioticCavity(d_model=16384, tau_veto=TAU, max_passes=8, kappa=0.0, seed=5)
    kill.teach_law(["photon"], ["expands"], ["wavepacket"])
    t3 = kill.step(["photon"], ["expands"], phase_error=1.2)
    record(
        "CAVITY kill control (kappa=0)", (not t3.converged) and t3.cos_trace[-1] < 0.95,
        f"converged={t3.converged} final_cos={t3.cos_trace[-1]:.6f} (bar not converged, <0.95); "
        f"trace={[round(c, 4) for c in t3.cos_trace]}",
    )

    t4 = cav.step(["granite"], ["expands"])
    novel_absent = all("granite" not in r.label for r in cav.ledger.records.values())
    record(
        "CAVITY compositional transfer",
        t4.converged and novel_absent and t4.boundary_cos >= 0.99,
        f"converged={t4.converged} boundary_cos={t4.boundary_cos:.6f} label={t4.label!r} "
        f"novel_entity_absent_from_ledger={novel_absent} (reported, not a generalization claim)",
    )


def main() -> int:
    np.random.seed(20261026)
    t0 = time.time()
    print("=== HENRI-v7-BTO-SYMBIONT VERIFICATION (NumPy complex128, CPU) ===")
    gate_zone_a()
    gate_zone_b()
    gate_zone_c()
    gate_cavity()
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    total = len(RESULTS)
    print()
    print(f"=== {passed}/{total} counted gates passed in {time.time() - t0:.1f}s ===")
    if passed != total:
        for name, ok, detail in RESULTS:
            if not ok:
                print(f"  FAILED {name}: {detail}")
    print("[INFO] NOT ESTABLISHED: OOD generality, 16.7M engram capacity, Turing completeness,")
    print("[INFO] BTO physical hardware, TimescaleDB, SLM, CUDA latency. None are claimed.")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
