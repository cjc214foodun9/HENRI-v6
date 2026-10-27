#!/usr/bin/env python3
"""Omega release verification suite (HENRI-v7-OMEGA).

Every counted gate states, in its own docstring, the observation that makes it
FAIL. Gates that cannot fail are not counted. Measurements that no realistic
sample can decide are REPORTED, not counted.

Run:  python scripts/verify_omega_release.py
Exit: 0 only when every counted gate passes.
"""
from __future__ import annotations

import ast
import math
import sys
import time

import numpy as np

from henri.egress_router import OmniModalEgressRouter
from henri.harmonic_lexicon import HarmonicLexicon
from henri.hhrg_memory import HHRGMemory
from henri.holographic_stack import HolographicCallStack
from henri.koopman_sheaf import PhaseGatedKoopmanSheaf, retract_to_sphere
from henri.omega_pipeline import HenriOmegaPipeline
from henri.omega_resonator import AdiabaticResonator
from henri.triadic_sagnac import TriadicSagnacGate

RESULTS = []


def record(gate: str, ok: bool, detail: str) -> None:
    RESULTS.append((gate, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {gate}: {detail}")


def rand_unit(d: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal(d) + 1j * rng.standard_normal(d)
    return v / np.linalg.norm(v)


def orth(d: int, r: int, rng: np.random.Generator) -> np.ndarray:
    a = rng.standard_normal((d, r)) + 1j * rng.standard_normal((d, r))
    q, _ = np.linalg.qr(a)
    return q


def wilson_lower(successes: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.0
    p = successes / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - margin) / denom)


# --------------------------------------------------------------------- OMEGA-1
def gate_omega_1() -> None:
    """Fails if diffusion does not move synonyms together, or moves unrelated tokens."""
    lex = HarmonicLexicon(vocab_size=256, d_slot=4096, d_sem=64, seed=11)
    a = lex.register_token("photon")
    b = lex.register_token("light")
    lex.register_token("granite")

    control = HarmonicLexicon(vocab_size=256, d_slot=4096, d_sem=64, seed=11)
    control.register_token("photon")
    control.register_token("light")
    control.register_token("granite")
    sim_before = control.similarity("photon", "light")

    lex.diffuse_ppmi_graph(
        np.array([a, b]), np.array([b, a]), np.array([3.0, 3.0]), steps=35
    )
    sim_syn = lex.similarity("photon", "light")
    sim_unrel = abs(lex.similarity("photon", "granite"))

    ok = (sim_syn >= 0.70) and (sim_unrel < 0.08) and (sim_before < 0.10)
    record(
        "OMEGA-1 lexicon",
        ok,
        f"syn={sim_syn:.6f} (bar 0.70), unrelated={sim_unrel:.6f} (bar 0.08), "
        f"no-diffusion control={sim_before:.6f} (must be < 0.10)",
    )


# --------------------------------------------------------------------- OMEGA-2
def gate_omega_2a() -> None:
    """Fails if peek returns the wrong frame or pop does not restore the parent."""
    rng = np.random.default_rng(5)
    d = 4096
    stack = HolographicCallStack(d_slot=d, seed=7)
    goals = np.stack([rand_unit(d, rng) for _ in range(6)], axis=0)

    s = np.zeros(d, dtype=np.complex128)
    for g in range(4):
        s = stack.push(s, goals[g])

    # Pop the top three frames. The FOURTH pop empties the stack by construction,
    # so the parent cosine must be measured BEFORE it, not after.
    peek_ok = True
    for g in (3, 2, 1):
        idx, _sim = stack.peek(s, goals)
        if idx != g:
            peek_ok = False
            break
        s = stack.pop(s, goals[g])

    recover = float(np.real(np.vdot(s / np.linalg.norm(s), goals[0])))
    empty_residual = float(np.linalg.norm(stack.pop(s, goals[0])))
    ok = peek_ok and recover >= 0.98 and empty_residual < 0.05
    record(
        "OMEGA-2a call-stack",
        ok,
        f"peek_order_correct={peek_ok}, recovered_parent_cos={recover:.6f} (bar 0.98), "
        f"final_pop_residual={empty_residual:.4f} (<0.05), decay_bound(D=4096)={stack.depth_bound()}",
    )


def gate_omega_2b() -> None:
    """Fails if the operator cannot reduce prediction error on a rank-r target.

    The target is constructed INSIDE the operator's image class. A negative arm
    with an orthogonal correction subspace must fail the same bar.
    """
    rng = np.random.default_rng(9)
    d_model, r = 16384, 32

    sheaf = PhaseGatedKoopmanSheaf(d_model=d_model, d_slot=d_model // 4, rank=r, num_charts=2, seed=3)
    psi = rand_unit(d_model, rng)

    identity_err = float(np.linalg.norm(psi - sheaf.step(psi, chart_idx=0)[0]))

    random_diag = sheaf.diag_m.copy()
    ang = rng.uniform(0, 2 * math.pi, (2, d_model))
    random_diag[:] = np.exp(1j * ang)
    sheaf.diag_m = random_diag
    broken_cos = float(np.real(np.vdot(sheaf.step(psi, chart_idx=0)[0], psi)))
    sheaf.diag_m = np.ones_like(random_diag)

    a_true = orth(d_model, r, rng)
    a_perp = np.linalg.qr(np.concatenate([a_true, orth(d_model, r, rng)], axis=1))[0][:, r : 2 * r]

    c = rng.standard_normal(r) + 1j * rng.standard_normal(r)
    c = c / np.linalg.norm(c)

    sheaf.stiefel_A = a_true
    target = retract_to_sphere(psi + a_true @ c)
    err_before = float(np.linalg.norm(target - sheaf.step(psi, chart_idx=0)[0]))
    for _ in range(4):
        sheaf.online_edmd_update(psi, target, chart_idx=0)
    err_after = float(np.linalg.norm(target - sheaf.step(psi, chart_idx=0)[0]))
    reduction = 1.0 - err_after / max(err_before, 1e-12)

    # Class ceiling. Any operator of the form retract(psi + A s) can reach at
    # most the projection of (target - psi) onto span(A). The bar is that
    # ceiling, not an arbitrary fraction: an unlearned operator sits at
    # err_before, a broken subspace sits at err_before, a correct learner sits
    # at the floor.
    floor_vec = retract_to_sphere(psi + a_true @ (a_true.conj().T @ (target - psi)))
    floor_err = float(np.linalg.norm(target - floor_vec))

    kill = PhaseGatedKoopmanSheaf(d_model=d_model, d_slot=d_model // 4, rank=r, num_charts=2, seed=3)
    kill.stiefel_A = a_perp
    kill_before = float(np.linalg.norm(target - kill.step(psi, chart_idx=0)[0]))
    for _ in range(4):
        kill.online_edmd_update(psi, target, chart_idx=0)
    kill_after = float(np.linalg.norm(target - kill.step(psi, chart_idx=0)[0]))
    kill_reduction = 1.0 - kill_after / max(kill_before, 1e-12)

    ok = (
        identity_err < 1e-9
        and broken_cos < 0.10
        and err_after <= floor_err * 1.02 + 1e-12
        and kill_reduction < 0.10
    )
    record(
        "OMEGA-2b koopman-sheaf",
        ok,
        f"identity_err={identity_err:.2e}, random-diag cos={broken_cos:+.4f} (<0.10), "
        f"err_after={err_after:.4e} vs class_floor={floor_err:.4e} (<=1.02x floor), "
        f"reduction={reduction:.4f}, orthogonal-subspace control={kill_reduction:+.4f} (<0.10)",
    )


# --------------------------------------------------------------------- OMEGA-3
def gate_omega_3() -> None:
    """Fails if a single observation anneals, or a valid axiom is not vetoed."""
    rng = np.random.default_rng(13)
    d = 16384
    gate = TriadicSagnacGate(d_model=d, num_slots=4, tau_veto=0.0431, anneal_rate=0.5, confirm_threshold=2)

    pred = rand_unit(d, rng)
    inv = rand_unit(d, rng)

    cold = gate.evaluate(pred, psi_inv=None)
    veto = gate.evaluate(pred, psi_inv=inv)

    gate2 = TriadicSagnacGate(d_model=d, num_slots=4, tau_veto=0.0431, anneal_rate=0.5, confirm_threshold=2)
    first = gate2.evaluate(pred, psi_inv=inv, psi_sens=pred)
    second = gate2.evaluate(pred, psi_inv=inv, psi_sens=pred)

    ok = (
        cold.epistemic_regime == "COLD_ANCHORED"
        and not cold.veto_fired
        and veto.veto_fired
        and veto.epistemic_regime == "HALLUCINATION_VETO"
        and first.epistemic_regime != "AXIOM_ANNEALED"
        and second.epistemic_regime == "AXIOM_ANNEALED"
    )
    record(
        "OMEGA-3 triadic-sagnac",
        ok,
        f"cold={cold.epistemic_regime}, veto={veto.epistemic_regime}, "
        f"one_observation={first.epistemic_regime} (must not anneal), "
        f"two_observations={second.epistemic_regime}",
    )


# --------------------------------------------------------------------- OMEGA-4
def gate_omega_4() -> None:
    """Fails if chart coherence exceeds 1e-6 or a stored key is not retrieved."""
    rng = np.random.default_rng(17)
    d = 4096
    mem = HHRGMemory(d_model=d, chart_capacity=32)
    keys = []
    cmax = 0.0
    for m in range(48):
        k = rand_unit(d, rng)
        keys.append(k)
        _, c = mem.ingest(k, k, label=f"engram_{m}")
        cmax = max(cmax, c)

    hit = mem.query(keys[5])
    miss = mem.query(rand_unit(d, rng))

    ok = (
        len(mem.charts) >= 2
        and cmax < 1e-6
        and hit.margin >= 0.70
        and hit.entropy_norm < 0.01
        and miss.margin < 0.10
    )
    record(
        "OMEGA-4 hhrg-memory",
        ok,
        f"charts={len(mem.charts)}, engrams={mem.total_engrams}, c_max={cmax:.2e} (<1e-6), "
        f"hit_margin={hit.margin:.4f} (>=0.70), hit_H={hit.entropy_norm:.4f} (<0.01), "
        f"negative_margin={miss.margin:.4f} (<0.10), negative_H={miss.entropy_norm:.4f}",
    )


# --------------------------------------------------------------------- OMEGA-5
def gate_omega_5() -> None:
    """Counted part: the exact arm must recover a planted 3-way factorization.

    Fails if the recovered indices differ from the planted ones.
    The iterative arm is REPORTED with its interval; n=8 cannot establish 99%.
    """
    rng = np.random.default_rng(19)
    d = 4096
    m = 16
    cbs = [np.exp(1j * rng.uniform(0, 2 * math.pi, (m, d))) for _ in range(3)]
    true = [3, 7, 11]
    comp = cbs[0][true[0]] * cbs[1][true[1]] * cbs[2][true[2]]

    res = AdiabaticResonator(d_slot=d, tau_lock=0.0431, seed=23)
    idx, err, _steps = res.factorize_exact(comp, cbs)
    ok = idx == true and err <= 1e-6
    record(
        "OMEGA-5a resonator-exact",
        ok,
        f"recovered={idx} (planted {true}), err={err:.2e} (<=1e-6)",
    )

    hits = 0
    trials = 8
    t0 = time.time()
    for _ in range(trials):
        i, e, _ = res.factorize_iterative(comp, cbs, num_restarts=3)
        if i == true and e <= 0.0431:
            hits += 1
    lower = wilson_lower(hits, trials)
    print(
        f"[INFO] OMEGA-5b resonator-iterative: {hits}/{trials} exact in {time.time() - t0:.1f}s, "
        f"Wilson95 lower bound {lower:.3f}. The specification's >=99% claim is NOT established at n={trials}."
    )

    other = cbs[0][1] * cbs[1][5] * cbs[2][9]
    idx2, err2, _ = res.factorize_exact(other, cbs)
    record(
        "OMEGA-5c resonator-kill-control",
        idx2 == [1, 5, 9] and err2 <= 1e-6,
        f"second planted triple recovered as {idx2} (planted [1, 5, 9]); a lookup table would return {true}",
    )


# --------------------------------------------------------------------- OMEGA-6
def gate_omega_6() -> None:
    """Fails if the grammar mask does not change an otherwise-invalid selection."""
    rng = np.random.default_rng(29)
    router = OmniModalEgressRouter(d_model=16384, seed=31)
    cb = router.ast_codebook if hasattr(router, "ast_codebook") else None
    if cb is None:
        raw = rng.standard_normal((5, 16384)) + 1j * rng.standard_normal((5, 16384))
        cb = np.stack([retract_to_sphere(r) for r in raw], axis=0)

    target = retract_to_sphere(cb[3] * 4.0 + np.sum(cb, axis=0))
    raw_sims = np.real(cb @ np.conj(target))
    raw_argmax = int(np.argmax(raw_sims))

    out = router.decode_ast(target, cb)
    ast.parse(str(out.content))

    ok = (
        raw_argmax == 3
        and out.selected[0] == 0
        and out.syntax_valid
        and out.notes == "mask_changed_selection"
    )
    record(
        "OMEGA-6 ast-mask",
        ok,
        f"unmasked_argmax={raw_argmax} (invalid for START), masked_selection={out.selected} "
        f"(START must be 0), syntax_valid={out.syntax_valid}, note={out.notes}",
    )


# --------------------------------------------------------------------- OMEGA-7
def gate_omega_7() -> None:
    """Fails if the cold-start text egress is wrong or the negative query is not refused."""
    pipe = HenriOmegaPipeline(d_model=16384, rank=32, num_charts=4, auto_bootstrap=True, seed=41)
    pos = pipe.step(query_tokens=["light"], egress_mode="text")
    neg = pipe.step(query_tokens=["granite"], egress_mode="text")

    ast_step = pipe.step(query_tokens=["python", "sort", "list"], egress_mode="code_ast")
    ast.parse(str(ast_step.output.content))
    motor = pipe.step(query_tokens=["unitary", "stiefel"], egress_mode="motor_efe")

    ok = (
        pos.output.content == "electromagnetic_wave_packet"
        and pos.output.margin >= 0.65
        and pos.thermostat_steps <= 3
        and pos.output.sagnac_verified
        and neg.output.content == "<abstain>"
        and ast_step.output.syntax_valid
        and np.all(np.abs(motor.output.content) <= 1.0)
        # The egress decision must be independent of the repair loop: it reads
        # the pre-thermostat retrieval. Otherwise a broken Zone B scores by
        # collapsing onto memory and the gate is vacuous.
        and abs(pos.output.margin - pos.retrieval.margin) < 1e-12
        and pos.retrieval_repaired is not None
        and pos.retrieval_repaired.max_similarity >= pos.retrieval.max_similarity
    )
    record(
        "OMEGA-7 omega-pipeline",
        ok,
        f"text='{pos.output.content}' margin={pos.output.margin:.4f} "
        f"thermostat_steps={pos.thermostat_steps} veto={not pos.output.sagnac_verified}; "
        f"egress_reads_pre_repair_margin={pos.output.margin:.4f}=={pos.retrieval.margin:.4f}; "
        f"repaired_sim={pos.retrieval_repaired.max_similarity:.4f}>=pre={pos.retrieval.max_similarity:.4f}; "
        f"negative='{neg.output.content}' margin={neg.output.margin:.4f}; "
        f"ast_ok={ast_step.output.syntax_valid} ast_sel={ast_step.output.selected}; "
        f"motor_max={float(np.max(np.abs(motor.output.content))):.4f}",
    )


def main() -> int:
    np.random.seed(20261026)
    t0 = time.time()
    print("=== HENRI-v7-OMEGA VERIFICATION (numpy complex128, CPU) ===")
    gate_omega_1()
    gate_omega_2a()
    gate_omega_2b()
    gate_omega_3()
    gate_omega_4()
    gate_omega_5()
    gate_omega_6()
    gate_omega_7()

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    total = len(RESULTS)
    print()
    print(f"=== {passed}/{total} counted gates passed in {time.time() - t0:.1f}s ===")
    if passed != total:
        for name, ok, detail in RESULTS:
            if not ok:
                print(f"  FAILED {name}: {detail}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
