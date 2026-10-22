"""Domain-routed Koopman wave transitions, coupled to cartridge recall.

WHY THIS MODULE EXISTS
----------------------
Phase 8, ACTION 2. `koopman.py`, `kuramoto.py`, `langevin.py` and
`active_inference_loop.py` all pass their own tests and none of them is
reached from `api.py`. This module is a thin adapter that puts the sealed
`KoopmanOperator` and the sealed `CartridgeBank` on the scoring path
WITHOUT editing any sealed file. Every kernel object here is imported, not
modified.

WHAT THE COUPLED SCORE IS
-------------------------
For a specification wave `Psi_Q` and one candidate wave `Psi_C`:

    s_static = Re<Psi_Q, Psi_C>                     the Phase 7 control
    s_trans  = Re<K_d Psi_Q, Psi_C>                 one Koopman step, domain d
    s_mem    = (1 - delta_sagnac(v_recall, Psi_C))   only when the bank HITS

    score = s_static + w_trans * s_trans + w_mem * s_mem * hit

`d` is the domain bucket. A domain is a bucket of the subject label, hashed
deterministically; the operators share nothing, so routing is a real choice
rather than a label.

WHAT IS NOT IMPLEMENTED, AND WHY
--------------------------------
No thermostat runs here. `langevin.py` and `anisotropic_thermostat.py` are
wired into the live path by `henri/robotics_loop.py`, where a trajectory
needs relaxing. A RANKING has no trajectory to relax, so running a Langevin
bath to reorder four options would be machinery without a mechanism. The
Sagnac gate does act here, as a DOWN-WEIGHT: a candidate whose recalled
consequent is incoherent with it loses score. That is the gate doing work,
not a bath being simulated.

G-KOOP1 PRE-REGISTRATION, FIXED BEFORE THE FIRST RUN
----------------------------------------------------
The weight grid is `W_GRID x W_MEM_GRID`. Weights are chosen on the FIT
split ONLY, and the tie-break prefers the SMALLEST total weight, so `w = 0`
wins every tie. That is what makes the gate able to fail.

  PASS  held-out top-1 with the fitted weights is STRICTLY greater than
        held-out top-1 with `w = 0`, and a paired McNemar test does not
        contradict it.
  FAIL  otherwise. On FAIL the module reports `w_trans = w_mem = 0.0` and
        the coupling stays INACTIVE for that domain. A transition that does
        not beat doing nothing is not installed; the negative is recorded.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from henri.config import WaveConfig
from henri.form_pin import INVARIANT_FORM
from henri.koopman import KoopmanOperator
from henri.sagnac_homodyne_gate import SagnacHomodyneGate
from henri.state import WaveState

TAU_VETO = 0.0431
DOMAIN_STRIDE = 7919
N_DOMAINS = 4

# Grid fixed before any run. `0.0` is present in both, so the static control
# is a member of the search space and a tie cannot silently favour coupling.
W_GRID = (0.0, 0.05, 0.10, 0.20, 0.40)
W_MEM_GRID = (0.0, 0.05, 0.10, 0.20)


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v)
    n = float(np.linalg.norm(v))
    return v if n == 0.0 else v / n


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    """Real cosine of two waves. Both are normalised here, never assumed."""
    return float(np.real(np.vdot(_unit(a), _unit(b))))


def domain_of(subject: str, n_domains: int = N_DOMAINS) -> int:
    """Deterministic domain bucket. sha256, not `hash()`.

    `hash()` on a str is salted per process in CPython, so a bucket derived
    from it would differ between the fitting run and the serving run. That
    is a quiet way to serve a different operator than the one calibrated.
    """
    h = hashlib.sha256((subject or "").strip().lower().encode("utf-8"))
    return int(h.hexdigest()[:8], 16) % int(max(1, n_domains))


def config_for(backbone, probe: str = "probe", **overrides) -> WaveConfig:
    """`WaveConfig` sized from the BACKBONE'S ACTUAL WAVE WIDTH.

    The wave is authoritative. Phase 7 measured this exact defect class:
    `api_server.py` sized a bank from `api.backbone.dim`, which does not
    exist, so every write would have raised. Guessing the width is the
    error; reading it from one encode is the repair.
    """
    w = int(np.asarray(backbone.encode(probe).psi).shape[0])
    return WaveConfig(dim=w, **overrides)


@dataclass
class RecallHit:
    """One cartridge lookup, with the numbers that justify the decision."""
    value: Optional[np.ndarray]
    margin: float
    t: float
    hit: bool
    cartridge: int
    rows_searched: int

    def to_dict(self) -> dict:
        return {"hit": bool(self.hit),
                "margin": round(float(self.margin), 9),
                "t": round(float(self.t), 9),
                "cartridge": int(self.cartridge),
                "rows_searched": int(self.rows_searched)}


class DomainKoopmanBank:
    """One sealed `KoopmanOperator` per domain, all sharing a config.

    Args:
        cfg: wave configuration. `dim` and `r_koopman` are read from it.
        n_domains: bucket count.
        seed: base seed. Domain `d` uses `seed + d * DOMAIN_STRIDE`, so the
            operators differ and the derivation is reproducible.
    """

    def __init__(self, cfg: WaveConfig, n_domains: int = N_DOMAINS,
                 seed: int = 20261021) -> None:
        self.cfg = cfg
        self.n_domains = int(max(1, n_domains))
        self.seed = int(seed)
        self._ops = [
            KoopmanOperator.build(cfg, seed=self.seed + d * DOMAIN_STRIDE)
            for d in range(self.n_domains)
        ]
        # Counters, so a receipt can show routing actually varied.
        self.routed: dict[int, int] = {d: 0 for d in range(self.n_domains)}

    def operator(self, domain: int) -> KoopmanOperator:
        d = int(domain) % self.n_domains
        self.routed[d] += 1
        return self._ops[d]

    def step(self, psi: WaveState, domain: int) -> WaveState:
        """One projected transition. `step` never returns a drifted norm."""
        return self.operator(domain).step(psi)

    def transition_score(self, spec_psi: WaveState, cand_psi: WaveState,
                         domain: int) -> float:
        """`cos(K_d . Psi_Q, Psi_C)`. One transition per query, not per pair."""
        moved = self.step(spec_psi, domain)
        return _cos(moved.psi, cand_psi.psi)


class CartridgeRecall:
    """`CartridgeBank` recall of the consequent wave, gated by a Sagnac check.

    `CartridgeBank.query` composes `DeltaMemory.read` inside the hot tier,
    so this class does not reach into the sealed memory directly.

    Args:
        dim: wave dimension; must match the backbone.
        rank: `DeltaMemory` rank, also the per-cartridge row cap.
        max_rows: associations held, partitioned at `rank` per cartridge.
    """

    def __init__(self, dim: int, rank: int = 32,
                 max_rows: int = 32, dtype=None) -> None:
        from henri.cartridge_bank import CartridgeBank

        self.dim = int(dim)
        self.rank = int(rank)
        self.max_rows = int(max_rows)
        self.bank = CartridgeBank(
            dim=self.dim, rank=self.rank,
            dtype=np.complex64 if dtype is None else dtype)
        self._keys: list[np.ndarray] = []
        self._values: list[np.ndarray] = []
        self.gate = SagnacHomodyneGate(tau_veto=TAU_VETO,
                                       cfg=WaveConfig(dim=self.dim),
                                       statistic=INVARIANT_FORM)
        self._finalized = False

    def add(self, key_wave: np.ndarray, value_wave: np.ndarray) -> None:
        if len(self._keys) >= self.max_rows:
            return
        self._keys.append(np.asarray(key_wave).ravel())
        self._values.append(np.asarray(value_wave).ravel())

    def finalize(self) -> int:
        n = len(self._keys)
        if n == 0:
            return 0
        for start in range(0, n, self.rank):
            stop = min(start + self.rank, n)
            k = np.stack(self._keys[start:stop], axis=0)
            v = np.stack(self._values[start:stop], axis=0)
            self.bank.add_cartridge(k, v)
        self.bank.finalize()
        self._finalized = True
        return n

    @property
    def rows(self) -> int:
        return len(self._keys)

    def recall(self, spec_psi: WaveState) -> RecallHit:
        """Route to a cartridge, then read. A miss reports `hit=False`."""
        if not self._finalized or not self._keys:
            return RecallHit(None, 0.0, 0.0, False, -1, 0)
        r = self.bank.query(np.asarray(spec_psi.psi))
        return RecallHit(value=r.value, margin=float(r.margin),
                         t=float(r.t), hit=bool(r.hits),
                         cartridge=int(r.cartridge),
                         rows_searched=int(r.rows_searched))

    def coherence(self, recalled: Optional[np.ndarray],
                  cand_psi: WaveState) -> float:
        """`1 - delta_sagnac(recalled, candidate)`, in `[0, 1]` on a pass.

        The gate is the sealed one, evaluated on the invariant form. A
        recalled wave that is incoherent with the candidate lowers the
        score; that is the veto acting as a down-weight rather than a pass
        or a rejection.
        """
        if recalled is None:
            return 0.0
        try:
            psi_r = WaveState.from_complex(np.asarray(recalled))
        except Exception:                        # noqa: BLE001
            return 0.0
        d = float(self.gate.statistic_value(psi_r, cand_psi))
        return max(0.0, min(1.0, 1.0 - d))


class KoopmanRouter:
    """Static cosine plus a gated transition term plus gated recall.

    Args:
        backbone: anything with `.encode(text) -> object` carrying `.psi`.
        cfg: wave configuration for the operators.
        recall: optional `CartridgeRecall`.
        n_domains: domain buckets.
        seed: base seed.
        w_trans / w_mem: installed weights. Default `0.0`, which means the
            router IS the static control until `fit_weights` says otherwise.
    """

    def __init__(self, backbone, cfg: WaveConfig,
                 recall: Optional[CartridgeRecall] = None,
                 n_domains: int = N_DOMAINS, seed: int = 20261021,
                 w_trans: float = 0.0, w_mem: float = 0.0,
                 encoder=None) -> None:
        self.backbone = backbone
        self.cfg = cfg
        self.recall_bank = recall
        # PLUGGABLE ENCODER. MEASURED DEFECT, repaired. This router first
        # encoded only through `SemanticBackbone`, which strips symbolic
        # operators, so `a + b` and `a - b` scored `s_static = 0.69013433`
        # to eight identical decimals. ACTION 1 built the operator channel
        # in `henri/operator_ingress.py` but nothing on the scoring path
        # used it. Accepting an encoder makes that channel reachable
        # without touching a sealed file.
        self.encoder = encoder
        self.domains = DomainKoopmanBank(cfg, n_domains=n_domains,
                                         seed=seed)
        self.w_trans = float(w_trans)
        self.w_mem = float(w_mem)
        self.active = bool(w_trans > 0.0 or w_mem > 0.0)

    # -- encoding -------------------------------------------------------
    def _psi(self, text: str) -> WaveState:
        """Text -> `WaveState`. `Encoded.psi` -> sphere. MEASURED, repaired.

        `SemanticBackbone.encode` returns an `Encoded`, whose fields are
        `psi, q0, q1, pol, active_slots` and which has NO `cfg`. The sealed
        `KoopmanOperator.apply_raw` reads `state.cfg.dim`, so passing the
        `Encoded` straight through raised

            AttributeError: 'Encoded' object has no attribute 'cfg'

        on the first real call. The wave is lifted here with the same config
        the operators were built from, and a width mismatch fails loudly
        instead of silently mixing two dimensions.
        """
        if self.encoder is not None:
            arr = np.asarray(self.encoder(text))
        else:
            arr = np.asarray(self.backbone.encode(text).psi)
        w = int(arr.shape[0]) if arr.ndim else 0
        if w != int(self.cfg.dim):
            raise ValueError(
                f"encoder emits width {w} but the operators were built for "
                f"dim {self.cfg.dim}. Build the router with "
                f"`cfg=config_for(...)` so the wave decides the width.")
        return WaveState.from_complex(arr, self.cfg)

    # -- scoring --------------------------------------------------------
    def score_components(self, spec_psi: WaveState, cand_psi: WaveState,
                         subject: str = "") -> dict:
        """Every term of the coupled score, separately. Never only the sum.

        A single blended number cannot show which mechanism moved. Each term
        is reported so a receipt can attribute a pass or a failure.
        """
        d = domain_of(subject, self.domains.n_domains)
        s_static = _cos(spec_psi.psi, cand_psi.psi)
        s_trans = self.domains.transition_score(spec_psi, cand_psi, d)

        s_mem = 0.0
        recall = RecallHit(None, 0.0, 0.0, False, -1, 0)
        if self.recall_bank is not None and self.w_mem > 0.0:
            recall = self.recall_bank.recall(spec_psi)
            if recall.hit:
                s_mem = self.recall_bank.coherence(recall.value, cand_psi)

        total = (s_static
                 + self.w_trans * s_trans
                 + self.w_mem * s_mem)
        return {
            "domain": int(d),
            "s_static": round(float(s_static), 9),
            "s_trans": round(float(s_trans), 9),
            "s_mem": round(float(s_mem), 9),
            "recall": recall.to_dict(),
            "total": round(float(total), 9),
        }

    def rank(self, spec: str, candidates: Sequence[str],
             subject: str = "") -> dict:
        """Order candidates by the coupled score, with full provenance."""
        spec_psi = self._psi(spec)
        book = [self._psi(c) for c in candidates]
        rows = []
        for i, psi_c in enumerate(book):
            comp = self.score_components(spec_psi, psi_c, subject=subject)
            rows.append((comp["total"], i, comp))
        rows.sort(key=lambda t: (-t[0], t[1]))
        totals = np.array([r[0] for r in rows], dtype=float)
        mx = float(totals.max()) if totals.size else 0.0
        tie_group = int(np.isclose(totals, mx, atol=1e-12).sum()) if totals.size else 0
        return {
            "top1_index": int(rows[0][1]) if rows else None,
            "top1_score": round(mx, 9),
            "tie_group_size": tie_group,
            "ranked": [{"index": r[1], **r[2]} for r in rows],
            "w_trans": self.w_trans,
            "w_mem": self.w_mem,
            "active": self.active,
        }

    # -- calibration ----------------------------------------------------
    def accuracy(self, items: Sequence[dict], w_trans: float,
                 w_mem: float, subject_key: str = "subject") -> dict:
        """Top-1 over `items`. Weights are passed, not installed, so a grid
        search cannot leave the object in a half-fitted state."""
        old = (self.w_trans, self.w_mem)
        self.w_trans, self.w_mem = float(w_trans), float(w_mem)
        try:
            hits, ties, uniq = 0, [], set()
            for it in items:
                r = self.rank(it["question"], it["options"],
                              subject=str(it.get(subject_key, "")))
                if r["top1_index"] == int(it["correct_idx"]):
                    hits += 1
                ties.append(r["tie_group_size"])
                for comp in r["ranked"]:
                    uniq.add(round(comp["total"], 9))
            n = len(items)
            return {"n": n, "correct": int(hits),
                    "acc": (hits / n) if n else 0.0,
                    "mean_tie_group": (float(np.mean(ties)) if ties else 0.0),
                    "distinct_scores": len(uniq)}
        finally:
            self.w_trans, self.w_mem = old

    def fit_weights(self, fit_items: Sequence[dict],
                    w_grid: Sequence[float] = W_GRID,
                    w_mem_grid: Sequence[float] = W_MEM_GRID,
                    subject_key: str = "subject") -> dict:
        """Choose weights on FIT rows only. Ties prefer the smaller weight.

        The parsimony tie-break is load bearing. Without it the grid could
        return a positive weight that scored exactly the same as zero, and
        the coupling would be reported as installed while adding nothing.
        """
        best = None
        table = []
        for w in w_grid:
            for wm in w_mem_grid:
                if wm > 0.0 and self.recall_bank is None:
                    continue
                r = self.accuracy(fit_items, w, wm, subject_key=subject_key)
                row = {"w_trans": float(w), "w_mem": float(wm),
                       "fit_acc": round(float(r["acc"]), 6),
                       "fit_correct": int(r["correct"]),
                       "mean_tie_group": round(float(r["mean_tie_group"]), 4)}
                table.append(row)
                key = (-r["acc"], w + wm, w, wm)
                if best is None or key < best[0]:
                    best = (key, row)
        return {"chosen": best[1] if best else None, "grid": table}

    def install(self, w_trans: float, w_mem: float) -> None:
        self.w_trans = float(w_trans)
        self.w_mem = float(w_mem)
        self.active = bool(self.w_trans > 0.0 or self.w_mem > 0.0)

    def stats(self) -> dict:
        return {"w_trans": self.w_trans, "w_mem": self.w_mem,
                "active": self.active,
                "n_domains": self.domains.n_domains,
                "domain_calls": dict(self.domains.routed),
                "recall_rows": (self.recall_bank.rows
                                if self.recall_bank is not None else 0),
                "tau_veto": TAU_VETO,
                "statistic_form": INVARIANT_FORM}


def mcnemar(arm_hits: Sequence[bool], ctrl_hits: Sequence[bool]) -> dict:
    """Paired McNemar on two boolean hit vectors.

    Reported because "accuracy went up" is not enough on the same items: the
    question is whether the arm wins on the items the control loses, and the
    converse. `b` and `c` are those two counts.
    """
    b = sum(1 for a, c in zip(arm_hits, ctrl_hits) if a and not c)
    c_ = sum(1 for a, c in zip(arm_hits, ctrl_hits) if (not a) and c)
    n = b + c_
    if n == 0:
        return {"arm_only": 0, "control_only": 0, "z": 0.0, "p": 1.0,
                "verdict": "NO_DIFFERENCE"}
    z = (b - c_) / float(np.sqrt(n))
    # Two-sided normal tail, no scipy dependency.
    from math import erfc, sqrt
    p = erfc(abs(z) / sqrt(2.0))
    verdict = ("ARM_BETTER" if b > c_ else
               "CONTROL_BETTER" if c_ > b else "NO_DIFFERENCE")
    return {"arm_only": int(b), "control_only": int(c_),
            "z": round(float(z), 4), "p": round(float(p), 6),
            "verdict": verdict}
