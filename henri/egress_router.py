"""Unified omni-modal egress with a hard grammar mask.

Omega Amendment 6.

Honest scope statements:
  - The "100% syntactic validity" claim is a property of the mask plus an
    explicit parse check. It is true by construction, not by learning.
  - The production bank below is a TOY grammar (five snippets). It is not a
    Python CFG. The gate tests mask enforcement and parseability, and says so.
  - `decode_motor` discards phase (it reads the real part of Slot 1). It is
    reported as phase_discarding, not as a calibrated torque decoder.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class EgressOutput:
    mode: str
    content: object
    margin: float
    entropy_norm: float
    syntax_valid: bool
    sagnac_verified: bool
    notes: str = ""
    selected: Optional[List[int]] = None


class HardASTGrammarMask:
    """Finite-state production mask: disallowed productions get -inf logit."""

    def __init__(self, production_snippets: Dict[int, str], valid_transitions: Dict[str, List[int]]) -> None:
        self.production_snippets = dict(production_snippets)
        self.valid_transitions = {k: list(v) for k, v in valid_transitions.items()}

    def mask_logits(self, similarities: np.ndarray, parser_state: str) -> np.ndarray:
        sims = np.asarray(similarities, dtype=np.float64).copy()
        allowed = self.valid_transitions.get(parser_state)
        if allowed is None:
            return sims
        mask = np.full_like(sims, -np.inf)
        for idx in allowed:
            if 0 <= int(idx) < sims.shape[-1]:
                mask[int(idx)] = 0.0
        return sims + mask

    def allowed_indices(self, parser_state: str, n: int) -> List[int]:
        allowed = self.valid_transitions.get(parser_state)
        if allowed is None:
            return list(range(n))
        return [int(i) for i in allowed if 0 <= int(i) < n]


class OmniModalEgressRouter:
    def __init__(
        self,
        d_model: int = 65536,
        d_action: int = 8,
        beta_hopfield: float = 26.10,
        seed: int = 42,
    ) -> None:
        self.d_model = int(d_model)
        self.d_slot = self.d_model // 4
        self.d_action = int(d_action)
        self.beta_hopfield = float(beta_hopfield)
        rng = np.random.default_rng(seed)
        self.motor_readout_proj = rng.standard_normal((self.d_action, self.d_slot)) / np.sqrt(
            float(self.d_slot)
        )

        self.ast_snippets: Dict[int, str] = {
            0: "def solve(x):\n",
            1: "    if x is None:\n        return 0\n",
            2: "    return [v * 2 for v in x]\n",
            3: "    return sum(x)\n",
            4: "    return sorted(x)\n",
        }
        self.ast_transitions: Dict[str, List[int]] = {
            "START": [0],
            "BODY_GUARD": [1],
            "RETURN_STMT": [2, 3, 4],
        }
        self.grammar_mask = HardASTGrammarMask(self.ast_snippets, self.ast_transitions)

    # --------------------------------------------------------------------- AST
    def decode_ast(
        self,
        settled_wave: np.ndarray,
        ast_codebook: np.ndarray,
        parser_states: Optional[Sequence[str]] = None,
    ) -> EgressOutput:
        wave = np.asarray(settled_wave, dtype=np.complex128).reshape(-1)
        cb = np.asarray(ast_codebook, dtype=np.complex128)
        n = min(cb.shape[0], len(self.ast_snippets))
        states = list(parser_states or ["START", "BODY_GUARD", "RETURN_STMT"])

        raw_sims = np.real(cb[:n] @ np.conj(wave))
        lines: List[str] = []
        selected: List[int] = []
        margins: List[float] = []
        entropies: List[float] = []
        masked_any = False

        for st in states:
            masked = self.grammar_mask.mask_logits(raw_sims, st)
            finite = np.isfinite(masked)
            if not np.any(finite):
                raise ValueError(f"grammar state {st!r} admits no production")
            logits = np.where(finite, self.beta_hopfield * masked, -np.inf)
            logits = logits - np.max(logits)
            probs = np.exp(logits)
            probs = probs / probs.sum()
            idx = int(np.argmax(probs))
            selected.append(idx)
            lines.append(self.ast_snippets[idx])
            if int(np.argmax(raw_sims)) != idx:
                masked_any = True

            allowed_vals = raw_sims[finite]
            if allowed_vals.size >= 2:
                top2 = np.sort(allowed_vals)[-2:]
                margins.append(float(top2[1] - top2[0]))
            else:
                margins.append(1.0)
            ent = -float(np.sum(probs * np.log(np.clip(probs, 1e-12, None))))
            entropies.append(ent)

        code = "".join(lines)
        try:
            ast.parse(code)
            syntax_ok = True
        except SyntaxError:
            syntax_ok = False

        return EgressOutput(
            mode="code_ast",
            content=code,
            margin=float(np.mean(margins)) if margins else 0.0,
            entropy_norm=float(np.mean(entropies)) if entropies else 0.0,
            syntax_valid=syntax_ok,
            sagnac_verified=True,
            notes="mask_changed_selection" if masked_any else "mask_inactive_on_this_wave",
            selected=selected,
        )

    # ------------------------------------------------------------------- motor
    def decode_motor(self, settled_wave: np.ndarray, sagnac_verified: bool = True) -> EgressOutput:
        wave = np.asarray(settled_wave, dtype=np.complex128).reshape(-1)
        if not sagnac_verified:
            return EgressOutput(
                mode="motor_efe",
                content=np.zeros(self.d_action, dtype=np.float64),
                margin=0.0,
                entropy_norm=1.0,
                syntax_valid=True,
                sagnac_verified=False,
                notes="vetoed_safe_zero",
            )
        slot1 = np.real(wave[self.d_slot : 2 * self.d_slot]) * np.sqrt(float(self.d_slot))
        torque = np.tanh(self.motor_readout_proj @ slot1)
        return EgressOutput(
            mode="motor_efe",
            content=torque,
            margin=float(np.linalg.norm(torque)),
            entropy_norm=0.0,
            syntax_valid=True,
            sagnac_verified=True,
            notes="phase_discarding_readout",
        )
