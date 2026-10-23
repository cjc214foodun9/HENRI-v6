"""Digital Twin API surface: three calls over the verified pipeline.

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
It is a self-contained Python interface to the pipeline that Phase 5 built:
`rank_options` scores candidates through the semantic backbone with lexical
stripping, `ingest_cartridge` writes exact associations into the multi-
cartridge Zone C bank, and `verify_seal` reports whether the sealed kernel
still matches its pinned digests.

It is NOT a web server, and the receipt is NOT a cryptographic signature.
The receipt carries a self-digest over its own payload, exactly as
`promotion_gate.py` does. That proves internal consistency and detects
tampering after the fact. It does not prove WHO produced the receipt. Call
it a self-digest, never a signature: an HMAC needs a secret, and this
module holds none.

HONEST TELEMETRY
----------------
Every field in the receipt comes from a real computation. `cartridge_hit`
is a boolean from the router, not an accuracy claim. `seal_status` is read
from `verify_kernel_integrity`, so a drifted kernel shows as `DRIFT` here
rather than being assumed absent.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from henri.cartridge_bank import CartridgeBank, cartridge_bytes
from henri.semantic_backbone import SemanticBackbone, strip_common_option_tokens

BETA_STAR = 26.1
"""Calibrated inverse temperature. Matches the default in
`henri.readout.entropy_at_beta`. Defined here because the sealed readout
exposes it as a default argument, not as a module constant."""

TAU_HIT = 0.55
"""Minimum matched-key similarity `t_key` for a cartridge to contribute.

Below it the bank contributes nothing, so an out-of-bank question cannot be
degraded by memory. This bound is a design choice, not a measurement; the
probe reports the observed `t_key` distribution so a reader can judge it.
"""

W_MEM_MAX = 0.5


def margin_gate_weight(t_key: float) -> float:
    """`w_mem = clip((t_key - tau_hit) / (1 - tau_hit), 0, W_MEM_MAX)`.

    Zero below `tau_hit`, ramping linearly to 0.5 at `t_key = 1`. The 0.5 cap
    means the recalled wave can never outvote the query's own content.
    """
    if t_key < TAU_HIT:
        return 0.0
    return float(min(W_MEM_MAX, (t_key - TAU_HIT) / (1.0 - TAU_HIT)))

__all__ = ["DigitalTwinAPI", "Receipt"]


def _digest(payload: dict) -> str:
    """Self-digest over a canonical JSON encoding of the payload."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


@dataclass
class Receipt:
    """Epistemic receipt for one decision. Self-digested, not signed."""

    payload: dict

    def to_dict(self) -> dict:
        body = dict(self.payload)
        body["self_digest"] = _digest(self.payload)
        body["digest_kind"] = "sha256_self_digest_not_signature"
        return body

    def to_json(self, **kw) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str, **kw)


class DigitalTwinAPI:
    """Ranking, ingestion, and seal verification over the v6 kernel.

    Args:
        dim: wave dimension. 65536 is production.
        rank: `DeltaMemory` rank. Also the per-cartridge exact capacity.
        dtype: complex128 is exact; complex64 halves the footprint.
        seed: pins the backbone projection frame.
        sigma: Bochner bandwidth. Do not tune this on an eval set.
        w_content: energy on the content slot. 0.5 keeps synonym and
            antonym gates both satisfied because the sign is applied to
            both slots; see `semantic_backbone.encode`.
    """

    def __init__(self, dim: int = 65536, rank: int = 64,
                 dtype=np.complex128, seed: int = 20261020,
                 sigma: float = float(np.sqrt(2.0)),
                 w_content: float = 0.5) -> None:
        self.dim = int(dim)
        self.rank = int(rank)
        self.dtype = np.dtype(dtype)
        self.backbone = SemanticBackbone(sigma=sigma, seed=seed,
                                         w_content=w_content)
        self.bank: Optional[CartridgeBank] = None

    # -- 1. ranking ------------------------------------------------------
    def rank_options(self, question: str, options: Sequence[str],
                     strip: bool = True,
                     signing_key: Optional[bytes] = None,
                     channel_weight: float = 0.0) -> Receipt:
        """Rank candidate options; return the winner and its telemetry.

        Stripping removes boilerplate before encoding, so the score linear
        functional is never modified. Ties break on content digest, never
        on position, so option order cannot influence the answer.

        Phase 6 Action 3 adds three things: the real magnitude-weighted
        Sagnac statistic on the winning wave, optional HMAC-SHA256 signing
        when `signing_key` is supplied, and margin-gated cartridge retrieval
        with `w_mem` blending. With no key the receipt stays a self-digest
        and says so.
        """
        opts = list(options)
        texts = strip_common_option_tokens(question, opts) if strip else opts
        q = self.backbone.encode(question).psi
        book = np.stack([self.backbone.encode(t).psi for t in texts], axis=0)
        scores = np.real(book @ np.conj(q))
        # -- PHASE 10 STEP 3: additive operator + morphism contrast --------
        #
        # `channel_weight = 0.0` (the default) leaves `scores` BYTE-IDENTICAL,
        # so every Phase 5/6 receipt stays numerically valid.
        #
        # THE CHANNEL IS ADDITIVE AND NEVER REWRITES `psi`. Step 1 measured
        # that populating a fifth slot rescales slots 0-2 (0.707107 ->
        # 0.632456) and moved G-AST2 top-1 0.4000 -> 0.3250. A second score
        # term cannot repeat that defect, because the wave keeps full energy.
        #
        # CONTRAST IS TAKEN AGAINST THE OPTION-MEAN. That is the only contrast
        # that makes the score permutation-equivariant, which is what makes
        # 4x rotation invariance exact rather than lucky.
        channel_meta = {"channel_weight": 0.0, "channel_added": False}
        if float(channel_weight) > 0.0:
            from henri.reason_channels import blended_scores
            scores, channel_meta = blended_scores(
                scores, question, texts, float(channel_weight))
        digests = [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts]
        order = sorted(range(len(scores)),
                       key=lambda i: (-round(float(scores[i]), 12), digests[i]))
        sel = int(order[0])
        top = float(scores[sel])
        second = float(scores[order[1]]) if len(order) > 1 else -1.0
        g = np.real(book @ book.conj().T)
        np.fill_diagonal(g, 0.0)
        c_max = float(np.max(g)) if book.shape[0] > 1 else 0.0
        pol = self.backbone.encode(question).pol

        # -- Action 3a: REAL Sagnac telemetry -----------------------------
        # The sealed statistic is `henri.sagnac.sagnac_delta`, whose
        # signature takes WaveState objects, not bare arrays. It computes the
        # MAGNITUDE-WEIGHTED mean phase divergence pinned in Phase 4.
        # A failure records `sagnac_error`; it never fabricates 0.0.
        sagnac_delta = None
        sagnac_error = None
        try:
            from henri.config import WaveConfig
            from henri.sagnac import sagnac_delta as _sagnac
            from henri.state import WaveState
            # DIM COMES FROM THE WAVE, NOT FROM self.dim. The backbone always
            # emits DIM=65536, so a WaveConfig built from a smaller API dim
            # raises on shape and the statistic silently reports None.
            _cfg = WaveConfig(dim=int(np.asarray(q).shape[0]))
            sagnac_delta = float(_sagnac(WaveState(book[sel], _cfg),
                                         WaveState(q, _cfg)))
        except Exception as exc:                # noqa: BLE001
            sagnac_error = f"{type(exc).__name__}: {exc}"

        # -- Action 3c: margin-gated cartridge readout --------------------
        w_mem = 0.0
        cartridge_hit = False
        cartridge_info = None
        if self.bank is not None and self.bank.cartridges:
            res = self.bank.query(q, k_route=1)
            if res.hits and res.value is not None:
                cartridge_info = res.to_dict()
                w_mem = margin_gate_weight(res.t)
                cartridge_hit = bool(res.t >= TAU_HIT)
                if w_mem > 0.0:
                    # Convex blend toward the recalled consequent, then
                    # renormalise. w_mem <= 0.5, so memory can never outvote
                    # the query's own content.
                    q2 = q + w_mem * (res.value - q)
                    n2 = float(np.linalg.norm(q2))
                    if n2 > 0:
                        q2 = q2 / n2
                    scores2 = np.real(book @ np.conj(q2))
                    order2 = sorted(
                        range(len(scores2)),
                        key=lambda i: (-round(float(scores2[i]), 12),
                                       digests[i]))
                    sel = int(order2[0])
                    top = float(scores2[sel])
                    second = (float(scores2[order2[1]])
                              if len(order2) > 1 else -1.0)

        payload = {
            "endpoint": "rank_options",
            "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
            "selected_index": sel,
            "selected_text": opts[sel],
            "selected_digest": digests[sel],
            "margin": round(top - second, 9),
            "t": round(top, 9),
            "second": round(second, 9),
            "c_max": round(c_max, 9),
            "beta_star": BETA_STAR,
            "question_polarity": int(pol),
            "stripped": bool(strip),
            "sagnac_delta": sagnac_delta,
            "sagnac_error": sagnac_error,
            "cartridge_hit": cartridge_hit,
            "cartridge": cartridge_info,
            "w_mem": round(w_mem, 6),
            "channel_weight": float(channel_meta.get("channel_weight", 0.0)),
            "channel_added": bool(channel_meta.get("channel_added", False)),
            "channel_n_op_live": channel_meta.get("n_op_live"),
            "channel_n_morph_live": channel_meta.get("n_morph_live"),
            "channel_question_has_operator": channel_meta.get(
                "question_has_operator"),
            "channel_question_has_morphism": channel_meta.get(
                "question_has_morphism"),
            "tau_hit": TAU_HIT,
            "seal_status": self.verify_seal()["status"],
        }
        # -- Action 3b: HMAC only when a key exists -----------------------
        # No key means no signature. The self-digest detects tampering; it
        # does not authenticate a producer, and the label says so.
        if signing_key is not None:
            import hmac as _hmac
            blob = json.dumps(payload, sort_keys=True,
                              separators=(",", ":"),
                              default=str).encode("utf-8")
            payload["hmac_sha256"] = _hmac.new(
                signing_key, blob, hashlib.sha256).hexdigest()
            payload["signature_kind"] = "hmac_sha256_with_explicit_key"
        else:
            payload["signature_kind"] = "sha256_self_digest_not_signature"
        return Receipt(payload)

    # -- 2. ingestion ----------------------------------------------------
    def ingest_cartridge(self, keys: Sequence[str], values: Sequence[str],
                         subject: str = "") -> Receipt:
        """Encode text pairs and write one cartridge. Cold path.

        Raises when the pair count exceeds the rank wall, because exact
        recall above it is impossible and silently degrading is worse.

        DIMENSION COMES FROM THE WAVE, NOT FROM self.dim. Measured defect:
        `DigitalTwinAPI(dim=4096)` built a bank at 4096 while `SemanticBackbone`
        always emits `DIM = 65536`, so `DeltaMemory._check` raised
        `expected length 4096, got 65536`. The encoded wave is authoritative.
        """
        if len(keys) != len(values):
            raise ValueError("keys and values must have equal length")
        if len(keys) > self.rank:
            raise ValueError(
                f"{len(keys)} pairs exceeds rank={self.rank}. Exact recall "
                "is impossible above the wall; split the cartridge.")
        k = np.stack([self.backbone.encode(t).psi for t in keys], axis=0)
        v = np.stack([self.backbone.encode(t).psi for t in values], axis=0)
        wave_dim = int(k.shape[1])
        if self.bank is None:
            self.bank = CartridgeBank(dim=wave_dim, rank=self.rank,
                                      dtype=self.dtype)
        elif self.bank.dim != wave_dim:
            raise ValueError(
                f"bank dim {self.bank.dim} != wave dim {wave_dim}")
        c = self.bank.add_cartridge(k, v, subject=subject)
        self.bank.finalize()
        return Receipt({
            "endpoint": "ingest_cartridge",
            "cartridge_index": c.index,
            "rows": c.rows,
            "subject": subject,
            "rank": self.rank,
            "dim": wave_dim,
            "bytes_this_cartridge": cartridge_bytes(wave_dim, self.rank,
                                                    self.dtype),
            "bank": self.bank.stats(),
            "seal_status": self.verify_seal()["status"],
        })

    # -- 3. seal ---------------------------------------------------------
    def verify_seal(self) -> dict:
        """Report the sealed-kernel status. Never assumes SEALED."""
        import subprocess
        import sys
        from pathlib import Path
        script = Path(__file__).resolve().parents[1] / "scripts" / \
            "verify_kernel_integrity.py"
        if not script.exists():
            return {"status": "UNKNOWN", "reason": "verifier missing"}
        p = subprocess.run([sys.executable, str(script), "--json"],
                           capture_output=True, text=True)
        try:
            return json.loads(p.stdout)
        except Exception:
            return {"status": "UNKNOWN",
                    "reason": f"verifier output not JSON (rc={p.returncode})"}
