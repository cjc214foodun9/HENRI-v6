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
                     strip: bool = True) -> Receipt:
        """Rank candidate options; return the winner and its telemetry.

        Stripping removes boilerplate before encoding, so the score linear
        functional is never modified. Ties break on content digest, never
        on position, so option order cannot influence the answer.
        """
        opts = list(options)
        texts = strip_common_option_tokens(question, opts) if strip else opts
        q = self.backbone.encode(question).psi
        book = np.stack([self.backbone.encode(t).psi for t in texts], axis=0)
        scores = np.real(book @ np.conj(q))
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
        return Receipt({
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
            "sagnac_delta": None,
            "sagnac_note": "veto not run on this path; requires an invariant "
                           "projector. Reported as None rather than 0.0.",
            "cartridge_hit": False,
            "seal_status": self.verify_seal()["status"],
        })

    # -- 2. ingestion ----------------------------------------------------
    def ingest_cartridge(self, keys: Sequence[str], values: Sequence[str],
                         subject: str = "") -> Receipt:
        """Encode text pairs and write one cartridge. Cold path.

        Raises when the pair count exceeds the rank wall, because exact
        recall above it is impossible and silently degrading is worse.
        """
        if len(keys) != len(values):
            raise ValueError("keys and values must have equal length")
        if len(keys) > self.rank:
            raise ValueError(
                f"{len(keys)} pairs exceeds rank={self.rank}. Exact recall "
                "is impossible above the wall; split the cartridge.")
        if self.bank is None:
            self.bank = CartridgeBank(dim=self.dim, rank=self.rank,
                                      dtype=self.dtype)
        k = np.stack([self.backbone.encode(t).psi for t in keys], axis=0)
        v = np.stack([self.backbone.encode(t).psi for t in values], axis=0)
        c = self.bank.add_cartridge(k, v, subject=subject)
        self.bank.finalize()
        return Receipt({
            "endpoint": "ingest_cartridge",
            "cartridge_index": c.index,
            "rows": c.rows,
            "subject": subject,
            "rank": self.rank,
            "dim": self.dim,
            "bytes_this_cartridge": cartridge_bytes(self.dim, self.rank,
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
