"""Zone C epistemic seeding from a real semantic encoder.

What this is: a seeded axiom bank. Each (subject, relation, object) triple
becomes one slot-routed boundary wave, committed through the existing HHRG
ledger. Retrieval is measured, not asserted.

What this is NOT: world knowledge, a benchmark score, or generalization. The
gate reports in-bank recall and a novelty margin, and it prints both.

Why MiniLM: the register already records that hash ingress reads paired 0.56733
against shuffled 0.56738 — no separation. A real encoder is required for the
bank to carry semantics at all.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from henri.harmonic_lexicon import HarmonicLexicon

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class SeededBank:
    ledger: object
    lexicon: HarmonicLexicon
    labels: List[str]
    d_model: int


def _load_encoder(device: str = "cpu"):
    from sentence_transformers import SentenceTransformer

    try:
        return SentenceTransformer(MODEL_NAME, device=device, local_files_only=True)
    except Exception:
        return SentenceTransformer(MODEL_NAME, device=device)


def default_axioms() -> List[Tuple[str, str, str]]:
    return [
        ("photon", "carries", "electromagnetic_force"),
        ("electron", "carries", "negative_charge"),
        ("proton", "carries", "positive_charge"),
        ("neutron", "lacks", "electric_charge"),
        ("gravity", "curves", "spacetime"),
        ("mass", "resists", "acceleration"),
        ("energy", "conserves", "total_quantity"),
        ("entropy", "increases", "disorder"),
        ("light", "travels_at", "finite_speed"),
        ("sound", "requires", "medium"),
        ("heat", "flows_to", "cold_body"),
        ("pressure", "acts_on", "surface_area"),
        ("voltage", "drives", "current"),
        ("current", "generates", "magnetic_field"),
        ("magnet", "attracts", "ferromagnetic_material"),
        ("lens", "refracts", "light_ray"),
        ("mirror", "reflects", "incident_wave"),
        ("prism", "disperses", "spectrum"),
        ("laser", "emits", "coherent_light"),
        ("antenna", "radiates", "radio_wave"),
        ("transistor", "amplifies", "electrical_signal"),
        ("capacitor", "stores", "electric_energy"),
        ("inductor", "stores", "magnetic_energy"),
        ("diode", "blocks", "reverse_current"),
        ("resistor", "dissipates", "heat"),
        ("algorithm", "terminates_in", "finite_steps"),
        ("hash", "maps_to", "fixed_width_digest"),
        ("cache", "reduces", "latency"),
        ("compiler", "translates", "source_code"),
        ("parser", "builds", "syntax_tree"),
        ("matrix", "represents", "linear_map"),
        ("eigenvector", "preserves", "direction"),
        ("gradient", "points_to", "steepest_ascent"),
        ("integral", "accumulates", "area"),
        ("derivative", "measures", "rate_of_change"),
        ("probability", "sums_to", "one"),
        ("entropy_rate", "bounds", "compression"),
        ("knot", "invariant_under", "isotopy"),
        ("group", "closed_under", "composition"),
        ("functor", "preserves", "structure"),
    ]


def novel_pairs() -> List[Tuple[str, str]]:
    """(subject, relation) pairs the bank never stored as a triple."""
    return [
        ("quasar", "emits"),
        ("neutrino", "carries"),
        ("superconductor", "expels"),
        ("turbine", "converts"),
        ("oscillator", "resonates_at"),
        ("waveguide", "confines"),
        ("modulator", "shifts"),
        ("detector", "measures"),
        ("lattice", "diffracts"),
        ("catalyst", "accelerates"),
    ]


def build_bank(
    d_model: int = 16384,
    vocab_size: int = 512,
    seed: int = 42,
    axioms: Optional[Sequence[Tuple[str, str, str]]] = None,
    include_novel: bool = True,
) -> SeededBank:
    from henri.holographic_ledger import HolographicAgenticLedger

    axioms = list(axioms if axioms is not None else default_axioms())
    d_slot = d_model // 4
    lex = HarmonicLexicon(vocab_size=vocab_size, d_slot=d_slot, d_sem=384, seed=seed)
    encoder = _load_encoder()

    # Lift the NOVEL vocabulary too, but commit no triple for it. Otherwise an
    # unseen word keeps its random phase bins and the novelty gap would measure
    # missing tokens rather than unseen (subject, relation) compositions.
    words = {w for triple in axioms for w in triple}
    if include_novel:
        words |= {w for pair in novel_pairs() for w in pair}
    ordered = sorted(words)
    ids = [lex.register_token(w) for w in ordered]
    emb = encoder.encode(ordered, normalize_embeddings=True, convert_to_numpy=True)
    lex.lift_semantic_embeddings(ids, np.asarray(emb, dtype=np.float64))

    led = HolographicAgenticLedger(d_model=d_model, seed=seed)
    labels: List[str] = []
    for subj, rel, obj in axioms:
        s_wave = lex.bins_to_sphere(lex.token_bins(subj))
        r_wave = lex.bins_to_sphere(lex.token_bins(rel))
        o_wave = lex.bins_to_sphere(lex.token_bins(obj))
        state = _route(d_model, d_slot, s_wave, r_wave, o_wave)
        key = _route(d_model, d_slot, s_wave, r_wave, None)
        led.commit(key, state, label=f"{subj}|{rel}|{obj}")
        labels.append(f"{subj}|{rel}|{obj}")

    return SeededBank(ledger=led, lexicon=lex, labels=labels, d_model=d_model)


def _route(d_model: int, d_slot: int, s0, s1, s2) -> np.ndarray:
    psi = np.zeros(d_model, dtype=np.complex128)
    for k, wave in enumerate((s0, s1, s2)):
        if wave is None:
            continue
        flat = np.asarray(wave, dtype=np.complex128).reshape(-1)
        n = float(np.linalg.norm(flat))
        if n > 1e-9:
            psi[k * d_slot : (k + 1) * d_slot] = flat / n
    n = float(np.linalg.norm(psi))
    return psi / n if n > 0 else psi


def query_pair(bank: SeededBank, subject: str, relation: str):
    d_slot = bank.d_model // 4
    s_wave = bank.lexicon.bins_to_sphere(bank.lexicon.token_bins(subject))
    r_wave = bank.lexicon.bins_to_sphere(bank.lexicon.token_bins(relation))
    q = _route(bank.d_model, d_slot, s_wave, r_wave, None)
    return bank.ledger.query(q)
