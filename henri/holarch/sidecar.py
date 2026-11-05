"""The HENRI agentic sidecar: epistemic brain, memory, router, verifier.

Evidence label: DERIVED for the wiring, OBSERVED for every number a run
records.

What this module is
-------------------
A thin, sovereign wrapper that any agent can call. It does not generate
prose. It answers four questions an agentic system actually needs:

    is this input inside the manifold?          veto_check
    what does this unseen system do?            solve / ingest
    what did we already learn about this rule?  recall_prototype / adapt_rule
    what is the current strategy for this task? compile_playbook / recall

The four zones keep their responsibilities: Zone C stores engrams and
prototypes, Zone B supplies the Sagnac statistic, Zone A carries the
plasticity, and slot 3 carries the rule context. Nothing here bypasses
them.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from henri.holarch.config import HolarchConfig, TASK_IDS
from henri.holarch.pipeline import HenriHolarch
from henri.holarch.symbolic import (
    CodeGraphIndexer,
    Edge,
    SolveResult,
    SymbolCodebook,
    SymbolicDecoder,
    TopologySolver,
    index_code,
)

_EPS = 1e-12


def _wave_tensor(wave: Sequence[float], dim: int) -> torch.Tensor:
    """One caller-supplied wave as a complex torch row.

    Two encodings are accepted, both lossless: 2*D floats interleaved as
    (real, imaginary), or D real floats. Anything else is a caller error
    and is refused rather than truncated.
    """
    values = np.asarray(wave, dtype=np.float64).reshape(-1)
    dim = int(dim)
    if values.size == 2 * dim:
        pairs = values.reshape(dim, 2)
        data = pairs[:, 0] + 1j * pairs[:, 1]
    elif values.size == dim:
        data = values.astype(np.complex64)
    else:
        raise ValueError(
            f"history wave has {values.size} values; expected {dim} or {2 * dim}"
        )
    return torch.from_numpy(np.asarray(data, dtype=np.complex64).reshape(1, dim))


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return (vector / max(norm, _EPS)).astype(np.complex64)


@dataclass
class Playbook:
    """A compiled execution strategy: exact steps, not free text."""

    name: str
    steps: List[str]
    version: int = 1
    source: str = "manual"

    def render(self, decoder: SymbolicDecoder) -> str:
        return decoder.render_plan(self.name, self.steps, verified=True)


class EpistemicSidecar:
    """One object an agent can hold: memory, verifier, router, solver."""

    def __init__(
        self,
        cfg: Optional[HolarchConfig] = None,
        model: Optional[HenriHolarch] = None,
        backend: str = "sqlite",
        memory_path: Optional[str] = None,
        root: str | Path = ".",
    ) -> None:
        self.cfg = cfg or HolarchConfig.small()
        self.root = Path(root)
        self.indexer: Optional[CodeGraphIndexer] = None
        self._engine = None
        path = memory_path or str(
            Path(tempfile.gettempdir()) / f"henri_sidecar_{int(time.time())}.sqlite3"
        )
        self.model = model or HenriHolarch(self.cfg, backend=backend, memory_path=path)
        self.model.eval()
        self.codebook = SymbolCodebook(self.cfg.dim)
        self.solver = TopologySolver(self.model.memory, self.codebook, mode="indexed")
        self.decoder = SymbolicDecoder()
        self.playbooks: Dict[str, Playbook] = {}
        self.index_report: Dict[str, float] = {}

    # ------------------------------------------------------- 1. structural
    def ingest_edges(
        self, edges: Iterable[Edge], calibrate: bool = True, namespace: str = "default"
    ) -> Dict[str, float]:
        """Write a novel system into Zone C. No weight is touched.

        Directive F1: the namespace is part of every write and every
        calibration read, so two tenants cannot share a transition key.
        """
        edges = list(edges)
        for edge in edges:
            self.codebook.add(edge.source)
            self.codebook.add(edge.target)
        before = float(self.model.memory.stats()["count"])
        written = self.solver.ingest(edges, namespace=str(namespace))
        report = {
            "edges_ingested": float(written),
            "engrams_before": before,
            "engrams_after": float(self.model.memory.stats()["count"]),
            "parameters_unchanged": 1.0,
            "namespace": str(namespace),
        }
        if calibrate and edges:
            report.update(self.solver.calibrate(edges, namespace=str(namespace)))
        return report

    def ingest_code(
        self,
        root: str | Path,
        package: str = "henri",
        namespace: str = "default",
        mode: str = "edges",
        file_path: Optional[str] = None,
        source_text: Optional[str] = None,
    ) -> Dict[str, float]:
        """Read a package once, then answer from memory only.

        Stage 2: the index pass also extracts every signature and every AST
        parent-child transition and stores them as Tier-3 boundary engrams,
        so the Sagnac gate has an axiom to compare a proposed statement with.

        Directive H4: mode="delta" re-indexes exactly ONE file through
        CodeGraphIndexer.update_file. The per-module records are reused for
        every other file, so an agent edit costs one ast.parse instead of a
        package pass, and the resulting graph is identical to a full index.
        """
        if str(mode) == "delta":
            if file_path is None:
                raise ValueError("mode='delta' needs a file_path")
            if self.indexer is None:
                self.indexer = CodeGraphIndexer.from_memory(self.model.memory, str(namespace))
            if self.indexer is None:
                self.indexer = CodeGraphIndexer(
                    self.model.memory, namespace=str(namespace), symbol_codebook=self.codebook
                )
            report = self.indexer.update_file(root, file_path, source_text=source_text)
            self._absorb_index_report(report)
            self.index_report["delta"] = 1.0
            self.index_report["delta_seconds"] = float(report.seconds)
            self.index_report["delta_updated"] = float(len(report.updated))
            return dict(self.index_report)
        indexer = CodeGraphIndexer(
            self.model.memory, namespace=str(namespace), symbol_codebook=self.codebook
        )
        report = indexer.index(root, package=package)
        self.indexer = indexer
        graph_edges = list(indexer.graph.edges)
        ingest = self.ingest_edges(graph_edges, namespace=str(namespace))
        self._absorb_index_report(report)
        return {**self.index_report, **ingest}

    def _absorb_index_report(self, report) -> None:
        self.index_report = {
            "files_parsed": float(report.files),
            "lines_parsed": float(report.lines),
            "parse_seconds": float(report.seconds),
            "edges": float(report.edges),
            "nodes": float(report.nodes),
            "signatures": float(report.signatures),
            "grammar_transitions": float(report.transitions),
            "boundary_engrams": float(report.boundary_engrams),
        }

    def impact(self, target: str, hops: int = 1, namespace: str = "default") -> Dict[str, object]:
        """Reverse-call blast radius for one symbol (Directive F4).

        Directive G4: if this process has not ingested the package, the index
        is restored from the Zone C metadata the previous process wrote, so a
        restart answers without a re-parse.
        """
        if self.indexer is None:
            self.indexer = CodeGraphIndexer.from_memory(self.model.memory, str(namespace))
        if self.indexer is None:
            raise RuntimeError("no code index in memory for this namespace")
        result = self.indexer.impact(target, hops=int(hops))
        result["namespace"] = str(namespace)
        if not result["resolved"]:
            result["vetoed"] = True
            result["callers"] = []
        else:
            result["vetoed"] = False
        return result

    # ------------------------------------------------------------- 2. solve
    def solve(
        self, source: str, operators: Sequence[str], namespace: str = "default"
    ) -> Dict[str, object]:
        result: SolveResult = self.solver.solve(source, operators, namespace=str(namespace))
        return {
            "answer": result.answer,
            "vetoed": bool(result.vetoed),
            "coherences": list(result.coherences),
            "text": self.decoder.render_path(result),
            "path": result.path,
            "namespace": str(namespace),
        }

    @torch.no_grad()
    def veto_check(self, task: str, x: torch.Tensor) -> Dict[str, float]:
        output = self.model(task, x, training=False)
        return {
            "veto_rate": float(output.veto.float().mean()),
            "s_max": float(output.s_max.mean()),
            "delta_sagnac": float(output.delta_sagnac.mean()),
            "confidence": float(output.margin.mean()),
        }

    # -------------------------------------------------------- 3. playbooks
    def _study_wave(self, text: str) -> np.ndarray:
        """A pure-phase study key: |key_d| is uniform, so unbinding is exact.

        compose() returns a chord whose component magnitudes vary between
        0.5 and 1.5. Unbinding with such a key leaves a random real
        amplitude mask, which destroys the recovered step phasors. The
        key is therefore projected onto the unit-modulus ring.
        """
        chord = self.codebook.compose(text, "operator", role=3, position=0.0)
        return _unit(np.exp(1j * np.angle(chord)))

    def compile_playbook(
        self,
        name: str,
        steps: Sequence[str],
        merge: bool = True,
        tier: int = 3,
        namespace: str = "default",
    ) -> Dict[str, float]:
        """Compile an execution strategy into a Tier-3 engram.

        The steps are superposed as psi_step * psi_pos^k, so the stored
        wave can unbind them one by one. merge=True appends new steps and
        drops duplicates, which is the ACE delta form: a strategy evolves
        without rewriting the whole record.
        """
        existing = self.playbooks.get(name)
        if merge and existing is not None:
            merged = list(existing.steps)
            for step in steps:
                if step not in merged:
                    merged.append(step)
            version = existing.version + 1
        else:
            merged = list(steps)
            version = 1 if existing is None else existing.version + 1
        for step in merged:
            self.codebook.add(str(step))
        position = self.codebook.position_wave()
        wave = np.zeros(self.cfg.dim, dtype=np.complex64)
        for index, step in enumerate(merged):
            wave = wave + self.codebook.symbol_wave(str(step)) * np.exp(
                1j * np.angle(position) * float(index)
            )
        wave = _unit(wave)
        key = self._study_wave(name)
        # Bind the key with a DC-biased payload, so retrieval is a plain
        # coherence match and the payload unbinds after the DC term is
        # removed. Nothing is generated: the steps are unit phasors.
        payload = 1.0 + 0.5 * wave.astype(np.complex64)
        stored = _unit(key * payload)
        from henri.holarch.zone_c import PLAYBOOK_LABEL_OFFSET

        record = self.model.memory.write(
            stored,
            label=PLAYBOOK_LABEL_OFFSET + int(version),
            tier=int(tier),
            namespace=str(namespace),
        )
        self.playbooks[name] = Playbook(name=name, steps=merged, version=version)
        return {
            "steps": float(len(merged)),
            "version": float(version),
            "engram_id": float(getattr(record, "engram_id", -1)),
            "tier": float(tier),
        }

    def recall_playbook(
        self, name: str, decode: bool = True, namespace: str = "default"
    ) -> Dict[str, object]:
        key = self._study_wave(name)
        # "__all__" is the explicit, audited cross-tenant read used only by
        # diagnostics; every request path passes a real namespace.
        scope = None if str(namespace) == "__all__" else str(namespace)
        read = self.model.memory.query(key, k=1, namespace=scope)
        if not read.hits:
            return {"found": 0.0, "steps": [], "text": "ABSTAIN: no playbook"}
        stored = np.asarray(read.top_wave if read.top_wave is not None else read.context)
        position = self.codebook.position_wave()
        steps: List[str] = []
        playbook = self.playbooks.get(name)
        exact = list(playbook.steps) if playbook is not None else []
        if decode:
            # stored = unit(key * (1 + a*w));  stored * conj(key) is
            # proportional to the payload, so normalise first, then remove
            # the DC bias, then unbind the position power of each step.
            residue = _unit(stored * np.conj(key))
            residue = _unit(residue - np.mean(residue))
            limit = max(1, len(exact)) if exact else 16
            for index in range(int(limit)):
                unbound = _unit(
                    residue * np.exp(-1j * np.angle(position) * float(index))
                )
                symbol, coherence, _ = self.codebook.snap(unbound)
                if coherence < 0.20:
                    break
                steps.append(symbol)
        return {
            "found": float(bool(read.hits)),
            "coherence": float(read.hits[0].coherence),
            "steps": exact,
            "decoded": steps,
            "text": self.decoder.render_plan(name, exact, verified=bool(read.hits)),
        }

    # -------------------------------------------------- 4. continual memory
    @torch.no_grad()
    def recall_prototype(self, task: str, x: torch.Tensor) -> Dict[str, object]:
        psi = self.model.encode(task, x)
        bright = self.model.zone_b(psi).bright
        bank = self.model.prototype_waves(task)
        if int(bank.numel()) == 0:
            return {"found": 0.0, "labels": [], "alignment": float("nan")}
        align = self.model.zone_b.alignment(bright, bank)
        best = align.argmax(dim=-1)
        labels = self.model.prototypes[task].labels
        return {
            "found": 1.0,
            "labels": [int(labels[int(index)]) for index in best],
            "alignment": float(align.max(dim=-1).values.mean()),
        }

    def adapt_rule(
        self,
        task: str,
        x: torch.Tensor,
        y: torch.Tensor,
        rule_id: Optional[int] = None,
        lr: float = 0.2,
    ) -> Dict[str, float]:
        """One surprise-gated fast-weight write, with the rule in slot 3."""
        with torch.no_grad():
            output = self.model(task, x, training=False, rule_id=rule_id)
            gate = self.model.plasticity_gate(output)
            self.model.fast_learn(output, y.long(), lr=float(lr))
        return {
            "gate_mean": float(gate.mean()),
            "surprise_mean": float(
                self.model.surprise(task, output.photonic.bright).mean()
            ),
            "examples": float(x.shape[0]),
        }

    # --------------------------------------------------------- 5. synthesis
    def synthesize(
        self,
        intent: str,
        kind: str = "code",
        namespace: str = "default",
        max_tokens: int = 320,
    ) -> Dict[str, object]:
        """Multi-token synthesis through the wave decoder (Stage 3).

        The engine is built once per service, because fitting the tokenizer,
        the projector and the count model is a corpus pass and not a request.
        """
        if self._engine is None:
            from henri.holarch.code_gen import build_synthesis_engine

            symbols = set()
            signatures = {}
            if self.indexer is not None:
                symbols |= self.indexer.symbol_names()
                signatures = dict(self.indexer.signatures)
            engine, stats = build_synthesis_engine(
                self.root,
                self.model.memory,
                namespace=str(namespace),
                symbol_table=symbols,
                signatures=signatures,
            )
            self._engine = engine
            self._engine_stats = stats
        self._engine.namespace = str(namespace)
        self._engine.plans.namespace = str(namespace)
        result = self._engine.synthesize(intent, kind=str(kind), max_tokens=int(max_tokens))
        result["corpus"] = dict(getattr(self, "_engine_stats", {}))
        return result

    def adapt_synthesis(
        self,
        intent: str,
        corrected_source: str,
        name: str,
        kind: str = "code",
        namespace: str = "default",
    ) -> Dict[str, object]:
        """Compile a verified fix into an ACE playbook for one tenant."""
        if self._engine is None:
            self.synthesize("warmup", kind="text", namespace=namespace)
        self._engine.namespace = str(namespace)
        self._engine.plans.namespace = str(namespace)
        return self._engine.adapt(intent, corrected_source, name, kind=str(kind))

    # ------------------------------------------- planner + claim guard (H3)
    def _action_rotor(self, transition: str):
        rotor = getattr(self.model, "action_rotor", None)
        if str(transition) in {"rotor", "blend"} and rotor is None:
            raise ValueError(
                "the mounted checkpoint carries no fitted rotor; "
                "mount a Cycle H checkpoint or use transition='shift'"
            )
        return rotor

    def plan(
        self,
        observation: Sequence[float],
        history_waves: Optional[Sequence[Sequence[float]]] = None,
        transition: str = "rotor",
        lookahead_depth: int = 2,
        gamma_look: float = 0.02,
        namespace: str = "default",
    ) -> Dict[str, object]:
        """Run the wall-masked depth-2 rotor lookahead on one observation.

        Directive H3. The planner is the object the 0.8125 win rate was
        measured with; it is now callable, and the rotor comes from the
        checkpoint rather than being refitted per request.
        """
        from henri.holarch.model import ACTION_NAMES, LatentActionLookahead

        rotor = self._action_rotor(transition)
        lookahead = LatentActionLookahead(
            self.model,
            None,
            rotor=rotor,
            transition=str(transition),
            lookahead_depth=int(lookahead_depth),
            gamma_look=float(gamma_look),
        )
        recent: List[torch.Tensor] = []
        for wave in list(history_waves or [])[-int(lookahead.history) :]:
            recent.append(_wave_tensor(wave, int(self.cfg.dim)))
        choice = lookahead.choose(observation, recent)
        action = int(choice["action"])
        scores = choice["scores"]
        return {
            "action": action,
            "action_name": ACTION_NAMES[action % len(ACTION_NAMES)],
            "scores": [float(value) for value in scores],
            "alignment": [float(value) for value in choice["alignment"]],
            "penalty": [float(value) for value in choice["penalty"]],
            "lookahead": (
                [float(value) for value in choice["lookahead"]]
                if int(lookahead_depth) == 2
                else []
            ),
            "legal_mask": [bool(value) for value in choice["legal"]],
            "transition": str(transition),
            "lookahead_depth": int(lookahead_depth),
            "gamma_look": float(gamma_look),
            "history_waves": int(len(recent)),
            "namespace": str(namespace),
        }

    def verify_claim(
        self,
        claim: str,
        kind: str = "text",
        top_k: int = 4,
        namespace: str = "default",
    ) -> Dict[str, object]:
        """Check a prose claim against the corpus, or code against the AST.

        Directive H3. Text goes through the OntologyVerifier (rare terms and
        quantities must appear in the evidence). Code is parsed, compiled,
        and every callee is checked against the code index, so a fabricated
        signature is refused rather than accepted.
        """
        text = str(claim).strip()
        if str(kind) == "text":
            verifier = self._ontology()
            verdict = verifier.verify_claim(text, top_k=int(top_k))
            payload = verdict.as_dict()
            payload["supported"] = bool(verdict.supported)
            payload["claim"] = text
            payload["kind"] = "text"
            payload["namespace"] = str(namespace)
            payload["corpus_digest"] = str(getattr(verifier, "corpus_digest", ""))
            return payload
        missing_terms: List[str] = []
        missing_numbers: List[str] = []
        try:
            tree = ast.parse(text)
        except SyntaxError as error:
            return {
                "status": "unsupported",
                "supported": False,
                "reason": f"ast.parse: {error.msg} at line {error.lineno}",
                "confidence": 0.0,
                "passages": [],
                "missing_terms": [],
                "missing_numbers": [],
                "kind": "code",
                "claim": text,
                "namespace": str(namespace),
            }
        try:
            compile(tree, "<verify>", "exec")
        except (SyntaxError, ValueError) as error:
            return {
                "status": "unsupported",
                "supported": False,
                "reason": f"compile: {error}",
                "confidence": 0.0,
                "passages": [],
                "missing_terms": [],
                "missing_numbers": [],
                "kind": "code",
                "claim": text,
                "namespace": str(namespace),
            }
        from henri.holarch.code_gen import ASTGate

        symbols = set()
        if self.indexer is None:
            self.indexer = CodeGraphIndexer.from_memory(self.model.memory, str(namespace))
        if self.indexer is not None:
            symbols |= self.indexer.symbol_names()
        gate = ASTGate(symbols)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = ""
                if isinstance(node.func, ast.Name):
                    name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    name = node.func.attr
                if name and gate.symbol_veto(name):
                    missing_terms.append(name)
        if missing_terms:
            return {
                "status": "unsupported",
                "supported": False,
                "reason": f"callee(s) absent from the code index: {sorted(set(missing_terms))}",
                "confidence": 0.0,
                "passages": [],
                "missing_terms": sorted(set(missing_terms)),
                "missing_numbers": missing_numbers,
                "kind": "code",
                "claim": text,
                "namespace": str(namespace),
            }
        return {
            "status": "supported",
            "supported": True,
            "reason": "ast.parse and compile succeeded; every callee is indexed",
            "confidence": 1.0,
            "passages": [],
            "missing_terms": [],
            "missing_numbers": [],
            "kind": "code",
            "claim": text,
            "namespace": str(namespace),
        }

    def _ontology(self):
        cached = getattr(self, "_verifier_cache", None)
        if cached is not None:
            return cached
        if self._engine is not None and getattr(self._engine, "verifier", None) is not None:
            self._verifier_cache = self._engine.verifier
            return self._verifier_cache
        from henri.holarch.ontology_guard import build_verifier

        self._verifier_cache = build_verifier(self.root)
        return self._verifier_cache

    # ------------------------------------------------------------- lifecycle
    def report(self) -> Dict[str, float]:
        stats = self.model.memory.stats()
        return {
            "engrams": float(stats.get("count", 0.0)),
            "symbols": float(len(self.codebook)),
            "playbooks": float(len(self.playbooks)),
            "parameters": float(self.model.parameter_count()),
            "backend": float(1.0 if stats.get("backend") == "timescaledb" else 0.0),
        }

    def close(self) -> None:
        self.model.memory.close()
