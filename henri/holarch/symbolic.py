"""Decision 8: a sovereign symbolic system solver.

Evidence label: DERIVED for the algebra, OBSERVED for every accuracy,
capacity and abstention number a run records.

Why this module exists
----------------------
The 24-character next-letter probe of the previous revision scored 0.223.
That probe asked a wave interferometer to be a novelist: it predicted
contingent spelling (D_c) from a flat window with orthogonal random
phases. HENRI is a system solver (D_u). This module implements the
ontological replacement:

    compose    qFHRR role-filler chords, with logical polarity
    ingest     a novel graph or rule system into Zone C at zero weight cost
    step       unbind one transition from the wave and snap to a symbol
    solve      multi-hop navigation with a fail-closed coherence veto
    index_code a Python package's call graph, read once and never again

The decoder is deterministic. It renders a solved path as exact symbols,
never as generated prose, because the target of an agentic solver is a
verified operator or a patch, not a paragraph.
"""

from __future__ import annotations

import ast
import hashlib
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from henri.holarch.zone_c import SYMBOL_LABEL_OFFSET

_EPS = 1e-12


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return (vector / max(norm, _EPS)).astype(np.complex64)


def _seed_for(text: str, salt: int = 0) -> int:
    digest = hashlib.blake2b(f"{salt}:{text}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**63 - 1)


class SymbolCodebook:
    """Deterministic unit phasors for symbols, primitives, roles, positions.

    Every symbol gets a fixed phasor from a hash of its name, so an unseen
    symbol needs no training: it is addressable the moment it is named.
    Primitives (function, class, guard, operator, ...) form a second small
    codebook, and the chord composer mixes them with a role phasor and a
    fractional position power.
    """

    ROLES = ("subject", "predicate", "object", "guard")

    def __init__(
        self,
        dim: int,
        symbols: Sequence[str] = (),
        primitives: Sequence[str] = (
            "function",
            "class",
            "module",
            "operator",
            "constant",
            "guard",
        ),
        seed: int = 0,
    ) -> None:
        self.dim = int(dim)
        self.seed = int(seed)
        self._symbols: List[str] = []
        self._index: Dict[str, int] = {}
        self._primitives = list(primitives)
        self._primitive_index = {name: i for i, name in enumerate(self._primitives)}
        for symbol in symbols:
            self.add(symbol)

    # ------------------------------------------------------------- building
    def add(self, symbol: str) -> int:
        if symbol in self._index:
            return self._index[symbol]
        index = len(self._symbols)
        self._symbols.append(str(symbol))
        self._index[str(symbol)] = index
        return index

    def __len__(self) -> int:
        return len(self._symbols)

    @property
    def symbols(self) -> List[str]:
        return list(self._symbols)

    def _phasor(self, text: str, salt: int) -> np.ndarray:
        generator = np.random.default_rng(_seed_for(text, salt) ^ self.seed)
        wave = generator.standard_normal(self.dim) + 1j * generator.standard_normal(self.dim)
        return _unit(wave)

    def symbol_wave(self, symbol: str) -> np.ndarray:
        self.add(symbol)
        return self._phasor(str(symbol), 11)

    def primitive_wave(self, primitive: str) -> np.ndarray:
        if primitive in self._primitive_index:
            return self._phasor(primitive, 23)
        return self._phasor(primitive, 23)

    def role_wave(self, role: int | str) -> np.ndarray:
        name = self.ROLES[int(role)] if isinstance(role, int) or str(role).isdigit() else str(role)
        return self._phasor(f"role:{name}", 37)

    def position_wave(self) -> np.ndarray:
        return self._phasor("position", 41)

    def operator_wave(self, operator: str) -> np.ndarray:
        return self._phasor(f"operator:{operator}", 53)

    # ------------------------------------------------------------ composers
    def compose(
        self,
        symbol: str,
        primitive: str = "operator",
        role: int | str = 1,
        position: float = 0.0,
        negative: bool = False,
        identity_weight: float = 0.5,
    ) -> np.ndarray:
        """One qFHRR chord:

            psi = e^(i*pi*neg) * (psi_primitive + a*psi_id) * psi_role * psi_pos^x

        The negation factor is a pi phase rotation, so an affirmative chord
        and its negation destructively interfere at the Sagnac dark port.
        The position power exp(i*theta*x) keeps the chord continuous in a
        continuous position.
        """
        identity = self.symbol_wave(symbol)
        primitive_wave = self.primitive_wave(primitive)
        chord = primitive_wave + float(identity_weight) * identity
        chord = chord * self.role_wave(role)
        theta = np.angle(self.position_wave())
        chord = chord * np.exp(1j * theta * float(position))
        if negative:
            chord = chord * np.exp(1j * math.pi)
        return _unit(chord)

    def matrix(self, symbols: Optional[Sequence[str]] = None) -> torch.Tensor:
        names = list(self._symbols if symbols is None else symbols)
        rows = [self.symbol_wave(name) for name in names]
        if not rows:
            return torch.zeros(0, self.dim, dtype=torch.complex64)
        return torch.from_numpy(np.stack(rows)).to(torch.complex64)

    # ---------------------------------------------------------------- snapping
    def snap(self, wave: np.ndarray, beta: float = 0.0) -> Tuple[str, float, np.ndarray]:
        """Continuous Modern Hopfield lexical snap onto the exact symbol."""
        matrix = self.matrix()
        if matrix.shape[0] == 0:
            return "", float("nan"), np.zeros(0, dtype=np.float32)
        query = torch.from_numpy(_unit(wave)).to(torch.complex64)
        align = torch.einsum("d,vd->v", query.conj(), matrix).real.numpy()
        index = int(np.argmax(align))
        return self._symbols[index], float(align[index]), align


@dataclass
class Edge:
    """One transition of a symbolic system."""

    source: str
    operator: str
    target: str
    guard: str = ""


@dataclass
class SolveResult:
    answer: Optional[str]
    path: List[Tuple[str, str]]
    coherences: List[float]
    vetoed: bool
    reason: str = ""

    def render(self) -> str:
        if self.answer is None:
            return f"ABSTAIN ({self.reason})"
        parts = [self.path[0][0]]
        for _, operator in self.path:
            parts.append(operator)
        parts.append(self.answer)
        text = parts[0]
        for operator in parts[1:-1]:
            text += f" --{operator}-->"
        return f"{text} {parts[-1]}"


class TopologySolver:
    """Zone C as a topological atlas, Zone B statistics as the veto.

    Two retrieval modes are implemented and both are measured:

    indexed    the wave key psi_src * psi_op is an engram; retrieval is a
               coherence search in Zone C and the payload is the target
               label. O(N) rows, no capacity ceiling.
    resonator  every transition is superposed into ONE bundle wave
               B = sum_e psi_src * psi_op * psi_tgt and the next symbol is
               read by unbinding, B * conj(psi_src * psi_op). This is the
               pure holographic form, and it saturates near D / (2 log V).

    The fail-closed decision uses the calibrated coherence of the winning
    engram. A query with no matching transition lands near chance
    coherence, so the solver abstains instead of inventing a symbol.
    """

    def __init__(
        self,
        memory,
        codebook: SymbolCodebook,
        mode: str = "indexed",
        tau_coherence: float = 0.0,
    ) -> None:
        self.memory = memory
        self.codebook = codebook
        self.mode = str(mode)
        if self.mode not in {"indexed", "resonator"}:
            raise ValueError("mode must be 'indexed' or 'resonator'")
        self.tau_coherence = float(tau_coherence)
        self.edges: List[Edge] = []
        self.bundle = np.zeros(codebook.dim, dtype=np.complex64)
        self.written = 0

    # --------------------------------------------------------------- ingest
    def key_wave(self, source: str, operator: str) -> np.ndarray:
        return _unit(self.codebook.symbol_wave(source) * self.codebook.operator_wave(operator))

    def transition_wave(self, edge: Edge) -> np.ndarray:
        return _unit(
            self.codebook.symbol_wave(edge.source)
            * self.codebook.operator_wave(edge.operator)
            * self.codebook.symbol_wave(edge.target)
        )

    def ingest(self, edges: Iterable[Edge], tier: int = 1) -> int:
        """Write an unseen system into Zone C. No neural weight changes."""
        count = 0
        for edge in edges:
            self.edges.append(edge)
            if self.mode == "indexed":
                key = self.key_wave(edge.source, edge.operator)
                # Decision D1: symbolic engrams live above the task blocks, so
                # a task-scoped query cannot retrieve them as task context.
                self.memory.write(
                    key,
                    label=SYMBOL_LABEL_OFFSET + int(self.codebook.add(edge.target)),
                    tier=int(tier),
                )
            else:
                self.bundle = _unit(self.bundle + self.transition_wave(edge))
                self.memory.write(self.transition_wave(edge), label=-1, tier=int(tier))
            count += 1
        self.written += count
        return count

    # ----------------------------------------------------------------- step
    def step(self, source: str, operator: str) -> Dict[str, object]:
        """Resolve one transition and snap to an exact symbol."""
        if self.mode == "indexed":
            key = self.key_wave(source, operator)
            read = self.memory.query(key, k=1)
            if not read.hits:
                return {"answer": None, "coherence": 0.0, "vetoed": True, "reason": "empty memory"}
            hit = read.hits[0]
            symbol_index = int(hit.label) - SYMBOL_LABEL_OFFSET
            valid_symbol = 0 <= symbol_index < len(self.codebook)
            answer = self.codebook.symbols[symbol_index] if valid_symbol else None
            vetoed = (not valid_symbol) or float(hit.coherence) < float(self.tau_coherence)
            return {
                "answer": None if vetoed else answer,
                "coherence": float(hit.coherence),
                "vetoed": vetoed,
                "reason": "coherence below tau" if vetoed else "",
            }
        query = self.key_wave(source, operator)
        unbound = _unit(self.bundle * np.conj(query))
        answer, coherence, align = self.codebook.snap(unbound)
        vetoed = float(coherence) < float(self.tau_coherence)
        return {
            "answer": None if vetoed else answer,
            "coherence": float(coherence),
            "vetoed": vetoed,
            "reason": "coherence below tau" if vetoed else "",
        }

    def solve(self, source: str, operators: Sequence[str]) -> SolveResult:
        """Multi-hop navigation. Abstains at the first unverifiable hop."""
        current = str(source)
        path: List[Tuple[str, str]] = []
        coherences: List[float] = []
        for operator in operators:
            result = self.step(current, operator)
            coherences.append(float(result["coherence"]))
            if result["answer"] is None:
                return SolveResult(
                    None, path, coherences, True, str(result.get("reason", "no transition"))
                )
            path.append((current, str(operator)))
            current = str(result["answer"])
        return SolveResult(current, path, coherences, False, "")

    # ------------------------------------------------------------ calibration
    def calibrate(self, edges: Sequence[Edge], quantile: float = 0.995, margin: float = 0.02) -> Dict[str, float]:
        """Set the abstention threshold from reachable transitions only."""
        values = [float(self.step(edge.source, edge.operator)["coherence"]) for edge in edges]
        if not values:
            return {"tau_coherence": 0.0, "n": 0.0}
        tau = float(np.quantile(values, float(quantile))) - float(margin)
        self.tau_coherence = max(0.0, tau)
        return {
            "tau_coherence": float(self.tau_coherence),
            "coherence_median": float(np.median(values)),
            "n": float(len(values)),
        }


class SymbolicTopologyTask:
    """Novel rule graphs, generated deterministically from a seed.

    A graph has V symbols, a small operator alphabet, and at most one
    outgoing edge per (node, operator) pair. A query is a start symbol plus
    an operator sequence. The answer is the terminal symbol, or unreachable
    when any hop has no matching edge.
    """

    name = "symbolic"

    def __init__(
        self,
        seed: int = 0,
        n_symbols: int = 24,
        operators: Sequence[str] = ("fwd", "map", "link", "lift", "fold", "swap"),
        out_degree: int = 3,
    ) -> None:
        self.seed = int(seed)
        self.n_symbols = int(n_symbols)
        self.operators = list(operators)
        self.out_degree = int(out_degree)
        self.symbols = [f"s{index:02d}" for index in range(self.n_symbols)]

    def graph(self, salt: int = 0) -> List[Edge]:
        generator = np.random.default_rng(self.seed + 1013 * int(salt))
        edges: List[Edge] = []
        for source in self.symbols:
            operators = list(generator.choice(self.operators, size=min(self.out_degree, len(self.operators)), replace=False))
            for operator in operators:
                target = self.symbols[int(generator.integers(0, self.n_symbols))]
                if target == source:
                    target = self.symbols[(self.symbols.index(source) + 1) % self.n_symbols]
                edges.append(Edge(source, str(operator), target))
        return edges

    def query(
        self,
        edges: Sequence[Edge],
        source: str,
        hops: int,
        salt: int = 0,
    ) -> Tuple[List[str], Optional[str]]:
        """Return the operator sequence and the true terminal symbol.

        The operator sequence is drawn from the graph itself, so every hop
        is guaranteed reachable at generation time. An unreachable query is
        produced separately by draw_unreachable.
        """
        generator = np.random.default_rng(self.seed + 7717 * int(salt) + hops)
        index = {edge.source: {} for edge in edges}
        for edge in edges:
            index.setdefault(edge.source, {}).setdefault(edge.operator, edge.target)
        current = str(source)
        operators: List[str] = []
        for _ in range(int(hops)):
            choices = sorted(index.get(current, {}).keys())
            if not choices:
                return operators, None
            operator = str(choices[int(generator.integers(0, len(choices)))])
            operators.append(operator)
            current = index[current][operator]
        return operators, current

    def draw_unreachable(self, edges: Sequence[Edge], salt: int = 0) -> Tuple[str, List[str]]:
        """A query whose first hop has no matching edge."""
        generator = np.random.default_rng(self.seed + 3331 * int(salt))
        present = {(edge.source, edge.operator) for edge in edges}
        for _ in range(256):
            source = self.symbols[int(generator.integers(0, self.n_symbols))]
            operator = self.operators[int(generator.integers(0, len(self.operators)))]
            if (source, operator) not in present:
                return source, [operator]
        return self.symbols[0], ["__none__"]


@dataclass
class CallGraph:
    """Structural index of a Python package: one pass, zero later reads."""

    root: str
    nodes: List[str] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)
    files: int = 0
    lines: int = 0
    seconds: float = 0.0


def _callee_name(node: ast.Call) -> Optional[str]:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def index_code(root: str | Path, package: str = "henri") -> CallGraph:
    """Parse every module under root once and emit (caller, CALLS, callee).

    The caller is the enclosing function or class, not the module, so the
    index supports call chains. After this runs a query never opens a file
    again: the topology lives in Zone C and the answer is a wave retrieval.
    """
    root = Path(root)
    start = time.perf_counter()
    graph = CallGraph(root=str(root))
    definitions: Dict[str, str] = {}
    local: Dict[str, Dict[str, str]] = {}
    calls: List[Tuple[str, str]] = []
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(text)
        except (SyntaxError, OSError):
            continue
        module = ".".join(path.relative_to(root).with_suffix("").parts)
        graph.files += 1
        graph.lines += int(text.count("\n")) + 1
        module_defs: Dict[str, str] = {}

        def walk(node, enclosing: str) -> None:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    qual = f"{module}:{child.name}"
                    module_defs.setdefault(child.name, qual)
                    definitions.setdefault(child.name, qual)
                    graph.nodes.append(qual)
                    walk(child, qual)
                else:
                    if isinstance(child, ast.Call):
                        name = _callee_name(child)
                        if name:
                            calls.append((enclosing, name))
                    walk(child, enclosing)

        walk(tree, module)
        local[module] = module_defs

    seen = set()
    for caller, callee in calls:
        module = caller.split(":")[0]
        target = local.get(module, {}).get(callee) or definitions.get(callee)
        if not target or target == caller:
            continue
        key = (caller, target)
        if key in seen:
            continue
        seen.add(key)
        graph.edges.append(Edge(caller, "CALLS", target))
    for name in dict.fromkeys(graph.nodes):
        if not any(edge.source == name for edge in graph.edges):
            graph.nodes.remove(name)
    graph.seconds = time.perf_counter() - start
    return graph


class CompositeSolver:
    """Bind several role waves into one key and read a target from Zone C.

    Decision C4. The topology solver keys on (source, operator) because a
    graph transition has two roles. A composition such as modular addition
    has three: the two operands and the operator. The key is the ordered
    product of their phasors, so the atlas stays indexed (one row per
    composition) and retrieval is exact at any edge count. No continuous
    weight is trained here: the composition lives in memory, and the
    perception that fills the operands lives in Zone B.
    """

    def __init__(self, memory, codebook: SymbolCodebook, tau_coherence: float = 0.0) -> None:
        self.memory = memory
        self.codebook = codebook
        self.tau_coherence = float(tau_coherence)
        self.entries = 0

    def key(self, parts: Sequence[str]) -> np.ndarray:
        wave = np.ones(self.codebook.dim, dtype=np.complex64)
        for part in parts:
            wave = wave * self.codebook.symbol_wave(str(part))
        return _unit(wave)

    def ingest(self, rows: Sequence[tuple], tier: int = 1) -> int:
        written = 0
        for parts, target in rows:
            self.memory.write(
                self.key(parts),
                label=SYMBOL_LABEL_OFFSET + int(self.codebook.add(str(target))),
                tier=int(tier),
            )
            written += 1
        self.entries += written
        return written

    def calibrate(self, rows: Sequence[tuple], margin: float = 0.02) -> Dict[str, float]:
        values: List[float] = []
        for parts, _ in rows:
            read = self.memory.query(self.key(parts), k=1)
            if read.hits:
                values.append(float(read.hits[0].coherence))
        if not values:
            return {"tau_coherence": 0.0, "n": 0.0, "coherence_median": 0.0}
        tau = max(0.0, float(np.min(values)) - float(margin))
        self.tau_coherence = tau
        return {
            "tau_coherence": tau,
            "n": float(len(values)),
            "coherence_median": float(np.median(values)),
        }

    def step(self, parts: Sequence[str]) -> Dict[str, object]:
        read = self.memory.query(self.key(parts), k=1)
        if not read.hits:
            return {"answer": None, "coherence": 0.0, "vetoed": True, "reason": "empty memory"}
        hit = read.hits[0]
        symbol_index = int(hit.label) - SYMBOL_LABEL_OFFSET
        valid_symbol = 0 <= symbol_index < len(self.codebook)
        vetoed = (not valid_symbol) or float(hit.coherence) < float(self.tau_coherence)
        answer = self.codebook.symbols[symbol_index] if valid_symbol and not vetoed else None
        return {
            "answer": answer,
            "coherence": float(hit.coherence),
            "vetoed": vetoed,
            "reason": "coherence below tau" if vetoed else "",
        }


class SymbolicDecoder:
    """Deterministic renderer. It emits exact symbols, never prose.

    The output of a solver is a path, a patch plan, or an abstention. A
    local symbolic decoder turns that structure into text with a fixed
    grammar, so no generative model sits between the verified answer and
    the caller.
    """

    def __init__(self, indent: str = "  ") -> None:
        self.indent = indent

    def render_path(self, result: SolveResult) -> str:
        if result.answer is None:
            return f"ABSTAIN: {result.reason}"
        lines = [f"SOLVED {result.answer}"]
        current = result.path[0][0] if result.path else result.answer
        for index, (_, operator) in enumerate(result.path):
            confidence = result.coherences[index] if index < len(result.coherences) else 0.0
            lines.append(
                f"{self.indent}{index + 1}. {current} --{operator}--> "
                f"coherence {confidence:.3f}"
            )
            current = result.answer if index + 1 == len(result.path) else "?"
        return "\n".join(lines)

    def render_plan(self, name: str, steps: Sequence[str], verified: bool) -> str:
        header = f"PLAN {name} [{'verified' if verified else 'unverified'}]"
        body = [f"{self.indent}{index + 1}. {step}" for index, step in enumerate(steps)]
        return "\n".join([header] + body)

    def render_patch_plan(self, file: str, intent: str, edits: Sequence[Tuple[str, str]]) -> str:
        lines = [f"PATCH {file}", f"{self.indent}intent: {intent}"]
        for anchor, replacement in edits:
            lines.append(f"{self.indent}- {anchor}")
            lines.append(f"{self.indent}+ {replacement}")
        return "\n".join(lines)
