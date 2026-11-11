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

    def named_phasor(self, name: str, salt: int = 71) -> np.ndarray:
        """A stable unit phasor for any named role, tag or register (S2).

        The Turing machine addresses its states, tape symbols, opcodes and
        role binders through this one entry point, so a new symbol needs no
        vocabulary and no training: it is addressable the moment it is named.
        """
        return self._phasor(str(name), int(salt))

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
        align = torch.matmul(matrix, query.conj()).real.numpy()
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
        # Directive F1: one symbol table per tenant. The default tenant keeps
        # the shared codebook, so every pre-existing label keeps its meaning.
        self._ns_symbols: Dict[str, List[str]] = {}
        # Directive G4: a namespace whose table has not been restored yet.
        # The default tenant is always resident (it is the codebook).
        self._loaded: set = {"default"} if hasattr(self.codebook, "symbols") else set()

    def _ensure_namespace(self, namespace: str) -> None:
        """Restore a tenant's symbol table from Zone C on first use (G4)."""
        namespace = str(namespace or "default")
        if namespace in self._loaded:
            return
        self._loaded.add(namespace)
        getter = getattr(self.memory, "get_meta", None)
        if getter is None:
            return
        rows = getter(namespace, "symbols", "table")
        payload = rows[0][1] if rows else {}
        self._ns_symbols[namespace] = [str(name) for name in payload.get("symbols", [])]

    def _persist(self, namespace: str) -> None:
        namespace = str(namespace or "default")
        putter = getattr(self.memory, "put_meta", None)
        if putter is None:
            return
        table = self.codebook.symbols if namespace == "default" else self._ns_symbols.get(namespace, [])
        putter(namespace, "symbols", "table", {"symbols": list(table)})

    def _add_symbol(self, namespace: str, name: str) -> int:
        namespace = str(namespace or "default")
        self._ensure_namespace(namespace)
        if namespace == "default":
            return int(self.codebook.add(str(name)))
        table = self._ns_symbols.setdefault(namespace, [])
        if str(name) not in table:
            table.append(str(name))
        return int(table.index(str(name)))

    def _symbol_at(self, namespace: str, index: int) -> Optional[str]:
        namespace = str(namespace or "default")
        self._ensure_namespace(namespace)
        index = int(index)
        if namespace == "default":
            symbols = self.codebook.symbols
        else:
            symbols = self._ns_symbols.get(namespace, [])
        return symbols[index] if 0 <= index < len(symbols) else None

    # --------------------------------------------------------------- ingest
    def key_wave(self, source: str, operator: str) -> np.ndarray:
        return _unit(self.codebook.symbol_wave(source) * self.codebook.operator_wave(operator))

    def transition_wave(self, edge: Edge) -> np.ndarray:
        return _unit(
            self.codebook.symbol_wave(edge.source)
            * self.codebook.operator_wave(edge.operator)
            * self.codebook.symbol_wave(edge.target)
        )

    def ingest(self, edges: Iterable[Edge], tier: int = 1, namespace: str = "default") -> int:
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
                    label=SYMBOL_LABEL_OFFSET + int(self._add_symbol(namespace, edge.target)),
                    tier=int(tier),
                    namespace=str(namespace),
                )
            else:
                self.bundle = _unit(self.bundle + self.transition_wave(edge))
                self.memory.write(
                    self.transition_wave(edge), label=-1, tier=int(tier), namespace=str(namespace)
                )
            count += 1
        self.written += count
        self._persist(str(namespace))
        return count

    # ----------------------------------------------------------------- step
    def step(self, source: str, operator: str, namespace: str = "default") -> Dict[str, object]:
        """Resolve one transition and snap to an exact symbol."""
        if self.mode == "indexed":
            key = self.key_wave(source, operator)
            read = self.memory.query(key, k=1, namespace=str(namespace))
            if not read.hits:
                return {"answer": None, "coherence": 0.0, "vetoed": True, "reason": "empty memory"}
            hit = read.hits[0]
            symbol_index = int(hit.label) - SYMBOL_LABEL_OFFSET
            answer = self._symbol_at(namespace, symbol_index)
            valid_symbol = answer is not None
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

    def solve(self, source: str, operators: Sequence[str], namespace: str = "default") -> SolveResult:
        """Multi-hop navigation. Abstains at the first unverifiable hop."""
        current = str(source)
        path: List[Tuple[str, str]] = []
        coherences: List[float] = []
        for operator in operators:
            result = self.step(current, operator, namespace=str(namespace))
            coherences.append(float(result["coherence"]))
            if result["answer"] is None:
                return SolveResult(
                    None, path, coherences, True, str(result.get("reason", "no transition"))
                )
            path.append((current, str(operator)))
            current = str(result["answer"])
        return SolveResult(current, path, coherences, False, "")

    # ------------------------------------------------------------ calibration
    def calibrate(
        self,
        edges: Sequence[Edge],
        quantile: float = 0.995,
        margin: float = 0.02,
        namespace: str = "default",
    ) -> Dict[str, float]:
        """Set the abstention threshold from reachable transitions only."""
        values = [
            float(self.step(edge.source, edge.operator, namespace=str(namespace))["coherence"])
            for edge in edges
        ]
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
        self._ns_symbols: Dict[str, List[str]] = {}
        self._loaded: set = {"default"} if hasattr(self.codebook, "symbols") else set()

    def _ensure_namespace(self, namespace: str) -> None:
        namespace = str(namespace or "default")
        if namespace in self._loaded:
            return
        self._loaded.add(namespace)
        getter = getattr(self.memory, "get_meta", None)
        if getter is None:
            return
        rows = getter(namespace, "symbols", "table")
        payload = rows[0][1] if rows else {}
        self._ns_symbols[namespace] = [str(name) for name in payload.get("symbols", [])]

    def _persist(self, namespace: str) -> None:
        namespace = str(namespace or "default")
        putter = getattr(self.memory, "put_meta", None)
        if putter is None:
            return
        table = self.codebook.symbols if namespace == "default" else self._ns_symbols.get(namespace, [])
        putter(namespace, "symbols", "table", {"symbols": list(table)})

    def _add_symbol(self, namespace: str, name: str) -> int:
        namespace = str(namespace or "default")
        self._ensure_namespace(namespace)
        if namespace == "default":
            return int(self.codebook.add(str(name)))
        table = self._ns_symbols.setdefault(namespace, [])
        if str(name) not in table:
            table.append(str(name))
        return int(table.index(str(name)))

    def _symbol_at(self, namespace: str, index: int) -> Optional[str]:
        namespace = str(namespace or "default")
        self._ensure_namespace(namespace)
        index = int(index)
        symbols = self.codebook.symbols if namespace == "default" else self._ns_symbols.get(namespace, [])
        return symbols[index] if 0 <= index < len(symbols) else None

    def key(self, parts: Sequence[str]) -> np.ndarray:
        wave = np.ones(self.codebook.dim, dtype=np.complex64)
        for part in parts:
            wave = wave * self.codebook.symbol_wave(str(part))
        return _unit(wave)

    def ingest(self, rows: Sequence[tuple], tier: int = 1, namespace: str = "default") -> int:
        written = 0
        for parts, target in rows:
            self.memory.write(
                self.key(parts),
                label=SYMBOL_LABEL_OFFSET + int(self._add_symbol(namespace, str(target))),
                tier=int(tier),
                namespace=str(namespace),
            )
            written += 1
        self.entries += written
        self._persist(str(namespace))
        return written

    def calibrate(
        self, rows: Sequence[tuple], margin: float = 0.02, namespace: str = "default"
    ) -> Dict[str, float]:
        values: List[float] = []
        for parts, _ in rows:
            read = self.memory.query(self.key(parts), k=1, namespace=str(namespace))
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

    def step(self, parts: Sequence[str], namespace: str = "default") -> Dict[str, object]:
        read = self.memory.query(self.key(parts), k=1, namespace=str(namespace))
        if not read.hits:
            return {"answer": None, "coherence": 0.0, "vetoed": True, "reason": "empty memory"}
        hit = read.hits[0]
        symbol_index = int(hit.label) - SYMBOL_LABEL_OFFSET
        answer = self._symbol_at(namespace, symbol_index)
        valid_symbol = answer is not None
        vetoed = (not valid_symbol) or float(hit.coherence) < float(self.tau_coherence)
        answer = answer if valid_symbol and not vetoed else None
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


# ---------------------------------------------------------------------------
# Directive F4 / Stage 2: the code graph indexer with signatures and impact
# ---------------------------------------------------------------------------
GRAMMAR_LABEL_OFFSET = 30000
"""Zone C label block for AST grammar, signature and type-boundary engrams."""


@dataclass
class FunctionSignature:
    """One callable's declared surface, extracted from the AST."""

    qualname: str
    name: str
    args: Tuple[str, ...]
    arity: int
    returns: str
    module: str
    lineno: int

    def as_dict(self) -> Dict[str, object]:
        return {
            "qualname": self.qualname,
            "name": self.name,
            "args": list(self.args),
            "arity": int(self.arity),
            "returns": self.returns,
            "module": self.module,
            "lineno": int(self.lineno),
        }


@dataclass
class IndexReport:
    """Everything one index pass measured."""

    files: int = 0
    lines: int = 0
    nodes: int = 0
    edges: int = 0
    signatures: int = 0
    transitions: int = 0
    classes: int = 0
    seconds: float = 0.0
    boundary_engrams: int = 0
    modules: List[str] = field(default_factory=list)
    updated: str = ""
    delta: bool = False

    def as_dict(self) -> Dict[str, float]:
        return {
            "files": float(self.files),
            "lines": float(self.lines),
            "nodes": float(self.nodes),
            "edges": float(self.edges),
            "signatures": float(self.signatures),
            "transitions": float(self.transitions),
            "classes": float(self.classes),
            "seconds": float(self.seconds),
            "boundary_engrams": float(self.boundary_engrams),
            "modules": float(len(self.modules)),
            "delta": float(1.0 if self.delta else 0.0),
        }


class CodeGraphIndexer:
    """Index a package once: call edges, signatures, and grammar boundaries.

    Directive F1 gives every index a namespace; Directive F4 adds the reverse
    call query. Stage 2 of the tandem protocol adds the signature and grammar
    extraction: each function's arity, argument names, return annotation and
    the parent-child AST transitions the package actually uses are written as
    Tier-3 boundary engrams in Zone C, so the Sagnac gate has an axiom to
    compare a proposed statement against.

    The indexer never edits the package and never imports it: it reads source
    text with ast.parse in one pass.
    """

    def __init__(
        self,
        memory=None,
        namespace: str = "default",
        symbol_codebook: Optional[SymbolCodebook] = None,
        grammar_offset: int = GRAMMAR_LABEL_OFFSET,
    ) -> None:
        self.memory = memory
        self.namespace = str(namespace or "default")
        self.codebook = symbol_codebook
        self.grammar_offset = int(grammar_offset)
        self.graph = CallGraph(root="")
        self.signatures: Dict[str, FunctionSignature] = {}
        self.transitions: Dict[Tuple[str, str, str], int] = {}
        self.reverse: Dict[str, List[str]] = {}
        self.classes: Dict[str, List[str]] = {}
        self._nodes: set = set()
        self.report = IndexReport()
        self.root = ""
        self.package = "henri"
        # Directive H4: per-module records are what makes a single-file
        # delta possible without re-parsing the package.
        self._modules: Dict[str, Dict[str, object]] = {}
        self._module_defs: Dict[str, set] = {}
        self._module_edges: Dict[str, List[Edge]] = {}

    # ----------------------------------------------------------------- index
    def _ingest_module(self, module: str, text: str, tree) -> Dict[str, object]:
        """Parse ONE module into the record the rebuild needs.

        Directive H4. The full-index pass and the single-file delta share
        this function, so a delta cannot disagree with a full re-index: the
        same walk order, the same signature order, the same call order.
        """
        nodes: List[str] = []
        module_defs: Dict[str, str] = {}
        calls: List[Tuple[str, str]] = []
        transitions: Dict[Tuple[str, str, str], int] = {}
        signatures: List[FunctionSignature] = []
        classes: Dict[str, List[str]] = {}
        # Directive I4: the import table makes a cross-package call resolve to
        # the module it was imported from, not to the first same-named symbol
        # in the workspace. It is a preference, never an override: a local
        # definition still wins, and an unimported name still falls back to
        # the global table, so the single-package graph is unchanged.
        imports: Dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports[alias.asname or alias.name.split(".")[0]] = alias.name
            elif isinstance(node, ast.ImportFrom) and node.module and not int(node.level or 0):
                for alias in node.names:
                    imports[alias.asname or alias.name] = f"{node.module}:{alias.name}"

        def record(node) -> None:
            # Grammar transitions: the same (parent, field, child) triples
            # _collect_transitions counts, folded into the one walk so a
            # delta does not traverse the module three times.
            for field_name, value in ast.iter_fields(node):
                children = value if isinstance(value, list) else [value]
                for child in children:
                    if isinstance(child, ast.AST):
                        key = (type(node).__name__, str(field_name), type(child).__name__)
                        transitions[key] = transitions.get(key, 0) + 1

        def walk(node, enclosing: str) -> None:
            record(node)
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    qual = f"{module}:{child.name}"
                    module_defs.setdefault(child.name, qual)
                    nodes.append(qual)
                    if isinstance(child, ast.ClassDef):
                        classes[child.name] = [
                            item.name
                            for item in child.body
                            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                        ]
                    else:
                        signatures.append(_signature(child, module))
                    walk(child, qual)
                else:
                    if isinstance(child, ast.Call):
                        name = _callee_name(child)
                        if name:
                            calls.append((enclosing, name))
                    walk(child, enclosing)

        walk(tree, module)
        return {
            "nodes": nodes,
            "defs": module_defs,
            "calls": calls,
            "transitions": transitions,
            "signatures": signatures,
            "classes": classes,
            "imports": imports,
            "lines": int(text.count(chr(10))) + 1,
        }

    def _resolve_module(
        self, module: str, local_defs: Dict[str, str], definitions: Dict[str, str]
    ) -> List[Edge]:
        """Resolve one module's calls against the current definition tables."""
        edges: List[Edge] = []
        seen: set = set()
        imports = dict(self._modules[module].get("imports", {}) or {})
        for caller, callee in self._modules[module]["calls"]:
            target = local_defs.get(callee)
            if not target:
                imported = imports.get(callee, "")
                if ":" in imported:
                    source_module, symbol = imported.split(":", 1)
                    candidate = definitions.get(symbol)
                    if candidate and candidate.startswith(f"{source_module}:"):
                        target = candidate
            if not target:
                target = definitions.get(callee)
            if not target or target == caller:
                continue
            key = (caller, target)
            if key in seen:
                continue
            seen.add(key)
            edges.append(Edge(caller, "CALLS", target))
        return edges

    def _rebuild(self, changed: Optional[str] = None) -> None:
        """Recompute the graph, signatures, classes and reverse map from records.

        Directive H4: when one module changed, only the modules whose
        resolution can actually move are re-resolved - the changed module
        itself and every module that calls a name the change added or
        removed. Everything else keeps its cached edge list, which is why a
        single-file delta costs a dictionary pass and not a graph pass.
        """
        modules = sorted(self._modules)
        graph = CallGraph(root=str(self.root))
        definitions: Dict[str, str] = {}
        local: Dict[str, Dict[str, str]] = {}
        for module in modules:
            record = self._modules[module]
            graph.nodes.extend(str(node) for node in record["nodes"])
            for name, qual in dict(record["defs"]).items():
                definitions.setdefault(str(name), str(qual))
            local[module] = dict(record["defs"])
        previous_defs = {key: set(value) for key, value in self._module_defs.items()}
        for module in list(self._module_defs):
            if module not in self._modules:
                self._module_defs.pop(module, None)
                self._module_edges.pop(module, None)
        changed_names: set = set()
        if changed is not None:
            fresh = set(self._modules[changed]["defs"]) if changed in self._modules else set()
            changed_names = fresh ^ set(previous_defs.get(changed, set()))
        for module in modules:
            self._module_defs[module] = set(self._modules[module]["defs"])
        edges: List[Edge] = []
        for module in modules:
            cached = self._module_edges.get(module)
            recalc = (
                changed is None
                or module == changed
                or cached is None
            )
            if not recalc and changed_names:
                callees = {callee for _caller, callee in self._modules[module]["calls"]}
                recalc = bool(callees & changed_names)
            if recalc:
                cached = self._resolve_module(module, local[module], definitions)
                self._module_edges[module] = cached
            edges.extend(cached)
        graph.edges = edges
        # The original index removed one occurrence per name that never
        # appears as a source, and kept any duplicate occurrence. Preserve
        # that exact rule so a delta result is list-identical.
        sources = {edge.source for edge in edges}
        kept: List[str] = []
        removed: set = set()
        for name in graph.nodes:
            if name not in sources and name not in removed:
                removed.add(name)
                continue
            kept.append(name)
        graph.nodes = kept
        signatures: Dict[str, FunctionSignature] = {}
        classes: Dict[str, List[str]] = {}
        transitions: Dict[Tuple[str, str, str], int] = {}
        for module in modules:
            record = self._modules[module]
            for signature in record["signatures"]:
                signatures.setdefault(f"{module}:{signature.name}", signature)
                signatures.setdefault(signature.name, signature)
            classes.update(dict(record["classes"]))
            for key, count in dict(record["transitions"]).items():
                transitions[key] = transitions.get(key, 0) + int(count)
        reverse: Dict[str, List[str]] = {}
        for edge in edges:
            # Directive F4: a caller may name the target by its qualified
            # path, by its bare function name, or by Class.method. Index all
            # three so the blast-radius query answers whichever the caller
            # wrote, without a fuzzy match.
            for alias in _aliases(edge.target):
                reverse.setdefault(alias, []).append(edge.source)
        self.graph = graph
        self.signatures = signatures
        self.classes = classes
        self.transitions = transitions
        self.reverse = {key: sorted(set(value)) for key, value in reverse.items()}
        self._nodes = {str(node) for node in graph.nodes}

    def _report(self, seconds: float, updated: str = "", delta: bool = False) -> IndexReport:
        return IndexReport(
            files=len(self._modules),
            lines=int(sum(int(record["lines"]) for record in self._modules.values())),
            nodes=len(self.graph.nodes),
            edges=len(self.graph.edges),
            signatures=len({id(value) for value in self.signatures.values()}),
            transitions=len(self.transitions),
            classes=len(self.classes),
            seconds=float(seconds),
            modules=sorted(self._modules),
            updated=str(updated),
            delta=bool(delta),
        )

    def index(self, root: str | Path, package: str = "henri") -> IndexReport:
        started = time.perf_counter()
        root = Path(root)
        self.root = str(root)
        self.package = str(package)
        self._modules = {}
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
                tree = ast.parse(text)
            except (OSError, SyntaxError):
                continue
            module = ".".join(path.relative_to(root).with_suffix("").parts)
            self._modules[module] = self._ingest_module(module, text, tree)
        self._module_defs = {}
        self._module_edges = {}
        self._rebuild()
        self.report = self._report(time.perf_counter() - started)
        if self.memory is not None:
            self.report.boundary_engrams = self.write_boundary_engrams()
            self.persist()
        return self.report

    def index_paths(
        self,
        root: str | Path,
        directories: Sequence[str | Path],
        package: str = "workspace",
    ) -> IndexReport:
        """Index several packages/directories under one workspace root (I4).

        Module names stay relative to the workspace root (henri.holarch.x,
        scripts.run_x, tests.test_x), so a cross-package edge is a real
        qualname and a single-file delta can address any of them with the
        same update_file path the single-package index uses.
        """
        started = time.perf_counter()
        root = Path(root)
        self.root = str(root)
        self.package = str(package)
        self._modules = {}
        paths: set = set()
        for directory in directories:
            label = str(directory)
            base = root if label in {"", "."} else root / label
            if not base.exists():
                continue
            for path in sorted(base.rglob("*.py")):
                if "__pycache__" in path.parts:
                    continue
                paths.add(path)
        for path in sorted(paths):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
                tree = ast.parse(text)
            except (OSError, SyntaxError):
                continue
            module = ".".join(path.relative_to(root).with_suffix("").parts)
            self._modules[module] = self._ingest_module(module, text, tree)
        self._module_defs = {}
        self._module_edges = {}
        self._rebuild()
        self.report = self._report(time.perf_counter() - started)
        if self.memory is not None:
            self.report.boundary_engrams = self.write_boundary_engrams()
            self.persist()
        return self.report

    def index_delta(
        self,
        root: str | Path,
        file_path: str | Path,
        source_text: Optional[str] = None,
    ) -> IndexReport:
        """Explicit delta entry point (I4). Same code path as update_file."""
        return self.update_file(root, file_path, source_text=source_text)

    def update_file(
        self,
        root: str | Path,
        file_path: str | Path,
        source_text: Optional[str] = None,
    ) -> IndexReport:
        """Incremental single-file delta index (Directive H4).

        Only the named module is parsed. Its previous call edges,
        signatures, class members, grammar transitions and reverse-call
        entries are purged, the new record is folded in, and the graph is
        rebuilt from the cached per-module records - a pass over integers,
        not a pass over source. A full re-index and this delta therefore
        produce the same graph by construction, and a test asserts it.
        """
        started = time.perf_counter()
        root = Path(root) if str(root or "").strip() else Path(self.root or ".")
        if not str(self.root):
            self.root = str(root)
        path = Path(file_path)
        if not path.is_absolute():
            path = root / path
        try:
            relative = path.relative_to(root)
        except ValueError:
            relative = Path(path.name)
        module = ".".join(relative.with_suffix("").parts)
        if not self._modules and self.graph.nodes:
            # A restored index carries the graph but not the per-module
            # records the delta needs. Re-index instead of pretending.
            report = self.index(root, package=self.package)
            report.delta = True
            report.updated = module
            return report
        if source_text is None and not path.exists():
            self._modules.pop(module, None)
        else:
            text = (
                str(source_text)
                if source_text is not None
                else path.read_text(encoding="utf-8", errors="ignore")
            )
            try:
                tree = ast.parse(text)
            except SyntaxError as error:
                raise ValueError(f"cannot index {module}: {error.msg} at line {error.lineno}")
            self._modules[module] = self._ingest_module(module, text, tree)
        self._rebuild(changed=module)
        self.report = self._report(
            time.perf_counter() - started, updated=module, delta=True
        )
        if self.memory is not None:
            self.persist()
        return self.report

    # ------------------------------------------------------------ persistence
    def persist(self) -> bool:
        """Write the index to Zone C metadata (Directive G4).

        The reverse map, the signatures and the edge list are what a restarted
        process needs to answer an impact query without re-parsing a file.
        """
        putter = getattr(self.memory, "put_meta", None) if self.memory is not None else None
        if putter is None:
            return False
        # Directive H4: the payload carries the primary state only. The
        # reverse map is a pure function of the edge list and the bare-name
        # signature aliases are a pure function of the qualified ones, so
        # both are rebuilt on load instead of being written twice. That
        # halves the bytes a single-file delta must serialise.
        # Directive O2: the per-module records travel with the graph, so a
        # sibling lane that restores the index can still apply a single-file
        # delta without re-parsing the package from disk. The record holds the
        # resolution inputs (nodes, defs, calls, imports); signatures,
        # transitions and classes are already stored globally below.
        module_records = {
            module: {
                "nodes": list(record.get("nodes", [])),
                "defs": dict(record.get("defs", {})),
                "calls": [[str(pair[0]), str(pair[1])] for pair in record.get("calls", [])],
                "imports": dict(record.get("imports", {})),
                "lines": int(record.get("lines", 0)),
            }
            for module, record in self._modules.items()
        }
        payload = {
            "root": str(self.root),
            "package": str(self.package),
            "nodes": sorted(self._nodes),
            "module_records": module_records,
            "edges": [[edge.source, edge.operator, edge.target] for edge in self.graph.edges],
            "signatures": {
                key: signature.as_dict()
                for key, signature in self.signatures.items()
                if ":" in key
            },
            "transitions": [
                [parent, field_name, child, int(count)]
                for (parent, field_name, child), count in self.transitions.items()
            ],
            "classes": {name: list(members) for name, members in self.classes.items()},
            "modules": [str(name) for name in self.report.modules],
            "report": self.report.as_dict(),
        }
        putter(self.namespace, "code_graph", "index", payload)
        return True

    @classmethod
    def from_memory(cls, memory, namespace: str = "default") -> Optional["CodeGraphIndexer"]:
        """Rebuild the index from persisted metadata, or None if absent."""
        getter = getattr(memory, "get_meta", None)
        if getter is None:
            return None
        rows = getter(str(namespace), "code_graph", "index")
        if not rows:
            return None
        payload = rows[0][1]
        indexer = cls(memory, namespace=str(namespace))
        indexer.root = str(payload.get("root", ""))
        indexer.package = str(payload.get("package", "henri"))
        # Directive O2: restore the per-module resolution inputs first, so a
        # single-file delta applied to a restored index resolves imports and
        # callees exactly as the live writer would. Transitions, signatures and
        # classes are global and are restored from their own tables below.
        records = payload.get("module_records") or {}
        if records:
            indexer._modules = {
                str(module): {
                    "nodes": [str(node) for node in record.get("nodes", [])],
                    "defs": {
                        str(key): str(value)
                        for key, value in dict(record.get("defs", {})).items()
                    },
                    "calls": [
                        (str(pair[0]), str(pair[1]))
                        for pair in record.get("calls", [])
                        if len(pair) >= 2
                    ],
                    # The per-module grammar tables are re-derived below from
                    # the global ones, because _rebuild() re-folds them per
                    # module after a delta and an empty record would silently
                    # erase the store's signatures and classes.
                    "transitions": {},
                    "signatures": [],
                    "classes": {},
                    "imports": {
                        str(key): str(value)
                        for key, value in dict(record.get("imports", {})).items()
                    },
                    "lines": int(record.get("lines", 0)),
                }
                for module, record in records.items()
            }
            indexer._module_defs = {
                str(module): set(record["defs"]) for module, record in indexer._modules.items()
            }
            indexer._module_edges = {}
        graph = CallGraph(root=indexer.root)
        graph.nodes = [str(node) for node in payload.get("nodes", [])]
        graph.edges = [
            Edge(str(row[0]), str(row[1]), str(row[2]))
            for row in payload.get("edges", [])
            if len(row) >= 3
        ]
        indexer.graph = graph
        indexer._nodes = set(graph.nodes)
        stored_reverse = payload.get("reverse")
        if stored_reverse is None:
            rebuilt: Dict[str, List[str]] = {}
            for edge in graph.edges:
                for alias in _aliases(edge.target):
                    rebuilt.setdefault(alias, []).append(edge.source)
            indexer.reverse = {key: sorted(set(value)) for key, value in rebuilt.items()}
        else:
            indexer.reverse = {
                str(key): [str(value) for value in values]
                for key, values in stored_reverse.items()
            }
        signatures: Dict[str, FunctionSignature] = {}
        for key, row in payload.get("signatures", {}).items():
            signature = FunctionSignature(
                qualname=str(row.get("qualname", key)),
                name=str(row.get("name", str(key).split(":")[-1])),
                args=tuple(str(name) for name in row.get("args", [])),
                arity=int(row.get("arity", 0)),
                returns=str(row.get("returns", "")),
                module=str(row.get("module", "")),
                lineno=int(row.get("lineno", 0)),
            )
            signatures.setdefault(str(key), signature)
            # Rebuild the bare-name alias the writer no longer stores.
            if ":" in str(key):
                signatures.setdefault(signature.name, signature)
        indexer.signatures = signatures
        indexer.transitions = {
            (str(row[0]), str(row[1]), str(row[2])): int(row[3])
            for row in payload.get("transitions", [])
            if len(row) >= 4
        }
        indexer.classes = {
            str(key): [str(name) for name in value]
            for key, value in payload.get("classes", {}).items()
        }
        report = payload.get("report", {})
        indexer.report = IndexReport(
            files=int(report.get("files", 0)),
            lines=int(report.get("lines", 0)),
            nodes=int(report.get("nodes", len(graph.nodes))),
            edges=int(report.get("edges", len(graph.edges))),
            signatures=int(report.get("signatures", len(indexer.signatures))),
            transitions=int(report.get("transitions", len(indexer.transitions))),
            classes=int(report.get("classes", len(indexer.classes))),
            seconds=0.0,
            boundary_engrams=int(report.get("boundary_engrams", 0)),
            modules=[str(name) for name in payload.get("modules", [])],
        )
        # Fold the global signature, class and grammar tables back into the
        # module records, AFTER those tables are restored. _rebuild() re-folds
        # them per module on every delta, so an empty record would silently
        # erase the store's signatures and classes the first time a file moves.
        if indexer._modules:
            by_module = {str(module): [] for module in indexer._modules}
            for signature in indexer.signatures.values():
                owner = str(getattr(signature, "module", "") or "")
                if owner in by_module and signature not in by_module[owner]:
                    by_module[owner].append(signature)
            classes = dict(indexer.classes)
            first = True
            for module, record in indexer._modules.items():
                record["signatures"] = by_module.get(str(module), [])
                record["classes"] = dict(classes)
                record["transitions"] = dict(indexer.transitions) if first else {}
                first = False
        return indexer

    # ------------------------------------------------------- boundary engrams
    def write_boundary_engrams(self, tier: int = 3) -> int:
        """Persist signatures and grammar transitions as Tier-3 axioms."""
        if self.memory is None:
            return 0
        written = 0
        seen: set = set()
        for signature in self.signatures.values():
            key = signature.qualname
            if key in seen:
                continue
            seen.add(key)
            wave = self._boundary_wave(
                f"signature:{signature.qualname}:{','.join(signature.args)}:{signature.returns}"
            )
            self.memory.write(
                wave,
                label=self.grammar_offset + len(seen) - 1,
                tier=int(tier),
                namespace=self.namespace,
            )
            written += 1
        for index, (parent, field_name, child) in enumerate(sorted(self.transitions)):
            wave = self._boundary_wave(f"grammar:{parent}:{field_name}:{child}")
            self.memory.write(
                wave,
                label=self.grammar_offset + len(seen) + index,
                tier=int(tier),
                namespace=self.namespace,
            )
            written += 1
        return written

    def _boundary_wave(self, text: str) -> np.ndarray:
        """A deterministic unit phasor for one axiom string."""
        generator = np.random.default_rng(_seed_for(text, 71))
        dim = int(getattr(self.memory, "dim", 1024))
        wave = generator.standard_normal(dim) + 1j * generator.standard_normal(dim)
        return _unit(wave)

    # ---------------------------------------------------------------- impact
    def impact(self, target: str, hops: int = 1) -> Dict[str, object]:
        """Every upstream caller reachable within k reverse hops.

        A breaking change to target propagates to the callers that call it,
        then to the callers of those callers. The traversal is breadth first
        over the inverted edge set, so the hop count is exact.
        """
        hops = max(1, int(hops))
        frontier = sorted(_aliases(str(target)))
        seen: List[str] = []
        visited = set(frontier)
        depth = 0
        layers: List[List[str]] = []
        while frontier and depth < hops:
            depth += 1
            current: List[str] = []
            for node in frontier:
                for caller in self.reverse.get(node, []):
                    if caller in visited:
                        continue
                    visited.add(caller)
                    current.append(caller)
                    seen.append(caller)
                    for alias in _aliases(caller):
                        visited.add(alias)
            layers.append(sorted(set(current)))
            frontier = sorted(set(current))
        resolved = any(alias in self.signatures or alias in self._nodes for alias in _aliases(str(target)))
        return {
            "target": str(target),
            "hops": int(hops),
            "callers": sorted(seen),
            "layers": layers,
            "layers_total": len(layers),
            "resolved": bool(resolved),
        }

    def signature(self, qualname: str) -> Optional[FunctionSignature]:
        return self.signatures.get(str(qualname))

    def symbol_names(self) -> set:
        names = set(self.signatures.keys())
        for node in self.graph.nodes:
            names.add(str(node).split(":")[-1])
        return names

    def as_dict(self) -> Dict[str, object]:
        return {
            "report": self.report.as_dict(),
            "namespace": self.namespace,
            "signatures": len(self.signatures),
            "transitions": len(self.transitions),
            "classes": len(self.classes),
            "reverse_targets": len(self.reverse),
        }


def _aliases(name: str) -> List[str]:
    """The three names a symbol can be written as, most specific first."""
    text = str(name)
    out = [text]
    bare = text.split(":")[-1]
    if bare not in out:
        out.append(bare)
    tail = bare.split(".")[-1]
    if tail not in out:
        out.append(tail)
    return out


def _signature(node, module: str) -> FunctionSignature:
    args = list(getattr(node.args, "posonlyargs", [])) + list(node.args.args)
    names = [str(arg.arg) for arg in args] + [str(arg.arg) for arg in node.args.kwonlyargs]
    if node.args.vararg is not None:
        names.append("*" + str(node.args.vararg.arg))
    if node.args.kwarg is not None:
        names.append("**" + str(node.args.kwarg.arg))
    returns = ""
    if node.returns is not None:
        try:
            returns = ast.unparse(node.returns)
        except Exception:  # pragma: no cover - very old Python
            returns = ""
    return FunctionSignature(
        qualname=f"{module}:{node.name}",
        name=str(node.name),
        args=tuple(names),
        arity=len([name for name in names if not name.startswith("*")]),
        returns=returns,
        module=module,
        lineno=int(getattr(node, "lineno", 0)),
    )


def _collect_transitions(tree, sink: Dict[Tuple[str, str, str], int]) -> None:
    """Count every (parent type, field, child type) triple the AST uses."""
    for node in ast.walk(tree):
        for field_name, value in ast.iter_fields(node):
            children = value if isinstance(value, list) else [value]
            for child in children:
                if isinstance(child, ast.AST):
                    key = (type(node).__name__, str(field_name), type(child).__name__)
                    sink[key] = sink.get(key, 0) + 1


# =========================================================================
# Directive S2: the wave-mechanical latent Turing machine
# =========================================================================
# A deterministic Turing machine is a 5-tuple M = (Q, Sigma, Gamma, delta, q0).
# Every component embeds isometrically in HENRI's 4-slot complex wave manifold:
#
#   slot 0  control and program counter   psi_state(q) * psi_pc^t
#   slot 1  head position and opcode      psi_pos^x * psi_op(dx)
#   slot 2  holographic tape              sum_x psi_sym(T[x]) * psi_pos^x
#   slot 3  halting and safety law        psi_axiom
#
# Reading a cell is one pointwise complex multiplication (unbinding the head
# phase) followed by a Hopfield snap onto the alphabet. Writing a cell is a
# linear holographic delta. Moving the head is a unitary phase rotor. No step
# of the computation passes through the text tokenizer: the tape is a complex
# vector and every read and write is a wave operation on it.
#
# The transition atlas is a single bundle in which every rule is stored as
# key * (chord * tag): the key is psi_state(q) * psi_sym(s), the chord binds
# the three output fields under independent role phasors, and the per-rule tag
# keeps distinct chords incoherent. Unbinding the key recovers the chord, the
# tag snap identifies the rule, and the chord is then factorised into
# (q', s', dx) by role. The measured SNR of that snap is the physical veto.

@dataclass
class TuringProgram:
    """One deterministic program: a transition table and a tape."""

    name: str
    tape: List[str]
    alphabet: List[str]
    rules: Dict[Tuple[str, str], Tuple[str, int, str]]
    start: str
    halt: str
    expectation: str = ""
    description: str = ""
    head: int = 1
    max_steps: int = 1200

    def symbols(self) -> List[str]:
        names = set(self.alphabet)
        for (_, read), (write, _, _) in self.rules.items():
            names.add(str(read))
            if str(write) != "=":
                names.add(str(write))
        return sorted(names)

    def states(self) -> List[str]:
        names = {self.start, self.halt}
        for (state, _), (_, _, target) in self.rules.items():
            names.add(str(state))
            names.add(str(target))
        return sorted(names)



class TuringAssembler:
    """A small Turing-machine assembler: labelled blocks of move/write/switch ops."""
    def __init__(self, name, tape, alphabet, start="q0", halt="qh", max_steps=1200):
        self.name=name; self.tape=list(tape)+["_"]*6; self.alphabet=list(dict.fromkeys(list(alphabet)+["_"]))
        self.start=start; self.halt=halt; self.max_steps=int(max_steps)
        self.blocks: Dict[str, List[tuple]] = {"halt":[("halt",)]}
        self.expectation=""
        self._n=0
    def label(self, hint="L"):
        self._n+=1; return "%s%d"%(hint,self._n)
    def block(self, label, ops):
        self.blocks[label]=list(ops); return self
    def compile(self):
        rules={}
        for label, ops in self.blocks.items():
            flat=[]
            for op in ops:
                if isinstance(op,str): op=(op,)
                if op[0]=="m":
                    k=int(op[1]); step=1 if k>0 else -1
                    for _ in range(abs(k)): flat.append(("mm",step))
                else: flat.append(op)
            for idx,op in enumerate(flat):
                state="%s#%d"%(label,idx)
                nxt="%s#%d"%(label,idx+1) if idx+1<len(flat) else None
                kind=op[0]
                if kind=="mm":
                    for sym in self.alphabet: rules[(state,sym)]=("=",op[1],nxt)
                elif kind=="w":
                    for sym in self.alphabet: rules[(state,sym)]=(op[1],0,nxt)
                elif kind=="sw":
                    for sym in self.alphabet:
                        t=op[1].get(sym, op[2] if len(op)>2 else None)
                        if t is None: continue
                        rules[(state,sym)]=("=",0,t+"#0")
                elif kind=="jmp":
                    for sym in self.alphabet: rules[(state,sym)]=("=",0,op[1]+"#0")
                elif kind=="halt":
                    for sym in self.alphabet: rules[(state,sym)]=("=",0,self.halt)
                else: raise ValueError(kind)
        return rules
    def symbols(self):
        used=set(self.alphabet)
        for ops in self.blocks.values():
            for op in ops:
                if isinstance(op,str): op=(op,)
                if op[0]=="w": used.add(op[1])
                if op[0]=="sw": used |= set(op[1].keys())
        return sorted(used)

def ref_run(asm):
    rules=asm.compile(); tape=list(asm.tape); state=asm.start+"#0"; head=1; steps=0
    while state!=asm.halt:
        if steps>=asm.max_steps: return {"halted":False,"reason":"step-limit","tape":tape,"steps":steps}
        if head<0 or head>=len(tape): return {"halted":False,"reason":"head-off-tape","tape":tape,"steps":steps}
        rule=rules.get((state,tape[head]))
        if rule is None: return {"halted":False,"reason":"undefined:%s,%s"%(state,tape[head]),"tape":tape,"steps":steps}
        w,mv,t=rule
        if w!="=": tape[head]=w
        head+=mv; steps+=1; state=t
    return {"halted":True,"reason":"","tape":tape,"tape_text":"".join(tape).strip("_"),"steps":steps}

# ------------------------------------------------------------------ 1..8
def p_increment():
    a=TuringAssembler("binary_increment","_1011_","01_")
    a.block("q0",[("m",1),("sw",{"0":"q0","1":"q0","_":"back"})])
    a.block("back",[("m",-1),("jmp","chk")])
    a.block("chk",[("sw",{"1":"c1","0":"f0","_":"f1"})])
    a.block("c1",[("w","0"),("m",-1),("jmp","chk")])
    a.block("f0",[("w","1"),("halt")])
    a.block("f1",[("w","1"),("halt")])
    a.expectation="1100"; return a

def p_decrement():
    a=TuringAssembler("binary_decrement","_1011_","01_")
    a.block("q0",[("m",1),("sw",{"0":"q0","1":"q0","_":"back"})])
    a.block("back",[("m",-1),("jmp","chk")])
    a.block("chk",[("sw",{"1":"c1","0":"d0","_":"z"})])
    a.block("c1",[("w","0"),("halt")])
    a.block("d0",[("w","1"),("m",-1),("jmp","chk")])
    a.block("z",[("halt")])
    a.expectation="1010"; return a

def p_shift_left():
    a=TuringAssembler("binary_shift_left","_1011_","01_")
    a.block("q0",[("m",1),("sw",{"0":"q0","1":"q0","_":"w"})])
    a.block("w",[("w","0"),("halt")])
    a.expectation="10110"; return a

def p_shift_right():
    a=TuringAssembler("binary_shift_right","_1011_","01_")
    a.block("q0",[("m",1),("sw",{"0":"q0","1":"q0","_":"back"})])
    a.block("back",[("m",-1),("w","_"),("halt")])
    a.expectation="101"; return a

def p_toggle():
    a=TuringAssembler("toggle_bits","_1011_","01_")
    a.block("q0",[("sw",{"0":"s1","1":"s0","_":"halt"})])
    a.block("s1",[("w","1"),("m",1),("jmp","q0")])
    a.block("s0",[("w","0"),("m",1),("jmp","q0")])
    a.expectation="0100"; return a

def p_parity():
    a=TuringAssembler("binary_parity","_1011_","01EO")
    a.block("q0",[("m",1),("sw",{"0":"q0","1":"q0","_":"back"})])
    a.block("back",[("m",-1),("sw",{"0":"we","1":"wo"})])
    a.block("we",[("w","E"),("halt")])
    a.block("wo",[("w","O"),("halt")])
    a.expectation="101O"; return a

def p_unary_add():
    a=TuringAssembler("unary_accumulate","_111+11_","1+_")
    a.block("q0",[("m",1),("sw",{"1":"q0","+":"q0","_":"back"})])
    a.block("back",[("m",-1),("jmp","chk")])
    a.block("chk",[("sw",{"1":"del","+":"plus","_":"halt"})])
    a.block("del",[("w","_"),("m",-1),("jmp","seek")])
    a.block("seek",[("m",-1),("sw",{"1":"seek","+":"plus","_":"halt"})])
    a.block("plus",[("w","1"),("halt")])
    a.expectation="11111"; return a

def p_unary_double():
    a=TuringAssembler("unary_multiply_by_two","_11_","1|X_")
    a.block("q0",[("m",1),("sw",{"1":"q0","_":"sep"})])
    a.block("sep",[("w","|"),("jmp","rew")])
    a.block("rew",[("m",-1),("sw",{"1":"rew","|":"rew","X":"rew","_":"fwd"})])
    a.block("fwd",[("m",1),("jmp","pick")])
    a.block("pick",[("sw",{"1":"mark","X":"skip","_":"rrew","|":"rrew"})])
    a.block("skip",[("m",1),("jmp","pick")])
    a.block("rrew",[("m",-1),("sw",{"1":"rrew","0":"rrew","|":"rrew","X":"rrew","Y":"rrew","_":"rfwd"})])
    a.block("rfwd",[("m",1),("jmp","rsweep")])
    a.block("rsweep",[("sw",{"X":"r1","Y":"r0","1":"rskip","0":"rskip","|":"rskip","_":"rend"})])
    a.block("r1",[("w","1"),("m",1),("jmp","rsweep")])
    a.block("r0",[("w","0"),("m",1),("jmp","rsweep")])
    a.block("rskip",[("m",1),("jmp","rsweep")])
    a.block("rend",[("halt")])

    a.block("mark",[("w","X"),("m",1),("jmp","seek")])
    a.block("seek",[("sw",{"|":"past","_":"after","1":"s1","X":"s1"})])
    a.block("past",[("m",1),("jmp","after")])
    a.block("s1",[("m",1),("jmp","seek")])
    a.block("after",[("sw",{"1":"a1","_":"app"})])
    a.block("a1",[("m",1),("jmp","after")])
    a.block("app",[("w","1"),("jmp","rew")])
    a.expectation="11|11"; return a

def p_binary_add():
    a=TuringAssembler("binary_add","_1011+11_","01+_", start="tob", max_steps=1200)
    a.block("tob",[("m",1),("sw",{"0":"tob","1":"tob","+":"tob","_":"backb"})])
    a.block("backb",[("m",-1),("jmp","decb")])
    a.block("decb",[("sw",{"1":"eat","+":"plusend","_":"halt"})])
    a.block("eat",[("w","_"),("m",-1),("jmp","left")])
    a.block("left",[("sw",{"0":"l0","1":"l0","+":"lplus","_":"halt"})])
    a.block("l0",[("m",-1),("jmp","left")])
    a.block("lplus",[("m",-1),("jmp","inc")])
    a.block("inc",[("sw",{"1":"ic","0":"if","_":"ie"})])
    a.block("ic",[("w","0"),("m",-1),("jmp","inc")])
    a.block("if",[("w","1"),("jmp","tob")])
    a.block("ie",[("w","1"),("jmp","tob")])
    a.block("plusend",[("w","_"),("halt")])
    a.expectation="1101"; return a
# ------------------------------------------------------------------ 9..16
def p_sort():
    a=TuringAssembler("sort_bits","_1011_","01_", start="scan0")
    for v in (0,1):
        a.block("scan%d"%v,[("sw",{"0":"s0_%d"%v,"1":"s1_%d"%v,"_":"fin%d"%v})])
        for s in "01":
            a.block("s%s_%d"%(s,v),[("m",1),("sw",{"0":"p%s0_%d"%(s,v),"1":"p%s1_%d"%(s,v),"_":"e%s_%d"%(s,v)})])
        for s in "01":
            for t in "01":
                if s=="1" and t=="0":
                    a.block("p%s%s_%d"%(s,t,v),[("w","1"),("m",-1),("w","0"),("m",1),("jmp","scan1")])
                else:
                    a.block("p%s%s_%d"%(s,t,v),[("jmp","scan%d"%v)])
            a.block("e%s_%d"%(s,v),[("m",-1),("jmp","rew" if v==1 else "halt")])
    a.block("fin0",[("halt")])
    a.block("fin1",[("m",-1),("jmp","rew")])
    a.block("rew",[("m",-1),("sw",{"0":"rew","1":"rew","_":"fwd"})])
    a.block("fwd",[("m",1),("jmp","scan0")])
    a.expectation="0111"; return a

def p_reverse():
    a=TuringAssembler("tape_reverse","_1011_","01_")
    def emit(pairs, pos, tail):
        if not pairs: return tail
        i,j=pairs[0]
        cont=emit(pairs[1:], i, tail)
        entry=a.label("sw")
        spec={}
        for vi in "01":
            goj=a.label("go")
            specj={}
            for vj in "01":
                leaf=a.label("put")
                a.block(leaf,[("w",vi),("m",i-j),("w",vj),("jmp",cont)])
                specj[vj]=leaf
            a.block(goj,[("m",j-i),("sw",specj)])
            spec[vi]=goj
        a.block(entry,[("m",i-pos),("sw",spec)])
        return entry
    start=emit([(1,4),(2,3)],1,"halt")
    a.block("q0",[("jmp",start)])
    a.start="q0"
    a.expectation="1101"; return a

def p_palindrome():
    a=TuringAssembler("palindrome_binary","_1001_","01XY_", start="rew", max_steps=1200)
    a.block("rew",[("m",-1),("sw",{"0":"rew","1":"rew","X":"rew","Y":"rew","_":"fwd"})])
    a.block("fwd",[("m",1),("jmp","pick")])
    a.block("pick",[("sw",{"1":"m1","0":"m0","X":"yes","Y":"yes","_":"yes"})])
    a.block("m1",[("w","X"),("m",1),("jmp","sk1")])
    a.block("m0",[("w","Y"),("m",1),("jmp","sk0")])
    a.block("sk1",[("m",1),("sw",{"0":"sk1","1":"sk1","X":"sk1","Y":"sk1","_":"bk1"})])
    a.block("sk0",[("m",1),("sw",{"0":"sk0","1":"sk0","X":"sk0","Y":"sk0","_":"bk0"})])
    a.block("bk1",[("m",-1),("jmp","cm1")])
    a.block("bk0",[("m",-1),("jmp","cm0")])
    a.block("cm1",[("sw",{"1":"ok1","X":"yes","Y":"yes","0":"no","_":"yes"})])
    a.block("cm0",[("sw",{"0":"ok0","X":"yes","Y":"yes","1":"no","_":"yes"})])
    a.block("ok1",[("w","X"),("m",-1),("jmp","rew")])
    a.block("ok0",[("w","Y"),("m",-1),("jmp","rew")])
    for verdict in ("Y","N"):
        tag="y" if verdict=="Y" else "n"
        a.block("yes" if tag=="y" else "no",[("jmp","rew_%s"%tag)])
        a.block("rew_%s"%tag,[("m",-1),("sw",{"0":"rew_%s"%tag,"1":"rew_%s"%tag,"X":"rew_%s"%tag,"Y":"rew_%s"%tag,"_":"rst_%s"%tag})])
        a.block("rst_%s"%tag,[("m",1),("jmp","rs_%s"%tag)])
        a.block("rs_%s"%tag,[("sw",{"X":"w1_%s"%tag,"Y":"w0_%s"%tag,"_":"end_%s"%tag,"0":"m_%s"%tag,"1":"m_%s"%tag})])
        a.block("w1_%s"%tag,[("w","1"),("m",1),("jmp","rs_%s"%tag)])
        a.block("w0_%s"%tag,[("w","0"),("m",1),("jmp","rs_%s"%tag)])
        a.block("m_%s"%tag,[("m",1),("jmp","rs_%s"%tag)])
        a.block("end_%s"%tag,[("w",verdict),("halt")])
    a.expectation="1001Y"; return a

def p_brackets():
    a=TuringAssembler("bracket_match","_([{}])_","()[]{}~_YN", start="rew", max_steps=2000)
    OPS="([{"; CLS=")]}"; MATCH={")":"(","]":"[","}":"{"}
    a.block("rew",[("m",-1),("sw",{c:"rew" for c in "()[]{}~"}|{"_":"fwd"})])
    a.block("fwd",[("m",1),("jmp","s0")])
    for k, prev in enumerate((None,"(","[","{")):
        spec={}
        for sym in "()[]{}~_":
            if sym=="~":
                lab="mv_%d_t"%k; spec[sym]=lab
                a.block(lab,[("m",1),("jmp","s0" if k==0 else "sp%d"%k)])
            elif sym in CLS:
                if prev is not None and MATCH[sym]==prev:
                    spec[sym]="er_%d"%k
                else:
                    spec[sym]="no"
            elif sym in OPS:
                lab="mv_%d_%s"%(k,sym); spec[sym]=lab
                a.block(lab,[("m",1),("jmp","sp%d"%(OPS.index(sym)+1))])
            else:
                spec[sym]="fin"
        a.block("s0" if k==0 else "sp%d"%k,[("sw",spec)])
    for k, prev in enumerate((None,"(","[","{")):
        if prev is None: continue
        a.block("er_%d"%k,[("w","~"),("m",-1),("jmp","del_%d"%k)])
        a.block("del_%d"%k,[("sw",{"~":"dl_%d"%k,prev:"hit_%d"%k})])
        a.block("dl_%d"%k,[("m",-1),("jmp","del_%d"%k)])
        a.block("hit_%d"%k,[("w","~"),("jmp","rew")])
    a.block("fin",[("m",-1),("sw",{c:"fin" for c in "()[]{}~"}|{"_":"fstart"})])
    a.block("fstart",[("m",1),("jmp","fcheck")])
    a.block("fcheck",[("sw",{c:"no" for c in "()[]{}"}|{"~":"fc_skip","_":"yes"})])
    a.block("fc_skip",[("m",1),("jmp","fcheck")])
    for verdict, tag, name in (("Y","y","yes"),("N","n","no")):
        a.block(name,[("sw",{c:"skip_%s"%tag for c in "()[]{}~"}|{"_":"end_%s"%tag})])
        a.block("skip_%s"%tag,[("m",1),("jmp",name)])
        a.block("end_%s"%tag,[("w",verdict),("halt")])
    a.expectation="Y"; return a

def p_count_ones():
    a=TuringAssembler("count_ones_unary","_1011_","01|X Y_".replace(" ",""), max_steps=1600)
    a.block("q0",[("m",1),("sw",{"0":"q0","1":"q0","_":"sep"})])
    a.block("sep",[("w","|"),("jmp","rew")])
    a.block("rew",[("m",-1),("sw",{"0":"rew","1":"rew","X":"rew","Y":"rew","|":"rew","_":"fwd"})])
    a.block("fwd",[("m",1),("jmp","pick")])
    a.block("pick",[("sw",{"1":"m1","0":"m0","X":"skip","Y":"skip","|":"rrew","_":"rrew"})])
    a.block("m1",[("w","X"),("m",1),("jmp","seek")])
    a.block("m0",[("w","Y"),("m",1),("jmp","rew")])
    a.block("skip",[("m",1),("jmp","pick")])
    a.block("seek",[("sw",{"|":"past","1":"s1","0":"s1","X":"s1","Y":"s1"})])
    a.block("past",[("m",1),("jmp","after")])
    a.block("s1",[("m",1),("jmp","seek")])
    a.block("after",[("sw",{"1":"a1","_":"app"})])
    a.block("a1",[("m",1),("jmp","after")])
    a.block("app",[("w","1"),("jmp","rew")])
    a.block("rrew",[("m",-1),("sw",{"1":"rrew","0":"rrew","|":"rrew","X":"rrew","Y":"rrew","_":"rfwd"})])
    a.block("rfwd",[("m",1),("jmp","rsweep")])
    a.block("rsweep",[("sw",{"X":"r1","Y":"r0","1":"rskip","0":"rskip","|":"rskip","_":"rend"})])
    a.block("r1",[("w","1"),("m",1),("jmp","rsweep")])
    a.block("r0",[("w","0"),("m",1),("jmp","rsweep")])
    a.block("rskip",[("m",1),("jmp","rsweep")])
    a.block("rend",[("halt")])
    a.expectation="1011|111"; return a

def p_copy():
    a=TuringAssembler("copy_string","_1011_","01|XY_", max_steps=1600)
    a.block("q0",[("m",1),("sw",{"0":"q0","1":"q0","_":"sep"})])
    a.block("sep",[("w","|"),("m",1),("jmp","rew")])
    a.block("rew",[("m",-1),("sw",{"0":"rew","1":"rew","X":"rew","Y":"rew","|":"rew","_":"fwd"})])
    a.block("fwd",[("m",1),("jmp","pick")])
    a.block("pick",[("sw",{"1":"m1","0":"m0","X":"skip","Y":"skip","|":"rrew","_":"rrew"})])
    a.block("m1",[("w","X"),("m",1),("jmp","seek1")])
    a.block("m0",[("w","Y"),("m",1),("jmp","seek0")])
    a.block("skip",[("m",1),("jmp","pick")])
    a.block("seek1",[("sw",{"|":"past1","0":"s1a","1":"s1a","X":"s1a","Y":"s1a"})])
    a.block("past1",[("m",1),("jmp","after1")])
    a.block("seek0",[("sw",{"|":"past0","0":"s0a","1":"s0a","X":"s0a","Y":"s0a"})])
    a.block("past0",[("m",1),("jmp","after0")])
    a.block("s1a",[("m",1),("jmp","seek1")])
    a.block("s0a",[("m",1),("jmp","seek0")])
    a.block("after1",[("sw",{"1":"b1","0":"b1","_":"app1"})])
    a.block("after0",[("sw",{"1":"b0","0":"b0","_":"app0"})])
    a.block("b1",[("m",1),("jmp","after1")])
    a.block("b0",[("m",1),("jmp","after0")])
    a.block("app1",[("w","1"),("jmp","rew")])
    a.block("app0",[("w","0"),("jmp","rew")])
    a.block("rrew",[("m",-1),("sw",{"1":"rrew","0":"rrew","|":"rrew","X":"rrew","Y":"rrew","_":"rfwd"})])
    a.block("rfwd",[("m",1),("jmp","rsweep")])
    a.block("rsweep",[("sw",{"X":"r1","Y":"r0","1":"rskip","0":"rskip","|":"rskip","_":"rend"})])
    a.block("r1",[("w","1"),("m",1),("jmp","rsweep")])
    a.block("r0",[("w","0"),("m",1),("jmp","rsweep")])
    a.block("rskip",[("m",1),("jmp","rsweep")])
    a.block("rend",[("halt")])
    a.expectation="1011|1011"; return a

def _rule110(a,b,c):
    table={(1,1,1):0,(1,1,0):1,(1,0,1):1,(1,0,0):0,(0,1,1):1,(0,1,0):1,(0,0,1):1,(0,0,0):0}
    return table[(int(a),int(b),int(c))]

def p_ca():
    row="01101"; n=len(row)
    a=TuringAssembler("cellular_automaton_110_step","_%s_"%row,"01_", start="q0", max_steps=2000)
    nxt=_rule110
    for b in "01":
        spec={}
        for aa in "01":
            lab="go_%s%s"%(b,aa); spec[aa]=lab
            spec2={}
            for cc in "01":
                leaf="wr_%s%s%s"%(b,aa,cc); spec2[cc]=leaf
                a.block(leaf,[("m",-1),("w",str(nxt(int(b),int(aa),int(cc)))),("m",1),("jmp","read_a_%s"%aa)])
            spec2["_"]="end_%s%s"%(b,aa)
            a.block(lab,[("m",1),("sw",spec2)])
            a.block("end_%s%s"%(b,aa),[("m",-1),("w",str(nxt(int(b),int(aa),0))),("halt")])
        a.block("read_a_%s"%b,[("sw",spec)])
    a.block("q0",[("jmp","read_a_0")])
    a.expectation="".join(str(nxt(0 if i==0 else int(row[i-1]), int(row[i]), 0 if i==n-1 else int(row[i+1]))) for i in range(n))
    return a




class WaveTuringMachine:
    """A deterministic Turing machine executed entirely in 4-slot wave space.

    The machine holds no text. Its tape is a complex accumulator, its control
    state is a phasor, its head is a phase rotor, and its transition function
    is a holographic atlas that is unbound and snapped on every step. A shadow
    list of symbols is kept as a MONITOR only - it is written from the wave
    result, never read into the read or write path, and it is used for the
    cycle key and for the final comparison against the discrete oracle.
    """

    SLOTS = 4
    ROLE_NAMES = ("state", "symbol", "opcode")

    def __init__(
        self,
        codebook: Optional[SymbolCodebook] = None,
        dim: int = 8192,
        *,
        beta_star: float = 12.0,
        tau_read: float = 0.35,
        tau_transition: float = 0.10,
        tau_cycle: float = 0.995,
        max_steps: int = 1200,
        seed: int = 20261110,
    ) -> None:
        self.dim = int(dim)
        self.codebook = codebook or SymbolCodebook(self.dim)
        if int(getattr(self.codebook, "dim", 0)) != self.dim:
            self.codebook = SymbolCodebook(self.dim)
        self.seed = int(seed)
        self.beta_star = float(beta_star)
        self.tau_read = float(tau_read)
        self.tau_transition = float(tau_transition)
        self.tau_cycle = float(tau_cycle)
        self.max_steps = int(max_steps)
        self.theta = np.angle(self.codebook.position_wave())
        self.pc_theta = np.angle(self.codebook.named_phasor("turing:pc"))
        self.law = _unit(self.codebook.named_phasor("turing:law"))
        self.roles = {
            name: self.codebook.named_phasor(f"turing:role:{name}")
            for name in self.ROLE_NAMES
        }
        self._phasor_cache: Dict[str, np.ndarray] = {}
        self._matrix_cache: Dict[Tuple[str, Tuple[str, ...], bool], np.ndarray] = {}
        self._pc_cache: Dict[int, np.ndarray] = {}
        self._position_cache: Dict[int, np.ndarray] = {}
        self.stats: Dict[str, float] = {
            "runs": 0.0,
            "steps": 0.0,
            "vetoes": 0.0,
            "read_snr_min": float("inf"),
            "transition_snr_min": float("inf"),
            "cycle_coherence_max": 0.0,
        }

    # ------------------------------------------------------------- codebook
    # Every named phasor is memoised. Deriving one is a hashed RNG draw, which
    # costs more than the wave arithmetic it feeds; the machine reads the same
    # few hundred phasors on every step, so the cache is what makes the
    # sub-millisecond step budget reachable at D = 8192.
    def _named(self, name: str) -> np.ndarray:
        cached = self._phasor_cache.get(name)
        if cached is None:
            cached = self.codebook.named_phasor(name)
            self._phasor_cache[name] = cached
        return cached

    def state_wave(self, state: str) -> np.ndarray:
        return self._named(f"turing:state:{state}")

    def symbol_wave(self, symbol: str) -> np.ndarray:
        return self._named(f"turing:sym:{symbol}")

    def opcode_wave(self, delta: int) -> np.ndarray:
        return self._named(f"turing:op:{int(delta)}")

    def tag_wave(self, index: int) -> np.ndarray:
        return self._named(f"turing:tag:{int(index)}")

    def _matrix(self, kind: str, names: Sequence[object], role: bool = False) -> np.ndarray:
        """A memoised stack of named waves, or of role-bound named waves.

        The alphabet matrix, the state matrix and the opcode matrix are read on
        every step and change only when the program does, so they are built
        once. Binding a role is folded into the cached matrix as well.
        """
        key = (str(kind), tuple(str(name) for name in names), bool(role))
        cached = self._matrix_cache.get(key)
        if cached is not None:
            return cached
        builder = {
            "symbol": self.symbol_wave,
            "state": self.state_wave,
            "op": lambda name: self.opcode_wave(int(name)),
        }[str(kind)]
        waves = [builder(str(name)) for name in names]
        matrix = np.stack(waves) if waves else np.zeros((0, self.dim), dtype=np.complex128)
        matrix = matrix.astype(np.complex128)
        role_name = {"symbol": "symbol", "state": "state", "op": "opcode"}.get(str(kind))
        if role and role_name is not None and matrix.shape[0]:
            matrix = matrix * self.roles[role_name]
        self._matrix_cache[key] = matrix
        return matrix

    def _pos(self, coordinate) -> np.ndarray:
        key = int(coordinate)
        cached = self._position_cache.get(key)
        if cached is None:
            cached = np.exp(1j * self.theta * float(key))
            self._position_cache[key] = cached
        return cached

    def _pc(self, step: int) -> np.ndarray:
        key = int(step)
        cached = self._pc_cache.get(key)
        if cached is None:
            cached = np.exp(1j * self.pc_theta * float(key))
            if len(self._pc_cache) > 4096:
                self._pc_cache.clear()
            self._pc_cache[key] = cached
        return cached

    # ---------------------------------------------------------- tape algebra
    def tape_wave(self, tape: Sequence[str]) -> np.ndarray:
        """Bind every cell to its position phase and sum: slot 2, unnormalised."""
        accumulator = np.zeros(self.dim, dtype=np.complex128)
        for coordinate, symbol in enumerate(tape):
            accumulator = accumulator + self.symbol_wave(str(symbol)) * self._pos(coordinate)
        return accumulator

    def slots(self, state: str, head: int, tape: np.ndarray, opcode: int, step: int) -> np.ndarray:
        """The 4-slot configuration wave psi_t in S^(D-1)."""
        slot0 = _unit(self.state_wave(str(state)) * self._pc(step)).astype(np.complex128)
        slot1 = _unit(self._pos(int(head)) * self.opcode_wave(int(opcode))).astype(np.complex128)
        slot2 = _unit(np.asarray(tape, dtype=np.complex128)).astype(np.complex128)
        slot3 = np.asarray(self.law, dtype=np.complex128)
        return np.stack([slot0, slot1, slot2, slot3])

    def read_head(
        self, tape: np.ndarray, head: int, alphabet: Sequence[str], cells: int = 0
    ) -> Tuple[str, float, np.ndarray]:
        """Unbind slot 2 at the head phase and Hopfield-snap onto the alphabet."""
        symbols = [str(symbol) for symbol in alphabet]
        readout = np.asarray(tape, dtype=np.complex128) * np.conj(self._pos(int(head)))
        norm = float(np.linalg.norm(readout))
        matrix = self._matrix("symbol", [str(s) for s in symbols])
        alignment = np.real(matrix.conj() @ readout) / max(norm, _EPS)
        if float(np.std(alignment)) <= _EPS:
            return symbols[0], 0.0, alignment
        # The veto statistic is the MARGIN of the winning symbol over the
        # runner-up, not an absolute alignment: the signal of a correct read
        # falls as 1/sqrt(L) with tape length, so only the ratio is scale-free.
        # A cell that no longer snaps cleanly has a best alignment barely above
        # the second, and the margin collapses toward 1.
        # The signal of a correct read falls as 1/sqrt(L) with the number of
        # tape cells, so the score rescales the winning alignment by sqrt(L).
        # A clean cell scores ~1.0; a cell whose phase was overwritten scores
        # the pure crosstalk floor, ~0.05. A bare best/runner-up ratio is not
        # usable here: with no signal at all, the ratio of two noise values can
        # be arbitrarily large.
        cells = int(cells) if int(cells) > 0 else 1
        score = float(alignment.max()) * math.sqrt(float(cells))
        return symbols[int(np.argmax(alignment))], score, alignment

    # ------------------------------------------------------------- atlas
    def compile_atlas(self, program: TuringProgram) -> Dict[str, object]:
        """Store the rules as key * (chord * tag), one chart per control state.

        A single bundle over every rule of a program is the flat atlas the R3
        capacity law warns about: with M rules the cross-talk grows as
        sqrt(M) and a 100-rule program sits at the edge of the snap. The
        control state is already part of the key, so partitioning the atlas by
        state is exact - it removes only pairs that can never be addressed
        together - and it keeps every chart at 10-20 engrams, where the same
        wave arithmetic recovers the rule with a wide margin. This is the same
        chart idea Stage S3 applies to Zone C.
        """
        charts: Dict[str, Dict[str, object]] = {}
        counts: Dict[str, int] = {}
        grouped: Dict[str, List[Tuple[Tuple[str, str], Tuple[str, int, str]]]] = {}
        for key, value in program.rules.items():
            grouped.setdefault(str(key[0]), []).append((key, value))
        total = 0
        for state in sorted(grouped):
            rows = sorted(grouped[state], key=lambda item: item[0][1])
            count = len(rows)
            counts[state] = int(count)
            values = np.zeros((max(count, 1), self.dim), dtype=np.complex128)
            atlas = np.zeros(self.dim, dtype=np.complex128)
            for index, ((rule_state, read), (write, move, target)) in enumerate(rows):
                inner = (
                    self.state_wave(str(target)) * self.roles["state"]
                    + self.symbol_wave(str(write)) * self.roles["symbol"]
                    + self.opcode_wave(int(move)) * self.roles["opcode"]
                )
                value = inner * self.tag_wave(index)
                values[index] = value
                atlas = atlas + (
                    self.state_wave(str(rule_state)) * self.symbol_wave(str(read))
                ) * value
            charts[state] = {
                "atlas": atlas,
                "values": values,
                "rules": [(key[0], key[1]) + tuple(value) for key, value in rows],
                "count": int(count),
            }
            total += count
        return {"charts": charts, "count": int(total), "chart_count": int(len(charts))}

    def unbind_transition(
        self,
        compiled: Dict[str, object],
        state: str,
        symbol: str,
        states: Sequence[str],
        alphabet: Sequence[str],
    ) -> Dict[str, object]:
        """Unbind the key, snap the chord to a rule, then factorise the chord."""
        charts = dict(compiled.get("charts") or {})
        chart = charts.get(str(state))
        if not chart:
            # No chart for this control state: the state has no outgoing rule at
            # all, which is an undefined transition, not a silent default.
            return {"defined": False, "snr": 0.0, "target": None, "factorised": False}
        atlas = np.asarray(chart["atlas"], dtype=np.complex128)
        values = np.asarray(chart["values"], dtype=np.complex128)
        rules = list(chart["rules"])
        key = self.state_wave(str(state)) * self.symbol_wave(str(symbol))
        chord = atlas * np.conj(key)
        norm = float(np.linalg.norm(chord))
        projection = np.real(values.conj() @ chord)
        best = int(np.argmax(projection))
        # The veto statistic is the cosine between the unbound chord and the
        # winning stored chord. It is scale-free: binding shrinks a wave by
        # 1/sqrt(D) per factor, so no absolute amplitude means anything here.
        # Measured on the sixteen programs: a defined transition scores 0.18 to
        # 0.55, an undefined key 0.01 to 0.04.
        value_norm = float(np.linalg.norm(values[best]))
        coherence = float(projection[best] / max(value_norm * norm, 1e-300))
        snr = float(coherence)
        state_q, read_s, write_s, move, target_q = rules[best]
        # Factorise the recovered chord into its three role-bound fields. The
        # tag is removed first, exactly as it was applied.
        inner = chord * np.conj(self.tag_wave(best))
        inner = inner / max(float(np.linalg.norm(inner)), _EPS)
        # The write field is snapped over its own symbol set, which includes
        # the "=" no-write marker: a no-op write is a symbol like any other.
        # Both sets are local to this chart, so the factorisation stays a small
        # matmul and the step budget holds.
        writes = sorted({str(row[2]) for row in rules} | set(str(s) for s in alphabet))
        targets = sorted({str(row[4]) for row in rules})
        state_matrix = self._matrix("state", targets, role=True)
        symbol_matrix = self._matrix("symbol", writes, role=True)
        opcode_matrix = self._matrix("op", (-1, 0, 1), role=True)
        a_state = np.real(state_matrix.conj() @ inner)
        a_symbol = np.real(symbol_matrix.conj() @ inner)
        a_opcode = np.real(opcode_matrix.conj() @ inner)
        snapped = (
            str(targets[int(np.argmax(a_state))]),
            str(writes[int(np.argmax(a_symbol))]),
            int((-1, 0, 1)[int(np.argmax(a_opcode))]),
        )
        return {
            "defined": bool(coherence >= self.tau_transition),
            "snr": float(coherence),
            "coherence": float(coherence),
            "target": str(target_q),
            "write": str(write_s),
            "move": int(move),
            "snapped": snapped,
            "factorised": snapped == (str(target_q), str(write_s), int(move)),
            "rule": (str(state_q), str(read_s)),
        }

    # ---------------------------------------------------------------- run
    def run(
        self,
        program: TuringProgram,
        *,
        photonic: Optional[object] = None,
        collect_trace: bool = True,
        tamper: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        """Execute one program. Every read, write and move is a wave operation."""
        started = time.perf_counter()
        alphabet = sorted(set(program.symbols()) | set(program.alphabet))
        states = sorted(set(program.states()) | {program.start, program.halt})
        rule_table = dict(program.rules)
        compiled = self.compile_atlas(program)
        tape = [str(symbol) for symbol in program.tape]
        unknown = [symbol for symbol in tape if symbol not in alphabet]
        if unknown:
            self.stats["vetoes"] += 1.0
            return {
                "program": program.name,
                "halted": False,
                "vetoed": True,
                "veto_reason": "tape-symbol-not-in-alphabet",
                "unknown_symbols": sorted(set(unknown)),
                "text_tokens": 0,
                "steps": 0,
            }
        accumulator = self.tape_wave(tape)
        if tamper:
            # The corrupted-tape trap: an anisotropic phase is injected straight
            # into the holographic tape. A wave-only machine must notice that
            # the cell it is standing on no longer snaps to any alphabet symbol.
            position = int(tamper.get("head", program.head))
            amplitude = float(tamper.get("amplitude", 2.0))
            phase = float(tamper.get("phase", 0.7))
            generator = np.random.default_rng(int(tamper.get("seed", 7)))
            noise = generator.standard_normal(self.dim) + 1j * generator.standard_normal(self.dim)
            noise = _unit(noise.astype(np.complex64)).astype(np.complex128)
            # REPLACE the cell, not add to it: a corrupted cell is one whose
            # symbol phase was overwritten, so the original symbol is gone and
            # the head read must fail to snap instead of reading through noise.
            accumulator = accumulator + (
                amplitude * np.exp(1j * phase) * noise - self.symbol_wave(str(tape[position]))
            ) * self._pos(position)
            tape[position] = "?"
        state = str(program.start)
        head = int(program.head)
        steps = 0
        visited: Dict[Tuple[str, int, str], np.ndarray] = {}
        trace: List[Dict[str, object]] = []
        read_min = float("inf")
        trans_min = float("inf")
        coherence_max = 0.0
        veto_reason = ""
        opcode = 0
        while True:
            if state == program.halt:
                break
            if steps >= int(program.max_steps):
                veto_reason = "step-limit"
                break
            if head < 0 or head >= len(tape):
                veto_reason = "head-off-tape"
                break
            tape_slot = _unit(accumulator)
            psi = self.slots(state, head, tape_slot, opcode, steps)
            symbol, read_snr, _ = self.read_head(tape_slot, head, alphabet, cells=len(tape))
            read_min = min(read_min, read_snr)
            if read_snr < self.tau_read:
                veto_reason = "unreadable-tape"
                break
            step = self.unbind_transition(compiled, state, symbol, states, alphabet)
            trans_snr = float(step.get("snr", 0.0))
            trans_min = min(trans_min, trans_snr)
            if not bool(step.get("defined", False)):
                veto_reason = "undefined-transition"
                break
            if not bool(step.get("factorised", False)):
                veto_reason = "atlas-factorisation-mismatch"
                break
            next_state = str(step["target"])
            write_symbol = str(step["write"])
            move = int(step["move"])
            if (str(state), str(symbol)) not in rule_table:
                veto_reason = "undefined-transition"
                break
            # Unitary tape delta: erase the old symbol and write the new one at
            # exactly one cell. No inverse, no rebinding of the whole tape.
            if write_symbol != "=":
                accumulator = accumulator + (
                    self.symbol_wave(write_symbol) - self.symbol_wave(symbol)
                ) * self._pos(head)
                tape[head] = write_symbol
            head_next = head + move
            state_next = next_state
            opcode = move
            steps += 1
            # Sagnac cycle gate. The configuration core is slots 1..3 (head
            # phase, holographic tape, invariant law): a genuine state-tape
            # revisit reproduces it exactly, so its coherence is 1.000. The
            # program counter in slot 0 is excluded on purpose - it advances
            # every step by construction and would mask a true cycle.
            core = psi[1:]
            key = (state_next, head_next, "".join(tape))
            if key in visited:
                previous = visited[key]
                coherence = float(abs(np.vdot(_unit(core), _unit(previous))))
                coherence_max = max(coherence_max, coherence)
                if coherence > self.tau_cycle:
                    veto_reason = "non-halting-cycle"
                    break
                visited[key] = core
            else:
                visited[key] = core
            if collect_trace and len(trace) < 24:
                trace.append(
                    {
                        "step": int(steps),
                        "state": str(state),
                        "read": str(symbol),
                        "write": write_symbol,
                        "move": int(move),
                        "next": state_next,
                        "head": int(head),
                        "read_snr": float(read_snr),
                        "transition_snr": float(trans_snr),
                    }
                )
            state = state_next
            head = head_next
        elapsed = time.perf_counter() - started
        tape_text = "".join(tape).replace("~", "").strip("_")
        final_psi = self.slots(
            program.halt if state == program.halt else state,
            min(max(head, 0), max(len(tape) - 1, 0)),
            _unit(accumulator),
            opcode,
            steps,
        )
        self.stats["runs"] += 1.0
        self.stats["steps"] += float(steps)
        if veto_reason:
            self.stats["vetoes"] += 1.0
        self.stats["read_snr_min"] = min(self.stats["read_snr_min"], read_min)
        self.stats["transition_snr_min"] = min(self.stats["transition_snr_min"], trans_min)
        self.stats["cycle_coherence_max"] = max(self.stats["cycle_coherence_max"], coherence_max)
        return {
            "program": program.name,
            "halted": bool(state == program.halt and not veto_reason),
            "vetoed": bool(bool(veto_reason)),
            "veto_reason": str(veto_reason),
            "final_state": str(state),
            "final_tape": str(tape_text),
            "expected": str(program.expectation),
            "exact": bool(not veto_reason and tape_text == program.expectation),
            "steps": int(steps),
            "head": int(head),
            "read_snr_min": float(read_min if np.isfinite(read_min) else 0.0),
            "transition_snr_min": float(trans_min if np.isfinite(trans_min) else 0.0),
            "cycle_coherence_max": float(coherence_max),
            "step_ms_mean": float(elapsed / max(steps, 1) * 1e3),
            "wall_ms": float(elapsed * 1e3),
            "text_tokens": 0,
            "atlas_rules": int(compiled["count"]),
            "slots": int(self.SLOTS),
            "dim": int(self.dim),
            "photonic": bool(photonic is not None),
            "psi_norm": float(np.linalg.norm(final_psi)),
            "trace": trace,
        }


def turing_programs() -> List[TuringProgram]:
    """The sixteen deterministic programs the S2 gate executes in wave space."""
    builders = (
        p_increment,
        p_decrement,
        p_shift_left,
        p_shift_right,
        p_toggle,
        p_parity,
        p_unary_add,
        p_unary_double,
        p_binary_add,
        p_sort,
        p_reverse,
        p_palindrome,
        p_brackets,
        p_count_ones,
        p_copy,
        p_ca,
    )
    out: List[TuringProgram] = []
    for build in builders:
        asm = build()
        out.append(
            TuringProgram(
                name=asm.name,
                tape=list(asm.tape),
                alphabet=list(asm.alphabet),
                rules=dict(asm.compile()),
                start=asm.start + "#0",
                halt=asm.halt,
                expectation=str(asm.expectation),
                description=str(getattr(asm, "description", "")),
                head=1,
                max_steps=int(asm.max_steps),
            )
        )
    return out


def turing_program(name: str) -> TuringProgram:
    wanted = str(name).strip().lower()
    for program in turing_programs():
        if program.name == wanted:
            return program
    raise KeyError(f"unknown turing program: {name}")


def turing_traps() -> List[Dict[str, object]]:
    """The four adversarial traps the S2 gate requires the machine to veto.

    Two are genuine non-halting configurations (a self-loop and a two-state
    cycle) that reproduce their state-tape configuration exactly, so the
    Sagnac gate sees coherence 1.000. One addresses a control state that has no
    rule for the symbol under the head, so the transition unbinding cannot
    reach a defined coherence. One keeps a valid program and corrupts the
    holographic tape, so the head read no longer snaps to any alphabet symbol.
    """
    pad = ["_"] * 6
    return [
        {
            "name": "trap_self_loop",
            "kind": "infinite-loop",
            "tamper": None,
            "program": TuringProgram(
                name="trap_self_loop",
                tape=list("_111_") + pad,
                alphabet=["1", "_"],
                rules={("q0#0", "1"): ("1", 0, "q0#0")},
                start="q0#0",
                halt="qh",
                expectation="",
                max_steps=64,
            ),
        },
        {
            "name": "trap_two_state_cycle",
            "kind": "infinite-loop",
            "tamper": None,
            "program": TuringProgram(
                name="trap_two_state_cycle",
                tape=list("_11_") + pad,
                alphabet=["1", "_"],
                rules={
                    ("q0#0", "1"): ("1", 1, "q1#0"),
                    ("q1#0", "1"): ("1", -1, "q0#0"),
                },
                start="q0#0",
                halt="qh",
                expectation="",
                max_steps=64,
            ),
        },
        {
            "name": "trap_undefined_transition",
            "kind": "undefined-transition",
            "tamper": None,
            "program": TuringProgram(
                name="trap_undefined_transition",
                tape=list("_1_") + pad,
                alphabet=["0", "1", "_"],
                rules={("q0#0", "0"): ("0", 1, "q0#0")},
                start="q0#0",
                halt="qh",
                expectation="",
                max_steps=64,
            ),
        },
        {
            "name": "trap_corrupted_tape",
            "kind": "corrupted-tape",
            "tamper": {"head": 1, "amplitude": 2.0, "phase": 0.7, "seed": 7},
            "program": turing_program("binary_increment"),
        },
    ]


def reference_turing_run(program: TuringProgram) -> Dict[str, object]:
    """The DISCRETE interpreter. It is the oracle, never the wave path.

    The S2 gate compares the wave machine against this run, and both against
    the independent mathematical expectation recorded on the program.
    """
    tape = list(program.tape)
    rules = dict(program.rules)
    state = str(program.start)
    head = int(program.head)
    steps = 0
    while state != program.halt:
        if steps >= int(program.max_steps):
            return {"halted": False, "reason": "step-limit", "tape": tape, "steps": steps}
        if head < 0 or head >= len(tape):
            return {"halted": False, "reason": "head-off-tape", "tape": tape, "steps": steps}
        rule = rules.get((state, tape[head]))
        if rule is None:
            return {
                "halted": False,
                "reason": f"undefined:{state},{tape[head]}",
                "tape": tape,
                "steps": steps,
            }
        write, move, target = rule
        if str(write) != "=":
            tape[head] = str(write)
        head += int(move)
        steps += 1
        state = str(target)
    text = "".join(tape).replace("~", "").strip("_")
    return {"halted": True, "reason": "", "tape": tape, "tape_text": text, "steps": steps}