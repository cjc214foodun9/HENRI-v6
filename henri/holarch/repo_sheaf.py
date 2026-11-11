
"""Demand-projected cellular sheaf over unannotated Python repositories (T4b).

A repository is a cellular sheaf (G, F): modules are vertices, imports and
calls are edges, and each edge carries a usage projector P_e = diag(r*)
built from the capability bits that the callee body actually dereferences.
The coboundary energy

    E_sheaf(s) = sum_e || P_e (r*_v - p_u) ||^2 + symbol + arity terms

is zero exactly when every caller provides every capability that the callee
demands.  An unannotated parameter is never a wildcard: its demand is read
from the operations inside the body and pulled back across the call graph.
"""

from __future__ import annotations

import ast
import hashlib
import itertools
import json
import math
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

CAP_NUM_ARITH = 0
CAP_INT_INDEX = 1
CAP_TEXT_OPS = 2
CAP_SIZED = 3
CAP_ITERABLE = 4
CAP_ELEM_NUM = 5
CAP_ELEM_TEXT = 6
CAP_ELEM_ATTR = 7
CAP_MAP_READ = 8
CAP_MAP_MUT = 9
CAP_SEQ_MUT = 10
CAP_SET_OPS = 11
CAP_CALLABLE = 12
CAP_HASHABLE = 13
CAP_NON_NONE = 14
CAP_CONFLICT = 15

CAP_NAMES = (
    "CAP_NUM_ARITH", "CAP_INT_INDEX", "CAP_TEXT_OPS", "CAP_SIZED",
    "CAP_ITERABLE", "CAP_ELEM_NUM", "CAP_ELEM_TEXT", "CAP_ELEM_ATTR",
    "CAP_MAP_READ", "CAP_MAP_MUT", "CAP_SEQ_MUT", "CAP_SET_OPS",
    "CAP_CALLABLE", "CAP_HASHABLE", "CAP_NON_NONE", "CAP_CONFLICT",
)
DIM = 16
ZERO = np.zeros(DIM, dtype=np.int64)


def vec(*caps: int) -> np.ndarray:
    out = ZERO.copy()
    for cap in caps:
        out[int(cap)] = 1
    return out


def names(vector: np.ndarray) -> List[str]:
    return [CAP_NAMES[index] for index in np.flatnonzero(np.asarray(vector))]


# ------------------------------------------------- static provision schema
TYPE_PROVISION: Dict[str, np.ndarray] = {
    "str": vec(CAP_TEXT_OPS, CAP_SIZED, CAP_ITERABLE, CAP_ELEM_TEXT, CAP_HASHABLE, CAP_NON_NONE),
    "bytes": vec(CAP_SIZED, CAP_ITERABLE, CAP_HASHABLE, CAP_NON_NONE),
    "int": vec(CAP_NUM_ARITH, CAP_INT_INDEX, CAP_HASHABLE, CAP_NON_NONE),
    "float": vec(CAP_NUM_ARITH, CAP_HASHABLE, CAP_NON_NONE),
    "bool": vec(CAP_NUM_ARITH, CAP_INT_INDEX, CAP_HASHABLE, CAP_NON_NONE),
    "list": vec(CAP_SIZED, CAP_ITERABLE, CAP_SEQ_MUT, CAP_NON_NONE),
    "tuple": vec(CAP_SIZED, CAP_ITERABLE, CAP_HASHABLE, CAP_NON_NONE),
    "dict": vec(CAP_SIZED, CAP_ITERABLE, CAP_MAP_READ, CAP_MAP_MUT, CAP_NON_NONE),
    "set": vec(CAP_SIZED, CAP_ITERABLE, CAP_SET_OPS, CAP_NON_NONE),
    "frozenset": vec(CAP_SIZED, CAP_ITERABLE, CAP_HASHABLE, CAP_NON_NONE),
    "None": ZERO.copy(),
}
ANNOTATION_NAMES = {
    "str": "str", "bytes": "bytes", "int": "int", "float": "float",
    "bool": "bool", "list": "list", "List": "list", "tuple": "tuple",
    "Tuple": "tuple", "dict": "dict", "Dict": "dict", "set": "set",
    "Set": "set", "frozenset": "frozenset", "None": "None",
}
ELEMENT_PROVISION = {
    "int": vec(CAP_ELEM_NUM), "float": vec(CAP_ELEM_NUM),
    "str": vec(CAP_ELEM_TEXT), "Attrs": vec(CAP_ELEM_ATTR),
}

METHOD_CAPS: Dict[str, int] = {
    "strip": CAP_TEXT_OPS, "split": CAP_TEXT_OPS, "lower": CAP_TEXT_OPS,
    "upper": CAP_TEXT_OPS, "replace": CAP_TEXT_OPS, "encode": CAP_TEXT_OPS,
    "startswith": CAP_TEXT_OPS, "endswith": CAP_TEXT_OPS, "format": CAP_TEXT_OPS,
    "get": CAP_MAP_READ, "keys": CAP_MAP_READ, "values": CAP_MAP_READ,
    "items": CAP_MAP_READ, "setdefault": CAP_MAP_MUT, "update": CAP_MAP_MUT,
    "pop": CAP_MAP_MUT, "append": CAP_SEQ_MUT, "extend": CAP_SEQ_MUT,
    "insert": CAP_SEQ_MUT, "sort": CAP_SEQ_MUT, "add": CAP_SET_OPS,
    "discard": CAP_SET_OPS, "union": CAP_SET_OPS, "intersection": CAP_SET_OPS,
    "join": CAP_ELEM_TEXT,
}
NUMERIC_CALLS = {"abs": CAP_NUM_ARITH, "float": CAP_NUM_ARITH,
                 "int": CAP_INT_INDEX, "round": CAP_NUM_ARITH}
ITERABLE_CALLS = {"sorted", "list", "set", "tuple", "min", "max", "iter",
                  "enumerate", "zip", "sum", "any", "all", "reversed"}


@dataclass
class FunctionProfile:
    module: str
    name: str
    params: List[str]
    annotations: Dict[str, np.ndarray]
    demand: Dict[str, np.ndarray]
    returns: Optional[str]
    calls: List[Tuple[str, List[ast.AST]]] = field(default_factory=list)
    guarded: List[str] = field(default_factory=list)
    return_type: Optional[str] = None
    line: int = 0


@dataclass
class ModuleInfo:
    name: str
    functions: Dict[str, FunctionProfile] = field(default_factory=dict)
    imports: Dict[str, str] = field(default_factory=dict)
    source: str = ""


@dataclass
class Violation:
    caller: str
    callee: str
    function: str
    missing: List[str]
    energy: float
    kind: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "caller": self.caller,
            "callee": self.callee,
            "function": self.function,
            "missing": list(self.missing),
            "energy": float(self.energy),
            "kind": self.kind,
        }


@dataclass
class SheafReport:
    energy: float
    violations: List[Violation]
    kappa: float
    modules: int
    edges: int
    active_capabilities: List[str]
    digest: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "energy": float(self.energy),
            "violations": [row.as_dict() for row in self.violations],
            "kappa_plus": float(self.kappa),
            "modules": int(self.modules),
            "edges": int(self.edges),
            "active_capabilities": list(self.active_capabilities),
            "digest": self.digest,
        }


# ------------------------------------------------------------------ demand
class _DemandVisitor(ast.NodeVisitor):
    """Read capability demands from the operations inside one function body."""

    def __init__(self, aliases: Dict[str, str], guarded: Set[str], leaf: Optional[str] = None) -> None:
        self.aliases = dict(aliases)
        self.leaf = leaf
        self.guarded = set(guarded)
        self.demand: Dict[str, np.ndarray] = {}
        self.conflicts: Dict[str, Set[int]] = {}
        self.element: Dict[str, str] = {}
        self.calls: List[Tuple[str, List[str]]] = []
        self.return_value: Optional[str] = None

    def _root(self, name: str) -> Optional[str]:
        seen = set()
        while name in self.aliases and name not in seen:
            seen.add(name)
            name = self.aliases[name]
        return name if name in self._params else None

    def bind(self, params: Sequence[str]) -> None:
        self._params = set(params)

    def _add(self, name: Optional[str], cap: int) -> None:
        root = self._root(name) if name else None
        if root is None:
            return
        self.demand.setdefault(root, ZERO.copy())[int(cap)] = 1

    def _finalize(self) -> None:
        """An intra-body contradiction: text operations and sequence mutation."""
        for root, values in self.demand.items():
            if values[CAP_TEXT_OPS] and values[CAP_SEQ_MUT]:
                values[CAP_CONFLICT] = 1
            if values[CAP_TEXT_OPS] and values[CAP_SET_OPS]:
                values[CAP_CONFLICT] = 1

    # ------------------------------------------------------------- visitors
    def visit_For(self, node: ast.For) -> None:
        if isinstance(node.target, ast.Name) and isinstance(node.iter, ast.Name):
            iterable = node.iter.id
            self._add(iterable, CAP_ITERABLE)
            self._add(iterable, CAP_NON_NONE)
            inner = _DemandVisitor(self.aliases, self.guarded, leaf=node.target.id)
            inner.bind(list(self._params) + [node.target.id])
            inner.aliases = dict(self.aliases)
            inner.aliases[node.target.id] = iterable
            for statement in node.body:
                inner.visit(statement)
            for root, values in inner.demand.items():
                if root == iterable:
                    continue
                for cap in np.flatnonzero(values):
                    if int(cap) in (CAP_ELEM_NUM, CAP_ELEM_TEXT, CAP_ELEM_ATTR):
                        self._add(iterable, int(cap))
                self.demand[root] = np.maximum(self.demand.get(root, ZERO), values)
            self.calls.extend(inner.calls)
        else:
            self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target = node.targets[0].id
            value = node.value
            if isinstance(value, ast.Name) and self._root(value.id):
                self.aliases[target] = value.id
            elif isinstance(value, ast.Subscript) and isinstance(value.value, ast.Name) and self._root(value.value.id):
                self.aliases[target] = value.value.id
                if isinstance(value.slice, ast.Constant) and isinstance(value.slice.value, str):
                    self._add(value.value.id, CAP_MAP_READ)
                else:
                    self._add(value.value.id, CAP_INT_INDEX)
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        ordered = any(isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq))
                      for op in node.ops)
        numeric = any(isinstance(n, ast.Constant) and isinstance(getattr(n, "value", None), (int, float))
                      and not isinstance(getattr(n, "value", None), bool)
                      for n in [node.left] + list(node.comparators))
        if ordered and numeric:
            for value in [node.left] + list(node.comparators):
                if isinstance(value, ast.Name):
                    self._add(value.id, CAP_NUM_ARITH)
                    self._add(value.id, CAP_NON_NONE)
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        operands = [node.left, node.right]
        stringy = any(isinstance(op, ast.Constant) and isinstance(getattr(op, "value", None), str) for op in operands)
        if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
            cap = CAP_TEXT_OPS if stringy else CAP_NUM_ARITH
        elif isinstance(node.op, (ast.Mod, ast.FloorDiv, ast.LShift, ast.RShift,
                                  ast.BitAnd, ast.BitOr, ast.BitXor)):
            cap = CAP_INT_INDEX
        else:
            cap = CAP_NUM_ARITH
        for operand in operands:
            if isinstance(operand, ast.Name):
                self._add(operand.id, cap)
                self._add(operand.id, CAP_NON_NONE)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.value, ast.Name):
            key = node.slice
            if isinstance(key, ast.Constant) and isinstance(getattr(key, "value", None), str):
                cap = CAP_MAP_MUT if isinstance(node.ctx, ast.Store) else CAP_MAP_READ
            else:
                cap = CAP_INT_INDEX
            self._add(node.value.id, cap)
            self._add(node.value.id, CAP_NON_NONE)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        positional = [arg for arg in node.args if isinstance(arg, ast.Name)]
        all_args = list(node.args)
        if isinstance(func, ast.Name):
            if func.id == "len" and positional:
                self._add(positional[0].id, CAP_SIZED)
                self._add(positional[0].id, CAP_NON_NONE)
            elif func.id == "sum" and positional:
                self._add(positional[0].id, CAP_ITERABLE)
                self._add(positional[0].id, CAP_ELEM_NUM)
            elif func.id in NUMERIC_CALLS and positional:
                self._add(positional[0].id, NUMERIC_CALLS[func.id])
            elif func.id in ITERABLE_CALLS and positional:
                self._add(positional[0].id, CAP_ITERABLE)
                self._add(positional[0].id, CAP_NON_NONE)
            elif func.id == "range":
                for arg in positional:
                    self._add(arg.id, CAP_INT_INDEX)
            elif self._root(func.id):
                self._add(func.id, CAP_CALLABLE)
            self.calls.append((func.id, all_args))
        elif isinstance(func, ast.Attribute):
            method = str(func.attr)
            receiver = func.value.id if isinstance(func.value, ast.Name) else None
            if method in METHOD_CAPS and receiver:
                self._add(receiver, METHOD_CAPS[method])
                self._add(receiver, CAP_NON_NONE)
            elif receiver:
                self._add(receiver, CAP_CALLABLE)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if (isinstance(node.value, ast.Name) and self.leaf is not None
                and node.value.id == self.leaf):
            # An attribute on a loop variable is an element-attribute demand
            # on the iterable that produced it.
            self._add(node.value.id, CAP_ELEM_ATTR)
        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict) -> None:
        for key in node.keys:
            if isinstance(key, ast.Name):
                self._add(key.id, CAP_HASHABLE)
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        value = node.value
        if isinstance(value, ast.Name) and self._root(value.id):
            self.return_value = self._root(value.id)
        elif isinstance(value, (ast.List, ast.Tuple)) and value.elts:
            first = value.elts[0]
            if isinstance(first, ast.Name) and self._root(first.id):
                self.return_value = "elem:" + str(self._root(first.id))
        elif isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name) and self._root(value.value.id):
            self.return_value = "attr:" + str(self._root(value.value.id))
        self.generic_visit(node)


def _annotation_vector(node: Optional[ast.AST]) -> Optional[np.ndarray]:
    if node is None:
        return None
    if isinstance(node, ast.Name):
        key = ANNOTATION_NAMES.get(node.id)
    elif isinstance(node, ast.Constant) and node.value is None:
        key = "None"
    elif isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
        key = ANNOTATION_NAMES.get(node.value.id)
    else:
        key = None
    return None if key is None else TYPE_PROVISION.get(key)


def _guards(tree: ast.AST, params: Sequence[str]) -> Set[str]:
    guarded: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            test = node.test
            if isinstance(test, ast.Compare) and isinstance(test.left, ast.Name):
                if any(isinstance(value, ast.Constant) and value.value is None for value in test.comparators):
                    guarded.add(test.left.id)
    return guarded


def profile_module(name: str, source: str) -> ModuleInfo:
    info = ModuleInfo(name=str(name), source=str(source))
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            module = str(node.module or "").split(".")[-1]
            for alias in node.names:
                info.imports[alias.asname or alias.name] = module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                info.imports[alias.asname or alias.name.split(".")[-1]] = alias.name.split(".")[-1]
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        params = [arg.arg for arg in node.args.args if arg.arg not in {"self", "cls"}]
        annotations: Dict[str, np.ndarray] = {}
        for arg in node.args.args:
            vector = _annotation_vector(arg.annotation)
            if vector is not None and arg.arg in params:
                annotations[arg.arg] = vector
        guarded = _guards(node, params)
        visitor = _DemandVisitor({}, guarded)
        visitor.bind(params)
        for statement in node.body:
            visitor.visit(statement)
        visitor._finalize()
        demand = {param: ZERO.copy() for param in params}
        for root, values in visitor.demand.items():
            if root in demand:
                demand[root] = np.maximum(demand[root], values)
        # A leading None guard discharges the NON_NONE demand.
        for param in guarded:
            if param in demand:
                demand[param][CAP_NON_NONE] = 0
        return_type = None
        if node.returns is not None:
            vector = _annotation_vector(node.returns)
            if vector is not None:
                for key, value in TYPE_PROVISION.items():
                    if np.array_equal(value, vector):
                        return_type = key
                        break
        profile = FunctionProfile(
            module=str(name), name=str(node.name), params=params,
            annotations=annotations, demand=demand,
            returns=visitor.return_value, calls=list(visitor.calls), line=int(node.lineno),
            guarded=sorted(guarded), return_type=return_type,
        )
        info.functions[str(node.name)] = profile
    return info


def _default_provision(expr: ast.AST, env: Dict[str, np.ndarray],
                       functions: Dict[str, FunctionProfile]) -> np.ndarray:
    """Confirmed minimum provision p_min of a call-site argument."""
    if isinstance(expr, ast.Constant):
        value = expr.value
        if isinstance(value, bool):
            return TYPE_PROVISION["bool"].copy()
        if isinstance(value, int):
            return TYPE_PROVISION["int"].copy()
        if isinstance(value, float):
            return TYPE_PROVISION["float"].copy()
        if isinstance(value, str):
            return TYPE_PROVISION["str"].copy()
        if value is None:
            return TYPE_PROVISION["None"].copy()
        return ZERO.copy()
    if isinstance(expr, ast.Name):
        return env.get(expr.id, ZERO.copy()).copy()
    if isinstance(expr, ast.List):
        base = TYPE_PROVISION["list"].copy()
        elements = [_default_provision(item, env, functions) for item in expr.elts]
        if elements:
            stacked = np.stack(elements)
            for cap, source in ((CAP_ELEM_NUM, CAP_NUM_ARITH),
                                (CAP_ELEM_TEXT, CAP_TEXT_OPS),
                                (CAP_ELEM_ATTR, CAP_NON_NONE)):
                if all(row[source] for row in stacked):
                    base[cap] = 1
        else:
            base[CAP_ELEM_NUM] = 0
            base[CAP_ELEM_TEXT] = 0
            base[CAP_ELEM_ATTR] = 0
        return base
    if isinstance(expr, ast.Dict):
        return TYPE_PROVISION["dict"].copy()
    if isinstance(expr, ast.Tuple):
        return TYPE_PROVISION["tuple"].copy()
    if isinstance(expr, ast.Set):
        return TYPE_PROVISION["set"].copy()
    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name):
        callee = functions.get(expr.func.id)
        if callee is not None and callee.return_type:
            return TYPE_PROVISION.get(callee.return_type, ZERO).copy()
        name = expr.func.id
        if name in {"str"}: return TYPE_PROVISION["str"].copy()
        if name in {"int"}: return TYPE_PROVISION["int"].copy()
        if name in {"float", "round", "abs", "sum"}: return TYPE_PROVISION["float"].copy()
        if name in {"len", "count"}: return TYPE_PROVISION["int"].copy()
        if name in {"list"}: return TYPE_PROVISION["list"].copy()
        if name in {"dict"}: return TYPE_PROVISION["dict"].copy()
        if name in {"set"}: return TYPE_PROVISION["set"].copy()
        if name in {"tuple"}: return TYPE_PROVISION["tuple"].copy()
    return ZERO.copy()


def _return_provision(profile: FunctionProfile) -> np.ndarray:
    """Confirmed return provision: the annotation, else the confirmed minimum."""
    if profile.return_type:
        return TYPE_PROVISION.get(profile.return_type, ZERO).copy()
    return ZERO.copy()


class RepoSheaf:
    """Analyze a multi-module repository as a cellular sheaf."""

    def __init__(self, tau_sheaf: float = 1.0) -> None:
        self.tau_sheaf = float(tau_sheaf)

    def analyze(self, files: Dict[str, str]) -> SheafReport:
        modules: Dict[str, ModuleInfo] = {}
        for path, source in files.items():
            name = Path(str(path)).stem
            modules[name] = profile_module(name, str(source))
        # Backward transitive demand pullback across the call graph.
        for _ in range(max(1, len(modules))):
            changed = False
            for module in modules.values():
                for profile in module.functions.values():
                    for callee_name, arguments in profile.calls:
                        target = self._resolve(module, modules, callee_name)
                        if target is None:
                            continue
                        callee = target.functions.get(callee_name)
                        if callee is None:
                            continue
                        for index, argument in enumerate(arguments):
                            if index >= len(callee.params):
                                break
                            transferred = callee.demand.get(callee.params[index], ZERO)
                            if not isinstance(argument, ast.Name):
                                continue
                            root = argument.id if argument.id in profile.demand else None
                            if root is None:
                                continue
                            merged = np.maximum(profile.demand[root], transferred)
                            if not np.array_equal(merged, profile.demand[root]):
                                profile.demand[root] = merged
                                changed = True
            if not changed:
                break
        # Edge energy and the capability Laplacian.
        violations: List[Violation] = []
        edges: List[Tuple[str, str, np.ndarray]] = []
        total = 0.0
        for name, module in modules.items():
            for profile in module.functions.values():
                env: Dict[str, np.ndarray] = dict(profile.annotations)
                pool = {key: value for other in modules.values() for key, value in other.functions.items()}
                for callee_name, arguments in profile.calls:
                    target = self._resolve(module, modules, callee_name)
                    if target is None:
                        continue
                    callee = target.functions.get(callee_name)
                    if callee is None:
                        total += 1.0
                        violations.append(Violation(name, target.name, callee_name,
                                                    ["CAP_MISSING_SYMBOL"], 1.0, "symbol"))
                        continue
                    if len(arguments) != len(callee.params):
                        total += 1.0
                        violations.append(Violation(name, target.name, callee_name,
                                                    ["ARITY"], 1.0, "arity"))
                        continue
                    for index, argument in enumerate(arguments):
                        demand = callee.demand.get(callee.params[index], ZERO)
                        # A literal None on a guarded parameter is accepted: the
                        # callee body handles the absent value before any use.
                        if (isinstance(argument, ast.Constant) and argument.value is None
                                and callee.params[index] in callee.guarded):
                            continue
                        if isinstance(argument, ast.Name) and argument.id in env:
                            provision = env[argument.id].copy()
                        else:
                            provision = _default_provision(argument, env, pool)
                        missing = demand & (1 - np.clip(provision, 0, 1))
                        if missing.any():
                            energy = float(missing.sum())
                            total += energy
                            violations.append(Violation(
                                name, target.name, callee_name,
                                names(missing), energy, "capability"))
                            edges.append((name, target.name, np.diag(missing).astype(float)))
                        elif demand.any():
                            edges.append((name, target.name, np.diag(demand).astype(float)))
        kappa = self._condition_number(modules, edges)
        active = sorted({CAP_NAMES[i] for _u, _v, mask in edges for i in np.flatnonzero(np.diag(mask))})
        digest = hashlib.sha256(json.dumps(
            {name: sorted(module.functions) for name, module in modules.items()},
            sort_keys=True).encode("utf-8")).hexdigest()[:16]
        return SheafReport(float(total), violations, float(kappa), len(modules),
                           len(edges), active, digest)

    @staticmethod
    def _resolve(module: ModuleInfo, modules: Dict[str, ModuleInfo], name: str) -> Optional[ModuleInfo]:
        if name in module.functions:
            return module
        target = module.imports.get(name)
        if target and target in modules:
            return modules[target]
        for other in modules.values():
            if name in other.functions:
                return other
        return None

    @staticmethod
    def _condition_number(modules: Dict[str, ModuleInfo], edges: List[Tuple[str, str, np.ndarray]]) -> float:
        names = sorted(modules)
        index = {name: i for i, name in enumerate(names)}
        size = len(names) * DIM
        if size == 0:
            return 1.0
        laplacian = np.zeros((size, size), dtype=np.float64)
        for caller, callee, mask in edges:
            if caller not in index or callee not in index:
                continue
            weights = np.diag(mask)
            for cap in np.flatnonzero(weights):
                a = index[caller] * DIM + int(cap)
                b = index[callee] * DIM + int(cap)
                w = float(weights[cap])
                laplacian[a, a] += w
                laplacian[b, b] += w
                laplacian[a, b] -= w
                laplacian[b, a] -= w
        # Restrict to the active support so isolated modules cannot create a
        # spurious zero eigenvalue.
        active = np.flatnonzero(np.abs(np.diag(laplacian)) > 1e-9)
        if active.size == 0:
            return 1.0
        block = laplacian[np.ix_(active, active)]
        eigenvalues = np.linalg.eigvalsh(block)
        positive = eigenvalues[eigenvalues > 1e-6]
        if positive.size == 0:
            return 1.0
        return float(positive.max() / positive.min())

# --------------------------------------------------------------- fixtures
@dataclass
class RepoFixture:
    name: str
    files: Dict[str, str]
    kind: str
    family: str


def _mismatch_family(family: str, index: int) -> RepoFixture:
    tag = f"{family}_{index}"
    if family == "str_to_sum":
        arg = ('"abc"', '"1 2 3"', '"xyz"')[index]
        files = {
            "storage.py": "def total(values):\n    return sum(values)\n",
            "api.py": f"from storage import total\n\ndef run():\n    return total({arg})\n",
        }
    elif family == "int_to_len":
        arg = ("12", "0", "7")[index]
        files = {
            "storage.py": "def size(value):\n    return len(value)\n",
            "api.py": f"from storage import size\n\ndef run():\n    return size({arg})\n",
        }
    elif family == "list_to_map":
        arg = ("[1, 2]", "[]", '["k"]')[index]
        files = {
            "storage.py": 'def lookup(cfg):\n    return cfg.get("key")\n',
            "api.py": f"from storage import lookup\n\ndef run():\n    return lookup({arg})\n",
        }
    elif family == "tuple_to_append":
        arg = ("(1, 2)", "(3,)", "(0, 1, 2)")[index]
        files = {
            "storage.py": "def push(buf, value):\n    buf.append(value)\n    return buf\n",
            "api.py": f"from storage import push\n\ndef run():\n    return push({arg}, 3)\n",
        }
    elif family == "list_as_key":
        arg = ("[1, 2]", "[]", '["k"]')[index]
        files = {
            "storage.py": "def keyed(value):\n    return {value: 1}\n",
            "api.py": f"from storage import keyed\n\ndef run():\n    return keyed({arg})\n",
        }
    elif family == "none_to_strip":
        arg = ("None", "None", "None")[index]
        files = {
            "storage.py": "def clean(text):\n    return text.strip()\n",
            "api.py": f"from storage import clean\n\ndef run():\n    return clean({arg})\n",
        }
    elif family == "transitive_three":
        files = {
            "storage.py": "def save(row):\n    row.append(1)\n    return row\n",
            "service.py": "from storage import save\n\ndef relay(row):\n    return save(row)\n",
            "api.py": "from service import relay\n\ndef run():\n    return relay((1, 2))\n",
        }
    else:  # dynamic_unknown
        files = {
            "storage.py": "def store(value):\n    value.append(1)\n    return value\n",
            "loader.py": "def fetch():\n    return []\n",
            "api.py": "from storage import store\nfrom loader import fetch\n\ndef run():\n    return store(fetch())\n",
        }
    return RepoFixture(tag, files, "mismatch", family)


def build_mismatch_battery() -> List[RepoFixture]:
    families = ("str_to_sum", "int_to_len", "list_to_map", "tuple_to_append",
                "list_as_key", "none_to_strip", "transitive_three", "dynamic_unknown")
    return [_mismatch_family(family, index) for family in families for index in range(3)]


def build_polymorphic_battery() -> List[RepoFixture]:
    rows: List[RepoFixture] = []
    sized = ('"abcd"', "[1, 2, 3]", "(1, 2)")
    for index, arg in enumerate(sized):
        rows.append(RepoFixture(
            f"poly_sized_{index}",
            {"util.py": "def first_two(seq):\n    return len(seq) >= 2\n",
             "api.py": f"from util import first_two\n\ndef run():\n    return first_two({arg})\n"},
            "polymorphic", "sized"))
    for index, arg in enumerate(("[1, 2]", "[1.5, 2.5]")):
        rows.append(RepoFixture(
            f"poly_numeric_{index}",
            {"util.py": "def mean(values):\n    return sum(values) / len(values)\n",
             "api.py": f"from util import mean\n\ndef run():\n    return mean({arg})\n"},
            "polymorphic", "numeric"))
    for index, arg in enumerate(('"abc"', "[1, 2]", '{"k": 1}')):
        rows.append(RepoFixture(
            f"poly_count_{index}",
            {"util.py": "def count_items(seq):\n    return len(seq)\n",
             "api.py": f"from util import count_items\n\ndef run():\n    return count_items({arg})\n"},
            "polymorphic", "sized"))
    guarded = [("None", "none"), ('"abc"', "text"), ("(1, 2)", "tuple"), ("[1]", "list")]
    for arg, tag in guarded:
        rows.append(RepoFixture(
            f"poly_guarded_{tag}",
            {"util.py": "def maybe(x):\n    if x is None:\n        return 0\n    return len(x)\n",
             "api.py": f"from util import maybe\n\ndef run():\n    return maybe({arg})\n"},
            "polymorphic", "guarded"))
    return rows

# ------------------------------------------------------------- synthesis
FUNCTION_KINDS: Dict[str, Dict[str, object]] = {
    "validate": {"param": ("payload", "list"), "returns": "bool",
                 "body": ["    return len(payload) > 0"]},
    "parse": {"param": ("text", "str"), "returns": "dict",
              "body": ['    return {"raw": text.strip()}']},
    "store": {"param": ("record", "dict"), "returns": "bool",
              "body": ['    record["stored"] = True', "    return True"]},
    "load": {"param": ("key", "str"), "returns": "dict",
             "body": ['    return {"key": key}']},
    "aggregate": {"param": ("values", "list"), "returns": "float",
                  "body": ["    return sum(values) / len(values)"]},
    "count": {"param": ("items", "list"), "returns": "int",
              "body": ["    return len(items)"]},
    "normalise": {"param": ("text", "str"), "returns": "str",
                  "body": ["    return text.strip()"]},
}
MORPHISM_KIND = {
    "validate": "validate", "verify": "validate", "check": "validate",
    "parse": "parse", "read": "normalise",
    "store": "store", "write": "store", "save": "store",
    "load": "load", "fetch": "load",
    "sort": "aggregate", "compute": "aggregate", "increment": "aggregate",
    "count": "count", "emit": "count", "copy": "count",
}
PACKAGE_SPECS = (
    "the parser validates the schema and the writer stores the record and the reader loads the table",
    "the service verifies the request and the worker parses the response and the index sorts the list",
    "the loader copies the buffer and the queue emits the event and the module checks the field",
    "the reader loads the index and the writer stores the value and the parser parses the input",
    "the counter increments the value and the worker verifies the event and the service validates the request",
    "the index sorts the table and the loader stores the record and the reader loads the schema",
)


class RepoSheafSynthesizer:
    """Synthesize a consistent multi-module package from a four-slot DAG."""

    def __init__(self) -> None:
        self.sheaf = RepoSheaf()

    @staticmethod
    def spec_from_dag(name: str, dag: Dict[str, List[str]]) -> Dict[str, object]:
        kinds: List[str] = []
        for morphism in dag.get("morphism", []):
            kind = MORPHISM_KIND.get(str(morphism))
            if kind and kind not in kinds:
                kinds.append(kind)
        while len(kinds) < 3:
            kinds.append(("validate", "parse", "store")[len(kinds)])
        objects = [str(value) for value in dag.get("object", [])]
        return {"name": name, "kinds": kinds[:5], "objects": objects}

    def synthesize(self, spec: Dict[str, object]) -> Dict[str, str]:
        kinds = [str(value) for value in spec.get("kinds") or ["validate", "parse", "store"]]
        objects = [str(value) for value in spec.get("objects") or []]
        core_kinds = kinds[:2]
        storage_kind = kinds[2] if len(kinds) > 2 else "store"
        files: Dict[str, str] = {}
        files["models.py"] = (
            '"""Generated models for the package."""\n\n'
            "from dataclasses import dataclass\n\n\n"
            "@dataclass\n"
            "class Record:\n"
            "    key: str = \"\"\n"
            "    payload: str = \"\"\n"
        )
        core_lines = ['"""Generated core functions."""', ""]
        core_names: List[str] = []
        for index, kind in enumerate(core_kinds):
            template = FUNCTION_KINDS[kind]
            param, param_type = template["param"]
            name = f"{kind}_{index}"
            core_names.append(name)
            core_lines.append(f"def {name}({param}: {param_type}) -> {template['returns']}:")
            core_lines.extend(list(template["body"]))
            core_lines.append("")
        files["core.py"] = "\n".join(core_lines).rstrip() + "\n"
        storage_template = FUNCTION_KINDS[storage_kind]
        storage_param, storage_type = storage_template["param"]
        files["storage.py"] = "\n".join(
            ['"""Generated storage functions."""', "",
             f"def {storage_kind}({storage_param}: {storage_type}) -> {storage_template['returns']}:",
             *list(storage_template["body"]), ""]
        )
        imports = []
        for index, name in enumerate(core_names):
            template = FUNCTION_KINDS[core_kinds[index]]
            param, param_type = template["param"]
            if param_type == "str":
                argument = '"payload"'
            elif param_type == "list":
                argument = "[1, 2, 3]"
            else:
                argument = '{"k": 1}'
            imports.append(f"    {name}({argument})")
        storage_import = f"    {storage_kind}(" + (
            '"key"' if storage_type == "str" else ("[1, 2]" if storage_type == "list" else '{"k": 1}')
        ) + ")"
        api_lines = [
            '"""Generated service entry point."""',
            "",
            f"from core import {', '.join(core_names)}",
            f"from storage import {storage_kind}",
            "",
            "",
            "def run(text: str) -> dict:",
            '    record = {"payload": text}',
        ]
        api_lines.extend(imports)
        api_lines.append(storage_import)
        api_lines.append("    return record")
        files["api.py"] = "\n".join(api_lines).rstrip() + "\n"
        files["test_suite.py"] = "\n".join([
            '"""Generated integration tests."""',
            "",
            "import api",
            "",
            "",
            "def test_run_returns_a_record():",
            '    result = api.run("sample")',
            "    assert isinstance(result, dict)",
            "",
            "",
            "def test_run_is_deterministic():",
            '    assert api.run("sample") == api.run("sample")',
            "",
        ])
        return files

    def verify(self, files: Dict[str, str]) -> SheafReport:
        return self.sheaf.analyze(files)

    def repair(self, files: Dict[str, str], max_rounds: int = 4) -> Tuple[Dict[str, str], SheafReport]:
        """CEGIS repair: widen a parameter type when its demand is unmet."""
        current = dict(files)
        report = self.verify(current)
        for _ in range(int(max_rounds)):
            if report.energy <= self.sheaf.tau_sheaf:
                break
            changed = False
            for violation in report.violations:
                if violation.kind != "capability":
                    continue
                source = current.get(f"{violation.caller}.py", "")
                # Widen a literal argument that cannot carry the missing bit.
                for capability in violation.missing:
                    if capability in {"CAP_SEQ_MUT", "CAP_MAP_READ", "CAP_ELEM_NUM"}:
                        replacement = "[1, 2]" if capability != "CAP_MAP_READ" else '{"k": 1}'
                        for literal in ('"key"', '"payload"', '"sample"'):
                            if literal in source:
                                source = source.replace(literal, replacement, 1)
                                changed = True
                                break
                current[f"{violation.caller}.py"] = source
            report = self.verify(current)
            if not changed:
                break
        return current, report
