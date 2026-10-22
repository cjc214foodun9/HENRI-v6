"""Cartridge-driven code synthesis: build a candidate pool from a bank.

WHY THIS MODULE EXISTS
----------------------
Phase 8, ACTION 3. `solve_verified_code` is a discriminator: it needs a
pool of candidate programs. The directive asks for the pool to be built
from a mounted bank when the caller supplies none.

THE STRUCTURAL PROBLEM, STATED PLAINLY
--------------------------------------
`CartridgeBank` stores *waves*. `DeltaMemory.read` returns a wave. A wave at
D=65536 is a geometric state, not a string. It cannot be turned back into
Python source. So "retrieve top-k code templates from the CartridgeBank"
is only implementable with a **text registry** beside the waves: the bank
finds *which* template, the registry returns *what* that template's source
text was. This module supplies that registry and says so, rather than
implying the wave core decoded text.

WHAT SYNTHESIS IS, AND IS NOT
-----------------------------
It is **AST recombination over retrieved templates**: rename the function
to the name the spec implies, and enumerate the operator variants the AST
permits. It is NOT generation from scratch. A spec whose solution is not
close to a stored template will produce candidates that all fail, and the
verdict then reads `NO_CANDIDATES_PASS` honestly.

The candidate count is bounded (`k * (1 + max_variants)`) because each
variant costs a sandboxed execution.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

# Operator antipodes, the same pairing the ingress module uses. A mutation
# walks this table; it never invents an operator that is not here.
OPPOSITE = {
    ast.Add: ast.Sub,
    ast.Sub: ast.Add,
    ast.Mult: ast.Div,
    ast.Div: ast.Mult,
    ast.FloorDiv: ast.Mult,
    ast.Mod: ast.Mult,
    ast.Lt: ast.Gt,
    ast.Gt: ast.Lt,
    ast.LtE: ast.GtE,
    ast.GtE: ast.LtE,
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
    ast.And: ast.Or,
    ast.Or: ast.And,
    ast.Is: ast.IsNot,
    ast.IsNot: ast.Is,
}


@dataclass
class Template:
    """One stored (docstring, source) pair. The registry's unit."""
    docstring: str
    source: str
    task_id: str = ""

    def to_dict(self) -> dict:
        return {"task_id": self.task_id,
                "docstring": self.docstring[:200],
                "source_sha256": _sha(self.source)}


def _sha(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class CodeTemplateBank:
    """Templates plus their waves, with the CARTRIDGE BANK doing storage.

    The wave side is a real `CartridgeBank`: it enforces the rank wall and
    gives the cold tier. The text side is this object's own list. Storing
    both is the honest design; a wave cannot be inverted into source.

    Args:
        dim: wave dimension. Must match the backbone.
        rank: `DeltaMemory` rank, also the per-cartridge row cap.
    """

    def __init__(self, dim: int, rank: int = 64) -> None:
        from henri.cartridge_bank import CartridgeBank

        self.dim = int(dim)
        self.rank = int(rank)
        self.templates: list[Template] = []
        self.bank = CartridgeBank(dim=self.dim, rank=self.rank)
        # Waves kept beside the texts so retrieval can score without
        # materialising a cartridge's hot operator per query.
        self._keys: list[np.ndarray] = []
        self._values: list[np.ndarray] = []

    # -- write ----------------------------------------------------------
    def add(self, backbone, template: Template) -> None:
        """Write one template: encode both sides, then store the pair."""
        k = backbone.encode(template.docstring).psi
        v = backbone.encode(template.source).psi
        self.templates.append(template)
        self._keys.append(np.asarray(k))
        self._values.append(np.asarray(v))

    def finalize(self) -> None:
        """Partition into cartridges of at most `rank` rows, then finalize.

        The rank wall is real: one cartridge holds `rank` exact
        associations. Exceeding it is a partitioning problem, not a
        tuning problem, so this method partitions rather than raising.
        """
        n = len(self.templates)
        if n == 0:
            raise ValueError("no templates")
        for start in range(0, n, self.rank):
            stop = min(start + self.rank, n)
            keys = np.stack(self._keys[start:stop], axis=0)
            vals = np.stack(self._values[start:stop], axis=0)
            self.bank.add_cartridge(keys, vals)
        self.bank.finalize()

    @property
    def rows(self) -> int:
        return len(self.templates)

    # -- read -----------------------------------------------------------
    def retrieve(self, backbone, spec: str, k: int = 8) -> list[dict]:
        """Top-`k` templates by wave resonance against `spec`.

        Routing bounds the search when the bank is large; at small sizes
        every cartridge is scored. Scores are full-wave cosines, matching
        how `rank_candidates` scores candidates, so the retrieved order and
        the final ranking use one statistic rather than two.
        """
        if not self.templates:
            return []
        q = np.asarray(backbone.encode(spec).psi)
        qn = q / max(float(np.linalg.norm(q)), 1e-30)
        scores = []
        for i, kk in enumerate(self._keys):
            kn = kk / max(float(np.linalg.norm(kk)), 1e-30)
            scores.append(float(np.real(np.vdot(qn, kn))))
        order = np.argsort(-np.asarray(scores))[:int(max(1, k))]
        out = []
        for rank, i in enumerate(order):
            out.append({"rank": rank + 1,
                        "template_index": int(i),
                        "wave_score": round(float(scores[i]), 9),
                        "template": self.templates[int(i)]})
        return out

    def to_dict(self) -> dict:
        return {"rows": self.rows,
                "rank": self.rank,
                "dim": self.dim,
                "cartridges": len(self.bank.cartridges)}


# -- AST synthesis --------------------------------------------------------

class _OpSwapper(ast.NodeTransformer):
    """Swap every swappable operator class. One variant per pass."""

    def __init__(self) -> None:
        self.changed = 0

    def _swap(self, node):
        cls = type(node.op)
        new = OPPOSITE.get(cls)
        if new is None:
            return node
        self.changed += 1
        node.op = new()
        return node

    def visit_BinOp(self, node):      # noqa: N802
        self.generic_visit(node)
        return self._swap(node)

    def visit_Compare(self, node):    # noqa: N802
        self.generic_visit(node)
        for i, op in enumerate(list(node.ops)):
            cls = type(op)
            new = OPPOSITE.get(cls)
            if new is not None:
                self.changed += 1
                node.ops[i] = new()
        return node

    def visit_BoolOp(self, node):     # noqa: N802
        self.generic_visit(node)
        return self._swap(node)


def operator_variants(source: str, limit: int = 2) -> list[str]:
    """Source texts with one operator class swapped.

    Returns `[]` when the source does not parse or holds no swappable
    operator. An empty list is a real answer: this template offers no
    operator channel, so no mutation is honest.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    out: list[str] = []
    seen = set()
    for node in ast.walk(tree):
        kinds = []
        if isinstance(node, ast.BinOp):
            kinds.append(type(node.op))
        elif isinstance(node, ast.Compare):
            kinds.extend(type(o) for o in node.ops)
        elif isinstance(node, ast.BoolOp):
            kinds.append(type(node.op))
        for cls in kinds:
            if cls not in OPPOSITE or cls in seen:
                continue
            seen.add(cls)
            # One class at a time. A full `NodeTransformer` pass would swap
            # every swappable node, blending several mutations into one
            # candidate and destroying the attribution a receipt needs.
            mutated = _swap_one(source, cls)
            if mutated and mutated not in out:
                out.append(mutated)
            if len(out) >= limit:
                return out
    return out


def _swap_one(source: str, cls) -> Optional[str]:
    """Swap every node of exactly one operator class. Others stay."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    new_cls = OPPOSITE.get(cls)
    if new_cls is None:
        return None
    hits = [0]

    class One(ast.NodeTransformer):
        def visit_BinOp(self, node):      # noqa: N802
            self.generic_visit(node)
            if type(node.op) is cls:
                node.op = new_cls()
                hits[0] += 1
            return node

        def visit_Compare(self, node):    # noqa: N802
            self.generic_visit(node)
            for i, op in enumerate(list(node.ops)):
                if type(op) is cls:
                    node.ops[i] = new_cls()
                    hits[0] += 1
            return node

        def visit_BoolOp(self, node):     # noqa: N802
            self.generic_visit(node)
            if type(node.op) is cls:
                node.op = new_cls()
                hits[0] += 1
            return node

    tree2 = One().visit(tree)
    if hits[0] == 0:
        return None
    ast.fix_missing_locations(tree2)
    try:
        return ast.unparse(tree2)
    except Exception:                     # noqa: BLE001
        return None


def rename_entry_function(source: str, name: str) -> Optional[str]:
    """Rename the first top-level FunctionDef so the spec's name is defined.

    A retrieved template usually defines a different name than the tests
    call. Without this, every execution fails on `NameError` and the
    synthesis arm would look broken for a trivial reason.
    """
    if not name or not name.isidentifier():
        return None
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if not fns:
        return None
    if fns[0].name == name:
        return source
    fns[0].name = name
    ast.fix_missing_locations(tree)
    try:
        return ast.unparse(tree)
    except Exception:                     # noqa: BLE001
        return None


def spec_entry_name(spec: str) -> Optional[str]:
    """Name a test asserts on, e.g. `assert add(2, 3) == 5` -> `add`.

    Read from the ASSERTIONS, not the docstring, because the assertions are
    what will actually call the function. Returns None when no call is
    found, and the caller must then skip renaming rather than guess.
    """
    import re
    m = re.search(r"assert\s+(?:not\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*\(",
                  spec)
    return m.group(1) if m else None


def synthesize_candidates(backbone, spec: str, bank: CodeTemplateBank,
                          tests: Sequence[str] = (), k: int = 8,
                          max_variants: int = 2) -> list[dict]:
    """Retrieved templates plus AST variants, each with provenance.

    Returns dicts `{text, origin, template_index, template_rank, mutation}`.
    `origin` is `retrieved` for the stored source and `mutated` for a
    variant, so a receipt can separate retrieval credit from mutation
    credit rather than blending them.
    """
    hits = bank.retrieve(backbone, spec, k=k)
    name = spec_entry_name("\n".join(list(tests)))
    out: list[dict] = []
    seen: set[str] = set()

    def push(text, origin, ti, trank, mutation):
        if not text or text in seen:
            return
        seen.add(text)
        out.append({"text": text, "origin": origin,
                    "template_index": ti, "template_rank": trank,
                    "mutation": mutation})

    for h in hits:
        src = h["template"].source
        renamed = rename_entry_function(src, name) if name else None
        base = renamed or src
        push(base, "retrieved", h["template_index"], h["rank"], "none")
        for vi, mut in enumerate(operator_variants(base, limit=max_variants)):
            push(mut, "mutated", h["template_index"], h["rank"],
                 f"operator_swap_{vi + 1}")
    return out
