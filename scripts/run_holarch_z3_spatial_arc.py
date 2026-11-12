"""Cycle Z3: spatial ARC-style induction over a 16-task 2-D battery.

    python scripts/run_holarch_z3_spatial_arc.py

Stage Z3 lifts the one-cell inducer into two dimensions:

  * a 2048-dimensional four-slot toroidal complex encoding whose inverse is
    exact on the grid subspace (the frame synthesis/analysis pair is an
    isometry and every demonstration grid round-trips with error 0);
  * a wave-delta router whose invariants are the shape/bounding-box delta, the
    D4 dihedral cross-spectral resonance, the 2-D Fourier translation phase
    peak, and the per-colour mass plus connected-component Euler/containment
    signature; the router ranks the eleven operator sketches from the training
    pairs alone and the routing path holds no task-name branch;
  * a CEGIS search over composable 1-2 stage operator chains under one
    200000-node budget, with REPULSOR pruning on refuted chains, and held-out
    generalisation measured at grid dimensions that never appear in the
    demonstrations.

The sixteen tasks and the four contradictory specs live in this script, not in
the package, so the inducer cannot special-case them.  DISCLOSURE: the task
battery and the operator library share an author.  The tasks are generated for
this stage, their expected outputs are computed by task-local transforms in
this file, and the inducer is given example pairs only.

Nothing is copied from an earlier receipt: every number is measured here.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_holarch_p_cycle as pc
from henri.holarch.onecell_audit import audit_task_name_branches
from henri.holarch.spatial_induction import (
    BACKGROUND,
    DECLARED_SHAPE_RULE,
    D4_ELEMENTS,
    SKETCH_LABELS,
    SPATIAL_SKETCHES,
    SpatialCegisInducer,
    SpatialSpec,
    ToroidalWaveEncoder,
    WaveDeltaSpatialRouter,
    encoding_audit,
    router_selftest,
)

RECEIPTS = ROOT / "receipts" / "holarch"
MODULE_PATH = ROOT / "henri" / "holarch" / "spatial_induction.py"
NODE_BUDGET = 200000
RECEIPT_NAME = "z3_spatial_arc_induction.json"


# --------------------------------------------------------------- grid builders
def _noise(height: int, width: int, seed: int,
           palette: Sequence[int] = (1, 2, 3), density: float = 0.38) -> np.ndarray:
    """A deterministic random colour field."""
    rng = random.Random(int(seed) * 7919 + int(height) * 31 + int(width))
    out = np.zeros((int(height), int(width)), dtype=np.int64)
    for row in range(int(height)):
        for col in range(int(width)):
            if rng.random() < float(density):
                out[row, col] = int(rng.choice(tuple(palette)))
    return out


def _edge(height: int, width: int, seed: int,
          palette: Sequence[int] = (1, 2, 3)) -> np.ndarray:
    """A noise field with content on the four border bands.

    A toroidal shift and a bounded shift differ exactly where the wrap lands,
    so a shift task needs material on the wrapped rows and columns.
    """
    grid = _noise(height, width, seed, palette)
    rng = random.Random(int(seed) + 104729)
    bands_rows = sorted({0, 1, height - 2, height - 1})
    bands_cols = sorted({0, 1, width - 2, width - 1})
    for row in bands_rows:
        for col in range(width):
            if grid[row, col] == BACKGROUND:
                grid[row, col] = int(rng.choice(tuple(palette)))
    for col in bands_cols:
        for row in range(height):
            if grid[row, col] == BACKGROUND:
                grid[row, col] = int(rng.choice(tuple(palette)))
    return grid


def _centered(height: int, width: int, seed: int,
              palette: Sequence[int] = (1, 2, 3)) -> np.ndarray:
    """A noise field whose bounding box is strictly inside the grid."""
    grid = _noise(height, width, seed, palette, density=0.55)
    top = 1 + int(seed) % 2
    bottom = 1 + (int(seed) // 3) % 2
    left = 1 + (int(seed) // 5) % 2
    right = 1 + (int(seed) // 7) % 2
    grid[:top, :] = BACKGROUND
    grid[height - bottom:, :] = BACKGROUND
    grid[:, :left] = BACKGROUND
    grid[:, width - right:] = BACKGROUND
    return grid


def _ring(height: int, width: int, seed: int,
          palette: Sequence[int] = (1, 2)) -> np.ndarray:
    """A noise field with a rectangular ring and an enclosed background hole."""
    grid = _noise(height, width, seed, palette)
    r0, c0 = max(1, height // 3), max(1, width // 3)
    r1, c1 = min(height - 2, height - 1 - height // 4), min(width - 2, width - 1 - width // 4)
    if r1 - r0 < 2 or c1 - c0 < 2:
        r0, c0, r1, c1 = 1, 1, height - 2, width - 2
    grid[r0:r1 + 1, c0:c1 + 1] = 1
    grid[r0 + 1:r1, c0 + 1:c1] = BACKGROUND
    return grid


def _quadrant(height: int, width: int, seed: int,
              palette: Sequence[int] = (1, 2, 3, 4), empty: int = 3) -> np.ndarray:
    """A noise field with exactly one empty quadrant."""
    grid = _noise(height, width, seed, palette, density=0.5)
    half_h, half_w = height // 2, width // 2
    boxes = [(0, half_h, 0, half_w), (0, half_h, half_w, width),
             (half_h, height, 0, half_w), (half_h, height, half_w, width)]
    for index, (r0, r1, c0, c1) in enumerate(boxes):
        if index == int(empty) % 4:
            grid[r0:r1, c0:c1] = BACKGROUND
        elif not bool((grid[r0:r1, c0:c1] != BACKGROUND).any()):
            grid[r0, c0] = int(palette[(index + 1) % len(palette)])
    return grid


# ------------------------------------------------- task-local transforms (data)
def _bbox(grid: np.ndarray) -> Tuple[int, int, int, int]:
    rows = np.flatnonzero((grid != BACKGROUND).any(axis=1))
    cols = np.flatnonzero((grid != BACKGROUND).any(axis=0))
    return int(rows[0]), int(cols[0]), int(rows[-1]), int(cols[-1])


def _enclosed(gap: np.ndarray) -> List[Tuple[int, int]]:
    """Background cells that no border-connected background region reaches."""
    height, width = gap.shape
    outside = np.zeros_like(gap, dtype=bool)
    stack: List[Tuple[int, int]] = []
    for row in range(height):
        for col in (0, width - 1):
            if bool(gap[row, col]) and not bool(outside[row, col]):
                outside[row, col] = True
                stack.append((row, col))
    for col in range(width):
        for row in (0, height - 1):
            if bool(gap[row, col]) and not bool(outside[row, col]):
                outside[row, col] = True
                stack.append((row, col))
    while stack:
        row, col = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = row + dr, col + dc
            if 0 <= nr < height and 0 <= nc < width and bool(gap[nr, nc]) \
                    and not bool(outside[nr, nc]):
                outside[nr, nc] = True
                stack.append((nr, nc))
    return [(int(row), int(col)) for row, col in zip(*np.nonzero(gap & ~outside))]


def _t_rot90(grid: np.ndarray) -> np.ndarray:
    return np.rot90(grid, 1).copy()


def _t_rot180(grid: np.ndarray) -> np.ndarray:
    return np.rot90(grid, 2).copy()


def _t_reflect_h(grid: np.ndarray) -> np.ndarray:
    return grid[:, ::-1].copy()


def _t_reflect_diag(grid: np.ndarray) -> np.ndarray:
    return np.swapaxes(grid, 0, 1).copy()


def _t_shift(dr: int, dc: int) -> Callable[[np.ndarray], np.ndarray]:
    def transform(grid: np.ndarray) -> np.ndarray:
        return np.roll(np.roll(grid, int(dr), axis=0), int(dc), axis=1)

    return transform


def _t_bshift(dr: int, dc: int) -> Callable[[np.ndarray], np.ndarray]:
    def transform(grid: np.ndarray) -> np.ndarray:
        height, width = grid.shape
        out = np.zeros_like(grid)
        rows_to = slice(max(0, int(dr)), min(height, height + int(dr)))
        rows_from = slice(max(0, -int(dr)), min(height, height - int(dr)))
        cols_to = slice(max(0, int(dc)), min(width, width + int(dc)))
        cols_from = slice(max(0, -int(dc)), min(width, width - int(dc)))
        out[rows_to, cols_to] = grid[rows_from, cols_from]
        return out

    return transform


def _t_cmap(mapping: Dict[int, int]) -> Callable[[np.ndarray], np.ndarray]:
    def transform(grid: np.ndarray) -> np.ndarray:
        out = grid.copy()
        for source, target in mapping.items():
            out[grid == int(source)] = int(target)
        return out

    return transform


def _t_crop(grid: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(grid)
    return grid[r0:r1 + 1, c0:c1 + 1].copy()


def _t_tile(grid: np.ndarray) -> np.ndarray:
    return np.tile(grid, (2, 2))


def _t_scale(grid: np.ndarray) -> np.ndarray:
    return np.kron(grid, np.ones((2, 2), dtype=np.int64))


def _t_flood(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    for row, col in _enclosed(grid == BACKGROUND):
        out[row, col] = 3
    return out


def _t_mask(grid: np.ndarray) -> np.ndarray:
    return np.where(grid == 2, grid, BACKGROUND).copy()


def _t_symmetry(grid: np.ndarray) -> np.ndarray:
    height, width = grid.shape
    half_h, half_w = height // 2, width // 2
    quads = [grid[:half_h, :half_w], grid[:half_h, half_w:],
             grid[half_h:, :half_w], grid[half_h:, half_w:]]
    empty = [index for index, quad in enumerate(quads) if not bool((quad != BACKGROUND).any())]
    if len(empty) != 1:
        return grid.copy()
    target = int(empty[0])
    opposite = (3, 2, 1, 0)[target]
    filled = np.rot90(quads[opposite], 2)
    out = grid.copy()
    slices = [(slice(0, half_h), slice(0, half_w)), (slice(0, half_h), slice(half_w, width)),
              (slice(half_h, height), slice(0, half_w)),
              (slice(half_h, height), slice(half_w, width))]
    out[slices[target]] = filled
    return out


def _t_constant(grid: np.ndarray) -> np.ndarray:
    return np.full(grid.shape, 5, dtype=np.int64)


def _t_rot90_swap(grid: np.ndarray) -> np.ndarray:
    rotated = np.rot90(grid, 1)
    out = rotated.copy()
    out[rotated == 1] = 2
    out[rotated == 2] = 1
    return out


# ------------------------------------------------------------------ task table
TaskRow = Dict[str, Any]


def _pairs(builder: Callable[..., np.ndarray], shapes: Sequence[Tuple[int, int]],
           transform: Callable[[np.ndarray], np.ndarray], seed: int,
           palette: Sequence[int]) -> List[Tuple[np.ndarray, np.ndarray]]:
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    for index, (height, width) in enumerate(shapes):
        grid = builder(height, width, int(seed) + 101 * index, palette)
        out.append((np.asarray(grid, dtype=np.int64),
                    np.asarray(transform(grid), dtype=np.int64)))
    return out


def build_tasks() -> List[TaskRow]:
    """The sixteen declared 2-D tasks: examples only, no rules."""
    table: List[TaskRow] = []

    def add(name: str, transform: Callable[[np.ndarray], np.ndarray],
            train: Sequence[Tuple[int, int]], test: Sequence[Tuple[int, int]],
            authored: str, family: str, builder: Callable[..., np.ndarray] = _noise,
            palette: Sequence[int] = (1, 2, 3), d4: Optional[str] = None,
            seed: int = 0) -> None:
        base = 1000 + 97 * len(table) + int(seed)
        examples = _pairs(builder, train, transform, base, palette)
        tests = _pairs(builder, test, transform, base + 500003, palette)
        table.append({
            "name": str(name),
            "spec": SpatialSpec(str(name), tuple(examples), tuple(tests)),
            "authored": str(authored),
            "family": str(family),
            "d4": d4,
            "train_shapes": [list(shape) for shape in train],
            "test_shapes": [list(shape) for shape in test],
        })

    add("z3_rot90_ccw", _t_rot90, [(7, 9), (8, 6), (9, 10)],
        [(6, 11), (10, 7), (6, 7)], "dihedral(rot90)", "dihedral", d4="rot90")
    add("z3_rot180_rect", _t_rot180, [(7, 9), (8, 6), (9, 10)],
        [(6, 8), (11, 7), (6, 9)], "dihedral(rot180)", "dihedral", d4="rot180")
    add("z3_reflect_h", _t_reflect_h, [(8, 9), (7, 6), (10, 11)],
        [(6, 7), (12, 10), (6, 12)], "dihedral(reflect_h)", "dihedral", d4="reflect_h")
    add("z3_reflect_diag", _t_reflect_diag, [(7, 9), (8, 6), (9, 10)],
        [(6, 11), (10, 7), (12, 9)], "dihedral(reflect_diag)", "dihedral", d4="reflect_diag")
    add("z3_toroidal_shift_1_2", _t_shift(1, 2), [(8, 9), (7, 10), (9, 7)],
        [(6, 11), (10, 8), (7, 12)], "toroidal_shift(1,2)", "toroidal_shift", builder=_edge)
    add("z3_bounded_shift_2_1", _t_bshift(2, 1), [(8, 9), (7, 8), (9, 10)],
        [(6, 7), (10, 12), (11, 6)], "bounded_shift(2,1,0)", "bounded_shift", builder=_edge)
    add("z3_colour_swap_1_2", _t_cmap({1: 2, 2: 1}), [(8, 9), (7, 6), (10, 8)],
        [(6, 11), (9, 12), (11, 7)], "colour_map(1<->2)", "colour_map")
    add("z3_colour_cycle_123", _t_cmap({1: 2, 2: 3, 3: 1}), [(8, 9), (7, 10), (9, 6)],
        [(6, 7), (11, 8), (12, 10)], "colour_map(1->2->3->1)", "colour_map")
    add("z3_crop_bbox", _t_crop, [(9, 10), (8, 11), (12, 12)],
        [(6, 7), (10, 9), (11, 13)], "crop_bbox()", "crop_bbox", builder=_centered)
    add("z3_tile_2x2", _t_tile, [(7, 7), (8, 6), (6, 10)],
        [(6, 9), (10, 8), (9, 6)], "tile(2,2)", "tile", seed=3)
    add("z3_scale_2", _t_scale, [(7, 9), (8, 6), (10, 6)],
        [(6, 10), (9, 7), (11, 8)], "scale(2)", "scale", seed=5)
    add("z3_interior_fill_3", _t_flood, [(8, 8), (7, 9), (9, 7)],
        [(6, 10), (10, 9), (12, 12)], "interior_flood_fill(3)",
        "interior_flood_fill", builder=_ring, palette=(1, 2))
    add("z3_mask_colour_2", _t_mask, [(8, 9), (7, 6), (10, 8)],
        [(6, 10), (9, 7), (11, 12)], "mask_by_colour(2)", "mask_by_colour")
    add("z3_symmetry_completion", _t_symmetry, [(8, 8), (10, 6), (6, 6)],
        [(6, 10), (12, 8), (8, 12)],
        "symmetry_completion()", "symmetry_completion",
        builder=lambda h, w, s, p: _quadrant(h, w, s, p, empty=(s // 2) % 4))
    add("z3_constant_fill_5", _t_constant, [(8, 9), (7, 6), (10, 11)],
        [(6, 10), (11, 7), (9, 12)], "constant_fill(5)", "constant_fill")
    add("z3_rot90_then_swap", _t_rot90_swap, [(8, 9), (7, 10), (9, 6)],
        [(6, 8), (10, 7), (12, 11)], "dihedral(rot90) >> colour_map(1<->2)",
        "dihedral", seed=7)
    return table


def build_contradictions() -> List[TaskRow]:
    """Four contradictory 2-D specs that must be refused fail-closed."""
    base_a = _noise(8, 8, 901, (1, 2, 3), 0.5)
    base_b = _noise(7, 9, 902, (1, 2, 3), 0.5)
    base_c = _noise(8, 8, 903, (1, 2, 3), 0.5)
    base_d = _centered(9, 7, 904, (1, 2, 3))
    rows: List[TaskRow] = [
        {
            "name": "z3_contradiction_two_outputs",
            "spec": SpatialSpec(
                "z3_contradiction_two_outputs",
                ((base_a, _t_shift(1, 1)(base_a)), (base_a, _t_rot90(base_a))),
            ),
            "authored": "one input, two different outputs",
            "family": "contradiction",
            "d4": None,
            "train_shapes": [[8, 8], [8, 8]],
            "test_shapes": [],
        },
        {
            "name": "z3_contradiction_colour_only",
            "spec": SpatialSpec(
                "z3_contradiction_colour_only",
                ((base_b, base_b.copy()), (base_b, _t_cmap({1: 2, 2: 1})(base_b))),
            ),
            "authored": "one input, two outputs differing in one colour",
            "family": "contradiction",
            "d4": None,
            "train_shapes": [[7, 9], [7, 9]],
            "test_shapes": [],
        },
        {
            "name": "z3_contradiction_shape_violation",
            "spec": SpatialSpec(
                "z3_contradiction_shape_violation",
                ((base_c, base_c.copy()), (base_c, _t_tile(base_c))),
            ),
            "authored": "one input, one shape-preserving and one tiled output",
            "family": "contradiction",
            "d4": None,
            "train_shapes": [[8, 8], [8, 8]],
            "test_shapes": [],
        },
        {
            "name": "z3_contradiction_declared_shape",
            "spec": SpatialSpec(
                "z3_contradiction_declared_shape",
                ((base_d, _t_crop(base_d)),),
                (),
                (DECLARED_SHAPE_RULE,),
            ),
            "authored": "an output shape that violates the declared shape rule",
            "family": "contradiction",
            "d4": None,
            "train_shapes": [[9, 7]],
            "test_shapes": [],
        },
    ]
    return rows


# ------------------------------------------------------------------- audits
def _module_literal_audit(path: Path, names: Sequence[str]) -> Dict[str, Any]:
    """Every string constant in the module, checked against the task names."""
    source = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    wanted = {str(value) for value in names}
    hits: List[Dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value in wanted:
                hits.append({"line": int(node.lineno), "literal": node.value})
    name_compares: List[Dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            operands = [node.left] + list(node.comparators)
            attribute = any(isinstance(item, ast.Attribute) and item.attr == "name"
                            for item in operands)
            strings = [item.value for item in operands
                       if isinstance(item, ast.Constant) and isinstance(item.value, str)]
            if attribute and strings:
                name_compares.append({"line": int(node.lineno), "strings": strings})
    return {
        "path": str(Path(path).relative_to(ROOT)).replace(chr(92), "/"),
        "task_names_checked": int(len(wanted)),
        "string_constants": int(sum(1 for node in ast.walk(tree)
                                    if isinstance(node, ast.Constant)
                                    and isinstance(node.value, str))),
        "task_name_literals_in_module": int(len(hits)),
        "task_name_literals": hits[:8],
        "name_attribute_comparisons": int(len(name_compares)),
        "name_attribute_examples": name_compares[:8],
        "passed": bool(not hits and not name_compares),
    }


FORBIDDEN_IMPORTS = ("requests", "httpx", "urllib", "urllib3", "openai", "anthropic",
                     "socket", "aiohttp", "http", "subprocess", "transformers", "torch.hub")


def _import_audit(path: Path) -> Dict[str, Any]:
    """No network client and no model client is imported by the inducer."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    found: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.append(node.module)
    blocked = sorted({name for name in found
                      if any(name == stem or name.startswith(stem + ".")
                             for stem in FORBIDDEN_IMPORTS)})
    return {
        "imports": sorted(found),
        "forbidden_imports": blocked,
        "llm_calls": 0,
        "network_calls": 0,
        "passed": bool(not blocked),
    }


def _text_token_count(tasks: Sequence[TaskRow], contradictions: Sequence[TaskRow]) -> int:
    """Task content that reaches the inducer as text rather than as a grid."""
    count = 0
    for row in list(tasks) + list(contradictions):
        arrays = [value for pair in row["spec"].examples for value in pair]
        arrays += [value for pair in row["spec"].tests for value in pair]
        count += sum(1 for value in arrays if isinstance(value, str))
    return int(count)


# ---------------------------------------------------------------- measurement
def measure() -> Dict[str, Any]:
    started = time.perf_counter()
    encoder = ToroidalWaveEncoder()
    router = WaveDeltaSpatialRouter(encoder, node_budget=NODE_BUDGET)
    inducer = SpatialCegisInducer(encoder, router, node_budget=NODE_BUDGET)
    tasks = build_tasks()
    contradictions = build_contradictions()
    names = [str(row["name"]) for row in tasks]

    rows: List[Dict[str, Any]] = []
    d4_evidence: List[Dict[str, Any]] = []
    signature_example_only = True
    ranking_applied = True
    for row in tasks:
        spec = row["spec"]
        outcome = inducer.induce(spec)
        router_row = dict(outcome.get("router") or {})
        ranked = [str(value) for value in list(router_row.get("ranked") or [])]
        family_order = [str(value) for value in
                        list(router_row.get("candidate_families") or [])]
        presented = [str(value) for value in ranked if str(value) in set(family_order)]
        stripped = dataclasses.replace(spec, tests=())
        again = router.rank(stripped)
        signature_example_only = bool(
            signature_example_only
            and str(again["signature_digest"]) == str(router_row.get("signature_digest", "")))
        ranking_applied = bool(
            ranking_applied and ranked and set(ranked) == set(SPATIAL_SKETCHES)
            and bool(family_order) and family_order == presented)
        signature = dict(router_row.get("signature") or {})
        if row["d4"] is not None:
            d4_evidence.append({
                "name": str(row["name"]),
                "authored_element": str(row["d4"]),
                "router_best_element": str(signature.get("d4_best_element", "")),
                "router_best_mean": float(signature.get("d4_best_mean", 0.0) or 0.0),
                "identified": bool(str(signature.get("d4_best_element", "")) == str(row["d4"])),
            })
        rows.append({
            "name": str(row["name"]),
            "authored": str(row["authored"]),
            "authored_family": str(row["family"]),
            "status": str(outcome.get("status", "")),
            "reason": str(outcome.get("reason", "")),
            "mechanism": " >> ".join(str(value) for value in outcome.get("chain") or []),
            "families": [str(value) for value in outcome.get("families") or []],
            "stages": int(outcome.get("stages", 0) or 0),
            "nodes": int(outcome.get("nodes", 0) or 0),
            "repulsors": int(outcome.get("repulsors", 0) or 0),
            "pruned_duplicates": int(outcome.get("pruned_duplicates", 0) or 0),
            "train_exact": bool(outcome.get("train_exact")),
            "train_total": int(outcome.get("train_total", 0) or 0),
            "held_out_exact": int(outcome.get("test_exact", 0) or 0),
            "held_out_total": int(outcome.get("test_total", 0) or 0),
            "generalized": bool(outcome.get("generalized")),
            "held_out_shapes": [
                {
                    "input": list(item.get("input_shape") or []),
                    "expected": list(item.get("expected_shape") or []),
                    "predicted": list(item.get("predicted_shape") or []),
                    "exact": bool(item.get("exact")),
                    "wave_exact": bool(item.get("wave_exact")),
                }
                for item in list(outcome.get("test_rows") or [])
            ],
            "authored_family_induced": bool(
                (outcome.get("families") or [""])[0] == str(row["family"])
                if outcome.get("families") else False),
            "held_out_shapes_unseen": bool(all(
                tuple(item.get("input") or ()) not in
                {tuple(shape) for shape in row["train_shapes"]}
                for item in list(outcome.get("test_rows") or []))),
            "train_examples": int(len(spec.examples)),
            "test_examples": int(len(spec.tests)),
            "router": {
                "signature_digest": str(router_row.get("signature_digest", "")),
                "ranked": ranked,
                "chosen_rank": int(router_row.get("chosen_rank", -1)),
                "candidate_families": [str(value) for value in
                                       list(router_row.get("candidate_families") or [])],
                "d4_best_element": str(signature.get("d4_best_element", "")),
                "d4_best_mean": float(signature.get("d4_best_mean", 0.0) or 0.0),
                "translation_peak": list(signature.get("translation_peak") or []),
                "translation_height": float(signature.get("translation_height", 0.0) or 0.0),
                "shape_preserved": bool(signature.get("shape_preserved")),
                "mass_preserved": bool(signature.get("mass_preserved")),
                "examples_used": int(router_row.get("examples_used", 0) or 0),
                "tests_consulted": bool(router_row.get("tests_consulted", True)),
            },
        })

    contradiction_rows: List[Dict[str, Any]] = []
    for row in contradictions:
        outcome = inducer.induce(row["spec"])
        contradiction_rows.append({
            "name": str(row["name"]),
            "authored": str(row["authored"]),
            "status": str(outcome.get("status", "")),
            "reason": str(outcome.get("reason", "")),
            "nodes": int(outcome.get("nodes", 0) or 0),
            "refused_fail_closed": bool(outcome.get("refused_fail_closed")),
            "mechanism": "",
        })
    refusals = sum(1 for row in contradiction_rows if row["refused_fail_closed"]
                   and row["status"] == "ABSTAIN")

    induced = sum(1 for row in rows if row["status"] == "INDUCED")
    generalized = sum(1 for row in rows if row["status"] == "INDUCED" and row["generalized"])
    failed = [row["name"] for row in rows if not (row["status"] == "INDUCED" and row["generalized"])]
    max_nodes = max(int(row["nodes"]) for row in rows)
    total_nodes = int(sum(int(row["nodes"]) for row in rows))

    grids: List[np.ndarray] = []
    for row in list(tasks) + list(contradictions):
        for src, dst in row["spec"].examples:
            grids.append(np.asarray(src, dtype=np.int64))
            grids.append(np.asarray(dst, dtype=np.int64))
        for src, dst in row["spec"].tests:
            grids.append(np.asarray(src, dtype=np.int64))
            grids.append(np.asarray(dst, dtype=np.int64))
    encoding = encoding_audit(encoder, grids)

    all_names = names + [str(row["name"]) for row in contradictions]
    ast_audit = audit_task_name_branches(str(MODULE_PATH), all_names)
    literal_audit = _module_literal_audit(MODULE_PATH, all_names)
    import_audit = _import_audit(MODULE_PATH)
    text_tokens = _text_token_count(tasks, contradictions)
    selftest = router_selftest(encoder)
    identified = sum(1 for row in d4_evidence if row["identified"])
    d4_elements_ok = all(len(row["router"]["ranked"]) == len(SPATIAL_SKETCHES) for row in rows)

    # Budget enforcement: a deliberately tiny budget must stop the search and
    # abstain, so the declared cap is a cap and not a report.
    stress_budget = 8
    stress = SpatialCegisInducer(encoder, router, node_budget=stress_budget)
    hardest = max(rows, key=lambda item: int(item["nodes"]))
    stress_task = next(row for row in tasks if str(row["name"]) == str(hardest["name"]))
    stress_outcome = stress.induce(stress_task["spec"])
    budget_enforcement = {
        "task": str(stress_task["name"]),
        "budget": int(stress_budget),
        "nodes": int(stress_outcome.get("nodes", 0) or 0),
        "status": str(stress_outcome.get("status", "")),
        "budget_exhausted": bool(stress_outcome.get("budget_exhausted")),
        "passed": bool(str(stress_outcome.get("status")) == "ABSTAIN"
                       and int(stress_outcome.get("nodes", 0) or 0) <= stress_budget),
    }
    unseen_tasks = sum(1 for row in rows if row["held_out_shapes_unseen"])
    train_pair_counts = sorted({int(row["train_examples"]) for row in rows})
    test_pair_counts = sorted({int(row["test_examples"]) for row in rows})
    stages = {
        "task_rows": int(len(rows)),
        "contradiction_rows": int(len(contradiction_rows)),
        "unique_names": int(len({row["name"] for row in rows})),
        "grids_round_tripped": int(encoding["grids"]),
        "d4_evidence_rows": int(len(d4_evidence)),
        "router_sketches": int(len(SPATIAL_SKETCHES)),
        "held_out_unseen_tasks": int(unseen_tasks),
        "train_pairs": train_pair_counts,
        "test_pairs": test_pair_counts,
    }
    stages_ok = bool(stages["task_rows"] == 16 and stages["contradiction_rows"] == 4
                     and stages["unique_names"] == 16
                     and stages["grids_round_tripped"] > 0
                     and stages["d4_evidence_rows"] == 4
                     and unseen_tasks == 16
                     and min(train_pair_counts) >= 3 and max(train_pair_counts) <= 4
                     and min(test_pair_counts) >= 2 and max(test_pair_counts) <= 3)

    gates = {
        "z3_runner_runs": bool(stages_ok),
        "z3_encoding_invertible": bool(
            encoding["invertible"] and int(encoding["max_cell_error"]) == 0
            and float(encoding["max_phase_residual"]) <= 1e-12
            and float(encoding["max_isometry_error"]) <= 1e-12
            and float(encoding["max_gram_error"]) <= 1e-12),
        "z3_zero_text_tokens": bool(text_tokens == 0),
        "z3_zero_llm_calls": bool(import_audit["passed"] and import_audit["llm_calls"] == 0),
        "z3_zero_task_name_branches": bool(ast_audit["passed"] and literal_audit["passed"]),
        "z3_induced_and_generalized_ge_14": bool(generalized >= 14),
        "z3_contradictions_4_of_4": bool(refusals == 4 and len(contradiction_rows) == 4),
        "z3_budget_le_200000": bool(
            max_nodes <= NODE_BUDGET
            and all(int(row["router"].get("examples_used", 0)) >= 3 for row in rows)
            and bool(budget_enforcement["passed"])),
        "z3_d4_router_used": bool(
            d4_elements_ok and d4_evidence and identified >= 3
            and ranking_applied and signature_example_only
            and bool(selftest["shift_exact"]) and bool(selftest["d4_exact"])),
    }
    payload: Dict[str, Any] = {
        "stage": "Z3",
        "title": "spatial ARC-style induction: four-slot toroidal wave encoding, "
                 "D4 invariant router, CEGIS operator chains",
        "node_budget": int(NODE_BUDGET),
        "encoding": {
            "dimension": int(encoding["dimension"]),
            "slots": int(encoding["slots"]),
            "slot_width": int(encoding["slot_width"]),
            "colours": 10,
            "basis": "fixed orthogonal toroidal Fourier basis per slot",
            "colour_phase": "exp(2*pi*i*g*s/10) per colour and slot",
            "cell_phase": "exp(2*pi*i*m*(s+1)/(4*H*W)) per cell and slot",
            "grids": int(encoding["grids"]),
            "exact_roundtrips": int(encoding["exact_roundtrips"]),
            "max_cell_error": int(encoding["max_cell_error"]),
            "max_phase_residual": float(encoding["max_phase_residual"]),
            "max_isometry_error": float(encoding["max_isometry_error"]),
            "max_gram_error": float(encoding["max_gram_error"]),
            "frame_gram": encoding["frame_gram"],
            "invertible": bool(encoding["invertible"]),
            "shapes_tested": len(encoding["shapes"]),
        },
        "router": {
            "sketches": list(SPATIAL_SKETCHES),
            "sketch_labels": dict(SKETCH_LABELS),
            "components": [
                "shape_and_bounding_box_delta",
                "d4_dihedral_cross_spectral_resonance",
                "fourier_translation_phase_peak",
                "colour_mass_and_euler_containment",
            ],
            "d4_elements": list(D4_ELEMENTS),
            "d4_evidence": d4_evidence,
            "d4_identified": int(identified),
            "selftest": selftest,
            "ranking_applied": bool(ranking_applied),
            "signature_invariant_to_tests": bool(signature_example_only),
            "examples_only": True,
        },
        "tasks": rows,
        "induced": int(induced),
        "generalized": int(generalized),
        "task_count": int(len(rows)),
        "failed_tasks": failed,
        "contradictions": {
            "rows": contradiction_rows,
            "refused": int(refusals),
            "declared": int(len(contradiction_rows)),
            "policy": "fail-closed: duplicate input/outputs or a violated declared "
                      "shape facet refuses before any search node is spent",
        },
        "budget": {
            "node_budget": int(NODE_BUDGET),
            "max_nodes_used": int(max_nodes),
            "total_nodes_used": int(total_nodes),
            "per_task_nodes": {row["name"]: int(row["nodes"]) for row in rows},
            "within_budget": bool(max_nodes <= NODE_BUDGET),
            "unified": "one budget per task across every ranked sketch and both stages",
            "node_definition": "one operator-chain execution on one training pair",
            "enforcement": budget_enforcement,
            "repulsor_policy": "a refuted chain is recorded as a REPULSOR axiom (cap 64) "
                               "and an identical chain is never executed twice; duplicate "
                               "proposals are pruned before they cost a node",
            "repulsors_recorded": int(sum(int(row["repulsors"]) for row in rows)),
            "duplicates_pruned": int(sum(int(row["pruned_duplicates"]) for row in rows)),
        },
        "ast_audit": {
            "onecell_audit": ast_audit,
            "module_literal_audit": literal_audit,
            "import_audit": import_audit,
            "task_name_branches": int(ast_audit["task_name_branches"])
            + int(literal_audit["task_name_literals_in_module"]),
        },
        "stages": stages,
        "wave_calls": {
            "encoder_calls": int(encoder.calls),
            "router_calls": int(router.calls),
            "note": "encoder invocations are counted by the encoder itself; the router "
                    "ran once per task, once per task again for the test-invariance "
                    "check, and once for the budget-enforcement probe",
        },
        "text_tokens": {
            "count": int(text_tokens),
            "source": "every task is passed as integer numpy arrays; no character "
                      "stream, tokenizer or vocabulary enters the inducer",
            "measured": "str-valued entries in the example and test pairs of all 20 specs",
        },
        "disclosure": "the sixteen tasks in this script and the operator library in "
                      "henri/holarch/spatial_induction.py share an author; the tasks are "
                      "stored in this script so the inducer cannot special-case them",
        "gate": pc._finalise(gates),
        "evidence": "OBSERVED",
        "git_head": pc._git_head(),
        "wall_seconds": float(time.perf_counter() - started),
    }
    payload["passed"] = bool(all(bool(value) for value in gates.values()))
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Cycle Z3 spatial ARC induction")
    parser.add_argument("--receipts", default=str(RECEIPTS))
    args = parser.parse_args(argv)
    directory = Path(args.receipts)
    directory.mkdir(parents=True, exist_ok=True)
    payload = measure()
    written = pc._write(directory, RECEIPT_NAME, payload)
    print("[z3] induced+generalized=%d/%d contradictions=%d/%d ast_branches=%d max_nodes=%d" % (
        payload["generalized"], payload["task_count"],
        payload["contradictions"]["refused"], payload["contradictions"]["declared"],
        payload["ast_audit"]["task_name_branches"], payload["budget"]["max_nodes_used"]))
    print("[z3] encoding_max_error=%d phase=%.3e isometry=%.3e gram=%.3e" % (
        payload["encoding"]["max_cell_error"], payload["encoding"]["max_phase_residual"],
        payload["encoding"]["max_isometry_error"], payload["encoding"]["max_gram_error"]))
    print("[z3] d4_identified=%d/4 router=%s passed=%s receipt=%s sha256=%s" % (
        payload["router"]["d4_identified"], payload["gate"]["z3_d4_router_used"],
        payload["passed"], written["path"], written["sha256"]))
    if payload["failed_tasks"]:
        print("[z3] failed=%s" % (",".join(payload["failed_tasks"]),))
    return 0 if bool(payload["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
