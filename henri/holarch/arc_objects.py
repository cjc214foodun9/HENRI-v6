"""Cycle AB3/AC3: object-centric 2D topological wave-graph induction for ARC-AGI.

The AA3 audit proved the wall: whole-grid D4 / cyclic-shift / global-colour
operators solve 1 of 48 official public ARC-AGI tasks.  Official tasks are
object-centric physics puzzles, so this module parses the grid into a scene
graph of connected objects, computes exact topological invariants, induces
the output canvas dimension morphism, and searches a library of object and
relational operators under a strict Leave-One-Out gate.

Cycle AC3 extends the library with four universal morphism families: the
dimension family (rank-predicate crops, mirror and fractal tiling, block
downscale, edge replication), the divider panel algebra (mesh templates, mesh
rank selection, OR/AND/XOR/NOR/DIFF/occlusion with a dihedral action), the
topology family (corner rays, glyph rays, collinear bridges, cavity fill,
unique shortest path, despeckle, contact gravity, ring completion) and a
Wave-invariant signature router that only orders the candidates.

No task id, benchmark function name or file name enters any branch.  The
scene parser, the invariants, the dimension inducer and every operator are
functions of the grid bytes only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

BACKGROUND = 0
COLOURS = 10
MAX_SIDE = 30


def as_grid(value: object) -> np.ndarray:
    grid = np.asarray(value, dtype=np.int64)
    if grid.ndim != 2 or grid.size == 0:
        raise ValueError("an ARC grid must be a non-empty 2D array")
    if int(grid.min()) < 0 or int(grid.max()) >= COLOURS:
        raise ValueError("ARC colours are 0..9")
    return np.ascontiguousarray(grid)


# ======================================================================
# Scene graph
# ======================================================================
@dataclass(frozen=True)
class ArcObject:
    """One connected component with its exact topological invariants."""

    cells: Tuple[Tuple[int, int], ...]
    colours: Tuple[int, ...]
    bbox: Tuple[int, int, int, int]
    mass: int
    holes: int
    shape_key: str
    centroid: Tuple[float, float]
    mask: Tuple[Tuple[int, ...], ...]
    mono: bool

    @property
    def colour(self) -> int:
        return int(self.colours[0]) if len(self.colours) == 1 else -1

    @property
    def height(self) -> int:
        return int(self.bbox[2] - self.bbox[0] + 1)

    @property
    def width(self) -> int:
        return int(self.bbox[3] - self.bbox[1] + 1)


def _components(mask: np.ndarray, connectivity: int) -> List[List[Tuple[int, int]]]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 8:
        offsets = [
            (-1, -1), (-1, 0), (-1, 1), (0, -1),
            (0, 1), (1, -1), (1, 0), (1, 1),
        ]
    else:
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    out: List[List[Tuple[int, int]]] = []
    for row in range(height):
        for col in range(width):
            if not mask[row, col] or seen[row, col]:
                continue
            stack = [(row, col)]
            seen[row, col] = True
            cells: List[Tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                cells.append((int(r), int(c)))
                for dr, dc in offsets:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(cells)
    return out


def _canonical_mask(mask: np.ndarray) -> str:
    """The D4-invariant canonical form of a boolean mask."""
    best: Optional[str] = None
    variants = [
        mask,
        np.rot90(mask, 1),
        np.rot90(mask, 2),
        np.rot90(mask, 3),
        mask[:, ::-1],
        mask[::-1, :],
        mask.T,
        np.rot90(mask.T, 2),
    ]
    for variant in variants:
        blob = np.packbits(np.ascontiguousarray(variant, dtype=np.uint8)).tobytes()
        text = "%dx%d:%s" % (variant.shape[0], variant.shape[1], blob.hex())
        if best is None or text < best:
            best = text
    return str(best)


def _hole_count(cells: Sequence[Tuple[int, int]], shape: Tuple[int, int]) -> int:
    """beta_1: interior background components enclosed by the object."""
    if not cells:
        return 0
    height, width = shape
    mask = np.zeros((height, width), dtype=bool)
    for row, col in cells:
        mask[row, col] = True
    free = ~mask
    # Flood the free space from the border; every free cell not reached is a hole.
    reached = np.zeros_like(free, dtype=bool)
    stack: List[Tuple[int, int]] = []
    for row in range(height):
        for col in (0, width - 1):
            if free[row, col] and not reached[row, col]:
                reached[row, col] = True
                stack.append((row, col))
    for col in range(width):
        for row in (0, height - 1):
            if free[row, col] and not reached[row, col]:
                reached[row, col] = True
                stack.append((row, col))
    while stack:
        r, c = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and free[nr, nc] and not reached[nr, nc]:
                reached[nr, nc] = True
                stack.append((nr, nc))
    interior = free & ~reached
    if not interior.any():
        return 0
    return int(len(_components(interior, 4)))


def extract_objects(
    grid: np.ndarray,
    background: int,
    *,
    connectivity: int = 4,
    mono: bool = True,
    min_mass: int = 1,
) -> List[ArcObject]:
    value = as_grid(grid)
    return list(
        _cached_objects(value, int(background), int(connectivity), bool(mono), int(min_mass))
    )


def _extract_objects_raw(
    grid: np.ndarray,
    background: int,
    connectivity: int = 4,
    mono: bool = True,
    min_mass: int = 1,
) -> List[ArcObject]:
    grid = as_grid(grid)
    height, width = grid.shape
    out: List[ArcObject] = []
    if mono:
        for colour in sorted(set(int(value) for value in np.unique(grid))):
            if int(colour) == int(background):
                continue
            for cells in _components(grid == int(colour), connectivity):
                if len(cells) < int(min_mass):
                    continue
                out.append(_build_object(grid, cells, background, mono=True))
    else:
        mask = grid != int(background)
        for cells in _components(mask, connectivity):
            if len(cells) < int(min_mass):
                continue
            out.append(_build_object(grid, cells, background, mono=False))
    out.sort(key=lambda item: (item.bbox[0], item.bbox[1], -item.mass, item.colours))
    return out


def _build_object(grid: np.ndarray, cells: List[Tuple[int, int]], background: int, *, mono: bool) -> ArcObject:
    rows = [row for row, _ in cells]
    cols = [col for _, col in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    colours = tuple(sorted(set(int(grid[row, col]) for row, col in cells)))
    sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for row, col in cells:
        sub[row - r0, col - c0] = True
    return ArcObject(
        cells=tuple(tuple(cell) for cell in cells),
        colours=colours,
        bbox=(int(r0), int(c0), int(r1), int(c1)),
        mass=int(len(cells)),
        holes=_hole_count(cells, grid.shape),
        shape_key=_canonical_mask(sub),
        centroid=(float(np.mean(rows)), float(np.mean(cols))),
        mask=tuple(tuple(int(value) for value in row) for row in sub),
        mono=bool(mono),
    )


def background_colour(grid: np.ndarray) -> int:
    """The dominant background: plurality, then border majority, then 0.

    The plurality rule is the standard ARC reading.  The border majority is
    the tie-break because a frame colour is the second most common
    background hypothesis, and 0 is the final default.
    """
    value = as_grid(grid)
    key = _signature_key(value)
    hit = _BG_CACHE.get(key)
    if hit is not None:
        return int(hit)
    answer = _background_colour_raw(value)
    if len(_BG_CACHE) > _CACHE_LIMIT:
        _BG_CACHE.clear()
    _BG_CACHE[key] = int(answer)
    return int(answer)


def _background_colour_raw(grid: np.ndarray) -> int:
    grid = as_grid(grid)
    counts = np.bincount(grid.ravel(), minlength=COLOURS)
    border = np.concatenate([grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1]])
    border_counts = np.bincount(border, minlength=COLOURS)
    cells = int(grid.size)
    # ARC convention: black (0) is the background whenever it is present as a
    # substantial field.  A foreground that merely contains 0 as a marker is
    # much rarer than the 20% floor, so the plurality rule then decides.
    if int(counts[BACKGROUND]) >= max(2, int(0.20 * cells)):
        return BACKGROUND
    best = int(np.argmax(counts))
    if int(counts[best]) > 0:
        tied = [int(value) for value in np.nonzero(counts == counts[best])[0]]
        if len(tied) > 1:
            border_best = max(tied, key=lambda value: (int(border_counts[value]), -int(value)))
            return int(border_best)
        return int(best)
    return int(np.argmax(border_counts))


@dataclass
class Scene:
    grid: np.ndarray
    background: int
    objects4: List[ArcObject]
    objects8: List[ArcObject]
    multicolour8: List[ArcObject]

    @property
    def objects(self) -> List[ArcObject]:
        return list(self.objects4)


def scene(grid: np.ndarray) -> Scene:
    value = as_grid(grid)
    background = background_colour(value)
    return Scene(
        grid=value,
        background=int(background),
        objects4=extract_objects(value, background, connectivity=4, mono=True),
        objects8=extract_objects(value, background, connectivity=8, mono=True),
        multicolour8=extract_objects(value, background, connectivity=8, mono=False),
    )


# ======================================================================
# Output canvas dimension induction
# ======================================================================
def _crop_frame(grid: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _bbox_of_colour(grid: np.ndarray, colour: int) -> Optional[Tuple[int, int, int, int]]:
    mask = grid == int(colour)
    if not mask.any():
        return None
    rows, cols = np.nonzero(mask)
    return (int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max()))


def _bbox_of_object(obj: ArcObject) -> Tuple[int, int, int, int]:
    return obj.bbox


def _non_background_bbox(grid: np.ndarray, background: int) -> Optional[Tuple[int, int, int, int]]:
    mask = grid != int(background)
    if not mask.any():
        return None
    rows, cols = np.nonzero(mask)
    return (int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max()))


def induce_dimensions(
    pairs: Sequence[Tuple[np.ndarray, np.ndarray]],
) -> List[Dict[str, Any]]:
    """Candidate output canvas morphisms read off the demonstration pairs only."""
    hypotheses: List[Dict[str, Any]] = []
    pairs = [(as_grid(src), as_grid(dst)) for src, dst in pairs]
    if not pairs:
        return hypotheses
    same = all(src.shape == dst.shape for src, dst in pairs)
    if same:
        hypotheses.append({"kind": "same_shape", "params": ()})
    shapes = {tuple(int(v) for v in dst.shape) for _src, dst in pairs}
    if len(shapes) == 1:
        only = tuple(int(v) for v in next(iter(shapes)))
        hypotheses.append({"kind": "fixed_shape", "params": only})
    factors = set()
    for src, dst in pairs:
        if src.shape[0] and src.shape[1]:
            if dst.shape[0] % src.shape[0] == 0 and dst.shape[1] % src.shape[1] == 0:
                factors.add((dst.shape[0] // src.shape[0], dst.shape[1] // src.shape[1]))
    for factor in sorted(factors):
        if factor != (1, 1):
            hypotheses.append({"kind": "scale", "params": (int(factor[0]),)})
            hypotheses.append({"kind": "tile", "params": (int(factor[0]), int(factor[1]))})
            hypotheses.append({"kind": "mirror_tile", "params": (int(factor[0]), int(factor[1]))})
            hypotheses.append({"kind": "kron_fractal", "params": (int(factor[0]), int(factor[1]))})
    pads = {
        (dst.shape[0] - src.shape[0], dst.shape[1] - src.shape[1])
        for src, dst in pairs
    }
    if len(pads) == 1:
        pad_r, pad_c = next(iter(pads))
        if pad_r == pad_c and pad_r > 0 and pad_r % 2 == 0:
            hypotheses.append({"kind": "pad_edges", "params": (int(pad_r // 2),)})
    for source in ("non_background", "largest", "smallest", "unique_shape", "unique_colour"):
        hypotheses.append({"kind": "crop", "params": (source,)})
    hypotheses.append({"kind": "crop_hole", "params": ()})
    hypotheses.append({"kind": "panel", "params": ()})
    return hypotheses


def apply_dimension(
    hypothesis: Dict[str, Any], src: np.ndarray
) -> Optional[np.ndarray]:
    """Return a canvas for the hypothesis, with the content still unbound."""
    kind = str(hypothesis["kind"])
    params = tuple(hypothesis.get("params") or ())
    value = as_grid(src)
    background = background_colour(value)
    if kind == "same_shape":
        return value.copy()
    if kind == "fixed_shape":
        return np.full((int(params[0]), int(params[1])), background, dtype=np.int64)
    if kind == "scale":
        factor = int(params[0])
        return np.kron(value, np.ones((factor, factor), dtype=np.int64))
    if kind == "tile":
        return np.tile(value, (int(params[0]), int(params[1])))
    if kind == "crop":
        source = str(params[0])
        box: Optional[Tuple[int, int, int, int]] = None
        objects = extract_objects(value, background, connectivity=4, mono=True)
        if source == "non_background":
            box = _non_background_bbox(value, background)
        elif source == "largest" and objects:
            box = _bbox_of_object(max(objects, key=lambda item: (item.mass, -item.bbox[0])))
        elif source == "smallest" and objects:
            box = _bbox_of_object(min(objects, key=lambda item: (item.mass, item.bbox[0])))
        elif source == "unique_shape" and objects:
            counts: Dict[str, int] = {}
            for item in objects:
                counts[item.shape_key] = counts.get(item.shape_key, 0) + 1
            unique = [item for item in objects if counts[item.shape_key] == 1]
            if unique:
                box = _bbox_of_object(max(unique, key=lambda item: item.mass))
        elif source == "unique_colour" and objects:
            colours = [item.colour for item in objects]
            unique = [item for item in objects if colours.count(item.colour) == 1]
            if unique:
                box = _bbox_of_object(max(unique, key=lambda item: item.mass))
        return None if box is None else _crop_frame(value, box)
    if kind == "crop_hole":
        holes = [item for item in extract_objects(value, background, connectivity=4, mono=True) if item.holes > 0]
        if not holes:
            return None
        return _crop_frame(value, _bbox_of_object(max(holes, key=lambda item: item.mass)))
    if kind == "panel":
        panels, _axis = split_panels(value)
        return None if not panels else panels[0].copy()
    if kind == "mirror_tile":
        return op_mirror_tile(value, int(params[0]), int(params[1]), "flip")
    if kind == "kron_fractal":
        return op_kron_fractal(value, int(params[0]), int(params[1]))
    if kind == "pad_edges":
        corner = int(background)
        out = op_pad_edges(value, int(params[0]), corner)
        return out
    return None


def split_panels(grid: np.ndarray) -> Tuple[List[np.ndarray], str]:
    """Split on full-row/full-column separator lines of a single colour."""
    grid = as_grid(grid)
    height, width = grid.shape
    rows = [row for row in range(height) if len(set(int(v) for v in grid[row])) == 1]
    cols = [col for col in range(width) if len(set(int(v) for v in grid[:, col])) == 1]
    if rows and not cols:
        bands: List[np.ndarray] = []
        start = 0
        for row in rows:
            if row > start:
                bands.append(grid[start:row])
            start = row + 1
        if start < height:
            bands.append(grid[start:])
        return ([band for band in bands if band.size], "rows")
    if cols and not rows:
        bands = []
        start = 0
        for col in cols:
            if col > start:
                bands.append(grid[:, start:col])
            start = col + 1
        if start < width:
            bands.append(grid[:, start:])
        return ([band for band in bands if band.size], "cols")
    return ([], "")


# ======================================================================
# Operators
# ======================================================================
def op_dihedral(grid: np.ndarray, index: int) -> Optional[np.ndarray]:
    grid = as_grid(grid)
    table = [
        grid,
        np.rot90(grid, 1),
        np.rot90(grid, 2),
        np.rot90(grid, 3),
        grid[:, ::-1],
        grid[::-1, :],
        grid.T,
        np.rot90(grid.T, 2),
    ]
    if not 0 <= int(index) < len(table):
        return None
    return np.ascontiguousarray(table[int(index)])


def op_colour_map(grid: np.ndarray, *lut: object) -> Optional[np.ndarray]:
    if len(lut) == 1 and isinstance(lut[0], (tuple, list)):
        lut = tuple(lut[0])
    if len(lut) != COLOURS:
        return None
    table = np.asarray([int(value) for value in lut], dtype=np.int64)
    return table[as_grid(grid)]


def op_crop_bbox(grid: np.ndarray) -> Optional[np.ndarray]:
    value = as_grid(grid)
    background = background_colour(value)
    box = _non_background_bbox(value, background)
    return None if box is None else _crop_frame(value, box)


def op_crop_colour(grid: np.ndarray, colour: int) -> Optional[np.ndarray]:
    value = as_grid(grid)
    box = _bbox_of_colour(value, int(colour))
    return None if box is None else _crop_frame(value, box)


def op_scale(grid: np.ndarray, factor: int) -> Optional[np.ndarray]:
    if int(factor) < 1:
        return None
    return np.kron(as_grid(grid), np.ones((int(factor), int(factor)), dtype=np.int64))


def op_tile(grid: np.ndarray, rows: int, cols: int) -> Optional[np.ndarray]:
    if int(rows) < 1 or int(cols) < 1:
        return None
    return np.tile(as_grid(grid), (int(rows), int(cols)))


def _sort_objects(objects: Sequence[ArcObject], rule: str) -> List[ArcObject]:
    items = list(objects)
    if rule == "mass_rank":
        items.sort(key=lambda item: (-item.mass, item.bbox[0], item.bbox[1]))
    elif rule == "mass_rank_asc":
        items.sort(key=lambda item: (item.mass, item.bbox[0], item.bbox[1]))
    elif rule == "holes":
        items.sort(key=lambda item: (-item.holes, -item.mass, item.bbox[0]))
    elif rule == "colour_rank":
        items.sort(key=lambda item: (item.colour, -item.mass))
    elif rule == "left_to_right":
        items.sort(key=lambda item: (item.bbox[1], item.bbox[0]))
    elif rule == "top_to_bottom":
        items.sort(key=lambda item: (item.bbox[0], item.bbox[1]))
    return items


def op_recolour_objects(grid: np.ndarray, rule: str, *colours: int) -> Optional[np.ndarray]:
    value = as_grid(grid)
    background = background_colour(value)
    objects = _sort_objects(extract_objects(value, background, connectivity=4, mono=True), str(rule))
    if not objects or not colours:
        return None
    out = value.copy()
    for index, obj in enumerate(objects):
        if index >= len(colours):
            break
        for row, col in obj.cells:
            out[row, col] = int(colours[index])
    return out


def op_recolour_by_property(
    grid: np.ndarray, rule: str, threshold: int, colour: int
) -> Optional[np.ndarray]:
    value = as_grid(grid)
    background = background_colour(value)
    objects = extract_objects(value, background, connectivity=4, mono=True)
    if not objects:
        return None
    out = value.copy()
    for obj in objects:
        keep = False
        if rule == "mass_ge":
            keep = obj.mass >= int(threshold)
        elif rule == "mass_le":
            keep = obj.mass <= int(threshold)
        elif rule == "holes_ge":
            keep = obj.holes >= int(threshold)
        elif rule == "holes_eq":
            keep = obj.holes == int(threshold)
        if keep:
            for row, col in obj.cells:
                out[row, col] = int(colour)
    return out


def op_delete_objects(grid: np.ndarray, rule: str) -> Optional[np.ndarray]:
    value = as_grid(grid)
    background = background_colour(value)
    objects = extract_objects(value, background, connectivity=4, mono=True)
    if not objects:
        return None
    keep: List[ArcObject] = []
    if rule == "largest":
        keep = [max(objects, key=lambda item: (item.mass, -item.bbox[0]))]
    elif rule == "smallest":
        keep = [min(objects, key=lambda item: (item.mass, item.bbox[0]))]
    elif rule == "unique_mass":
        masses = [item.mass for item in objects]
        keep = [item for item in objects if masses.count(item.mass) == 1]
    elif rule == "unique_shape":
        counts: Dict[str, int] = {}
        for item in objects:
            counts[item.shape_key] = counts.get(item.shape_key, 0) + 1
        keep = [item for item in objects if counts[item.shape_key] == 1]
    elif rule == "unique_colour":
        colours = [item.colour for item in objects]
        keep = [item for item in objects if colours.count(item.colour) == 1]
    out = np.full_like(value, int(background))
    for obj in keep:
        for row, col in obj.cells:
            out[row, col] = value[row, col]
    return out


def op_fill_holes(grid: np.ndarray, colour: int) -> Optional[np.ndarray]:
    value = as_grid(grid)
    background = background_colour(value)
    mask = value == int(background)
    out = value.copy()
    for region in _components(mask, 4):
        rows = [row for row, _ in region]
        cols = [col for _, col in region]
        if min(rows) == 0 or min(cols) == 0 or max(rows) == value.shape[0] - 1 or max(cols) == value.shape[1] - 1:
            continue
        for row, col in region:
            out[row, col] = int(colour)
    return out


def op_stamp(grid: np.ndarray, mapping: Sequence[Sequence[object]]) -> Optional[np.ndarray]:
    """Paint a marker-specific template around every cell of each marker colour."""
    value = as_grid(grid)
    if not mapping:
        return None
    table: Dict[int, Tuple[Tuple[int, int, int], ...]] = {}
    for entry in mapping:
        marker, offsets = int(entry[0]), entry[1]
        table[marker] = tuple((int(dr), int(dc), int(colour)) for dr, dc, colour in offsets)
    out = value.copy()
    height, width = value.shape
    for row in range(height):
        for col in range(width):
            offsets = table.get(int(value[row, col]))
            if not offsets:
                continue
            for dr, dc, colour in offsets:
                nr, nc = row + int(dr), col + int(dc)
                if 0 <= nr < height and 0 <= nc < width:
                    out[nr, nc] = int(colour)
    return out


def op_rays(
    grid: np.ndarray, markers: Sequence[int], directions: Sequence[Sequence[int]], colour: int
) -> Optional[np.ndarray]:
    value = as_grid(grid)
    if not directions:
        return None
    out = value.copy()
    height, width = value.shape
    marker_set = {int(item) for item in markers}
    for row in range(height):
        for col in range(width):
            if int(value[row, col]) not in marker_set:
                continue
            for dr, dc in directions:
                nr, nc = row + int(dr), col + int(dc)
                while 0 <= nr < height and 0 <= nc < width:
                    out[nr, nc] = int(colour)
                    nr += int(dr)
                    nc += int(dc)
    return out


def op_slide(grid: np.ndarray, dr: int, dc: int) -> Optional[np.ndarray]:
    value = as_grid(grid)
    background = background_colour(value)
    objects = extract_objects(value, background, connectivity=4, mono=True)
    if not objects:
        return None
    out = np.full_like(value, int(background))
    occupied = np.zeros_like(value, dtype=bool)
    dr, dc = int(dr), int(dc)
    order = sorted(objects, key=lambda item: (-(item.bbox[0] * dr + item.bbox[1] * dc), item.bbox[0]))
    for obj in order:
        shift = 0
        while True:
            candidate = [(row + (shift + 1) * dr, col + (shift + 1) * dc) for row, col in obj.cells]
            if any(
                row < 0 or col < 0 or row >= value.shape[0] or col >= value.shape[1] or occupied[row, col]
                for row, col in candidate
            ):
                break
            shift += 1
        for row, col in obj.cells:
            nr, nc = row + shift * dr, col + shift * dc
            out[nr, nc] = value[row, col]
            occupied[nr, nc] = True
    return out


def op_symmetry_completion(grid: np.ndarray, axis: str) -> Optional[np.ndarray]:
    value = as_grid(grid)
    out = value.copy()
    height, width = value.shape
    if axis in ("vertical", "both"):
        for row in range(height):
            for col in range(width // 2):
                mirror = width - 1 - col
                if out[row, col] == out[row, mirror]:
                    continue
                if out[row, col] == BACKGROUND:
                    out[row, col] = out[row, mirror]
                elif out[row, mirror] == BACKGROUND:
                    out[row, mirror] = out[row, col]
    if axis in ("horizontal", "both"):
        for col in range(width):
            for row in range(height // 2):
                mirror = height - 1 - row
                if out[row, col] == out[mirror, col]:
                    continue
                if out[row, col] == BACKGROUND:
                    out[row, col] = out[mirror, col]
                elif out[mirror, col] == BACKGROUND:
                    out[mirror, col] = out[row, col]
    return out


def op_overlay_panels(grid: np.ndarray, mode: str) -> Optional[np.ndarray]:
    panels, _axis = split_panels(as_grid(grid))
    if len(panels) < 2:
        return None
    shapes = {panel.shape for panel in panels}
    if len(shapes) != 1:
        return None
    stack = np.stack(panels)
    background = BACKGROUND
    if mode == "and":
        out = np.where((stack != background).all(axis=0), stack[0], background)
    elif mode == "or":
        out = np.zeros_like(panels[0])
        for panel in panels:
            out = np.where(panel != background, panel, out)
    elif mode == "xor":
        out = np.zeros_like(panels[0])
        for panel in panels:
            out = np.where(panel != background, np.where(out == background, panel, background), out)
    else:  # majority
        out = np.zeros_like(panels[0])
        for row in range(out.shape[0]):
            for col in range(out.shape[1]):
                values = [int(panel[row, col]) for panel in panels]
                counts: Dict[int, int] = {}
                for item in values:
                    counts[item] = counts.get(item, 0) + 1
                out[row, col] = max(counts, key=lambda key: (counts[key], -key))
    return out


def op_local_rule(grid: np.ndarray, table: Dict[str, int], radius: int) -> Optional[np.ndarray]:
    value = as_grid(grid)
    background = background_colour(value)
    pad = int(radius)
    padded = np.pad(value, pad, mode="constant", constant_values=int(background))
    out = value.copy()
    height, width = value.shape
    for row in range(height):
        for col in range(width):
            patch = padded[row : row + 2 * pad + 1, col : col + 2 * pad + 1]
            key = patch.tobytes()
            if key in table:
                out[row, col] = int(table[key])
    return out


def op_constant(grid: np.ndarray, colour: int) -> Optional[np.ndarray]:
    return np.full_like(as_grid(grid), int(colour))


def op_cross_center(grid: np.ndarray, ring_colour: int, line_colour: int) -> Optional[np.ndarray]:
    """Draw the full centre row and column of every ring of one colour."""
    value = as_grid(grid)
    background = background_colour(value)
    out = value.copy()
    height, width = value.shape
    rings = [
        obj
        for obj in extract_objects(value, background, connectivity=4, mono=True)
        if obj.colour == int(ring_colour)
    ]
    if not rings:
        return None
    for obj in rings:
        r0, c0, r1, c1 = obj.bbox
        row = (r0 + r1) // 2
        col = (c0 + c1) // 2
        for cc in range(width):
            if int(out[row, cc]) == int(background):
                out[row, cc] = int(line_colour)
        for rr in range(height):
            if int(out[rr, col]) == int(background):
                out[rr, col] = int(line_colour)
    return out


def op_fill_interior(grid: np.ndarray, colour: int) -> Optional[np.ndarray]:
    """Fill the one-cell-eroded bounding-box interior of every object."""
    value = as_grid(grid)
    background = background_colour(value)
    objects = extract_objects(value, background, connectivity=4, mono=True)
    if not objects:
        return None
    out = value.copy()
    for obj in objects:
        r0, c0, r1, c1 = obj.bbox
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        out[r0 + 1 : r1, c0 + 1 : c1] = int(colour)
    return out


# ======================================================================
# Cycle AC3: multiscale morphism families
#
# Family A (dimension): rank-predicate crops, mirror tiling, block
#   downscale, self-Kronecker fractal tiling.
# Family B (divider algebra): mesh template propagation and the cell-wise
#   OR / AND / XOR / NOR / DIFF / occlusion panel operators.
# Family C (topology): corner rays, glyph rays, collinear bridges, cavity
#   fill, unique shortest path, isolated-colour deletion, ring completion.
# Family D (signature routing): a Wave-invariant signature orders the
#   candidate families; it never removes one.
#
# Every operator is a function of the grid bytes and of parameters that the
# demonstrations themselves supply.  No branch reads a task id, a file name
# or a benchmark index.
# ======================================================================
_CACHE_LIMIT = 20000
_BG_CACHE: Dict[Tuple[int, int, bytes], int] = {}
_OBJ_CACHE: Dict[Tuple[int, int, bytes, int, bool, int], Tuple[ArcObject, ...]] = {}


def _signature_key(grid: np.ndarray) -> Tuple[int, int, bytes]:
    return (int(grid.shape[0]), int(grid.shape[1]), grid.tobytes())


def _cached_objects(
    grid: np.ndarray, background: int, connectivity: int, mono: bool, min_mass: int
) -> Tuple[ArcObject, ...]:
    key = _signature_key(grid) + (int(background), int(connectivity), bool(mono), int(min_mass))
    hit = _OBJ_CACHE.get(key)
    if hit is not None:
        return hit
    if len(_OBJ_CACHE) > _CACHE_LIMIT:
        _OBJ_CACHE.clear()
        _BG_CACHE.clear()
    found = tuple(_extract_objects_raw(grid, background, connectivity, mono, min_mass))
    _OBJ_CACHE[key] = found
    return found


def _dominant_colour(grid: np.ndarray, skip: Sequence[int] = ()) -> Optional[int]:
    counts = np.bincount(np.asarray(grid, dtype=np.int64).ravel(), minlength=COLOURS)
    for value in skip:
        counts[int(value)] = 0
    if int(counts.sum()) <= 0:
        return None
    return int(np.argmax(counts))


def _uniform_lines(value: np.ndarray) -> Tuple[List[int], List[int]]:
    """Candidate divider lines: uniform rows/columns that sit strictly inside the grid.

    A uniform first or last row is a border, not a divider, so it is excluded.
    A line painted in the background colour is a blank margin rather than a
    drawn divider, so it is excluded too.  What remains is a line that was
    deliberately drawn to separate panels.
    """
    height, width = value.shape
    background = int(background_colour(value))
    rows = [
        row
        for row in range(1, height - 1)
        if len(set(int(v) for v in value[row])) == 1 and int(value[row, 0]) != background
    ]
    cols = [
        col
        for col in range(1, width - 1)
        if len(set(int(v) for v in value[:, col])) == 1 and int(value[0, col]) != background
    ]
    return rows, cols


def _equal_panels(value: np.ndarray) -> Tuple[List[np.ndarray], str]:
    """Split on dividers into at least two congruent panels on one axis."""
    rows, cols = _uniform_lines(value)
    height, width = value.shape
    if rows and cols:
        return ([], "")
    if rows:
        bands: List[np.ndarray] = []
        start = 0
        for row in rows:
            if row > start:
                bands.append(value[start:row])
            start = row + 1
        if start < height:
            bands.append(value[start:])
        bands = [band for band in bands if band.size]
        if len(bands) >= 2 and len({band.shape for band in bands}) == 1:
            return (bands, "rows")
        return ([], "")
    if cols:
        bands = []
        start = 0
        for col in cols:
            if col > start:
                bands.append(value[:, start:col])
            start = col + 1
        if start < width:
            bands.append(value[:, start:])
        bands = [band for band in bands if band.size]
        if len(bands) >= 2 and len({band.shape for band in bands}) == 1:
            return (bands, "cols")
        return ([], "")
    return ([], "")


def _panel_grid(value: np.ndarray) -> Optional[Dict[str, Any]]:
    """The divider mesh: equal-shaped panels cut out by full separator lines."""
    height, width = value.shape
    rows, cols = _uniform_lines(value)
    for band in (rows, cols):
        if any(b - a == 1 for a, b in zip(band, band[1:])):
            return None
    bands_r: List[Tuple[int, int]] = []
    start = 0
    for row in rows:
        if row > start:
            bands_r.append((start, row))
        start = row + 1
    if start < height:
        bands_r.append((start, height))
    bands_c: List[Tuple[int, int]] = []
    start = 0
    for col in cols:
        if col > start:
            bands_c.append((start, col))
        start = col + 1
    if start < width:
        bands_c.append((start, width))
    if not bands_r or not bands_c:
        return None
    if len(bands_r) < 2 and len(bands_c) < 2:
        return None
    shapes = {(b - a, d - c) for a, b in bands_r for c, d in bands_c}
    if len(shapes) != 1:
        return None
    panel_h, panel_w = next(iter(shapes))
    if panel_h < 1 or panel_w < 1:
        return None
    return {
        "rows": bands_r,
        "cols": bands_c,
        "panel": (int(panel_h), int(panel_w)),
        "separators": (rows, cols),
    }


# ---------------------------------------------------------------- Family A
def op_mirror_tile(grid: np.ndarray, rows: int, cols: int, mode: str) -> Optional[np.ndarray]:
    """s x s dihedral tiling: alternate mirroring of the source tile."""
    value = as_grid(grid)
    rows, cols = int(rows), int(cols)
    if rows < 1 or cols < 1 or rows * cols < 2:
        return None
    height, width = value.shape
    out = np.zeros((rows * height, cols * width), dtype=np.int64)
    for i in range(rows):
        for j in range(cols):
            if mode == "flip":
                tile = value[::-1, :] if i % 2 else value
                tile = tile[:, ::-1] if j % 2 else tile
            elif mode == "rows":
                tile = value[:, ::-1] if i % 2 else value
            elif mode == "rot180":
                tile = np.rot90(value, 2) if (i + j) % 2 else value
            elif mode == "rot90":
                tile = np.rot90(value, (i + j) % 4)
            else:
                return None
            out[i * height : (i + 1) * height, j * width : (j + 1) * width] = tile
    return out


def op_kron_fractal(grid: np.ndarray, rows: int, cols: int) -> Optional[np.ndarray]:
    """Self-Kronecker fractal tiling: Y[ir+u, jc+v] = X[u,v] * I[X[i,j] != 0]."""
    value = as_grid(grid)
    rows, cols = int(rows), int(cols)
    if rows < 1 or cols < 1:
        return None
    height, width = value.shape
    background = background_colour(value)
    out = np.zeros((rows * height, cols * width), dtype=np.int64)
    for i in range(min(rows, height)):
        for j in range(min(cols, width)):
            if int(value[i, j]) == int(background):
                continue
            out[i * height : (i + 1) * height, j * width : (j + 1) * width] = np.where(
                value != int(background), value, int(background)
            )
    return out


def op_downscale_blocks(grid: np.ndarray, factor: int, mode: str) -> Optional[np.ndarray]:
    """Block-mode / majority downscale by an integer factor."""
    value = as_grid(grid)
    factor = int(factor)
    if factor < 2:
        return None
    height, width = value.shape
    if height % factor or width % factor:
        return None
    rows = height // factor
    cols = width // factor
    out = np.zeros((rows, cols), dtype=np.int64)
    for row in range(rows):
        for col in range(cols):
            block = value[row * factor : (row + 1) * factor, col * factor : (col + 1) * factor]
            values = [int(item) for item in block.ravel()]
            counts: Dict[int, int] = {}
            for item in values:
                counts[item] = counts.get(item, 0) + 1
            if mode == "block":
                out[row, col] = max(counts, key=lambda key: counts[key])
            elif mode == "majority":
                out[row, col] = max(counts, key=lambda key: (counts[key], -key))
            else:
                return None
    return out


def op_crop_rank(grid: np.ndarray, rule: str) -> Optional[np.ndarray]:
    """Crop to the bounding box of the object a universal rank predicate selects."""
    value = as_grid(grid)
    background = background_colour(value)
    objects = extract_objects(value, background, connectivity=4, mono=True)
    box: Optional[Tuple[int, int, int, int]] = None
    if rule == "non_background":
        box = _non_background_bbox(value, background)
    elif rule == "largest" and objects:
        box = max(objects, key=lambda item: (item.mass, -item.bbox[0], -item.bbox[1])).bbox
    elif rule == "smallest" and objects:
        box = min(objects, key=lambda item: (item.mass, item.bbox[0], item.bbox[1])).bbox
    elif rule == "most_holes" and objects:
        box = max(objects, key=lambda item: (item.holes, item.mass)).bbox
    elif rule == "least_holes" and objects:
        box = min(objects, key=lambda item: (item.holes, -item.mass)).bbox
    elif rule == "unique_colour" and objects:
        colours = [item.colour for item in objects]
        unique = [item for item in objects if item.colour >= 0 and colours.count(item.colour) == 1]
        if unique:
            box = max(unique, key=lambda item: item.mass).bbox
    elif rule == "unique_shape" and objects:
        counts: Dict[str, int] = {}
        for item in objects:
            counts[item.shape_key] = counts.get(item.shape_key, 0) + 1
        unique = [item for item in objects if counts[item.shape_key] == 1]
        if unique:
            box = max(unique, key=lambda item: item.mass).bbox
    elif rule == "most_symmetric" and objects:
        scored = [(_symmetry_order(item), item.mass, item) for item in objects]
        best = max(scored, key=lambda entry: (entry[0], entry[1]))
        box = best[2].bbox
    elif rule == "framed" and objects:
        box = _framed_box(value, background)
    return None if box is None else _crop_frame(value, box)


def _symmetry_order(obj: ArcObject) -> int:
    mask = np.asarray(obj.mask, dtype=bool)
    total = 0
    for variant in (
        mask,
        np.rot90(mask, 1),
        np.rot90(mask, 2),
        np.rot90(mask, 3),
        mask[:, ::-1],
        mask[::-1, :],
        mask.T,
        np.rot90(mask.T, 2),
    ):
        if variant.shape == mask.shape and np.array_equal(variant, mask):
            total += 1
    return int(total)


def _framed_box(value: np.ndarray, background: int) -> Optional[Tuple[int, int, int, int]]:
    """The interior object that sits one cell inside a rectangle of another colour."""
    objects = extract_objects(value, background, connectivity=4, mono=True)
    framed = [item for item in objects if item.holes > 0]
    best: Optional[Tuple[int, int, int, int]] = None
    for frame in framed:
        r0, c0, r1, c1 = frame.bbox
        if r1 - r0 < 3 or c1 - c0 < 3:
            continue
        inner = [
            item
            for item in objects
            if item is not frame
            and item.bbox[0] >= r0 + 1
            and item.bbox[2] <= r1 - 1
            and item.bbox[1] >= c0 + 1
            and item.bbox[3] <= c1 - 1
        ]
        if inner and best is None:
            best = frame.bbox
    return best


# ---------------------------------------------------------------- Family B
def op_mesh_template(grid: np.ndarray) -> Optional[np.ndarray]:
    """Propagate the largest panel's mask into every panel, in that panel's colour."""
    value = as_grid(grid)
    layout = _panel_grid(value)
    if layout is None:
        return None
    background = background_colour(value)
    panel_h, panel_w = layout["panel"]
    bands_r = layout["rows"]
    bands_c = layout["cols"]
    panels: List[List[np.ndarray]] = []
    for r0, _r1 in bands_r:
        row: List[np.ndarray] = []
        for c0, _c1 in bands_c:
            row.append(value[r0 : r0 + panel_h, c0 : c0 + panel_w])
        panels.append(row)
    best: Optional[Tuple[int, int]] = None
    best_mass = -1
    for i, row in enumerate(panels):
        for j, panel in enumerate(row):
            mass = int(np.count_nonzero(panel != int(background)))
            if mass > best_mass:
                best_mass = mass
                best = (i, j)
    if best is None or best_mass <= 0:
        return None
    ti, tj = best
    template = panels[ti][tj]
    template_colour = _dominant_colour(template, skip=(int(background),))
    if template_colour is None:
        return None
    mask = template == int(template_colour)
    if not bool(mask.any()):
        return None
    out = value.copy()
    for i, (r0, _r1) in enumerate(bands_r):
        candidates: List[int] = []
        for j, (c0, _c1) in enumerate(bands_c):
            if i == ti and j == tj:
                continue
            colour = _dominant_colour(panels[i][j], skip=(int(background),))
            if colour is not None:
                candidates.append(int(colour))
        if not candidates:
            for j, (c0, _c1) in enumerate(bands_c):
                colour = _dominant_colour(panels[i][j], skip=(int(background),))
                if colour is not None:
                    candidates.append(int(colour))
        if not candidates:
            for row in panels:
                for panel in row:
                    colour = _dominant_colour(panel, skip=(int(background),))
                    if colour is not None:
                        candidates.append(int(colour))
        if not candidates:
            return None
        paint = max(candidates, key=lambda key: (candidates.count(key), -key))
        for c0, _c1 in bands_c:
            out[r0 : r0 + panel_h, c0 : c0 + panel_w] = np.where(
                mask, int(paint), int(background)
            )
    return out


_PANEL_MODES = ("or", "and", "xor", "nor", "diff", "first", "last")
_PANEL_TRANSFORMS = ("none", "h", "v", "rot180")


def _panel_transform(panel: np.ndarray, action: str) -> Optional[np.ndarray]:
    if action == "none":
        return panel
    if action == "h":
        return panel[:, ::-1]
    if action == "v":
        return panel[::-1, :]
    if action == "rot180":
        return np.rot90(panel, 2)
    return None


def op_panel_op(grid: np.ndarray, mode: str, transform: str) -> Optional[np.ndarray]:
    """Divider-split panel algebra with an optional dihedral action on the last panel."""
    panels, _axis = _equal_panels(as_grid(grid))
    if len(panels) < 2:
        return None
    shapes = {panel.shape for panel in panels}
    if len(shapes) != 1:
        return None
    stack = [panel for panel in panels[:-1]]
    tail = _panel_transform(panels[-1], str(transform))
    if tail is None:
        return None
    stack.append(tail)
    background = int(background_colour(as_grid(grid)))
    out = np.full(panels[0].shape, background, dtype=np.int64)
    for row in range(out.shape[0]):
        for col in range(out.shape[1]):
            values = [int(panel[row, col]) for panel in stack]
            foreground = [value for value in values if value != background]
            if mode == "or":
                out[row, col] = foreground[0] if foreground else background
            elif mode == "and":
                out[row, col] = values[0] if len(foreground) == len(values) else background
            elif mode == "xor":
                out[row, col] = foreground[0] if len(foreground) % 2 == 1 else background
            elif mode == "nor":
                out[row, col] = background if foreground else values[0]
            elif mode == "diff":
                rest = [value for value in values[1:] if value != background]
                out[row, col] = values[0] if (values[0] != background and not rest) else background
            elif mode == "first":
                out[row, col] = foreground[0] if foreground else background
            elif mode == "last":
                out[row, col] = foreground[-1] if foreground else background
            else:
                return None
    return out


# ---------------------------------------------------------------- Family C
def op_corner_ray(grid: np.ndarray) -> Optional[np.ndarray]:
    """Every 2x2-minus-one L piece casts a diagonal ray from its missing corner."""
    value = as_grid(grid)
    background = background_colour(value)
    objects = extract_objects(value, background, connectivity=8, mono=True)
    height, width = value.shape
    out = value.copy()
    found = False
    for obj in objects:
        r0, c0, r1, c1 = obj.bbox
        if (int(r1 - r0), int(c1 - c0)) != (1, 1) or int(obj.mass) != 3:
            continue
        if int(obj.colour) < 0:
            continue
        cells = {(int(r), int(c)) for r, c in obj.cells}
        missing = [
            (r0 + dr, c0 + dc)
            for dr in (0, 1)
            for dc in (0, 1)
            if (r0 + dr, c0 + dc) not in cells
        ]
        if len(missing) != 1:
            continue
        found = True
        mr, mc = missing[0]
        cr, cc = obj.centroid
        step_r = 1 if mr > cr else -1
        step_c = 1 if mc > cc else -1
        row, col = mr + step_r, mc + step_c
        while 0 <= row < height and 0 <= col < width and int(value[row, col]) == int(background):
            out[row, col] = int(obj.colour)
            row += step_r
            col += step_c
    return out if found else None


def _solid_squares(value: np.ndarray, size: int) -> List[Tuple[int, int, np.ndarray]]:
    background = background_colour(value)
    height, width = value.shape
    out: List[Tuple[int, int, np.ndarray]] = []
    for r0 in range(height - int(size) + 1):
        for c0 in range(width - int(size) + 1):
            block = value[r0 : r0 + int(size), c0 : c0 + int(size)]
            if bool(np.all(block != int(background))):
                out.append((r0, c0, block.copy()))
    return out


def op_glyph_rays(grid: np.ndarray, size: int) -> Optional[np.ndarray]:
    """A solid s x s glyph emits one ray from each border cell, away from its centre."""
    value = as_grid(grid)
    size = int(size)
    if size < 3:
        return None
    squares = _solid_squares(value, size)
    if len(squares) != 1:
        return None
    r0, c0, block = squares[0]
    background = background_colour(value)
    height, width = value.shape
    out = value.copy()
    centre = size // 2
    for dr in range(size):
        for dc in range(size):
            if dr == centre and dc == centre:
                continue
            step_r = 0 if dr == centre else (1 if dr > centre else -1)
            step_c = 0 if dc == centre else (1 if dc > centre else -1)
            if step_r == 0 and step_c == 0:
                continue
            colour = int(block[dr, dc])
            row, col = r0 + dr + step_r, c0 + dc + step_c
            while (
                0 <= row < height and 0 <= col < width and int(out[row, col]) == int(background)
            ):
                out[row, col] = colour
                row += step_r
                col += step_c
    return out


def op_ring_fill(grid: np.ndarray, period: int, *cycle: int) -> Optional[np.ndarray]:
    """Concentric ring completion inside the hollow frame, with a periodic colour cycle."""
    value = as_grid(grid)
    period = int(period)
    if len(cycle) == 1 and isinstance(cycle[0], (tuple, list)):
        cycle = tuple(cycle[0])
    cycle = tuple(int(item) for item in cycle)
    if period < 1 or len(cycle) != period:
        return None
    box = _ring_box(value)
    if box is None:
        return None
    r0, c0, r1, c1 = box
    out = value.copy()
    for row in range(r0, r1 + 1):
        for col in range(c0, c1 + 1):
            distance = min(row - r0, r1 - row, col - c0, c1 - col)
            out[row, col] = int(cycle[distance % period])
    return out


def _ring_box(value: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    background = background_colour(value)
    objects = extract_objects(value, background, connectivity=4, mono=True)
    holes = [item for item in objects if item.holes > 0]
    if not holes:
        return None
    frame = max(holes, key=lambda item: (item.mass, -item.bbox[0]))
    r0, c0, r1, c1 = frame.bbox
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None
    return frame.bbox


def op_delete_isolated_colour(grid: np.ndarray) -> Optional[np.ndarray]:
    """Despeckle: erase each colour whose every cell is an isolated singleton.

    Each erased cell takes the majority non-background colour of its eight
    neighbours, so a speck that sits on an object dissolves into that object
    and a speck in open space returns to the background.
    """
    value = as_grid(grid)
    background = background_colour(value)
    height, width = value.shape
    out = value.copy()
    removed = False
    for colour in sorted(int(item) for item in np.unique(value)):
        if int(colour) == int(background):
            continue
        cells = [(int(r), int(c)) for r, c in zip(*np.nonzero(value == int(colour)))]
        if not cells:
            continue
        isolated = True
        for row, col in cells:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = row + dr, col + dc
                if 0 <= nr < height and 0 <= nc < width and int(value[nr, nc]) == int(colour):
                    isolated = False
                    break
            if not isolated:
                break
        if not isolated:
            continue
        members = [
            item
            for item in extract_objects(value, background, connectivity=4, mono=True)
            if int(item.colour) not in (-1, int(colour))
        ]
        for row, col in cells:
            candidates = []
            for item in members:
                if not (
                    item.bbox[0] <= row <= item.bbox[2] and item.bbox[1] <= col <= item.bbox[3]
                ):
                    continue
                touching = False
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = row + dr, col + dc
                        if 0 <= nr < height and 0 <= nc < width and int(value[nr, nc]) == int(item.colour):
                            touching = True
                            break
                    if touching:
                        break
                if touching:
                    candidates.append(item)
            if candidates:
                chosen = min(
                    candidates,
                    key=lambda item: (
                        (item.bbox[2] - item.bbox[0] + 1) * (item.bbox[3] - item.bbox[1] + 1),
                        -item.mass,
                    ),
                )
                out[row, col] = int(chosen.colour)
                continue
            out[row, col] = int(background)
        removed = True
    return out if removed else None


def op_shortest_path(grid: np.ndarray, colour: int) -> Optional[np.ndarray]:
    """Paint the unique shortest 4-connected corridor between the two rarest markers."""
    value = as_grid(grid)
    background = background_colour(value)
    height, width = value.shape
    target_colour: Optional[int] = None
    pair: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
    for candidate in sorted(int(item) for item in np.unique(value)):
        if int(candidate) == int(background):
            continue
        cells = [(int(r), int(c)) for r, c in zip(*np.nonzero(value == int(candidate)))]
        if len(cells) == 2:
            target_colour = int(candidate)
            pair = (cells[0], cells[1])
            break
    if pair is None:
        return None
    start, goal = pair
    from collections import deque

    distance = {start: 0}
    paths = {start: 1}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = (current[0] + dr, current[1] + dc)
            if not (0 <= nxt[0] < height and 0 <= nxt[1] < width):
                continue
            if int(value[nxt]) != int(background) and nxt != goal:
                continue
            if nxt not in distance:
                distance[nxt] = distance[current] + 1
                paths[nxt] = paths[current]
                queue.append(nxt)
            elif distance[nxt] == distance[current] + 1:
                paths[nxt] = min(2, paths[nxt] + paths[current])
    if goal not in distance or paths.get(goal, 0) != 1:
        return None
    step = goal
    out = value.copy()
    while step != start:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            previous = (step[0] + dr, step[1] + dc)
            if distance.get(previous, -1) == distance[step] - 1:
                if previous != start:
                    out[previous] = int(colour)
                step = previous
                break
        else:
            return None
    return out


def op_ray_bridge(grid: np.ndarray, axes: str, colour: int) -> Optional[np.ndarray]:
    """Bridge collinear same-colour pairs across background with a colour."""
    value = as_grid(grid)
    background = background_colour(value)
    height, width = value.shape
    out = value.copy()
    painted = False
    directions: List[Tuple[int, int]] = []
    if "h" in axes:
        directions.append((0, 1))
    if "v" in axes:
        directions.append((1, 0))
    if "d" in axes:
        directions.extend([(1, 1), (1, -1)])
    for dr, dc in directions:
        for row in range(height):
            for col in range(width):
                if int(value[row, col]) == int(background):
                    continue
                colour_here = int(value[row, col])
                pr, pc = row - dr, col - dc
                if 0 <= pr < height and 0 <= pc < width and int(value[pr, pc]) == colour_here:
                    continue
                step = 1
                while True:
                    nr, nc = row + step * dr, col + step * dc
                    if not (0 <= nr < height and 0 <= nc < width):
                        break
                    value_here = int(value[nr, nc])
                    if value_here == colour_here:
                        for k in range(1, step):
                            out[row + k * dr, col + k * dc] = int(colour)
                            painted = True
                        break
                    if value_here != int(background):
                        break
                    step += 1
    return out if painted else None


def op_cavity_object_fill(grid: np.ndarray) -> Optional[np.ndarray]:
    """Fill each enclosed cavity with the colour of the object that encloses it."""
    value = as_grid(grid)
    background = background_colour(value)
    height, width = value.shape
    out = value.copy()
    filled = False
    for region in _components(value == int(background), 4):
        rows = [row for row, _ in region]
        cols = [col for _, col in region]
        if min(rows) == 0 or min(cols) == 0 or max(rows) == height - 1 or max(cols) == width - 1:
            continue
        neighbours: List[int] = []
        for row, col in region:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = row + dr, col + dc
                if 0 <= nr < height and 0 <= nc < width and int(value[nr, nc]) != int(background):
                    neighbours.append(int(value[nr, nc]))
        if not neighbours:
            continue
        counts: Dict[int, int] = {}
        for item in neighbours:
            counts[item] = counts.get(item, 0) + 1
        paint = max(counts, key=lambda key: (counts[key], -key))
        for row, col in region:
            out[row, col] = int(paint)
            filled = True
    return out if filled else None


def op_fill_holes_by_size(grid: np.ndarray, small: int, big: int) -> Optional[np.ndarray]:
    """Fill every hollow box interior; a one-cell cavity takes `small`, else `big`."""
    value = as_grid(grid)
    background = background_colour(value)
    objects = extract_objects(value, background, connectivity=4, mono=True)
    out = value.copy()
    touched = False
    for obj in objects:
        r0, c0, r1, c1 = obj.bbox
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        interior = value[r0 + 1 : r1, c0 + 1 : c1]
        if not bool(np.all(interior == int(background))):
            continue
        paint = int(small) if int(interior.size) == 1 else int(big)
        out[r0 + 1 : r1, c0 + 1 : c1] = paint
        touched = True
    return out if touched else None


def op_mesh_select_fill(grid: np.ndarray, rule: str) -> Optional[np.ndarray]:
    """Fill every mesh panel that carries the extreme number of marks.

    The panel content is discarded and replaced by the panel's own colour, so
    the operator answers "which panel holds the most (or fewest) marks" rather
    than "what is inside it".  Every panel tied at the extreme is filled, which
    is what a rank predicate with no tie-break must do.
    """
    value = as_grid(grid)
    layout = _panel_grid(value)
    if layout is None:
        return None
    background = background_colour(value)
    panel_h, panel_w = layout["panel"]
    scored: List[Tuple[int, int, int, Optional[int]]] = []
    for r0, _r1 in layout["rows"]:
        for c0, _c1 in layout["cols"]:
            block = value[r0 : r0 + panel_h, c0 : c0 + panel_w]
            mask = block != int(background)
            cells = int(mask.sum())
            paint = _dominant_colour(block, skip=(int(background),))
            scored.append((cells, r0, c0, paint))
    if not scored:
        return None
    counts = [entry[0] for entry in scored]
    if rule == "most_marks":
        target = max(counts)
        chosen = [entry for entry in scored if entry[0] == target]
    elif rule == "fewest_marks":
        target = min(counts)
        chosen = [entry for entry in scored if entry[0] == target]
    else:
        return None
    if target <= 0:
        return None
    out = np.full_like(value, int(background))
    sep_rows, sep_cols = layout["separators"]
    for row in sep_rows:
        out[row, :] = value[row, :]
    for col in sep_cols:
        out[:, col] = value[:, col]
    for _cells, r0, c0, paint in chosen:
        if paint is None:
            return None
        out[r0 : r0 + panel_h, c0 : c0 + panel_w] = int(paint)
    return out


def op_gravity_colour(grid: np.ndarray, colour: int, dr: int, dc: int) -> Optional[np.ndarray]:
    """Contact gravity for one colour: each cell falls until a wall or obstacle stops it."""
    value = as_grid(grid)
    background = background_colour(value)
    height, width = value.shape
    dr, dc = int(dr), int(dc)
    if (dr, dc) not in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        return None
    moving = value == int(colour)
    if not bool(moving.any()):
        return None
    cells = [(int(r), int(c)) for r, c in zip(*np.nonzero(moving))]
    occupied = (value != int(background)) & ~moving
    cells.sort(key=lambda cell: (-(cell[0] * dr + cell[1] * dc), cell[0], cell[1]))
    out = value.copy()
    out[moving] = int(background)
    for row, col in cells:
        r, c = row, col
        while True:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < height and 0 <= nc < width) or bool(occupied[nr, nc]):
                break
            r, c = nr, nc
        out[r, c] = int(colour)
        occupied[r, c] = True
    return out


def op_pad_edges(grid: np.ndarray, pad: int, corner: int) -> Optional[np.ndarray]:
    """Grow the canvas by `pad`, replicating each edge outward, corners at `corner`."""
    value = as_grid(grid)
    pad = int(pad)
    if pad < 1 or pad > 4:
        return None
    height, width = value.shape
    out = np.full((height + 2 * pad, width + 2 * pad), int(corner), dtype=np.int64)
    out[pad : pad + height, pad : pad + width] = value
    for index in range(height):
        out[pad + index, 0] = value[index, 0]
        out[pad + index, width + 2 * pad - 1] = value[index, width - 1]
    for index in range(width):
        out[0, pad + index] = value[0, index]
        out[height + 2 * pad - 1, pad + index] = value[height - 1, index]
    return out


def op_swatch(grid: np.ndarray, height: int, width: int, rule: str) -> Optional[np.ndarray]:
    """A solid canvas in the colour of the object the rank predicate selects."""
    value = as_grid(grid)
    background = background_colour(value)
    objects = [item for item in extract_objects(value, background, connectivity=4, mono=True) if item.colour >= 0]
    if not objects:
        return None
    if rule == "largest":
        chosen = max(objects, key=lambda item: (item.mass, -item.bbox[0], -item.bbox[1]))
    elif rule == "smallest":
        chosen = min(objects, key=lambda item: (item.mass, item.bbox[0], item.bbox[1]))
    elif rule == "most_holes":
        chosen = max(objects, key=lambda item: (item.holes, item.mass))
    else:
        return None
    return np.full((int(height), int(width)), int(chosen.colour), dtype=np.int64)


def wave_signature(pairs: Sequence[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
    """The Family-D signature: it orders candidate families and never gates one."""
    ratios: List[Tuple[float, float]] = []
    shapes: List[Tuple[int, int]] = []
    conserved = 0
    novel = 0
    reduced = 0
    divider = 0
    panels = 0
    euler_delta = 0.0
    for src, dst in pairs:
        src, dst = as_grid(src), as_grid(dst)
        shapes.append((int(dst.shape[0]), int(dst.shape[1])))
        if src.shape[0] and src.shape[1]:
            ratios.append((dst.shape[0] / src.shape[0], dst.shape[1] / src.shape[1]))
        src_palette = {int(item) for item in np.unique(src)}
        dst_palette = {int(item) for item in np.unique(dst)}
        if dst_palette <= src_palette:
            conserved += 1
        if dst_palette - src_palette:
            novel += 1
        if src_palette - dst_palette:
            reduced += 1
        layout = _panel_grid(src)
        if layout is not None:
            divider += 1
            panels = max(panels, len(layout["rows"]) * len(layout["cols"]))
        src_objects = extract_objects(src, background_colour(src), connectivity=4, mono=True)
        dst_objects = extract_objects(dst, background_colour(dst), connectivity=4, mono=True)
        euler_delta += float(len(dst_objects) - len(src_objects))
    return {
        "lattice_ratio": ratios[0] if ratios and len(set(ratios)) == 1 else None,
        "constant_output_shape": shapes[0] if len(set(shapes)) == 1 else None,
        "divider_pairs": int(divider),
        "panel_count": int(panels),
        "palette_conserved": int(conserved),
        "palette_novel": int(novel),
        "palette_reduced": int(reduced),
        "euler_delta": float(euler_delta / max(len(pairs), 1)),
        "pairs": int(len(pairs)),
    }


def route_operators(ops: Sequence[ArcOp], signature: Dict[str, Any]) -> List[ArcOp]:
    """Reorder candidates by the Wave signature.  Ordering only; nothing is removed."""
    lattice = signature.get("lattice_ratio")
    constant = signature.get("constant_output_shape")
    panels = int(signature.get("panel_count") or 0)
    reduced = int(signature.get("palette_reduced") or 0)
    preferred: List[str] = []
    if lattice and tuple(lattice) != (1.0, 1.0):
        preferred.extend(["mirror_tile", "kron_fractal", "scale", "tile", "downscale_blocks"])
    if panels >= 4:
        preferred.extend(["mesh_template", "panel_op"])
    elif panels >= 2:
        preferred.extend(["panel_op", "overlay_panels", "mesh_template"])
    if constant is not None:
        preferred.extend(["swatch", "crop_rank", "crop_colour"])
    if reduced:
        preferred.extend(["delete_isolated_colour", "crop_rank"])
    if not preferred:
        return list(ops)
    rank = {name: index for index, name in enumerate(preferred)}
    indexed = list(enumerate(ops))
    indexed.sort(key=lambda entry: (rank.get(entry[1].family, len(preferred)), entry[0]))
    return [op for _index, op in indexed]


# ======================================================================
# Cycle AD2: relational and topological morphisms
# ======================================================================
def op_swap_object_palette(grid: np.ndarray) -> Optional[np.ndarray]:
    """Swap the two colours of every two-colour connected component."""
    value = as_grid(grid)
    background = background_colour(value)
    out = value.copy()
    for obj in extract_objects(value, background, connectivity=4, mono=False):
        if len(obj.colours) != 2:
            continue
        first, second = int(obj.colours[0]), int(obj.colours[1])
        for row, col in obj.cells:
            out[row, col] = second if int(value[row, col]) == first else first
    return out


def _hollow_square_frame(value: np.ndarray, colour: int) -> Optional[Tuple[int, int, int, int]]:
    """The bounding box of a colour whose border is closed and whose interior is not."""
    positions = np.argwhere(value == int(colour))
    if positions.size == 0:
        return None
    r0, c0 = (int(positions[:, 0].min()), int(positions[:, 1].min()))
    r1, c1 = (int(positions[:, 0].max()), int(positions[:, 1].max()))
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None
    border = 0
    total = 0
    for row in range(r0, r1 + 1):
        for col in range(c0, c1 + 1):
            if row in (r0, r1) or col in (c0, c1):
                total += 1
                if int(value[row, col]) == int(colour):
                    border += 1
    if total == 0 or border != total:
        return None
    return (r0, c0, r1, c1)


def op_concentric_squares(
    grid: np.ndarray, frame: int, ring: int, step: int, shell: int
) -> Optional[np.ndarray]:
    """Nested square shells: a two-ring outer shell, then period-four recursion."""
    value = as_grid(grid)
    frame = int(frame)
    ring = int(ring)
    step = max(1, int(step))
    shell = max(1, int(shell))
    box = _hollow_square_frame(value, frame)
    if box is None:
        return None
    out = value.copy()
    r0, c0, r1, c1 = box
    side = min(r1 - r0 + 1, c1 - c0 + 1)
    if side < 3:
        return None
    squares: List[Tuple[int, int, int, int, bool]] = [
        (r0, c0, r0 + side - 1, c0 + side - 1, True)
    ]
    depth_guard = 0
    while squares and depth_guard < 24:
        depth_guard += 1
        a0, b0, a1, b1, outermost = squares.pop(0)
        s = min(a1 - a0 + 1, b1 - b0 + 1)
        if s < 3:
            continue
        depth = (s - 1) // 2
        limit = min(shell, depth) if outermost else depth
        for distance in range(0, limit + 1):
            slot = distance % 4
            colour = frame if slot in (0, 2) else (ring if slot == 1 else 0)
            for row in range(a0 + distance, a1 - distance + 1):
                for col in range(b0 + distance, b1 - distance + 1):
                    if (
                        row in (a0 + distance, a1 - distance)
                        or col in (b0 + distance, b1 - distance)
                    ):
                        out[row, col] = int(colour)
        if s - 2 * step >= 3:
            squares.append((a0 + step, b0 + step, a1 - step, b1 - step, False))
    return out


def _enclosed_background(value: np.ndarray, background: int) -> List[List[Tuple[int, int]]]:
    """Background components that never touch the grid border (Jordan interior)."""
    mask = value == int(background)
    height, width = value.shape
    out: List[List[Tuple[int, int]]] = []
    for cells in _components(mask, 4):
        if any(
            row in (0, height - 1) or col in (0, width - 1) for row, col in cells
        ):
            continue
        out.append(cells)
    return out


def op_fill_hole_lattice(grid: np.ndarray, outer: int, inner: int) -> Optional[np.ndarray]:
    """Paint enclosed cavities by their position in the cavity lattice."""
    value = as_grid(grid)
    background = background_colour(value)
    holes = _enclosed_background(value, background)
    if len(holes) < 3:
        return None
    centres = [
        (float(np.mean([row for row, _ in cells])), float(np.mean([col for _, col in cells])))
        for cells in holes
    ]
    rows = [centre[0] for centre in centres]
    cols = [centre[1] for centre in centres]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    out = value.copy()
    for (row, col), cells in zip(centres, holes):
        boundary = (
            abs(row - r0) < 1e-9
            or abs(row - r1) < 1e-9
            or abs(col - c0) < 1e-9
            or abs(col - c1) < 1e-9
        )
        colour = int(outer) if boundary else int(inner)
        for r, c in cells:
            out[r, c] = colour
    return out


def _border_line(value: np.ndarray) -> Optional[Tuple[str, int]]:
    """The colour of a full border row or column, with its side."""
    height, width = value.shape
    for side, line in (
        ("left", value[:, 0]),
        ("right", value[:, -1]),
        ("top", value[0, :]),
        ("bottom", value[-1, :]),
    ):
        first = int(line[0])
        if first == 0:
            continue
        if np.all(line == first):
            return side, first
    return None


def op_border_line_completion(
    grid: np.ndarray, diagonal: int, row_fill: int
) -> Optional[np.ndarray]:
    """Given a full border line, draw the anti-diagonal and fill the far border row."""
    value = as_grid(grid)
    found = _border_line(value)
    if found is None:
        return None
    side, _colour = found
    out = value.copy()
    height, width = value.shape

    def canonical(array: np.ndarray) -> np.ndarray:
        if side == "left":
            return array
        if side == "right":
            return np.fliplr(array)
        if side == "top":
            return array.T
        return np.flipud(array.T)

    def invert(array: np.ndarray) -> np.ndarray:
        if side == "left":
            return array
        if side == "right":
            return np.fliplr(array)
        if side == "top":
            return array.T
        return np.flipud(array.T).T

    work = canonical(out)
    h, w = work.shape
    for index in range(0, max(h - 1, 0)):
        row = index
        col = w - 1 - index
        if 0 <= row < h and 0 <= col < w:
            if int(work[row, col]) == 0:
                work[row, col] = int(diagonal)
    for col in range(1, w):
        if int(work[h - 1, col]) == 0:
            work[h - 1, col] = int(row_fill)
    return invert(work)


def op_extend_objects(grid: np.ndarray, dr: int, dc: int) -> Optional[np.ndarray]:
    """Extend every object along a cardinal vector until another object or the border."""
    value = as_grid(grid)
    background = background_colour(value)
    height, width = value.shape
    dr = int(dr)
    dc = int(dc)
    if abs(dr) + abs(dc) != 1:
        return None
    objects = sorted(
        extract_objects(value, background, connectivity=4, mono=True),
        key=lambda item: (-item.mass, item.bbox),
    )
    out = value.copy()
    for obj in objects:
        colour = int(obj.colour)
        if colour == int(background) or colour < 1:
            continue
        for row, col in obj.cells:
            r, c = int(row) + dr, int(col) + dc
            while 0 <= r < height and 0 <= c < width:
                if int(value[r, c]) != int(background):
                    break
                out[r, c] = colour
                r += dr
                c += dc
    return out


def op_extend_to_wall(grid: np.ndarray) -> Optional[np.ndarray]:
    """Extend every object toward the full border line of the canvas."""
    value = as_grid(grid)
    found = _border_line(value)
    if found is None:
        return None
    side, _colour = found
    direction = {"left": (0, -1), "right": (0, 1), "top": (-1, 0), "bottom": (1, 0)}[side]
    return op_extend_objects(value, int(direction[0]), int(direction[1]))


def _full_lines(value: np.ndarray, colour: int) -> List[Tuple[str, int]]:
    """Every full row or column of one colour."""
    out: List[Tuple[str, int]] = []
    for row in range(value.shape[0]):
        if np.all(value[row, :] == int(colour)):
            out.append(("row", int(row)))
    for col in range(value.shape[1]):
        if np.all(value[:, col] == int(colour)):
            out.append(("col", int(col)))
    return out


def op_marker_wall_impact(
    grid: np.ndarray, marker: int, wall: int
) -> Optional[np.ndarray]:
    """Cast a perpendicular marker ray at the nearest full wall and thicken the impact."""
    value = as_grid(grid)
    marker = int(marker)
    wall = int(wall)
    lines = _full_lines(value, wall)
    if not lines:
        return None
    out = value.copy()
    height, width = value.shape
    rows = [index for orientation, index in lines if orientation == "row"]
    cols = [index for orientation, index in lines if orientation == "col"]
    for row in range(height):
        for col in range(width):
            if int(value[row, col]) != marker:
                continue
            targets: List[Tuple[str, int]] = []
            above = [index for index in rows if index < row]
            below = [index for index in rows if index > row]
            left = [index for index in cols if index < col]
            right = [index for index in cols if index > col]
            if above:
                targets.append(("row", max(above)))
            if below:
                targets.append(("row", min(below)))
            if left:
                targets.append(("col", max(left)))
            if right:
                targets.append(("col", min(right)))
            for orientation, index in targets:
                if orientation == "row":
                    direction = 1 if index > row else -1
                    for r in range(max(0, index - 1), min(height, index + 2)):
                        for c in range(max(0, col - 1), min(width, col + 2)):
                            out[r, c] = wall
                    out[index, col] = marker
                    r = row + direction
                    while 0 <= r < height and abs(r - index) > 1:
                        out[r, col] = marker
                        r += direction
                else:
                    direction = 1 if index > col else -1
                    for c in range(max(0, index - 1), min(width, index + 2)):
                        for r in range(max(0, row - 1), min(height, row + 2)):
                            out[r, c] = wall
                    out[row, index] = marker
                    c = col + direction
                    while 0 <= c < width and abs(c - index) > 1:
                        out[row, c] = marker
                        c += direction
    return out


_FAMILY_APPLY = {
    "dihedral": op_dihedral,
    "colour_map": op_colour_map,
    "crop_bbox": op_crop_bbox,
    "crop_colour": op_crop_colour,
    "scale": op_scale,
    "tile": op_tile,
    "recolour_objects": op_recolour_objects,
    "recolour_by_property": op_recolour_by_property,
    "delete_objects": op_delete_objects,
    "fill_holes": op_fill_holes,
    "stamp": op_stamp,
    "rays": op_rays,
    "slide": op_slide,
    "symmetry_completion": op_symmetry_completion,
    "overlay_panels": op_overlay_panels,
    "cross_center": op_cross_center,
    "fill_interior": op_fill_interior,
    "local_rule": op_local_rule,
    "constant": op_constant,
    # Cycle AC3 families
    "mirror_tile": op_mirror_tile,
    "kron_fractal": op_kron_fractal,
    "downscale_blocks": op_downscale_blocks,
    "crop_rank": op_crop_rank,
    "mesh_template": op_mesh_template,
    "panel_op": op_panel_op,
    "corner_ray": op_corner_ray,
    "glyph_rays": op_glyph_rays,
    "ring_fill": op_ring_fill,
    "delete_isolated_colour": op_delete_isolated_colour,
    "shortest_path": op_shortest_path,
    "ray_bridge": op_ray_bridge,
    "cavity_object_fill": op_cavity_object_fill,
    "gravity_colour": op_gravity_colour,
    "mesh_select_fill": op_mesh_select_fill,
    "fill_holes_by_size": op_fill_holes_by_size,
    "pad_edges": op_pad_edges,
    "swatch": op_swatch,
    # Cycle AD2 relational families
    "swap_object_palette": op_swap_object_palette,
    "concentric_squares": op_concentric_squares,
    "fill_hole_lattice": op_fill_hole_lattice,
    "border_line_completion": op_border_line_completion,
    "extend_objects": op_extend_objects,
    "extend_to_wall": op_extend_to_wall,
    "marker_wall_impact": op_marker_wall_impact,
}


@dataclass(frozen=True)
class ArcOp:
    family: str
    params: Tuple[object, ...] = ()

    def label(self) -> str:
        if not self.params:
            return str(self.family)
        return "%s(%s)" % (self.family, ",".join(str(value) for value in self.params))

    def apply(self, grid: np.ndarray) -> Optional[np.ndarray]:
        runner = _FAMILY_APPLY.get(str(self.family))
        if runner is None:
            return None
        try:
            value = runner(as_grid(grid), *self.params)
        except Exception:
            return None
        if value is None:
            return None
        value = np.asarray(value, dtype=np.int64)
        if value.ndim != 2 or value.size == 0:
            return None
        if int(value.min()) < 0 or int(value.max()) >= COLOURS:
            return None
        if value.shape[0] > MAX_SIDE * 6 or value.shape[1] > MAX_SIDE * 6:
            return None
        return np.ascontiguousarray(value)


def apply_chain(ops: Sequence[ArcOp], grid: np.ndarray) -> Optional[np.ndarray]:
    value: Optional[np.ndarray] = as_grid(grid)
    for op in ops:
        if value is None:
            return None
        value = op.apply(value)
    return value


# ======================================================================
# Proposal from pairs (the LOO search uses only the K-1 pairs it is given)
# ======================================================================
def _joint_colour_map(pairs: Sequence[Tuple[np.ndarray, np.ndarray]]) -> Optional[ArcOp]:
    lut = list(range(COLOURS))
    for src, dst in pairs:
        src, dst = as_grid(src), as_grid(dst)
        if src.shape != dst.shape:
            return None
        for colour in np.unique(src):
            values = np.unique(dst[src == int(colour)])
            if values.size != 1:
                return None
            target = int(values[0])
            if lut[int(colour)] != int(colour) and lut[int(colour)] != target:
                return None
            lut[int(colour)] = target
    return ArcOp("colour_map", tuple(int(value) for value in lut))


def _recolour_proposals(pairs: Sequence[Tuple[np.ndarray, np.ndarray]]) -> List[ArcOp]:
    out: List[ArcOp] = []
    for rule in ("mass_rank", "mass_rank_asc", "holes", "colour_rank", "left_to_right", "top_to_bottom"):
        ranks: Dict[int, int] = {}
        consistent = True
        max_rank = 0
        for src, dst in pairs:
            src, dst = as_grid(src), as_grid(dst)
            if src.shape != dst.shape:
                consistent = False
                break
            background = background_colour(src)
            objects = _sort_objects(extract_objects(src, background, connectivity=4, mono=True), rule)
            for index, obj in enumerate(objects):
                values = np.unique(dst[[row for row, _ in obj.cells], [col for _, col in obj.cells]])
                if values.size != 1:
                    consistent = False
                    break
                target = int(values[0])
                if index in ranks and ranks[index] != target:
                    consistent = False
                    break
                ranks[index] = target
                max_rank = max(max_rank, index)
            if not consistent:
                break
        if consistent and ranks:
            out.append(ArcOp("recolour_objects", (rule, tuple(ranks[index] for index in range(max_rank + 1)))))
    return out


def _property_proposals(pairs: Sequence[Tuple[np.ndarray, np.ndarray]]) -> List[ArcOp]:
    """Threshold-sweep recolouring proposals, verified on the given pairs.

    A threshold read off one object does not transfer across folds (the held
    pair's object may have a different mass).  The sweep names every
    threshold, so the fold intersection can agree on the one that is actually
    constant across the demonstrations.
    """
    out: List[ArcOp] = []
    if not pairs or any(as_grid(a).shape != as_grid(b).shape for a, b in pairs):
        return out
    masses = set()
    holes = set()
    colours = set()
    for src, dst in pairs:
        src = as_grid(src)
        background = background_colour(src)
        for obj in extract_objects(src, background, connectivity=4, mono=True):
            masses.add(int(obj.mass))
            holes.add(int(obj.holes))
        colours.update(int(value) for value in np.unique(src))
        colours.update(int(value) for value in np.unique(as_grid(dst)))
    thresholds = sorted(masses)[:8] + [2, 3, 4, 5]
    for rule in ("mass_ge", "mass_le"):
        for threshold in sorted(set(thresholds)):
            for colour in sorted(colours):
                op = ArcOp("recolour_by_property", (rule, int(threshold), int(colour)))
                if all(
                    (lambda value: value is not None and value.shape == as_grid(dst).shape and np.array_equal(value, as_grid(dst)))(
                        op.apply(as_grid(src))
                    )
                    for src, dst in pairs
                ):
                    out.append(op)
    for rule in ("holes_ge", "holes_eq"):
        for threshold in sorted(set(holes) | {0, 1, 2}):
            for colour in sorted(colours):
                op = ArcOp("recolour_by_property", (rule, int(threshold), int(colour)))
                if all(
                    (lambda value: value is not None and value.shape == as_grid(dst).shape and np.array_equal(value, as_grid(dst)))(
                        op.apply(as_grid(src))
                    )
                    for src, dst in pairs
                ):
                    out.append(op)
    return out


def _stamp_proposal(pairs: Sequence[Tuple[np.ndarray, np.ndarray]]) -> Optional[ArcOp]:
    """One template per marker colour, learned from the input-output diff."""
    per_marker: Dict[int, set] = {}
    for src, dst in pairs:
        src, dst = as_grid(src), as_grid(dst)
        if src.shape != dst.shape:
            return None
        background = background_colour(src)
        changed = dst != src
        rows, cols = np.nonzero(changed)
        if rows.size == 0:
            return None
        markers = np.argwhere(src != int(background))
        if markers.size == 0:
            return None
        for row, col in zip(rows.tolist(), cols.tolist()):
            distances = np.abs(markers[:, 0] - row) + np.abs(markers[:, 1] - col)
            nearest = int(np.argmin(distances))
            distance = int(distances[nearest])
            if distance > 2:
                return None
            mr, mc = int(markers[nearest][0]), int(markers[nearest][1])
            marker_colour = int(src[mr, mc])
            if marker_colour == int(background):
                return None
            offset = (row - mr, col - mc, int(dst[row, col]))
            per_marker.setdefault(marker_colour, set()).add(offset)
    if not per_marker:
        return None
    mapping = tuple(
        (int(colour), tuple(sorted(per_marker[colour]))) for colour in sorted(per_marker)
    )
    return ArcOp("stamp", (mapping,))


def _ray_proposals(pairs: Sequence[Tuple[np.ndarray, np.ndarray]]) -> List[ArcOp]:
    out: List[ArcOp] = []
    directions = [
        ((1, 0),), ((-1, 0),), ((0, 1),), ((0, -1),),
        ((1, 1), (-1, -1)), ((1, -1), (-1, 1)),
        ((1, 0), (-1, 0)), ((0, 1), (0, -1)),
    ]
    marker_values = sorted(
        set(int(value) for src, _dst in pairs for value in np.unique(as_grid(src)) if int(value) != BACKGROUND)
    )
    for marker in marker_values[:6]:
        for direction in directions:
            out.append(ArcOp("rays", ((marker,), direction, marker)))
    return out


def _dimension_ops(pairs: Sequence[Tuple[np.ndarray, np.ndarray]]) -> List[ArcOp]:
    out: List[ArcOp] = []
    for hypothesis in induce_dimensions(pairs):
        if hypothesis["kind"] in ("same_shape", "fixed_shape", "crop", "crop_hole", "panel"):
            # Dimension-only hypotheses are carried by the content operators.
            continue
        if hypothesis["kind"] == "scale":
            out.append(ArcOp("scale", (int(hypothesis["params"][0]),)))
        if hypothesis["kind"] == "tile":
            out.append(ArcOp("tile", (int(hypothesis["params"][0]), int(hypothesis["params"][1]))))
        if hypothesis["kind"] == "mirror_tile":
            for mode in ("rows", "flip", "rot180"):
                out.append(
                    ArcOp(
                        "mirror_tile",
                        (
                            int(hypothesis["params"][0]),
                            int(hypothesis["params"][1]),
                            str(mode),
                        ),
                    )
                )
        if hypothesis["kind"] == "kron_fractal":
            out.append(
                ArcOp(
                    "kron_fractal",
                    (int(hypothesis["params"][0]), int(hypothesis["params"][1])),
                )
            )
        if hypothesis["kind"] == "pad_edges":
            out.append(ArcOp("pad_edges", (int(hypothesis["params"][0]), 0)))
    return out


def _ring_cycles(pairs: Sequence[Tuple[np.ndarray, np.ndarray]]) -> List[Tuple[int, Tuple[int, ...]]]:
    """Periodic ring cycles that every given pair exhibits, shortest period first."""
    observed: Dict[int, int] = {}
    for src, dst in pairs:
        src, dst = as_grid(src), as_grid(dst)
        if src.shape != dst.shape:
            return []
        box = _ring_box(src)
        if box is None:
            return []
        r0, c0, r1, c1 = box
        if r1 - r0 < 2 or c1 - c0 < 2:
            return []
        for row in range(r0, r1 + 1):
            for col in range(c0, c1 + 1):
                distance = min(row - r0, r1 - row, col - c0, c1 - col)
                colour = int(dst[row, col])
                if distance in observed and observed[distance] != colour:
                    return []
                observed[distance] = colour
    if not observed:
        return []
    out: List[Tuple[int, Tuple[int, ...]]] = []
    for period in range(1, 7):
        cycle: Dict[int, int] = {}
        ok = True
        for distance in sorted(observed):
            slot = distance % period
            if slot in cycle and cycle[slot] != observed[distance]:
                ok = False
                break
            cycle[slot] = observed[distance]
        if ok and len(cycle) == period:
            out.append((period, tuple(int(cycle[slot]) for slot in range(period))))
    return out


def _ac3_proposals(pairs: Sequence[Tuple[np.ndarray, np.ndarray]]) -> List[ArcOp]:
    """Cycle AC3 candidate morphisms, all read off the pairs they are given."""
    out: List[ArcOp] = []
    if not pairs:
        return out
    grids = [(as_grid(src), as_grid(dst)) for src, dst in pairs]
    palette: List[int] = []
    for src, dst in grids:
        for grid in (src, dst):
            for value in np.unique(grid):
                if int(value) not in palette:
                    palette.append(int(value))
    palette.sort()
    same_shape = all(src.shape == dst.shape for src, dst in grids)

    out.append(ArcOp("corner_ray", ()))
    out.append(ArcOp("mesh_template", ()))
    for rule in ("most_marks", "fewest_marks"):
        out.append(ArcOp("mesh_select_fill", (str(rule),)))
    for mode in _PANEL_MODES:
        for transform in _PANEL_TRANSFORMS:
            out.append(ArcOp("panel_op", (str(mode), str(transform))))
    if same_shape:
        out.append(ArcOp("delete_isolated_colour", ()))
        out.append(ArcOp("cavity_object_fill", ()))
        out.append(ArcOp("swap_object_palette", ()))
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            out.append(ArcOp("extend_objects", (int(dr), int(dc))))
        out.append(ArcOp("extend_to_wall", ()))
        background_value = background_colour(grids[0][0])
        wall_source = {int(value) for src, _dst in grids for value in np.unique(src)}
        for wall_colour in sorted(wall_source):
            if int(wall_colour) == int(background_value):
                continue
            if not all(_full_lines(src, int(wall_colour)) for src, _dst in grids):
                continue
            for marker_colour in sorted(wall_source):
                if int(marker_colour) == int(wall_colour):
                    continue
                if int(marker_colour) == int(background_value):
                    continue
                out.append(
                    ArcOp("marker_wall_impact", (int(marker_colour), int(wall_colour)))
                )
        source_colours = {
            int(value) for src, _dst in grids for value in np.unique(src)
        }
        target_colours = {
            int(value) for _src, dst in grids for value in np.unique(dst)
        }
        new_colours = sorted(target_colours - source_colours)
        for outer_colour in new_colours:
            for inner_colour in new_colours:
                if int(inner_colour) == int(outer_colour):
                    continue
                out.append(
                    ArcOp("fill_hole_lattice", (int(outer_colour), int(inner_colour)))
                )
        if all(_border_line(src) is not None for src, _dst in grids):
            for diagonal_colour in new_colours:
                for row_colour in new_colours:
                    if int(diagonal_colour) == int(row_colour):
                        continue
                    out.append(
                        ArcOp(
                            "border_line_completion",
                            (int(diagonal_colour), int(row_colour)),
                        )
                    )
        for frame_colour in sorted(source_colours):
            if not any(_hollow_square_frame(src, frame_colour) for src, _dst in grids):
                continue
            for ring_colour in sorted(target_colours - source_colours):
                if int(ring_colour) == int(frame_colour):
                    continue
                for step in (4,):
                    out.append(
                        ArcOp(
                            "concentric_squares",
                            (int(frame_colour), int(ring_colour), int(step), 2),
                        )
                    )
        for size in (3, 5):
            out.append(ArcOp("glyph_rays", (int(size),)))
        for period, cycle in _ring_cycles(grids):
            out.append(ArcOp("ring_fill", (int(period),) + tuple(int(item) for item in cycle)))
        for colour in palette:
            out.append(ArcOp("shortest_path", (int(colour),)))
        for colour in palette:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                out.append(ArcOp("gravity_colour", (int(colour), int(dr), int(dc))))
        for axes in ("h", "v", "d", "hv", "hd", "vd", "hvd"):
            for colour in palette:
                out.append(ArcOp("ray_bridge", (str(axes), int(colour))))
        for small in palette:
            for big in palette:
                if int(small) == int(big):
                    continue
                out.append(ArcOp("fill_holes_by_size", (int(small), int(big))))

    for rule in (
        "non_background",
        "largest",
        "smallest",
        "most_holes",
        "least_holes",
        "unique_colour",
        "unique_shape",
        "most_symmetric",
        "framed",
    ):
        out.append(ArcOp("crop_rank", (str(rule),)))

    shapes = {tuple(int(item) for item in dst.shape) for _src, dst in grids}
    if len(shapes) == 1:
        only = next(iter(shapes))
        for rule in ("largest", "smallest", "most_holes"):
            out.append(ArcOp("swatch", (int(only[0]), int(only[1]), str(rule))))

    factors = set()
    for src, dst in grids:
        if src.shape[0] and src.shape[1] and src.shape != dst.shape:
            if dst.shape[0] % src.shape[0] == 0 and dst.shape[1] % src.shape[1] == 0:
                factors.add((dst.shape[0] // src.shape[0], dst.shape[1] // src.shape[1]))
            if src.shape[0] % dst.shape[0] == 0 and src.shape[1] % dst.shape[1] == 0:
                shrink = (src.shape[0] // dst.shape[0], src.shape[1] // dst.shape[1])
                if shrink != (1, 1):
                    for mode in ("block", "majority"):
                        out.append(ArcOp("downscale_blocks", (int(shrink[0]), str(mode))))
    pads = {(dst.shape[0] - src.shape[0], dst.shape[1] - src.shape[1]) for src, dst in grids}
    if len(pads) == 1:
        pad_r, pad_c = next(iter(pads))
        if pad_r == pad_c and pad_r > 0 and pad_r % 2 == 0:
            corners = sorted({int(value) for _src, dst in grids for value in np.unique(dst)})
            for corner in corners:
                out.append(ArcOp("pad_edges", (int(pad_r // 2), int(corner))))
    return out


def propose_operators(
    pairs: Sequence[Tuple[np.ndarray, np.ndarray]]
) -> List[ArcOp]:
    """Every candidate the K-1 demonstration pairs support, deterministically."""
    cached = _PROPOSAL_CACHE.get(_pair_key(pairs))
    if cached is not None:
        return list(cached)
    out: List[ArcOp] = []
    seen: set = set()

    def push(op: Optional[ArcOp]) -> None:
        if op is None:
            return
        key = op.label()
        if key in seen:
            return
        seen.add(key)
        out.append(op)

    for index in range(8):
        push(ArcOp("dihedral", (int(index),)))
    push(_joint_colour_map(pairs))
    push(_stamp_proposal(pairs))
    if pairs:
        push(ArcOp("crop_bbox", ()))
        first = as_grid(pairs[0][0])
        for colour in sorted(set(int(value) for value in np.unique(first))):
            push(ArcOp("crop_colour", (int(colour),)))
        for rule in ("largest", "smallest", "unique_mass", "unique_shape", "unique_colour"):
            push(ArcOp("delete_objects", (str(rule),)))
        target_palette = sorted(
            set(int(value) for value in np.unique(first))
            | set(int(value) for value in np.unique(as_grid(pairs[0][1])))
        )
        for colour in target_palette:
            push(ArcOp("fill_holes", (int(colour),)))
            push(ArcOp("fill_interior", (int(colour),)))
        palette = target_palette
        background = background_colour(first)
        for ring in palette:
            for line in palette:
                if int(line) == int(ring) or int(line) == int(background):
                    continue
                push(ArcOp("cross_center", (int(ring), int(line))))
        for axis in ("vertical", "horizontal", "both"):
            push(ArcOp("symmetry_completion", (axis,)))
        for mode in ("and", "or", "xor", "majority"):
            push(ArcOp("overlay_panels", (mode,)))
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            push(ArcOp("slide", (int(dr), int(dc))))
        for colour in sorted(set(int(value) for value in np.unique(first))):
            push(ArcOp("constant", (int(colour),)))
    for op in _dimension_ops(pairs):
        push(op)
    for op in _recolour_proposals(pairs):
        push(op)
    for op in _property_proposals(pairs):
        push(op)
    for op in _ray_proposals(pairs):
        push(op)
    for op in _local_rule_proposals(pairs):
        push(op)
    for op in _ac3_proposals(pairs):
        push(op)
    signature = wave_signature(pairs)
    ordered = route_operators(out, signature)
    if len(_PROPOSAL_CACHE) > 4096:
        _PROPOSAL_CACHE.clear()
    _PROPOSAL_CACHE[_pair_key(pairs)] = list(ordered)
    return ordered


def _pair_key(pairs: Sequence[Tuple[np.ndarray, np.ndarray]]) -> Tuple[Any, ...]:
    key: List[Any] = []
    for src, dst in pairs:
        key.append(_signature_key(as_grid(src)))
        key.append(_signature_key(as_grid(dst)))
    return tuple(key)


_PROPOSAL_CACHE: Dict[Tuple[Any, ...], List[ArcOp]] = {}


def _local_rule_proposals(pairs: Sequence[Tuple[np.ndarray, np.ndarray]]) -> List[ArcOp]:
    out: List[ArcOp] = []
    for radius in (1, 2, 3):
        table: Dict[str, int] = {}
        consistent = True
        for src, dst in pairs:
            src, dst = as_grid(src), as_grid(dst)
            if src.shape != dst.shape:
                consistent = False
                break
            background = background_colour(src)
            padded = np.pad(src, radius, mode="constant", constant_values=background)
            for row in range(src.shape[0]):
                for col in range(src.shape[1]):
                    patch = padded[row : row + 2 * radius + 1, col : col + 2 * radius + 1]
                    key = patch.tobytes()
                    target = int(dst[row, col])
                    if key in table and table[key] != target:
                        consistent = False
                        break
                    table[key] = target
                if not consistent:
                    break
            if not consistent:
                break
        if consistent and table:
            out.append(ArcOp("local_rule", (dict(table), int(radius))))
    return out


# ======================================================================
# Strict Leave-One-Out inducer
# ======================================================================
class ObjectLooInducer:
    """Depth-1/2 object-program search under a strict Leave-One-Out gate.

    A program is admitted only when, for EVERY fold k, a parameter setting
    derived from the other K-1 pairs reproduces pair k exactly.  The test
    pairs are read once, after the program is admitted, to score it.
    """

    def __init__(
        self,
        node_budget: int = 250000,
        head_cap: int = 16,
        depth: int = 2,
        time_budget: float = 20.0,
    ) -> None:
        self.node_budget = int(node_budget)
        self.head_cap = int(head_cap)
        self.depth = int(depth)
        self.time_budget = float(time_budget)
        self._deadline = 0.0

    def _expired(self) -> bool:
        import time

        return bool(self._deadline and time.perf_counter() > self._deadline)

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _matches(op: ArcOp, pairs: Sequence[Tuple[np.ndarray, np.ndarray]]) -> bool:
        for src, dst in pairs:
            value = op.apply(as_grid(src))
            target = as_grid(dst)
            if value is None or value.shape != target.shape or not np.array_equal(value, target):
                return False
        return True

    def _fold_keys(
        self, pairs: Sequence[Tuple[np.ndarray, np.ndarray]], state: Dict[str, int]
    ) -> Optional[set]:
        """Program keys that transfer to every held-out fold (depth 1)."""
        folds = len(pairs)
        admitted: Optional[set] = None
        for fold in range(folds):
            train = [pair for index, pair in enumerate(pairs) if index != fold]
            held = [pairs[fold]]
            keys = set()
            for op in propose_operators(train):
                if int(state["nodes"]) >= self.node_budget or self._expired():
                    state["exhausted"] = 1
                    return None
                state["nodes"] = int(state["nodes"]) + 1
                if self._matches(op, held):
                    keys.add(op.label())
            admitted = keys if admitted is None else (admitted & keys)
            if not admitted:
                return set()
        return admitted or set()

    def _depth_two(
        self,
        pairs: Sequence[Tuple[np.ndarray, np.ndarray]],
        state: Dict[str, int],
    ) -> List[Tuple[ArcOp, ArcOp]]:
        """Depth-2 residual composition.

        ``T_1`` must strictly reduce the total Hamming mismatch over the K
        pairs; ``T_2`` is then searched on ``(T_1(X_k), Y_k)`` and must be
        exact on all K pairs.  Nothing is returned unless both stages hold,
        so a returned composition is a candidate for the leave-one-out
        confirmation that ``induce`` runs afterwards.
        """
        return self._residual_search(pairs, state, None, None)

    @staticmethod
    def _mismatch(value: Optional[np.ndarray], target: np.ndarray) -> int:
        """Cell mismatch, with a shape penalty so a wrong canvas cannot look cheap."""
        if value is None:
            return int(target.size) + 1
        if value.shape != target.shape:
            return (
                int(target.size)
                + int(abs(value.shape[0] - target.shape[0]))
                + int(abs(value.shape[1] - target.shape[1]))
                + 1
            )
        return int(np.count_nonzero(value != target))

    def _residual_search(
        self,
        pairs: Sequence[Tuple[np.ndarray, np.ndarray]],
        state: Optional[Dict[str, int]],
        head_family: Optional[str],
        second_family: Optional[str],
    ) -> List[Tuple[ArcOp, ArcOp]]:
        index = [(as_grid(src), as_grid(dst)) for src, dst in pairs]
        total0 = sum(self._mismatch(src, dst) for src, dst in index)
        heads: List[Tuple[int, ArcOp, List[np.ndarray]]] = []
        for head in propose_operators(pairs):
            if head_family is not None and head.family != head_family:
                continue
            if state is not None:
                if int(state["nodes"]) >= self.node_budget or self._expired():
                    state["exhausted"] = 1
                    return []
                state["nodes"] = int(state["nodes"]) + 1
            middles: List[np.ndarray] = []
            usable = True
            for src, _dst in index:
                middle = head.apply(src)
                if middle is None:
                    usable = False
                    break
                middles.append(middle)
            if not usable:
                continue
            total1 = sum(
                self._mismatch(middle, dst) for middle, (_src, dst) in zip(middles, index)
            )
            if total1 >= total0:
                continue
            heads.append((total1, head, middles))
        heads.sort(key=lambda entry: (entry[0], entry[1].label()))
        found: List[Tuple[ArcOp, ArcOp]] = []
        for total1, head, middles in heads[: max(int(self.head_cap), 1)]:
            targets = [dst for _src, dst in index]
            for second in propose_operators(list(zip(middles, targets))):
                if second_family is not None and second.family != second_family:
                    continue
                if state is not None:
                    if int(state["nodes"]) >= self.node_budget or self._expired():
                        state["exhausted"] = 1
                        return found
                    state["nodes"] = int(state["nodes"]) + 1
                exact = True
                for middle, target in zip(middles, targets):
                    value = second.apply(middle)
                    if (
                        value is None
                        or value.shape != target.shape
                        or not np.array_equal(value, target)
                    ):
                        exact = False
                        break
                if exact:
                    found.append((head, second))
                    break
            if len(found) >= 4:
                break
        return found

    # ------------------------------------------------------------- induce
    def induce(self, examples: Sequence[Tuple[np.ndarray, np.ndarray]], tests: Sequence[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
        import time

        self._deadline = time.perf_counter() + float(self.time_budget)
        pairs = [(as_grid(src), as_grid(dst)) for src, dst in examples]
        state: Dict[str, int] = {"nodes": 0, "exhausted": 0}
        result: Dict[str, Any] = {
            "status": "ABSTAIN",
            "reason": "",
            "chain": [],
            "nodes": 0,
            "folds": int(len(pairs)),
            "loo_passed": False,
            "test_exact": 0,
            "test_total": int(len(tests)),
            "test_rows": [],
            "emitted": False,
            "solved": False,
            "budget_exhausted": False,
        }
        if len(pairs) < 2:
            result["reason"] = "ABSTAIN: fewer than two demonstration pairs"
            return result
        keys = self._fold_keys(pairs, state)
        chosen: Optional[List[ArcOp]] = None
        if keys:
            # Re-propose from all pairs and select an admitted key.
            for op in propose_operators(pairs):
                if op.label() in keys and self._matches(op, pairs):
                    chosen = [op]
                    break
        if chosen is None and self.depth >= 2:
            for head, second in self._depth_two(pairs, state):
                if not self._matches_chain(head, second, pairs):
                    continue
                if not self._loo_chain(pairs, head, second):
                    continue
                chosen = [head, second]
                break
        result["nodes"] = int(state["nodes"])
        result["budget_exhausted"] = bool(state["exhausted"])
        if chosen is None:
            result["reason"] = (
                "ABSTAIN: node budget exhausted"
                if state["exhausted"]
                else "ABSTAIN: no object program survives the Leave-One-Out gate"
            )
            return result
        # Strict LOO confirmation: re-derive from K-1 and score the held pair.
        loo_ok = True
        for fold in range(len(pairs)):
            train = [pair for index, pair in enumerate(pairs) if index != fold]
            derived = self._rederive(train, chosen, pairs[fold])
            if not derived:
                loo_ok = False
                break
        result["loo_passed"] = bool(loo_ok)
        if not loo_ok:
            result["reason"] = "ABSTAIN: Leave-One-Out confirmation failed"
            return result
        result["status"] = "INDUCED"
        result["chain"] = [op.label() for op in chosen]
        result["emitted"] = True
        exact = 0
        rows = []
        for src, dst in tests:
            value = apply_chain(chosen, as_grid(src))
            target = as_grid(dst)
            ok = bool(value is not None and value.shape == target.shape and np.array_equal(value, target))
            exact += int(ok)
            rows.append(
                {
                    "input_shape": [int(src.shape[0]), int(src.shape[1])],
                    "expected_shape": [int(target.shape[0]), int(target.shape[1])],
                    "predicted_shape": None if value is None else [int(value.shape[0]), int(value.shape[1])],
                    "exact": bool(ok),
                }
            )
        result["test_rows"] = rows
        result["test_exact"] = int(exact)
        result["solved"] = bool(rows and exact == len(rows))
        return result

    def _matches_chain(self, head: ArcOp, second: ArcOp, pairs) -> bool:
        for src, dst in pairs:
            middle = head.apply(src)
            if middle is None:
                return False
            value = second.apply(middle)
            target = as_grid(dst)
            if value is None or value.shape != target.shape or not np.array_equal(value, target):
                return False
        return True

    def _loo_chain(self, pairs, head: ArcOp, second: ArcOp) -> bool:
        """Depth-2 leave-one-out: every fold re-derives the same family pair."""
        for fold in range(len(pairs)):
            train = [pair for index, pair in enumerate(pairs) if index != fold]
            if not self._rederive(train, [head, second], pairs[fold]):
                return False
        return True

    def _rederive(self, train, chosen, held) -> bool:
        """Re-derive the chosen program family from K-1 pairs and score the held pair."""
        src, dst = as_grid(held[0]), as_grid(held[1])
        if len(chosen) == 1:
            family = chosen[0].family
            for op in propose_operators(train):
                if op.family != family:
                    continue
                value = op.apply(src)
                if value is not None and value.shape == dst.shape and np.array_equal(value, dst):
                    return True
            return False
        head_family = chosen[0].family
        second_family = chosen[1].family
        for head, second in self._residual_search(
            train, None, head_family, second_family
        ):
            middle = head.apply(src)
            if middle is None:
                continue
            value = second.apply(middle)
            if value is not None and value.shape == dst.shape and np.array_equal(value, dst):
                return True
        return False
