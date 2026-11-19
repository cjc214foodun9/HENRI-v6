"""Spatial wave induction (Project HENRI Cycle Z, Stage Z3).

The one-cell inducer in henri.holarch.induction reasons over a tape and a head.
This module lifts the same three mechanisms to two dimensions: a wave encoding
with an exact inverse, an invariant router that ranks operator sketches from
the demonstration pairs alone, and a CEGIS search over composable grid
operators with a single node budget and REPULSOR pruning.

Encoding
--------
A grid G in {0..9}^{H x W} becomes one complex wave of dimension D = 2048 laid
out as four contiguous slots of D/4 = 512 cells.  Each slot carries the same
fixed orthogonal toroidal Fourier basis

    phi_m(r, c) = exp(2*pi*i*(a_m*r/H + b_m*c/W)) / sqrt(H*W),
    (a_m, b_m) = (m // W, m % W)

and every cell contributes one unit wave per slot.  Two unit phase factors
modulate the coefficient:

    colour phase   v_s(g) = exp(2*pi*i*g*s/10)           (per colour, per slot)
    cell phase     p_s(m) = exp(2*pi*i*m*(s+1)/(4*H*W))  (per cell, per slot)

so that

    psi = sum_{s=0..3} sum_{m=0..H*W-1} 0.5 * v_s(G_m) * p_s(m) * phi_{s,m} e_s

is a finite sum of 4*H*W unit waves.  Because the phi_{s,m} are orthonormal,
the coefficient field is exactly the 2-D DFT of the slot field and the
synthesis/analysis pair is an isometry on the frame span (E^dagger E = I).
The colour is read back from slot 1 by
g = round(angle(v_1) * 10 / 2*pi) mod 10 and slots 2 and 3 verify the read.
The colour map is injective on the ten colours, so the grid round-trip is
exact: max |G - decode(encode(G))| = 0.

Router
------
WaveDeltaSpatialRouter computes a Wave-Delta signature from the 3-4 training
pairs only.  It has exactly the declared components: the shape and bounding-box
delta, the D4 dihedral cross-spectral resonance of the eight dihedral elements,
the 2-D Fourier translation phase peak modulo (H, W), and the per-colour mass
plus connected-component Euler/containment signature.  The signature scores
every operator sketch and returns one ranked list.  No sketch is selected by a
task name, and the routing path holds no task-name literal.

The two geometric invariants are read from the colour-phase field that the
analysis recovers: dividing out the two unit factors leaves exp(2*pi*i*G/10) per
cell, on which a grid permutation acts as a pure permutation.  The translation
peak is the cross-spectrum of that field and the D4 resonance is its normalised
overlap with each dihedral image, so a shift and a dihedral element both report
exactly 1.0 when they are the whole rule.

Search
------
SpatialCegisInducer proposes concrete operators from the training pairs
(parameters are derived from the wave invariants, never from a task name),
verifies them against every training pair in the router's ranked order, records
refuted hypotheses as REPULSOR axioms, and accepts the first consistent 1-stage
or 2-stage chain.  The held-out pairs are never consulted during the search;
the caller measures generalisation afterwards.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

DIM = 2048
N_SLOTS = 4
SLOT_WIDTH = DIM // N_SLOTS
MAX_CELLS = SLOT_WIDTH
BACKGROUND = 0
COLOURS = 10
_EPS = 1e-12

D4_ELEMENTS = (
    "identity",
    "rot90",
    "rot180",
    "rot270",
    "reflect_h",
    "reflect_v",
    "reflect_diag",
    "reflect_antidiag",
)

SPATIAL_SKETCHES = (
    "dihedral",
    "toroidal_shift",
    "bounded_shift",
    "colour_map",
    "crop_bbox",
    "tile",
    "scale",
    "interior_flood_fill",
    "mask_by_colour",
    "symmetry_completion",
    "constant_fill",
)
"""The eleven declared operator sketches.  Order is the tie-break order."""

SKETCH_LABELS = {
    "dihedral": "D4 dihedral transform",
    "toroidal_shift": "toroidal shift (wrap)",
    "bounded_shift": "bounded shift (constant fill)",
    "colour_map": "colour permutation / mapping",
    "crop_bbox": "bounding-box crop",
    "tile": "tile / replicate",
    "scale": "nearest-neighbour scale",
    "interior_flood_fill": "interior flood fill",
    "mask_by_colour": "mask by colour",
    "symmetry_completion": "symmetry completion",
    "constant_fill": "constant fill",
}

DECLARED_SHAPE_RULE = "output_shape_equals_input_shape"
"""The one declared shape facet a spec may carry as data."""


# --------------------------------------------------------------- grid tools
def as_grid(value: object) -> np.ndarray:
    """Validate one grid: a 2-D integer array with colours in 0..9."""
    grid = np.asarray(value)
    if grid.ndim != 2:
        raise ValueError("a grid must be two-dimensional")
    grid = grid.astype(np.int64, copy=True)
    if grid.size == 0:
        raise ValueError("a grid must not be empty")
    if int(grid.min()) < 0 or int(grid.max()) >= COLOURS:
        raise ValueError("grid colours must lie in 0..9")
    return grid


def palette(grid: np.ndarray) -> Tuple[int, ...]:
    """Every colour present in one grid, sorted."""
    return tuple(int(value) for value in np.unique(np.asarray(grid, dtype=np.int64)))


def foreground_palette(grid: np.ndarray) -> Tuple[int, ...]:
    """Every non-background colour present in one grid, sorted."""
    return tuple(value for value in palette(grid) if value != BACKGROUND)


def bounding_box(grid: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """(r0, c0, r1, c1) inclusive bounding box of the non-background cells."""
    mask = np.asarray(grid, dtype=np.int64) != BACKGROUND
    if not bool(mask.any()):
        return None
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return int(rows[0]), int(cols[0]), int(rows[-1]), int(cols[-1])


def _regions(mask: np.ndarray) -> List[List[Tuple[int, int]]]:
    """Four-connected regions of a boolean mask, in scan order."""
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    regions: List[List[Tuple[int, int]]] = []
    for row in range(height):
        for col in range(width):
            if not bool(mask[row, col]) or bool(seen[row, col]):
                continue
            stack = [(row, col)]
            seen[row, col] = True
            cells: List[Tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width and bool(mask[nr, nc]) \
                            and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            regions.append(cells)
    return regions


def component_signature(grid: np.ndarray) -> Dict[int, Dict[str, int]]:
    """Per-colour component count, enclosed-hole count and Euler number."""
    grid = np.asarray(grid, dtype=np.int64)
    height, width = grid.shape
    out: Dict[int, Dict[str, int]] = {}
    for colour in palette(grid):
        mask = grid == int(colour)
        regions = _regions(mask)
        holes = 0
        for region in _regions(~mask):
            touches = any(
                row == 0 or row == height - 1 or col == 0 or col == width - 1
                for row, col in region
            )
            holes += int(not touches)
        out[int(colour)] = {
            "components": int(len(regions)),
            "holes": int(holes),
            "euler": int(len(regions) - holes),
            "mass": int(mask.sum()),
        }
    return out


def _contains(outer: Sequence[int], inner: Sequence[int]) -> bool:
    """Whether the inclusive box 'outer' strictly contains 'inner'."""
    return bool(outer[0] <= inner[0] and outer[1] <= inner[1]
                and outer[2] >= inner[2] and outer[3] >= inner[3]
                and tuple(outer) != tuple(inner))


def containment_signature(grid: np.ndarray) -> List[Tuple[int, int]]:
    """Pairs (outer, inner) where the outer colour's box contains the inner's."""
    grid = np.asarray(grid, dtype=np.int64)
    boxes: Dict[int, Tuple[int, int, int, int]] = {}
    for colour in foreground_palette(grid):
        box = bounding_box(np.where(grid == int(colour), int(colour), BACKGROUND))
        if box is not None:
            boxes[int(colour)] = box
    pairs: List[Tuple[int, int]] = []
    for outer in sorted(boxes):
        for inner in sorted(boxes):
            if outer != inner and _contains(boxes[outer], boxes[inner]):
                pairs.append((int(outer), int(inner)))
    return pairs


# ------------------------------------------------------------------ encoder
class ToroidalWaveEncoder:
    """Four-slot toroidal complex wave encoding with an exact inverse."""

    def __init__(self, dim: int = DIM, n_slots: int = N_SLOTS, colours: int = COLOURS) -> None:
        if int(dim) % int(n_slots) != 0:
            raise ValueError("the wave dimension must split evenly into slots")
        self.dim = int(dim)
        self.n_slots = int(n_slots)
        self.width = self.dim // self.n_slots
        self.colours = int(colours)
        self.calls = 0

    # ------------------------------------------------------------- geometry
    def check_shape(self, shape: Sequence[int]) -> Tuple[int, int]:
        height, width = int(shape[0]), int(shape[1])
        if height < 1 or width < 1:
            raise ValueError("an empty grid has no wave")
        if height * width > self.width:
            raise ValueError(
                "a %dx%d grid needs %d cells but one slot holds %d"
                % (height, width, height * width, self.width)
            )
        return height, width

    def cell_phase(self, shape: Sequence[int], slot: int) -> np.ndarray:
        """The per-cell unit phase of one slot."""
        height, width = self.check_shape(shape)
        cells = np.arange(height * width, dtype=np.float64).reshape(height, width)
        return np.exp(2j * np.pi * cells * float(int(slot) + 1) / (4.0 * height * width))

    def colour_phase(self, grid: np.ndarray, slot: int) -> np.ndarray:
        """The per-colour unit phase of one slot."""
        values = np.asarray(grid, dtype=np.float64)
        return np.exp(2j * np.pi * values * float(int(slot)) / float(self.colours))

    # ----------------------------------------------------------- transform
    def coefficients(self, grid: np.ndarray) -> np.ndarray:
        """The (4, H, W) frame coefficient field of one grid."""
        value = as_grid(grid)
        if int(value.max()) >= self.colours:
            raise ValueError("colour outside the declared alphabet")
        height, width = self.check_shape(value.shape)
        out = np.empty((self.n_slots, height, width), dtype=np.complex128)
        for slot in range(self.n_slots):
            out[slot] = 0.5 * self.colour_phase(value, slot) * self.cell_phase(value.shape, slot)
        return out

    def synthesise(self, coefficients: np.ndarray) -> np.ndarray:
        """Sum the unit waves of one coefficient field into a D-dim wave."""
        field = np.asarray(coefficients, dtype=np.complex128)
        if field.ndim != 3 or field.shape[0] != self.n_slots:
            raise ValueError("coefficients must be (n_slots, H, W)")
        height, width = self.check_shape(field.shape[1:])
        cells = height * width
        psi = np.zeros(self.dim, dtype=np.complex128)
        for slot in range(self.n_slots):
            piece = np.sqrt(float(cells)) * np.fft.ifft2(field[slot])
            start = slot * self.width
            psi[start:start + cells] = piece.reshape(-1)
        return psi

    def encode(self, grid: np.ndarray) -> np.ndarray:
        """Encode one grid as a finite sum of unit waves."""
        self.calls += 1
        return self.synthesise(self.coefficients(grid))

    def analyse(self, psi: np.ndarray, shape: Sequence[int]) -> np.ndarray:
        """Project one wave onto the frame of a declared grid shape."""
        height, width = self.check_shape(shape)
        cells = height * width
        wave = np.asarray(psi, dtype=np.complex128)
        if wave.shape[0] < self.dim:
            raise ValueError("the wave is shorter than the declared dimension")
        out = np.empty((self.n_slots, height, width), dtype=np.complex128)
        for slot in range(self.n_slots):
            start = slot * self.width
            field = wave[start:start + cells].reshape(height, width)
            out[slot] = np.fft.fft2(field) / np.sqrt(float(cells))
        return out

    def decode(self, psi: np.ndarray, shape: Sequence[int]) -> Tuple[np.ndarray, float]:
        """Read the grid back and return it with the worst phase residual."""
        coefficients = self.analyse(psi, shape)
        height, width = self.check_shape(shape)
        phase_one = self.cell_phase((height, width), 1)
        phase_two = self.cell_phase((height, width), 2)
        v1 = 2.0 * coefficients[1] * np.conj(phase_one)
        angle = np.angle(v1)
        grid = np.mod(
            np.round(angle * float(self.colours) / (2.0 * np.pi)).astype(np.int64),
            self.colours,
        )
        predicted = np.exp(2j * np.pi * grid.astype(np.float64) * 1.0 / float(self.colours))
        v2 = 2.0 * coefficients[2] * np.conj(phase_two)
        predicted_two = np.exp(2j * np.pi * grid.astype(np.float64) * 2.0 / float(self.colours))
        residual = float(max(
            np.max(np.abs(v1 - predicted)) if v1.size else 0.0,
            np.max(np.abs(v2 - predicted_two)) if v2.size else 0.0,
        ))
        return grid, residual

    def colour_field(self, psi: np.ndarray, shape: Sequence[int]) -> np.ndarray:
        """The dephased colour-phase field read from the wave (unit modulus).

        The analysis recovers the slot-1 frame coefficients and divides out the
        encoder's per-cell phase, so the field is exactly exp(2*pi*i*G/10) and
        every grid permutation acts on it as a pure permutation.  This is the
        field the router correlates; the correlation itself is a cross-spectrum.
        """
        coefficients = self.analyse(psi, shape)
        return 2.0 * coefficients[1] * np.conj(self.cell_phase(shape, 1))

    def field_of(self, grid: np.ndarray) -> np.ndarray:
        """Encode one grid and return its colour-phase field."""
        value = as_grid(grid)
        return self.colour_field(self.encode(value), value.shape)

    def roundtrip(self, grid: np.ndarray) -> Dict[str, object]:
        """Encode, decode, and measure one exact round trip."""
        value = as_grid(grid)
        psi = self.encode(value)
        recovered, residual = self.decode(psi, value.shape)
        return {
            "shape": [int(value.shape[0]), int(value.shape[1])],
            "max_cell_error": int(np.max(np.abs(recovered - value))),
            "phase_residual": float(residual),
            "exact": bool(np.array_equal(recovered, value)),
            "wave_norm": float(np.linalg.norm(psi)),
            "unit_waves": int(self.n_slots * value.size),
        }

    def frame_matrix(self, shape: Sequence[int]) -> np.ndarray:
        """The (4*H*W, D) synthesis matrix of the frame (rows are unit waves)."""
        height, width = self.check_shape(shape)
        cells = height * width
        rows = np.arange(height)
        cols = np.arange(width)
        basis_rows = np.exp(2j * np.pi * np.outer(rows, rows) / float(height))
        basis_cols = np.exp(2j * np.pi * np.outer(cols, cols) / float(width))
        block = np.einsum("ar,bc->abrc", basis_rows, basis_cols).reshape(cells, cells)
        block = block / np.sqrt(float(cells))
        frame = np.zeros((self.n_slots * cells, self.dim), dtype=np.complex128)
        for slot in range(self.n_slots):
            start = slot * self.width
            frame[slot * cells:(slot + 1) * cells, start:start + cells] = block
        return frame

    def frame_gram_error(self, shape: Sequence[int]) -> float:
        """max |F^H F - I| for the declared frame of one grid shape."""
        frame = self.frame_matrix(shape)
        gram = frame @ frame.conj().T
        return float(np.max(np.abs(gram - np.eye(gram.shape[0], dtype=np.complex128))))

    def isometry_error(self, shape: Sequence[int], seed: int = 20261111) -> float:
        """max |analyse(synthesise(a)) - a| for a deterministic random field."""
        height, width = self.check_shape(shape)
        generator = np.random.default_rng(int(seed) ^ (height * 4096 + width))
        raw = generator.standard_normal((self.n_slots, height, width)) + 1j * generator.standard_normal(
            (self.n_slots, height, width)
        )
        psi = self.synthesise(raw)
        back = self.analyse(psi, (height, width))
        return float(np.max(np.abs(back - raw)))


# ----------------------------------------------------- wave-domain invariants
def d4_coefficients(op: str, coefficients: np.ndarray) -> np.ndarray:
    """Apply one D4 element to a colour-phase field (a cell permutation)."""
    field = np.asarray(coefficients)
    if op == "identity":
        return field
    if op == "rot90":
        return np.rot90(field, 1)
    if op == "rot180":
        return np.rot90(field, 2)
    if op == "rot270":
        return np.rot90(field, 3)
    if op == "reflect_h":
        return field[:, ::-1]
    if op == "reflect_v":
        return field[::-1, :]
    if op == "reflect_diag":
        return np.swapaxes(field, 0, 1)
    if op == "reflect_antidiag":
        return np.rot90(np.swapaxes(field, 0, 1), 2)
    raise ValueError("unknown dihedral element %r" % (op,))


def d4_resonance(field_in: np.ndarray, field_out: np.ndarray) -> Dict[str, Optional[float]]:
    """Normalised cross-spectral resonance of the eight dihedral elements.

    The argument is the colour-phase field read from the wave, so a grid that
    is exactly a dihedral image of the input resonates at 1.0 on that element
    and every shape-mismatched element reports None.
    """
    out = np.asarray(field_out)
    energy_out = float(np.sum(np.abs(out) ** 2))
    result: Dict[str, Optional[float]] = {}
    for op in D4_ELEMENTS:
        moved = d4_coefficients(op, field_in)
        if tuple(moved.shape) != tuple(out.shape):
            result[op] = None
            continue
        energy_in = float(np.sum(np.abs(moved) ** 2))
        if energy_in <= _EPS or energy_out <= _EPS:
            result[op] = 0.0
            continue
        overlap = float(np.abs(np.sum(np.conj(out) * moved)))
        result[op] = float(overlap / np.sqrt(energy_in * energy_out))
    return result


def translation_peaks(field_in: np.ndarray, field_out: np.ndarray,
                      limit: int = 4) -> List[Tuple[int, int, float]]:
    """The 2-D Fourier translation phase peaks (dr, dc, height) of one pair.

    np.roll(input, (dr, dc)) is the grid that the peak predicts.  The
    correlation is the cross-spectrum of the two colour-phase fields, so a
    toroidal shift shows a single peak of height 1.0.
    """
    src = np.asarray(field_in)
    dst = np.asarray(field_out)
    if src.shape != dst.shape:
        return []
    height, width = src.shape
    correlation = np.fft.ifft2(np.conj(np.fft.fft2(dst)) * np.fft.fft2(src))
    energy = float(np.sqrt(np.sum(np.abs(src) ** 2) * np.sum(np.abs(dst) ** 2)))
    if energy <= _EPS:
        return []
    magnitude = np.abs(correlation) / energy
    peaks: List[Tuple[int, int, float]] = []
    work = magnitude.copy()
    for _ in range(int(limit)):
        index = int(np.argmax(work))
        row, col = divmod(index, width)
        value = float(magnitude[row, col])
        # C[d] = sum conj(out)[m] in[m + d]; in[m + d] = out[m] means the input
        # must be rolled by -d, so the reported shift is the negated index.
        peaks.append((int((-row) % height), int((-col) % width), round(value, 6)))
        work[max(0, row - 1):row + 2, max(0, col - 1):col + 2] = -1.0
    return peaks


# ----------------------------------------------------------------- operators
@dataclass(frozen=True)
class SpatialOp:
    """One concrete operator: a sketch name plus its declared parameters."""

    family: str
    params: Tuple[object, ...] = ()

    def label(self) -> str:
        if not self.params:
            return str(self.family)
        rendered = ",".join(str(value) for value in self.params)
        return "%s(%s)" % (self.family, rendered)

    def key(self) -> str:
        return self.label()

    def apply(self, grid: np.ndarray) -> Optional[np.ndarray]:
        return apply_family(self.family, self.params, grid)


def _op_dihedral(grid: np.ndarray, index: int) -> Optional[np.ndarray]:
    if int(index) == 0:
        return grid.copy()
    if int(index) == 1:
        return np.rot90(grid, 1)
    if int(index) == 2:
        return np.rot90(grid, 2)
    if int(index) == 3:
        return np.rot90(grid, 3)
    if int(index) == 4:
        return grid[:, ::-1].copy()
    if int(index) == 5:
        return grid[::-1, :].copy()
    if int(index) == 6:
        return np.swapaxes(grid, 0, 1).copy()
    if int(index) == 7:
        return np.rot90(np.swapaxes(grid, 0, 1), 2).copy()
    return None


def _op_toroidal_shift(grid: np.ndarray, dr: int, dc: int) -> np.ndarray:
    return np.roll(np.roll(grid, int(dr), axis=0), int(dc), axis=1)


def _op_bounded_shift(grid: np.ndarray, dr: int, dc: int, fill: int) -> Optional[np.ndarray]:
    height, width = grid.shape
    out = np.full((height, width), int(fill), dtype=grid.dtype)
    src_r0 = max(0, -int(dr))
    src_r1 = min(height, height - int(dr))
    src_c0 = max(0, -int(dc))
    src_c1 = min(width, width - int(dc))
    if src_r1 <= src_r0 or src_c1 <= src_c0:
        return None
    out[src_r0 + int(dr):src_r1 + int(dr), src_c0 + int(dc):src_c1 + int(dc)] = \
        grid[src_r0:src_r1, src_c0:src_c1]
    return out


def _op_colour_map(grid: np.ndarray, *values: object) -> Optional[np.ndarray]:
    """A ten-entry lookup table, passed flat or as one tuple."""
    lut: Sequence[object] = values
    if len(lut) == 1 and isinstance(lut[0], (tuple, list)):
        lut = tuple(lut[0])
    if len(lut) != COLOURS:
        return None
    table = np.asarray(tuple(int(value) for value in lut), dtype=grid.dtype)
    if table.shape[0] <= int(grid.max()):
        return None
    return table[grid]


def _op_crop_bbox(grid: np.ndarray) -> np.ndarray:
    box = bounding_box(grid)
    if box is None:
        return grid.copy()
    r0, c0, r1, c1 = box
    return grid[r0:r1 + 1, c0:c1 + 1].copy()


def _op_tile(grid: np.ndarray, fy: int, fx: int) -> Optional[np.ndarray]:
    if int(fy) < 1 or int(fx) < 1:
        return None
    return np.tile(grid, (int(fy), int(fx)))


def _op_scale(grid: np.ndarray, factor: int) -> Optional[np.ndarray]:
    if int(factor) < 1:
        return None
    return np.kron(grid, np.ones((int(factor), int(factor)), dtype=grid.dtype))


def _op_interior_flood_fill(grid: np.ndarray, colour: int) -> np.ndarray:
    height, width = grid.shape
    mask = grid == BACKGROUND
    out = grid.copy()
    for region in _regions(mask):
        touches = any(
            row == 0 or row == height - 1 or col == 0 or col == width - 1
            for row, col in region
        )
        if touches:
            continue
        for row, col in region:
            out[row, col] = int(colour)
    return out


def _op_mask_by_colour(grid: np.ndarray, colour: int) -> np.ndarray:
    return np.where(grid == int(colour), grid, BACKGROUND).astype(grid.dtype)


def _op_symmetry_completion(grid: np.ndarray) -> Optional[np.ndarray]:
    """Copy the diagonally opposite quadrant into the single empty quadrant."""
    height, width = grid.shape
    if height % 2 or width % 2:
        return None
    half_h, half_w = height // 2, width // 2
    quads = [
        grid[:half_h, :half_w],
        grid[:half_h, half_w:],
        grid[half_h:, :half_w],
        grid[half_h:, half_w:],
    ]
    empty = [index for index, quad in enumerate(quads) if not bool((quad != BACKGROUND).any())]
    if len(empty) != 1:
        return None
    target = int(empty[0])
    opposite = (3, 2, 1, 0)[target]
    filled = np.rot90(quads[opposite], 2)
    out = grid.copy()
    if target == 0:
        out[:half_h, :half_w] = filled
    elif target == 1:
        out[:half_h, half_w:] = filled
    elif target == 2:
        out[half_h:, :half_w] = filled
    else:
        out[half_h:, half_w:] = filled
    return out


def _op_constant_fill(grid: np.ndarray, colour: int) -> np.ndarray:
    return np.full(grid.shape, int(colour), dtype=grid.dtype)


_FAMILY_APPLY = {
    "dihedral": _op_dihedral,
    "toroidal_shift": _op_toroidal_shift,
    "bounded_shift": _op_bounded_shift,
    "colour_map": _op_colour_map,
    "crop_bbox": _op_crop_bbox,
    "tile": _op_tile,
    "scale": _op_scale,
    "interior_flood_fill": _op_interior_flood_fill,
    "mask_by_colour": _op_mask_by_colour,
    "symmetry_completion": _op_symmetry_completion,
    "constant_fill": _op_constant_fill,
}


def apply_family(family: str, params: Sequence[object], grid: np.ndarray) -> Optional[np.ndarray]:
    """Run one operator family on one grid; None when it does not apply."""
    runner = _FAMILY_APPLY.get(str(family))
    if runner is None:
        return None
    try:
        value = runner(as_grid(grid), *params)
    except Exception:
        return None
    if value is None:
        return None
    value = np.asarray(value, dtype=np.int64)
    if value.ndim != 2 or value.size == 0:
        return None
    if int(value.min()) < 0 or int(value.max()) >= COLOURS:
        return None
    return value


def apply_chain(ops: Sequence[SpatialOp], grid: np.ndarray) -> Optional[np.ndarray]:
    """Run a 1-2 stage composition left to right."""
    value: Optional[np.ndarray] = as_grid(grid)
    for op in ops:
        if value is None:
            return None
        value = op.apply(value)
    return value


# --------------------------------------------------------- candidate proposal
def _colour_map_from_pair(src: np.ndarray, dst: np.ndarray) -> Optional[SpatialOp]:
    if src.shape != dst.shape:
        return None
    lut = list(range(COLOURS))
    for colour in palette(src):
        values = np.unique(dst[src == int(colour)])
        if values.size != 1:
            return None
        lut[int(colour)] = int(values[0])
    return SpatialOp("colour_map", tuple(int(value) for value in lut))


def _colour_permutations(src: np.ndarray) -> List[SpatialOp]:
    colours = foreground_palette(src)
    if not colours or len(colours) > 4:
        return []
    from itertools import permutations

    out: List[SpatialOp] = []
    base = list(range(COLOURS))
    for order in permutations(colours):
        lut = list(base)
        for source, target in zip(colours, order):
            lut[int(source)] = int(target)
        if tuple(order) == tuple(colours):
            continue
        out.append(SpatialOp("colour_map", tuple(int(value) for value in lut)))
    return out


def _shift_candidates(src: np.ndarray, dst: np.ndarray,
                      encoder: ToroidalWaveEncoder,
                      limit: int = 3) -> List[Tuple[int, int, float]]:
    if src.shape != dst.shape:
        return []
    peaks = translation_peaks(encoder.field_of(src), encoder.field_of(dst), limit=limit)
    return [peak for peak in peaks if float(peak[2]) > 0.0]


def propose_family(family: str, src: np.ndarray, dst: np.ndarray,
                   encoder: ToroidalWaveEncoder) -> List[SpatialOp]:
    """Concrete operators of one family that could map src to dst."""
    src = as_grid(src)
    dst = as_grid(dst)
    out: List[SpatialOp] = []
    if family == "dihedral":
        out = [SpatialOp("dihedral", (index,)) for index in range(8)]
    elif family == "toroidal_shift":
        out = [SpatialOp("toroidal_shift", (int(dr), int(dc)))
               for dr, dc, _height in _shift_candidates(src, dst, encoder)]
    elif family == "bounded_shift":
        fills = sorted(set(palette(dst)) | {BACKGROUND})[:4]
        out = [SpatialOp("bounded_shift", (int(dr), int(dc), int(fill)))
               for dr, dc, _height in _shift_candidates(src, dst, encoder)
               for fill in fills]
    elif family == "colour_map":
        derived = _colour_map_from_pair(src, dst)
        if derived is not None:
            out.append(derived)
        out.extend(_colour_permutations(src))
    elif family == "crop_bbox":
        out = [SpatialOp("crop_bbox", ())]
    elif family == "tile":
        if src.shape[0] and src.shape[1] and dst.shape[0] % src.shape[0] == 0 \
                and dst.shape[1] % src.shape[1] == 0:
            factor_h = dst.shape[0] // src.shape[0]
            factor_w = dst.shape[1] // src.shape[1]
            if (factor_h, factor_w) != (1, 1):
                out.append(SpatialOp("tile", (int(factor_h), int(factor_w))))
        out.extend(SpatialOp("tile", (int(fy), int(fx)))
                   for fy, fx in ((2, 2), (2, 1), (1, 2), (3, 3)))
    elif family == "scale":
        if dst.shape[0] % src.shape[0] == 0 and dst.shape[1] % src.shape[1] == 0:
            factor_h = dst.shape[0] // src.shape[0]
            factor_w = dst.shape[1] // src.shape[1]
            if factor_h == factor_w and factor_h > 1:
                out.append(SpatialOp("scale", (int(factor_h),)))
        out.extend(SpatialOp("scale", (int(factor),)) for factor in (2, 3))
    elif family == "interior_flood_fill":
        out = [SpatialOp("interior_flood_fill", (int(colour),))
               for colour in foreground_palette(dst)[:4]]
    elif family == "mask_by_colour":
        out = [SpatialOp("mask_by_colour", (int(colour),))
               for colour in foreground_palette(src)[:4]]
    elif family == "symmetry_completion":
        out = [SpatialOp("symmetry_completion", ())]
    elif family == "constant_fill":
        out = [SpatialOp("constant_fill", (int(colour),))
               for colour in palette(dst)[:4]]
    return out


def propose_joint(family: str, pairs: Sequence[Tuple[np.ndarray, np.ndarray]],
                  encoder: ToroidalWaveEncoder) -> List[SpatialOp]:
    """Concrete operators of one family proposed from the whole pair set.

    A per-pair derivation under-constrains a colour map: a colour absent from
    one pair leaves the lookup at the identity, so the per-pair operator does
    not transfer to the other pairs.  The joint derivation intersects the
    constraints of every pair, which is what makes the colour family a
    hypothesis about the rule instead of a lookup for one example.
    """
    out: List[SpatialOp] = []
    seen: set = set()

    def push(op: SpatialOp) -> None:
        key = op.key()
        if key in seen:
            return
        seen.add(key)
        out.append(op)

    if family == "colour_map":
        lut = list(range(COLOURS))
        consistent = True
        for src, dst in pairs:
            src = as_grid(src)
            dst = as_grid(dst)
            if src.shape != dst.shape:
                consistent = False
                break
            for colour in palette(src):
                values = np.unique(dst[src == int(colour)])
                if values.size != 1:
                    consistent = False
                    break
                target = int(values[0])
                if lut[int(colour)] != int(colour) and lut[int(colour)] != target:
                    consistent = False
                    break
                lut[int(colour)] = target
            if not consistent:
                break
        if consistent:
            push(SpatialOp("colour_map", tuple(int(value) for value in lut)))
    for src, dst in pairs:
        for op in propose_family(str(family), as_grid(src), as_grid(dst), encoder):
            push(op)
    return out


# -------------------------------------------------------------------- router
class WaveDeltaSpatialRouter:
    """Deterministic I/O wave-delta signature and operator-sketch ranking."""

    def __init__(self, encoder: Optional[ToroidalWaveEncoder] = None,
                 node_budget: int = 200000) -> None:
        self.encoder = encoder if encoder is not None else ToroidalWaveEncoder()
        self.node_budget = int(node_budget)
        self.calls = 0

    # ------------------------------------------------------------- features
    def features(self, spec: "SpatialSpec") -> Dict[str, object]:
        pairs = [(as_grid(x), as_grid(y)) for x, y in spec.examples]
        self.calls += 1
        shape_rows: List[List[int]] = []
        bbox_rows: List[List[int]] = []
        d4_rows: List[Dict[str, Optional[float]]] = []
        peak_rows: List[List[Tuple[int, int, float]]] = []
        mass_rows: List[List[Dict[str, int]]] = []
        euler_rows: List[Dict[int, int]] = []
        containment_rows: List[List[Tuple[int, int]]] = []
        for src, dst in pairs:
            psi_in = self.encoder.encode(src)
            psi_out = self.encoder.encode(dst)
            field_in = self.encoder.colour_field(psi_in, src.shape)
            field_out = self.encoder.colour_field(psi_out, dst.shape)
            shape_rows.append([int(src.shape[0]), int(src.shape[1]),
                               int(dst.shape[0]), int(dst.shape[1])])
            box_in = bounding_box(src)
            box_out = bounding_box(dst)
            bbox_rows.append([-1 if box_in is None else int(box_in[2] - box_in[0] + 1),
                              -1 if box_in is None else int(box_in[3] - box_in[1] + 1),
                              -1 if box_out is None else int(box_out[2] - box_out[0] + 1),
                              -1 if box_out is None else int(box_out[3] - box_out[1] + 1)])
            d4_rows.append(d4_resonance(field_in, field_out))
            peak_rows.append(translation_peaks(field_in, field_out))
            signature_in = component_signature(src)
            signature_out = component_signature(dst)
            mass_rows.append([
                {"in": int(signature_in.get(colour, {}).get("mass", 0)),
                 "out": int(signature_out.get(colour, {}).get("mass", 0)),
                 "colour": int(colour)}
                for colour in sorted(set(signature_in) | set(signature_out))
            ])
            euler_rows.append({
                int(colour): int(signature_out.get(colour, {}).get("euler", 0))
                - int(signature_in.get(colour, {}).get("euler", 0))
                for colour in sorted(set(signature_in) | set(signature_out))
            })
            containment_rows.append(containment_signature(dst))

        count = max(1, len(pairs))
        shapes_preserved = all(row[0] == row[2] and row[1] == row[3] for row in shape_rows)
        shape_deltas = [[row[2] - row[0], row[3] - row[1]] for row in shape_rows]
        delta_unique = sorted({(int(delta[0]), int(delta[1])) for delta in shape_deltas})
        ratios = sorted({
            (float(row[2]) / float(row[0]), float(row[3]) / float(row[1])) for row in shape_rows
        })
        candidate_ratios = [ratio[0] for ratio in ratios if ratio[0] == ratio[1]]
        d4_mean: Dict[str, float] = {}
        for element in D4_ELEMENTS:
            values = [float(row[element]) for row in d4_rows if row.get(element) is not None]
            d4_mean[element] = round(float(np.mean(values)) if values else 0.0, 6)
        best_element = max(D4_ELEMENTS, key=lambda name: (d4_mean[name],
                                                          -D4_ELEMENTS.index(name)))
        peak_consensus: Dict[Tuple[int, int], float] = {}
        first_peak_rows = [row[0] for row in peak_rows if row]
        for dr, dc, _height in first_peak_rows:
            key = (int(dr), int(dc))
            peak_consensus[key] = peak_consensus.get(key, 0.0) + 1.0
        consensus_peak: Optional[Tuple[int, int]] = None
        consensus_fraction = 0.0
        if peak_consensus:
            consensus_peak = max(peak_consensus, key=lambda key: peak_consensus[key])
            consensus_fraction = round(peak_consensus[consensus_peak] / float(count), 6)
        heights = [float(row[2]) for row in first_peak_rows]
        mean_peak_height = round(float(np.mean(heights)) if heights else 0.0, 6)
        palette_equal = all(
            palette(pairs[index][1]) == palette(pairs[index][0]) for index in range(len(pairs)))
        mass_preserved = all(
            sorted((row["colour"], row["in"]) for row in rows)
            == sorted((row["colour"], row["out"]) for row in rows) for rows in mass_rows)
        foreground_in = set()
        foreground_out = set()
        for src, dst in pairs:
            foreground_in.update(foreground_palette(src))
            foreground_out.update(foreground_palette(dst))
        constant_output = all(len(palette(dst)) == 1 for _src, dst in pairs)
        empty_quadrant = True
        for src, _dst in pairs:
            if src.shape[0] % 2 or src.shape[1] % 2:
                empty_quadrant = False
                break
            half_h, half_w = src.shape[0] // 2, src.shape[1] // 2
            quads = [src[:half_h, :half_w], src[:half_h, half_w:],
                     src[half_h:, :half_w], src[half_h:, half_w:]]
            empties = sum(1 for quad in quads if not bool((quad != BACKGROUND).any()))
            empty_quadrant = bool(empty_quadrant and empties == 1)
        enclosed = 0
        for src, _dst in pairs:
            for colour, row in component_signature(src).items():
                if int(colour) != BACKGROUND:
                    enclosed += int(row["holes"])
        mask_reduction = 0.0
        for rows in mass_rows:
            for row in rows:
                if row["colour"] != BACKGROUND:
                    mask_reduction += float(max(0, int(row["in"]) - int(row["out"])))
        flood_gain = 0.0
        for rows in mass_rows:
            for row in rows:
                if row["colour"] != BACKGROUND:
                    flood_gain += float(max(0, int(row["out"]) - int(row["in"])))
        signature: Dict[str, object] = {
            "examples_used": int(len(pairs)),
            "shapes": shape_rows,
            "shape_preserved": bool(shapes_preserved),
            "shape_deltas": shape_deltas,
            "shape_delta_unique": int(len(delta_unique)),
            "bbox_sizes": bbox_rows,
            "bbox_growth": bool(all(row[2] >= row[0] and row[3] >= row[1]
                                    for row in bbox_rows if row[0] >= 0)),
            "bbox_shrink": bool(any(row[2] < row[0] or row[3] < row[1]
                                    for row in bbox_rows if row[0] >= 0)),
            "bbox_matches_output": bool(all(
                row[2] == shape_rows[index][2] and row[3] == shape_rows[index][3]
                for index, row in enumerate(bbox_rows))),
            "area_ratios": [[round(ratio[0], 6), round(ratio[1], 6)] for ratio in ratios],
            "scale_ratio": float(candidate_ratios[0]) if len(candidate_ratios) == 1 else 0.0,
            "scale_ratio_consensus": bool(len(candidate_ratios) == 1
                                          and candidate_ratios[0] > 1.0),
            "tile_ratio_consensus": bool(len(ratios) == 1 and all(
                abs(ratio[0] - round(ratio[0])) < 1e-9 and abs(ratio[1] - round(ratio[1])) < 1e-9
                for ratio in ratios)),
            "d4_elements": list(D4_ELEMENTS),
            "d4_resonance": {name: [d4_rows[index].get(name) for index in range(len(d4_rows))]
                             for name in D4_ELEMENTS},
            "d4_mean": d4_mean,
            "d4_best_element": str(best_element),
            "d4_best_mean": float(d4_mean[best_element]),
            "d4_identity_mean": float(d4_mean["identity"]),
            "translation_peaks": [[[int(row[0]), int(row[1]), float(row[2])] for row in peaks]
                                  for peaks in peak_rows],
            "translation_peak": [-1, -1] if consensus_peak is None else list(consensus_peak),
            "translation_consensus": float(consensus_fraction),
            "translation_height": float(mean_peak_height),
            "palette_in": sorted(int(value) for value in foreground_in),
            "palette_out": sorted(int(value) for value in foreground_out),
            "palette_equal": bool(palette_equal),
            "mass_preserved": bool(mass_preserved),
            "mass_gain": round(float(flood_gain), 6),
            "mass_loss": round(float(mask_reduction), 6),
            "mass_rows": mass_rows,
            "euler_delta": euler_rows,
            "containment": containment_rows,
            "constant_output": bool(constant_output),
            "empty_quadrant_input": bool(empty_quadrant),
            "enclosed_background": int(enclosed),
        }
        return signature

    # --------------------------------------------------------------- scores
    def scores(self, signature: Dict[str, object]) -> Dict[str, float]:
        preserve = 1.0 if signature["shape_preserved"] else 0.0
        best = float(signature["d4_best_mean"])
        nonzero_d4 = 1.0 if (signature["d4_best_element"] != "identity"
                             and best >= 0.75) else 0.0
        peak = float(signature["translation_consensus"]) * float(signature["translation_height"])
        mass = 1.0 if signature["mass_preserved"] else 0.0
        same_palette = 1.0 if signature["palette_equal"] else 0.0
        constant = 1.0 if signature["constant_output"] else 0.0
        shrink = 1.0 if signature["bbox_shrink"] else 0.0
        crop_hit = 1.0 if (signature["bbox_matches_output"] and shrink) else 0.0
        scale = 1.0 if signature["scale_ratio_consensus"] else 0.0
        tile = 1.0 if signature["tile_ratio_consensus"] else 0.0
        enclosed = 1.0 if int(signature["enclosed_background"]) > 0 else 0.0
        empty_quad = 1.0 if signature["empty_quadrant_input"] else 0.0
        loss = 1.0 if float(signature["mass_loss"]) > 0.0 else 0.0
        gain = 1.0 if float(signature["mass_gain"]) > 0.0 else 0.0
        return {
            "dihedral": round(3.0 + 9.0 * best * nonzero_d4 + 2.0 * preserve, 6),
            "toroidal_shift": round(2.0 + 8.0 * peak * preserve + 2.0 * mass, 6),
            "bounded_shift": round(2.0 + 7.0 * peak * preserve + 1.0 * mass, 6),
            "colour_map": round(3.0 + 5.0 * mass + 2.0 * same_palette + 2.0 * preserve, 6),
            "crop_bbox": round(2.0 + 8.0 * crop_hit + 1.0 * shrink, 6),
            "tile": round(2.0 + 6.0 * tile * (1.0 - preserve) + 2.0 * scale, 6),
            "scale": round(2.0 + 8.0 * scale * (1.0 - preserve) + 1.0 * tile, 6),
            "interior_flood_fill": round(2.0 + 7.0 * enclosed + 3.0 * gain + 1.0 * preserve, 6),
            "mask_by_colour": round(2.0 + 5.0 * loss + 2.0 * preserve + 1.0 * mass, 6),
            "symmetry_completion": round(2.0 + 9.0 * empty_quad + 2.0 * preserve, 6),
            "constant_fill": round(1.0 + 10.0 * constant + 1.0 * preserve, 6),
        }

    # ----------------------------------------------------------------- rank
    def rank(self, spec: "SpatialSpec") -> Dict[str, object]:
        signature = self.features(spec)
        scores = self.scores(signature)
        order = {name: index for index, name in enumerate(SPATIAL_SKETCHES)}
        ranked = sorted(scores, key=lambda name: (-scores[name], order.get(name, 999)))
        digest = hashlib.sha256(
            json.dumps(signature, sort_keys=True, default=float).encode("utf-8")
        ).hexdigest()
        return {
            "signature": signature,
            "signature_digest": digest,
            "scores": scores,
            "ranked": list(ranked),
            "examples_used": int(len(spec.examples)),
            "tests_consulted": False,
            "budget_nodes": int(self.node_budget),
            "components": [
                "shape_and_bounding_box_delta",
                "d4_dihedral_cross_spectral_resonance",
                "fourier_translation_phase_peak",
                "colour_mass_and_euler_containment",
            ],
        }


# ------------------------------------------------------------------- inducer
@dataclass(frozen=True)
class SpatialSpec:
    """One spatial induction problem: example grid pairs only, no rules."""

    name: str
    examples: Sequence[Tuple[np.ndarray, np.ndarray]]
    tests: Sequence[Tuple[np.ndarray, np.ndarray]] = ()
    declared: Tuple[str, ...] = ()


class SpatialCegisInducer:
    """Ranked CEGIS over 1-2 stage operator chains, one node budget.

    A node is one operator-chain execution on one training pair.  The search
    is deterministic: candidates are proposed from the training pairs, ordered
    by the router's ranked sketch list, and the first chain that reproduces
    every training pair is accepted.  Refuted chains are recorded as REPULSOR
    axioms (bounded, deduplicated) and an identical chain is never executed
    twice.  The held-out pairs are never read.
    """

    def __init__(self, encoder: Optional[ToroidalWaveEncoder] = None,
                 router: Optional[WaveDeltaSpatialRouter] = None,
                 node_budget: int = 200000, repulsor_cap: int = 64,
                 stage_one_cap: int = 140) -> None:
        self.encoder = encoder if encoder is not None else ToroidalWaveEncoder()
        self.router = router if router is not None else WaveDeltaSpatialRouter(self.encoder)
        self.node_budget = int(node_budget)
        self.repulsor_cap = int(repulsor_cap)
        self.stage_one_cap = int(stage_one_cap)

    # ------------------------------------------------------------- refusal
    def refuse(self, spec: SpatialSpec) -> Optional[str]:
        """Fail-closed checks that need no search at all."""
        pairs = [(as_grid(x), as_grid(y)) for x, y in spec.examples]
        if not pairs:
            return "ABSTAIN: no examples"
        seen: Dict[str, set] = {}
        for src, dst in pairs:
            key = "%dx%d:%s" % (src.shape[0], src.shape[1],
                                hashlib.sha256(np.ascontiguousarray(src).tobytes()).hexdigest())
            seen.setdefault(key, set()).add(
                "%dx%d:%s" % (dst.shape[0], dst.shape[1],
                              hashlib.sha256(np.ascontiguousarray(dst).tobytes()).hexdigest())
            )
        if any(len(values) > 1 for values in seen.values()):
            return "ABSTAIN: unsatisfiable specification (one input, two outputs)"
        for facet in tuple(spec.declared):
            if str(facet) == DECLARED_SHAPE_RULE:
                if any(src.shape != dst.shape for src, dst in pairs):
                    return "ABSTAIN: a pair violates the declared shape rule"
        return None

    # ----------------------------------------------------------- proposals
    def _candidates(self, spec: SpatialSpec, ranked: Sequence[str]) -> List[SpatialOp]:
        pairs = [(as_grid(x), as_grid(y)) for x, y in spec.examples]
        out: List[SpatialOp] = []
        seen: set = set()
        for family in ranked:
            for op in propose_joint(str(family), pairs, self.encoder):
                key = op.key()
                if key in seen:
                    continue
                seen.add(key)
                out.append(op)
        return out

    def _second_stage(self, spec: SpatialSpec, head: SpatialOp,
                      ranked: Sequence[str]) -> List[SpatialOp]:
        middles: List[Tuple[np.ndarray, np.ndarray]] = []
        for src, dst in spec.examples:
            middle = head.apply(as_grid(src))
            if middle is None:
                continue
            middles.append((middle, as_grid(dst)))
        out: List[SpatialOp] = []
        seen: set = set()
        for family in ranked:
            for op in propose_joint(str(family), middles, self.encoder):
                key = op.key()
                if key in seen:
                    continue
                seen.add(key)
                out.append(op)
        return out

    # -------------------------------------------------------------- search
    def _execute(self, chain: Sequence[SpatialOp], spec: SpatialSpec,
                 state: Dict[str, object]) -> bool:
        for src, dst in spec.examples:
            if int(state["nodes"]) >= self.node_budget:
                state["exhausted"] = True
                return False
            state["nodes"] = int(state["nodes"]) + 1
            value = apply_chain(chain, as_grid(src))
            if value is None:
                return False
            target = as_grid(dst)
            if value.shape != target.shape or not np.array_equal(value, target):
                return False
        return True

    def induce(self, spec: SpatialSpec) -> Dict[str, object]:
        import time

        started = time.perf_counter()
        refusal = self.refuse(spec)
        if refusal is not None:
            return {
                "name": str(spec.name),
                "status": "ABSTAIN",
                "reason": str(refusal),
                "chain": [],
                "families": [],
                "stages": 0,
                "nodes": 0,
                "repulsors": 0,
                "repulsor_keys": [],
                "pruned_duplicates": 0,
                "train_exact": False,
                "train_rows": 0,
                "train_total": int(len(spec.examples)),
                "test_rows": [],
                "test_exact": 0,
                "test_total": int(len(spec.tests)),
                "generalized": False,
                "router": {},
                "node_budget": int(self.node_budget),
                "budget_exhausted": False,
                "refused_fail_closed": True,
                "wall_seconds": float(time.perf_counter() - started),
            }
        ranking = dict(self.router.rank(spec))
        ranked = [str(name) for name in list(ranking.get("ranked") or [])]
        candidates = self._candidates(spec, ranked)
        candidate_families: List[str] = []
        for op in candidates:
            if op.family not in candidate_families:
                candidate_families.append(op.family)
        state: Dict[str, object] = {
            "nodes": 0,
            "repulsors": [],
            "pruned": 0,
            "exhausted": False,
            "attempts": {},
        }
        refuted: set = set()
        found: List[SpatialOp] = []

        def note(family: str) -> None:
            attempts = state["attempts"]
            attempts[family] = int(attempts.get(family, 0)) + 1

        for op in candidates:
            key = op.key()
            if key in refuted:
                state["pruned"] = int(state["pruned"]) + 1
                continue
            note(op.family)
            if self._execute([op], spec, state):
                found = [op]
                break
            if len(state["repulsors"]) < self.repulsor_cap:
                refuted.add(key)
                state["repulsors"].append(key)
        if not found:
            heads = candidates[:self.stage_one_cap]
            for head in heads:
                if int(state["nodes"]) >= self.node_budget:
                    state["exhausted"] = True
                    break
                note(head.family)
                seconds = self._second_stage(spec, head, ranked)
                for second in seconds:
                    key = "%s>>%s" % (head.key(), second.key())
                    if key in refuted:
                        state["pruned"] = int(state["pruned"]) + 1
                        continue
                    note("%s>>%s" % (head.family, second.family))
                    if self._execute([head, second], spec, state):
                        found = [head, second]
                        break
                    if len(state["repulsors"]) < self.repulsor_cap:
                        refuted.add(key)
                        state["repulsors"].append(key)
                if found:
                    break
        outcome: Dict[str, object] = {
            "name": str(spec.name),
            "status": "INDUCED" if found else "ABSTAIN",
            "reason": "" if found else (
                "ABSTAIN: node budget exhausted" if state["exhausted"]
                else "ABSTAIN: no consistent operator chain"),
            "chain": [op.label() for op in found],
            "families": [op.family for op in found],
            "stages": int(len(found)),
            "nodes": int(state["nodes"]),
            "repulsors": int(len(state["repulsors"])),
            "repulsor_keys": list(state["repulsors"])[:16],
            "pruned_duplicates": int(state["pruned"]),
            "train_exact": bool(found),
            "train_rows": int(len(spec.examples)) if found else 0,
            "train_total": int(len(spec.examples)),
            "test_rows": [],
            "test_exact": 0,
            "test_total": int(len(spec.tests)),
            "generalized": False,
            "router": {
                "signature_digest": str(ranking.get("signature_digest", "")),
                "ranked": ranked,
                "scores": dict(ranking.get("scores") or {}),
                "signature": dict(ranking.get("signature") or {}),
                "examples_used": int(ranking.get("examples_used", 0) or 0),
                "tests_consulted": False,
                "budget_nodes": int(ranking.get("budget_nodes", 0) or 0),
                "chosen_rank": (ranked.index(found[0].family)
                                if found and found[0].family in ranked else -1),
                "candidate_families": list(candidate_families),
                "attempts": dict(state["attempts"]),
            },
            "node_budget": int(self.node_budget),
            "budget_exhausted": bool(state["exhausted"]),
            "refused_fail_closed": False,
            "wall_seconds": float(time.perf_counter() - started),
        }
        if found:
            rows: List[Dict[str, object]] = []
            exact = 0
            for src, dst in spec.tests:
                value = apply_chain(found, as_grid(src))
                target = as_grid(dst)
                ok = bool(value is not None and value.shape == target.shape
                          and np.array_equal(value, target))
                exact += int(ok)
                rows.append({
                    "input_shape": [int(src.shape[0]), int(src.shape[1])],
                    "expected_shape": [int(target.shape[0]), int(target.shape[1])],
                    "predicted_shape": (None if value is None
                                        else [int(value.shape[0]), int(value.shape[1])]),
                    "exact": bool(ok),
                    "wave_exact": bool(ok and self._wave_equal(value, target)),
                })
            outcome["test_rows"] = rows
            outcome["test_exact"] = int(exact)
            outcome["generalized"] = bool(rows and exact == len(rows))
        return outcome

    def _wave_equal(self, left: np.ndarray, right: np.ndarray) -> bool:
        """Compare two grids through the wave encoding (Parseval isometry)."""
        if left.shape != right.shape:
            return False
        if left.size > MAX_CELLS:
            return bool(np.array_equal(left, right))
        psi_left = self.encoder.encode(left)
        psi_right = self.encoder.encode(right)
        return bool(float(np.max(np.abs(psi_left - psi_right))) <= 1e-9)


# -------------------------------------------------------------------- audits
def encoding_audit(encoder: ToroidalWaveEncoder, grids: Sequence[np.ndarray],
                   gram_shapes: Sequence[Sequence[int]] = ((4, 6), (5, 7))) -> Dict[str, object]:
    """Round-trip every supplied grid and measure the frame isometry."""
    worst_cell = 0
    worst_phase = 0.0
    exact = 0
    shapes: Dict[str, Dict[str, int]] = {}
    for grid in grids:
        report = encoder.roundtrip(grid)
        worst_cell = max(worst_cell, int(report["max_cell_error"]))
        worst_phase = max(worst_phase, float(report["phase_residual"]))
        exact += int(bool(report["exact"]))
        key = "%dx%d" % (report["shape"][0], report["shape"][1])
        if key not in shapes:
            shapes[key] = {"count": 0, "max_cell_error": 0}
        shapes[key]["count"] = int(shapes[key]["count"]) + 1
        shapes[key]["max_cell_error"] = max(int(shapes[key]["max_cell_error"]),
                                            int(report["max_cell_error"]))
    isometry: Dict[str, Dict[str, float]] = {}
    for shape in sorted({tuple(int(value) for value in grid.shape) for grid in grids}):
        isometry["%dx%d" % shape] = {
            "isometry_max_error": float(encoder.isometry_error(shape)),
        }
    grams: Dict[str, Dict[str, object]] = {}
    for shape in gram_shapes:
        grams["%dx%d" % (int(shape[0]), int(shape[1]))] = {
            "frame_rows": int(encoder.n_slots * int(shape[0]) * int(shape[1])),
            "gram_max_error": float(encoder.frame_gram_error(shape)),
        }
    return {
        "grids": int(len(grids)),
        "exact_roundtrips": int(exact),
        "max_cell_error": int(worst_cell),
        "max_phase_residual": float(worst_phase),
        "invertible": bool(exact == len(grids) and worst_cell == 0 and worst_phase <= 1e-12),
        "shapes": shapes,
        "isometry": isometry,
        "max_isometry_error": float(max(
            (row["isometry_max_error"] for row in isometry.values()), default=0.0)),
        "frame_gram": grams,
        "max_gram_error": float(max(
            (row["gram_max_error"] for row in grams.values()), default=0.0)),
        "dimension": int(encoder.dim),
        "slots": int(encoder.n_slots),
        "slot_width": int(encoder.width),
    }


def router_selftest(encoder: Optional[ToroidalWaveEncoder] = None) -> Dict[str, object]:
    """Recover a declared toroidal shift and a declared dihedral element."""
    engine = encoder if encoder is not None else ToroidalWaveEncoder()
    base = np.array([[0, 1, 2, 0, 0, 3],
                     [4, 0, 0, 5, 0, 0],
                     [0, 6, 0, 0, 7, 0],
                     [0, 0, 8, 0, 0, 9]], dtype=np.int64)
    shift = (2, 3)
    moved = np.roll(np.roll(base, shift[0], axis=0), shift[1], axis=1)
    peaks = translation_peaks(engine.field_of(base), engine.field_of(moved))
    recovered = (int(peaks[0][0]), int(peaks[0][1])) if peaks else (-1, -1)
    height = float(peaks[0][2]) if peaks else 0.0
    square = np.array([[0, 1, 2, 3],
                       [4, 5, 0, 6],
                       [0, 7, 8, 0],
                       [9, 0, 0, 1]], dtype=np.int64)
    rotated = np.rot90(square, 1)
    resonance = d4_resonance(engine.field_of(square), engine.field_of(rotated))
    best = max(D4_ELEMENTS,
               key=lambda name: (-1.0 if resonance.get(name) is None else float(resonance[name])))
    return {
        "declared_shift": [int(shift[0]), int(shift[1])],
        "recovered_shift": [int(recovered[0]), int(recovered[1])],
        "shift_exact": bool(recovered == shift),
        "shift_height": round(height, 9),
        "declared_d4": "rot90",
        "recovered_d4": str(best),
        "d4_exact": bool(best == "rot90"),
        "d4_resonance": {name: (None if value is None else round(float(value), 9))
                         for name, value in resonance.items()},
        "elements": list(D4_ELEMENTS),
    }
