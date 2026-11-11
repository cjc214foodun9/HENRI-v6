"""Foveated harmonic tape: constant read SNR as the tape grows (Sovereign T3a).

Slot 2 is partitioned into 16 disjoint harmonic comb sub-bands of 16 complex
channels. One page holds 16 cells, one cell owns one sub-band, so the inner
product between two cells of the same page is exactly zero up to float32
rounding. Only three contiguous pages live in the active window; outer pages
are committed to Zone C and paged back by a velocity-predicted prefetch.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

D_SLOT = 256
B_PAGE = 16
D_CELL = D_SLOT // B_PAGE
_EPS = 1e-12


class HarmonicComb:
    """Disjoint 16-channel comb codes for one page of 16 cells."""

    def __init__(self, seed: int = 20261111, alphabet: Sequence[str] = ("0", "1", "#", "_")) -> None:
        self.seed = int(seed)
        self.alphabet = [str(symbol) for symbol in alphabet]
        generator = np.random.default_rng(self.seed)
        prototypes = generator.standard_normal((len(self.alphabet), D_CELL)) + 1j * generator.standard_normal(
            (len(self.alphabet), D_CELL)
        )
        norms = np.linalg.norm(prototypes, axis=1, keepdims=True)
        self.prototypes = prototypes / np.maximum(norms, _EPS)
        self.symbol_index = {symbol: index for index, symbol in enumerate(self.alphabet)}

    def embed(self, symbol: str, cell: int) -> np.ndarray:
        vector = np.zeros(D_SLOT, dtype=np.complex128)
        index = self.symbol_index.get(str(symbol), self.symbol_index.get("_", 0))
        start = int(cell) * D_CELL
        vector[start : start + D_CELL] = self.prototypes[index]
        return vector

    def page_wave(self, symbols: Sequence[str]) -> np.ndarray:
        wave = np.zeros(D_SLOT, dtype=np.complex128)
        for cell, symbol in enumerate(symbols[:B_PAGE]):
            wave = wave + self.embed(str(symbol), cell)
        return wave / np.sqrt(float(B_PAGE))

    def read(self, wave: np.ndarray, cell: int) -> Tuple[str, float]:
        start = int(cell) * D_CELL
        channel = np.asarray(wave[start : start + D_CELL], dtype=np.complex128)
        norm = float(np.linalg.norm(channel))
        if norm <= _EPS:
            return "_", 0.0
        alignment = np.real(self.prototypes.conj() @ channel) / norm
        best = int(np.argmax(alignment))
        second = float(np.max(np.delete(alignment, best))) if alignment.size > 1 else 0.0
        margin = float(alignment[best] - second)
        return self.alphabet[best], margin

    def decode(self, wave: np.ndarray) -> List[str]:
        return [self.read(wave, cell)[0] for cell in range(B_PAGE)]

    def cell_crosstalk(self, symbols: Sequence[str]) -> float:
        """Maximum inter-cell coherence. Disjoint sub-bands give exactly 0."""
        codes = np.stack(
            [self.embed(str(symbol), cell) for cell, symbol in enumerate(symbols[:B_PAGE])]
        )
        gram = np.abs(codes @ codes.conj().T) ** 2
        np.fill_diagonal(gram, 0.0)
        return float(np.sqrt(gram.max())) if gram.size else 0.0


class PagedHarmonicTape:
    """A 3-page fovea over an unbounded paged harmonic tape."""

    def __init__(
        self,
        *,
        memory=None,
        length: int = 256,
        alphabet: Sequence[str] = ("0", "1", "#", "_"),
        namespace: str = "tape_pages",
        seed: int = 20261111,
    ) -> None:
        self.length = int(length)
        self.alphabet = [str(symbol) for symbol in alphabet]
        self.memory = memory
        self.namespace = str(namespace)
        self.comb = HarmonicComb(seed=seed, alphabet=self.alphabet)
        self.pages: Dict[int, List[str]] = {}
        self.fovea: Dict[int, np.ndarray] = {}
        self.page_engram: Dict[int, int] = {}
        self.head = 0
        self.active_page = 0
        self.page_ins = 0
        self.page_outs = 0
        self.prefetches = 0
        self.moves = 0
        self.read_margins: List[float] = []
        for page in range(max(1, (self.length + B_PAGE - 1) // B_PAGE)):
            self.pages[page] = ["_"] * B_PAGE
        self._ensure_fovea(0)

    # --------------------------------------------------------------- pages
    def _page_wave(self, page: int) -> np.ndarray:
        return self.comb.page_wave(self.pages[page])

    def _ensure_fovea(self, page: int) -> None:
        for offset in (-1, 0, 1):
            index = int(page) + offset
            if index < 0 or index not in self.pages:
                continue
            if index in self.fovea:
                continue
            if self.memory is not None and index in self.page_engram:
                try:
                    wave = self.memory.fetch_wave(int(self.page_engram[index]))
                    self.fovea[index] = np.asarray(wave, dtype=np.complex128)
                    self.pages[index] = self.comb.decode(self.fovea[index])
                    self.page_ins += 1
                    continue
                except Exception:
                    pass
            self.fovea[index] = self._page_wave(index)
        # Evict pages outside the 3-page halo.
        for index in list(self.fovea):
            if abs(index - int(page)) > 1:
                self._page_out(index)
                self.fovea.pop(index, None)

    def _page_out(self, page: int) -> None:
        if self.memory is None:
            return
        wave = self._page_wave(page)
        try:
            record = self.memory.write(
                wave.astype(np.complex64), label=int(page), tier=2, namespace=self.namespace
            )
            self.page_engram[int(page)] = int(record.engram_id)
            self.page_outs += 1
        except Exception:
            pass

    def prefetch(self, velocity: int) -> None:
        if int(velocity) == 0:
            return
        target = int(self.active_page) + (1 if int(velocity) > 0 else -1) * 2
        if target in self.pages and target not in self.fovea:
            self._ensure_fovea(int(self.active_page))
            self.prefetches += 1

    # ---------------------------------------------------------------- head
    def page_of(self, cell: int) -> int:
        return int(cell) // B_PAGE

    def offset_of(self, cell: int) -> int:
        return int(cell) % B_PAGE

    def seek(self, cell: int) -> None:
        target = int(cell)
        velocity = 1 if target >= self.head else -1
        while self.head != target:
            self.move(velocity)

    def move(self, delta: int) -> None:
        target = self.head + int(delta)
        if target < 0 or target >= self.length:
            raise IndexError("head off tape")
        old_page = self.active_page
        self.head = target
        self.moves += 1
        self.active_page = self.page_of(target)
        if self.active_page != old_page:
            self._ensure_fovea(self.active_page)
            self.prefetch(int(delta))

    def read(self, cell: Optional[int] = None) -> str:
        if cell is not None:
            self.seek(int(cell))
        page = self.page_of(self.head)
        offset = self.offset_of(self.head)
        self._ensure_fovea(page)
        wave = self.fovea.get(page)
        if wave is None:
            return "_"
        symbol, margin = self.comb.read(wave, offset)
        self.read_margins.append(float(margin))
        return symbol

    def write(self, symbol: str, cell: Optional[int] = None) -> None:
        if cell is not None:
            self.seek(int(cell))
        page = self.page_of(self.head)
        offset = self.offset_of(self.head)
        self._ensure_fovea(page)
        self.pages[page][offset] = str(symbol)
        self.fovea[page] = self._page_wave(page)

    def tape_text(self) -> str:
        text = "".join(symbol for page in sorted(self.pages) for symbol in self.pages[page])
        return text[: self.length]

    def load(self, text: str) -> None:
        symbols = list(str(text)) + ["_"] * max(0, self.length - len(str(text)))
        for index, symbol in enumerate(symbols[: self.length]):
            page, offset = self.page_of(index), self.offset_of(index)
            self.pages.setdefault(page, ["_"] * B_PAGE)[offset] = symbol
        for page in list(self.fovea):
            self.fovea[page] = self._page_wave(page)

    # --------------------------------------------------------------- gates
    def intra_page_crosstalk(self) -> float:
        symbols = (self.alphabet * 4)[:B_PAGE]
        return float(self.comb.cell_crosstalk(symbols))

    def min_read_margin(self) -> float:
        return float(min(self.read_margins)) if self.read_margins else 0.0

    def stats(self) -> Dict[str, object]:
        return {
            "length": int(self.length),
            "pages": int(len(self.pages)),
            "page_capacity": int(B_PAGE),
            "d_slot": int(D_SLOT),
            "d_cell": int(D_CELL),
            "head": int(self.head),
            "page_ins": int(self.page_ins),
            "page_outs": int(self.page_outs),
            "prefetches": int(self.prefetches),
            "moves": int(self.moves),
            "reads": int(len(self.read_margins)),
            "min_read_margin": float(self.min_read_margin()),
            "intra_page_crosstalk": float(self.intra_page_crosstalk()),
        }


def reverse_tape(tape: PagedHarmonicTape, length: int) -> str:
    for index in range(int(length) // 2):
        left = tape.read(index)
        right = tape.read(int(length) - 1 - index)
        tape.write(right, index)
        tape.write(left, int(length) - 1 - index)
    return tape.tape_text()[: int(length)]


def palindrome_check(tape: PagedHarmonicTape, length: int) -> bool:
    for index in range(int(length) // 2):
        if tape.read(index) != tape.read(int(length) - 1 - index):
            return False
    return True


def binary_add(tape: PagedHarmonicTape, length: int) -> str:
    """Read A#B, write A#B#A+B on the paged tape."""
    text = "".join(tape.read(index) for index in range(int(length))).strip("_")
    if "#" not in text:
        raise ValueError("binary_add needs A#B")
    left, right = text.split("#", 1)
    left = left or "0"
    right = right.split("#", 1)[0] or "0"
    total = int(left, 2) + int(right, 2)
    result = bin(total)[2:]
    output = f"{left}#{right}#{result}"
    tape.load(output)
    return output
