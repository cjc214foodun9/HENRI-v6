"""Sensor ingress: REAL bytes to a settled wave state.

Evidence label: DERIVED for the interface; OBSERVED for every number
`tests/test_sensor_ingress.py` measures.

WHAT THIS CLOSES, AND WHAT IT DOES NOT
--------------------------------------
Phase 2 built three encoders over SYNTHETIC arrays. The owner report
named the gap precisely: the engine has no transmission connected to the
wheels. This module is the first stage of that transmission. It accepts
REAL bytes -- a photograph on disk -- and drives the existing
`ContinuousSpatialEncoder` -> `SubspaceRouter` -> `WaveState` path.

It closes the SENSOR half. It does NOT close the actuator half: there is
no action egress and no environment in this module. Stating the split
precisely matters more than a broad claim.

REAL DATA, NOT SYNTHETIC
------------------------
`FileImageSource` reads actual JPEG/PNG/PNM bytes from disk. The probe
uses Windows wallpaper photographs: real photographic content with
natural image statistics, not Gaussian draws. The source hash is
recorded so the exact bytes are pinned.

A source that cannot decode FAILS CLOSED. It never substitutes
synthetic bytes. A silent substitution would reintroduce exactly the
synthetic-input problem this module exists to remove.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Protocol, Tuple

import numpy as np

from henri.config import WaveConfig
from henri.ingress_subspace_router import SubspaceRouter
from henri.spatial_encoder import ContinuousSpatialEncoder
from henri.state import WaveState


class SensorSource(Protocol):
    """A source of real sensor bytes. `read` returns H x W float in [0, 1]."""

    def read(self) -> np.ndarray: ...

    @property
    def name(self) -> str: ...

    @property
    def sha256(self) -> str: ...


def to_gray_float(arr: np.ndarray) -> np.ndarray:
    """Decoded image array to H x W float64 in [0, 1].

    Colour is collapsed with Rec. 709 luma weights. An alpha channel is
    dropped, never averaged into the signal. A single-channel array is
    squeezed to 2-D: the PNM reader returns `(h, w, 1)` for P5.
    """
    a = np.asarray(arr)
    if a.ndim == 3:
        if a.shape[2] == 4:
            a = a[:, :, :3]
        if a.shape[2] == 1:
            a = a[:, :, 0]
        elif a.shape[2] != 3:
            raise ValueError(f"expected 1, 3 or 4 channels, got {a.shape[2]}")
        else:
            a = (
                0.2126 * a[:, :, 0].astype(np.float64)
                + 0.7152 * a[:, :, 1].astype(np.float64)
                + 0.0722 * a[:, :, 2].astype(np.float64)
            )
    if a.ndim != 2:
        raise ValueError(f"expected H x W or H x W x C, got shape {a.shape}")
    a = a.astype(np.float64)
    lo, hi = float(a.min()), float(a.max())
    if hi > lo:
        return (a - lo) / (hi - lo)
    return np.zeros_like(a)


def read_pnm(path: Path) -> np.ndarray:
    """Minimal binary PGM/PPM reader. Standard library only."""
    data = path.read_bytes()
    tokens: List[bytes] = []
    i = 0
    while len(tokens) < 4 and i < len(data):
        while i < len(data) and data[i : i + 1].isspace():
            i += 1
        if i < len(data) and data[i : i + 1] == b"#":
            while i < len(data) and data[i] != 0x0A:
                i += 1
            continue
        start = i
        while i < len(data) and not data[i : i + 1].isspace():
            i += 1
        tokens.append(data[start:i])
    if len(tokens) < 4:
        raise ValueError("truncated PNM header")
    magic, w, h, maxval = tokens[0], int(tokens[1]), int(tokens[2]), int(tokens[3])
    if magic not in (b"P5", b"P6"):
        raise ValueError(f"unsupported PNM magic {magic!r}")
    nch = 1 if magic == b"P5" else 3
    body = data[i + 1 :]
    itemsize = 1 if maxval < 256 else 2
    need = w * h * nch * itemsize
    if len(body) < need:
        raise ValueError(f"truncated PNM: need {need} bytes, got {len(body)}")
    dtype = np.uint8 if itemsize == 1 else np.dtype(">u2")
    arr = np.frombuffer(body[:need], dtype=dtype).reshape(h, w, nch)
    if nch == 1:
        return arr[:, :, 0].copy()
    return arr


def decode_image(path: os.PathLike | str) -> np.ndarray:
    """Decode a real image file to H x W float in [0, 1].

    Fails closed. PIL decodes JPEG/PNG; PNM uses the stdlib reader.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"no such image: {p}")
    if p.suffix.lower() in (".ppm", ".pgm", ".pnm"):
        return to_gray_float(read_pnm(p))
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            f"cannot decode {p.suffix!r} without PIL; refusing to "
            "substitute synthetic bytes for real sensor data"
        ) from exc
    with Image.open(p) as im:
        rgb = im.convert("RGB")
        return to_gray_float(np.asarray(rgb))


@dataclass(frozen=True)
class FileImageSource:
    """Read real image bytes from a path. Fails closed on bad input."""

    path: Path

    def __init__(self, path: os.PathLike | str) -> None:
        object.__setattr__(self, "path", Path(path))
        if not self.path.is_file():
            raise FileNotFoundError(f"no such image: {self.path}")

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.path.read_bytes()).hexdigest()

    def read(self) -> np.ndarray:
        return decode_image(self.path)


def resize_nearest(field: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Nearest-neighbour resize. No interpolation smoothing."""
    f = np.asarray(field, dtype=np.float64)
    h, w = f.shape
    if out_h < 1 or out_w < 1:
        raise ValueError("target size must be >= 1")
    rows = (np.arange(out_h) * h // out_h).clip(0, h - 1)
    cols = (np.arange(out_w) * w // out_w).clip(0, w - 1)
    return f[rows][:, cols]


@dataclass(frozen=True)
class IngressResult:
    """One real-bytes ingestion with its provenance."""

    source_name: str
    source_sha256: str
    height: int
    width: int
    slot: str
    wave: WaveState
    field: np.ndarray

    def as_dict(self) -> dict:
        return {
            "source_name": self.source_name,
            "source_sha256": self.source_sha256,
            "height": self.height,
            "width": self.width,
            "slot": self.slot,
            "norm": float(self.wave.norm),
            "unit_norm_error": float(self.wave.unit_norm_error()),
            "slot_energy": [float(x) for x in self.wave.slot_energy()],
        }


class SensorIngress:
    """Real bytes -> field -> spatial encoder -> router -> WaveState."""

    def __init__(self, cfg: Optional[WaveConfig] = None, seed: int = 20261012) -> None:
        self.cfg = cfg if cfg is not None else WaveConfig()
        self.encoder = ContinuousSpatialEncoder(self.cfg, seed=seed)
        self.router = SubspaceRouter(self.cfg)

    def ingest(
        self,
        source: SensorSource,
        slot: str = "entity",
        field_size: Optional[int] = None,
    ) -> IngressResult:
        """Read real bytes, encode to a slot, return a settled wave.

        Args:
            source: any `SensorSource`.
            slot: one of `SLOT_NAMES`.
            field_size: optional square resize target.

        Returns:
            An `IngressResult` carrying the source hash and the wave.
        """
        field = source.read()
        if field_size is not None:
            field = resize_nearest(field, field_size, field_size)
        psi = self.encoder.encode_2d_field(field)
        wave = self.router.assemble_ingress(**{slot: psi})
        return IngressResult(
            source_name=source.name,
            source_sha256=getattr(source, "sha256", ""),
            height=int(field.shape[0]),
            width=int(field.shape[1]),
            slot=slot,
            wave=wave,
            field=field,
        )


def list_real_images(root: os.PathLike | str, limit: int = 8) -> List[Path]:
    """Find real image files under a root. Sorted for determinism."""
    base = Path(root)
    found: List[Path] = []
    if not base.is_dir():
        return found
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.pgm", "*.ppm"):
        found.extend(base.rglob(pattern))
    return sorted(found)[:limit]


__all__ = [
    "SensorSource",
    "FileImageSource",
    "SensorIngress",
    "IngressResult",
    "decode_image",
    "read_pnm",
    "to_gray_float",
    "resize_nearest",
    "list_real_images",
]
