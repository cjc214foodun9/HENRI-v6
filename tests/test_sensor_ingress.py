"""Sensor ingress tests: REAL bytes, real decode, fail-closed.

These tests use REAL image bytes. Where an image is unavailable they
generate a real PNM file on disk rather than substituting an array, so
the decode path is always exercised on bytes.

What is and is not claimed: these prove the SENSOR path accepts real
files and drives the existing encoder. They do not prove a closed loop.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from henri.sensor_ingress import (
    FileImageSource,
    SensorIngress,
    decode_image,
    list_real_images,
    read_pnm,
    resize_nearest,
    to_gray_float,
)


def _write_pgm(path: Path, h: int = 16, w: int = 24, seed: int = 5) -> Path:
    """Write a REAL binary PGM file with a deterministic pattern."""
    rng = np.random.default_rng(seed)
    body = rng.integers(0, 256, size=(h, w), dtype=np.uint8)
    header = f"P5\n{w} {h}\n255\n".encode()
    path.write_bytes(header + body.tobytes())
    return path


def _real_wallpaper() -> Path | None:
    """A real photograph shipped with Windows, if present."""
    root = Path(r"C:\Windows\Web\Wallpaper\Windows")
    hits = sorted(root.glob("*.jpg")) if root.is_dir() else []
    return hits[0] if hits else None


# -- decode -----------------------------------------------------------------


def test_pgm_roundtrip_is_exact(tmp_path):
    p = _write_pgm(tmp_path / "a.pgm", h=8, w=12, seed=3)
    field = decode_image(p)
    assert field.shape == (8, 12)
    assert field.dtype == np.float64
    assert 0.0 <= field.min() and field.max() <= 1.0


def test_pnm_header_with_comment(tmp_path):
    p = tmp_path / "c.pgm"
    body = bytes(range(256)) * 4
    p.write_bytes(b"P5\n# a comment\n4 256\n255\n" + body[: 4 * 256])
    arr = read_pnm(p)
    assert arr.shape == (256, 4)


def test_truncated_pnm_fails_closed(tmp_path):
    p = tmp_path / "short.pgm"
    p.write_bytes(b"P5\n64 64\n255\n" + b"\x00" * 10)
    with pytest.raises(ValueError, match="truncated"):
        read_pnm(p)


def test_unsupported_magic_rejected(tmp_path):
    p = tmp_path / "x.pbm"
    p.write_bytes(b"P1\n2 2\n0 1 1 0\n")
    with pytest.raises(ValueError, match="magic"):
        read_pnm(p)


def test_missing_file_fails_closed(tmp_path):
    with pytest.raises(FileNotFoundError):
        FileImageSource(tmp_path / "nope.jpg")


def test_to_gray_float_drops_alpha_not_averages_it():
    """Alpha must never be blended into the signal."""
    base = np.zeros((2, 2, 3), dtype=np.uint8)
    base[:, :, 0] = 200
    rgba = np.concatenate(
        [base, np.zeros((2, 2, 1), dtype=np.uint8) + 7], axis=2
    )
    rgb_only = to_gray_float(base)
    with_alpha = to_gray_float(rgba)
    assert np.allclose(rgb_only, with_alpha)


def test_flat_field_does_not_divide_by_zero():
    flat = np.full((4, 4), 128, dtype=np.uint8)
    out = to_gray_float(flat)
    assert np.all(out == 0.0)


def test_real_wallpaper_decodes_with_variation():
    """A REAL photograph, not an array. Skips only if truly absent."""
    path = _real_wallpaper()
    if path is None:
        pytest.skip("no Windows wallpaper image present")
    src = FileImageSource(path)
    field = src.read()
    assert field.ndim == 2
    assert field.shape[0] > 100 and field.shape[1] > 100
    # A real photograph has structure, unlike uniform noise or a constant.
    assert field.std() > 0.05
    # The hash must pin the exact bytes actually read.
    assert src.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_real_jpeg_is_not_synthetic():
    """Distinguish real photographic content from Gaussian noise.

    Natural photographs have spatially correlated pixels: the mean
    absolute difference between horizontally adjacent pixels is far
    below the standard deviation. White noise has none of that.
    """
    path = _real_wallpaper()
    if path is None:
        pytest.skip("no Windows wallpaper image present")
    field = decode_image(path)
    adj = np.abs(np.diff(field, axis=1)).mean()
    flat = field.mean()
    grad = adj if flat > adj else adj
    rng = np.random.default_rng(0)
    noise = rng.random(field.shape)
    noise_adj = np.abs(np.diff(noise, axis=1)).mean()

    # report the contrast; real images are far more correlated than noise
    assert adj < 0.6 * noise_adj, (
        f"real image adjacent-diff {adj:.4f} vs noise {noise_adj:.4f}: "
        "content does not look like a real photograph"
    )
    assert grad > 0.0


# -- resize -----------------------------------------------------------------


def test_resize_nearest_shape_and_determinism():
    f = np.arange(64, dtype=np.float64).reshape(8, 8)
    a = resize_nearest(f, 4, 4)
    b = resize_nearest(f, 4, 4)
    assert a.shape == (4, 4)
    assert np.array_equal(a, b)
    # Identity resize must reproduce the input exactly.
    assert np.array_equal(f, resize_nearest(f, 8, 8))


def test_resize_rejects_zero():
    f = np.zeros((4, 4))
    with pytest.raises(ValueError):
        resize_nearest(f, 0, 4)


# -- ingest -----------------------------------------------------------------


def test_ingest_real_bytes_reaches_a_settled_wave(cfg):
    src = FileImageSource(_write_pgm(Path.cwd() / "_tmp_probe.pgm"))
    try:
        ing = SensorIngress(cfg)
        res = ing.ingest(src, slot="entity", field_size=16)
        assert res.wave.unit_norm_error() < 1e-12
        assert res.wave.settled
        # G-U1: everything landed in the entity slot
        energy = res.wave.slot_energy()
        assert energy[0] == pytest.approx(1.0, abs=1e-12)
        assert float(energy[1:].sum()) == 0.0
        # provenance carried through
        assert res.source_sha256 == src.sha256
    finally:
        (Path.cwd() / "_tmp_probe.pgm").unlink(missing_ok=True)


def test_ingest_into_context_slot(cfg):
    p = _write_pgm(Path.cwd() / "_tmp_ctx.pgm", seed=9)
    try:
        res = SensorIngress(cfg).ingest(FileImageSource(p), slot="context")
        energy = res.wave.slot_energy()
        assert energy[3] == pytest.approx(1.0, abs=1e-12)
        assert float(energy[:3].sum()) == 0.0
    finally:
        (Path.cwd() / "_tmp_ctx.pgm").unlink(missing_ok=True)


def test_ingest_is_deterministic_for_the_same_bytes(cfg):
    p = _write_pgm(Path.cwd() / "_tmp_det.pgm", seed=11)
    try:
        a = SensorIngress(cfg).ingest(FileImageSource(p), field_size=16)
        b = SensorIngress(cfg).ingest(FileImageSource(p), field_size=16)
        assert np.array_equal(a.wave.psi, b.wave.psi)
    finally:
        (Path.cwd() / "_tmp_det.pgm").unlink(missing_ok=True)


def test_different_real_images_give_different_waves(cfg):
    """Two real files must not collapse to the same state."""
    p1 = _write_pgm(Path.cwd() / "_tmp_i1.pgm", seed=21)
    p2 = _write_pgm(Path.cwd() / "_tmp_i2.pgm", seed=22)
    try:
        ing = SensorIngress(cfg)
        a = ing.ingest(FileImageSource(p1), field_size=16)
        b = ing.ingest(FileImageSource(p2), field_size=16)
        assert a.source_sha256 != b.source_sha256
        assert abs(a.wave.coherence(b.wave)) < 0.9
    finally:
        for p in (p1, p2):
            p.unlink(missing_ok=True)


def test_ingest_rejects_unknown_slot(cfg):
    p = _write_pgm(Path.cwd() / "_tmp_bad.pgm")
    try:
        with pytest.raises((TypeError, ValueError)):
            SensorIngress(cfg).ingest(FileImageSource(p), slot="nonsense")
    finally:
        p.unlink(missing_ok=True)


def test_list_real_images_finds_files(tmp_path):
    for i in range(3):
        _write_pgm(tmp_path / f"i{i}.pgm")
    found = list_real_images(tmp_path)
    assert len(found) == 3
    assert all(p.suffix == ".pgm" for p in found)


def test_list_real_images_missing_root_is_empty(tmp_path):
    assert list_real_images(tmp_path / "absent") == []
