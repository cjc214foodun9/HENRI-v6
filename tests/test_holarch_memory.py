"""Zone C persistence, retrieval, decay, and compaction."""

from __future__ import annotations

import time

import numpy as np
import pytest

from henri.holarch.config import HolarchConfig
from henri.holarch.zone_c import ZoneCMemory


def _waves(cfg, count, seed=3, spread=1.0):
    generator = np.random.default_rng(seed)
    out = []
    for _ in range(count):
        wave = generator.standard_normal(cfg.dim) + 1j * generator.standard_normal(cfg.dim)
        out.append((spread * wave / np.linalg.norm(wave)).astype(np.complex64))
    return out


def test_sqlite_roundtrip_survives_reopen(tmp_path):
    cfg = HolarchConfig.small()
    path = tmp_path / "zone_c.sqlite3"
    memory = ZoneCMemory(cfg, backend="sqlite", path=str(path))
    waves = _waves(cfg, 12)
    for index, wave in enumerate(waves):
        memory.write(wave, label=index % 4, ts=1000.0 + index)
    read = memory.query(waves[5], k=3)
    assert read.hits and read.hits[0].engram_id == 6
    memory.close()

    reopened = ZoneCMemory(cfg, backend="sqlite", path=str(path))
    assert reopened.stats()["count"] == 12
    again = reopened.query(waves[5], k=3)
    assert again.hits[0].engram_id == 6
    reopened.close()


def test_context_shape_is_invariant_in_store_size(tmp_path):
    cfg = HolarchConfig.small()
    memory = ZoneCMemory(cfg, backend="sqlite", path=str(tmp_path / "c.sqlite3"))
    waves = _waves(cfg, 40)
    shapes = []
    for index, wave in enumerate(waves):
        memory.write(wave, label=index % 5)
        if index in (0, 9, 39):
            prefix, _ = memory.context_tensor(wave, k=4)
            shapes.append(prefix.shape)
    assert len(set(shapes)) == 1
    assert shapes[0] == (cfg.context_tokens, cfg.context_dim)
    memory.close()


def test_retrieval_recall_and_scores(tmp_path):
    cfg = HolarchConfig.small()
    memory = ZoneCMemory(cfg, backend="sqlite", path=str(tmp_path / "r.sqlite3"))
    waves = _waves(cfg, 30)
    for index, wave in enumerate(waves):
        memory.write(wave, label=index % 3)
    correct = 0
    for index, wave in enumerate(waves):
        read = memory.query(wave, k=2)
        correct += int(bool(read.hits) and read.hits[0].engram_id == index + 1)
    assert correct / len(waves) > 0.9
    assert memory.query(waves[0], k=2).latency_ms >= 0.0
    memory.close()


def test_viscoelastic_decay_and_prune_protect_axioms(tmp_path):
    cfg = HolarchConfig.small()
    memory = ZoneCMemory(cfg, backend="sqlite", path=str(tmp_path / "d.sqlite3"))
    waves = _waves(cfg, 3)
    for index, wave in enumerate(waves):
        memory.write(wave, label=index, ts=100.0, tier=3 if index == 0 else 1)
    weight = memory.decay(now=100.0 + 10_000_000.0)
    assert weight[0] == pytest.approx(1.0)
    assert weight[1] < 1e-4
    report = memory.prune(now=100.0 + 10_000_000.0)
    assert report["pruned"] == 2
    assert report["protected_axiomatic"] == 1
    assert memory.stats()["count"] == 1
    memory.close()


def test_consolidation_reduces_gram_crosstalk(tmp_path):
    cfg = HolarchConfig.small()
    memory = ZoneCMemory(cfg, backend="sqlite", path=str(tmp_path / "g.sqlite3"))
    base = _waves(cfg, 1, seed=8)[0]
    generator = np.random.default_rng(9)
    for _ in range(24):
        noise = generator.standard_normal(cfg.dim) + 1j * generator.standard_normal(cfg.dim)
        noise = (noise / np.linalg.norm(noise)).astype(np.complex64)
        wave = base + 0.6 * noise
        memory.write((wave / np.linalg.norm(wave)).astype(np.complex64), label=1)
    report = memory.consolidate(chart_size=24)
    assert report["crosstalk_after"] < report["crosstalk_before"]
    assert report["crosstalk_after"] < 1e-5
    memory.close()


def test_timescaledb_roundtrip_when_reachable():
    cfg = HolarchConfig.small()
    try:
        from henri.holarch.zone_c import TimescaleEngramStore

        store = TimescaleEngramStore(table="zone_c_engram_unit_test")
    except Exception as exc:
        pytest.skip(f"TimescaleDB unavailable: {exc}")
    store.reset()
    memory = ZoneCMemory(cfg, store=store)
    waves = _waves(cfg, 6)
    for index, wave in enumerate(waves):
        memory.write(wave, label=index % 2)
    assert memory.stats()["count"] == 6
    read = memory.query(waves[2], k=2)
    assert read.hits and read.hits[0].engram_id == 3
    store.reset()
    store.close()


def test_no_dense_projection_is_formed():
    cfg = HolarchConfig.small()
    assert cfg.gram_dims < cfg.dim
    assert cfg.cache_capacity < 1024
