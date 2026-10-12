"""Config tests. Pin the dimension and slot layout decisions."""

from __future__ import annotations

import pytest

from henri.config import BLOCK_DIM, N_SLOTS, SLOT_NAMES, WaveConfig


def test_default_dimension_is_theory_value():
    assert WaveConfig().dim == 65536


def test_block_count_is_pinned():
    """8192 blocks at the default dimension. Layout is a frozen choice."""
    assert WaveConfig().blocks == 8192


def test_slot_layout_is_pinned():
    """Four slots tile the vector exactly, in structural order."""
    cfg = WaveConfig(dim=512)
    assert cfg.slot_span == 512 // N_SLOTS
    ranges = cfg.slot_ranges()
    assert len(ranges) == N_SLOTS
    assert ranges[0].start == 0
    assert ranges[-1].stop == 512
    for i in range(N_SLOTS - 1):
        assert ranges[i].stop == ranges[i + 1].start


def test_slot_names_order():
    assert SLOT_NAMES == ("entity", "action", "target", "context")


def test_dim_must_divide_slot_span():
    with pytest.raises(ValueError, match="divisible"):
        WaveConfig(dim=N_SLOTS * BLOCK_DIM + 1)


def test_bad_rank_rejected():
    with pytest.raises(ValueError, match="low-rank"):
        WaveConfig(dim=512, r_koopman=0)


def test_kernel_arrays_must_align():
    with pytest.raises(ValueError, match="align"):
        WaveConfig(dim=512, kernel_taus=(1e-3,), kernel_weights=(0.5, 0.5))


def test_slot_slice_out_of_range():
    cfg = WaveConfig(dim=512)
    with pytest.raises(IndexError):
        cfg.slot_slice(4)


def test_config_is_immutable():
    cfg = WaveConfig(dim=512)
    with pytest.raises(Exception):
        cfg.dim = 1024  # type: ignore[misc]


def test_small_preset_keeps_structure():
    small = WaveConfig.small()
    assert small.dim == 512
    assert small.slot_span == 128
    assert small.blocks == 64
