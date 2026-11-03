"""Configuration invariants for the holarch."""

from __future__ import annotations

import pytest

from henri.holarch.config import HolarchConfig


def test_small_and_production_profiles_are_valid():
    small = HolarchConfig.small()
    production = HolarchConfig.production()
    assert small.dim % (small.n_slots * 8) == 0
    assert production.dim % (production.n_slots * 8) == 0
    assert small.d_slot * small.n_slots == small.dim
    assert production.chunk * production.context_tokens == production.dim


def test_digest_is_stable_and_sensitive():
    a = HolarchConfig.small()
    b = HolarchConfig.small()
    c = HolarchConfig.small(dim=512)
    assert a.digest() == b.digest()
    assert a.digest() != c.digest()


def test_slot_slices_tile_the_vector():
    cfg = HolarchConfig.small()
    slices = [cfg.slot_slice(index) for index in range(cfg.n_slots)]
    assert slices[0].start == 0
    assert slices[-1].stop == cfg.dim
    for left, right in zip(slices, slices[1:]):
        assert left.stop == right.start


@pytest.mark.parametrize(
    "overrides",
    [
        {"dim": 100},
        {"n_slots": 2},
        {"t_max": 1.0},
        {"pockels_gain": 0.0},
        {"top_k": 99},
        {"context_tokens": 7},
        {"crosstalk_max": 1.0},
        {"device": "tpu"},
    ],
)
def test_invalid_configuration_is_refused(overrides):
    with pytest.raises(ValueError):
        HolarchConfig(**overrides)


def test_task_label_blocks_are_disjoint():
    cfg = HolarchConfig.small()
    blocks = [
        (0, cfg.n_vision_classes),
        (cfg.n_vision_classes, cfg.n_vision_classes + cfg.vocab_size),
        (
            cfg.n_vision_classes + cfg.vocab_size,
            cfg.n_vision_classes + cfg.vocab_size + cfg.n_actions,
        ),
        (
            cfg.n_vision_classes + cfg.vocab_size + cfg.n_actions,
            cfg.n_vision_classes + cfg.vocab_size + cfg.n_actions + cfg.relational_classes,
        ),
    ]
    for left, right in zip(blocks, blocks[1:]):
        assert left[1] <= right[0]
