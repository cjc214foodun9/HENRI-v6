"""Readout tests: tie-robust selection, no smuggled floor, typed envelopes."""

from __future__ import annotations

import numpy as np
import pytest

from henri.readout import (
    ACTION_KINDS,
    centroid_snap,
    make_envelope,
    max_group_indices,
    normalized_entropy,
    scores_of,
    select,
)
from henri.state import WaveState


def test_max_group_returns_whole_tied_set(cfg):
    """Constraint C11: a plateau must yield the full tied set."""
    state = WaveState.random(cfg, seed=1)
    book = np.vstack([state.psi, state.psi, state.psi]).astype(np.complex128)
    scores = scores_of(state, book)
    group = max_group_indices(scores)
    assert len(group) == 3


def test_monotone_transform_cannot_split_a_tie(cfg):
    """FALSIFIED: softmax(s/k) is monotone in s, so argmax is unchanged.

    This is the measured reason a temperature cannot fix a tie plateau.
    """
    state = WaveState.random(cfg, seed=2)
    book = np.vstack([state.psi, state.psi]).astype(np.complex128)
    raw_scores = scores_of(state, book)

    for kappa in (0.038316, 1.0, 100.0):
        z = raw_scores / kappa
        p = np.exp(z - np.max(z))
        assert int(np.argmax(p)) == int(np.argmax(raw_scores))
        assert np.argmax(p) == np.argmax(raw_scores)


def test_select_uses_group_key_to_break_tie(cfg):
    state = WaveState.random(cfg, seed=3)
    book = np.vstack([state.psi, state.psi, state.psi]).astype(np.complex128)
    key = np.array([0.1, 0.9, 0.2])
    result = select(state, book, group_key=key)
    assert result.group_size == 3
    assert result.index == 1


def test_select_group_key_wrong_length(cfg):
    state = WaveState.random(cfg, seed=4)
    book = np.vstack([state.psi, -state.psi]).astype(np.complex128)
    with pytest.raises(ValueError, match="group_key must have length"):
        select(state, book, group_key=np.array([1.0]))


def test_no_floor_means_no_abstention(cfg, codebook):
    state = WaveState.random(cfg, seed=5)
    result = select(state, codebook, floor=None)
    assert result.abstained is False
    assert result.index >= 0
    assert result.meta["floor"] is None


def test_floor_suppresses_emission(cfg, codebook):
    state = WaveState.random(cfg, seed=6)
    result = select(state, codebook, floor=1e9)
    assert result.abstained is True
    assert result.index == -1


def test_margin_is_nonnegative(cfg, codebook):
    state = WaveState.random(cfg, seed=7)
    assert select(state, codebook).margin >= 0.0


def test_entropy_bounds(cfg, codebook):
    state = WaveState.random(cfg, seed=8)
    e = select(state, codebook).entropy
    assert 0.0 <= e <= 1.0


def test_entropy_is_max_for_flat_scores():
    flat = np.zeros(32)
    assert abs(normalized_entropy(flat) - 1.0) < 1e-12


def test_entropy_is_min_for_peaked_scores():
    peaked = np.array([10.0, 0.0, 0.0, 0.0])
    assert normalized_entropy(peaked) < 0.1


def test_codebook_shape_validated(cfg):
    state = WaveState.random(cfg, seed=9)
    with pytest.raises(ValueError, match="codebook must be"):
        scores_of(state, np.zeros((3, 7), dtype=np.complex128))


def test_centroid_snap_cosine_picks_correct(cfg):
    c = np.zeros((3, cfg.dim), dtype=np.complex128)
    c[1, 0] = 1.0
    query = np.zeros(cfg.dim, dtype=np.complex128)
    query[0] = 1.0
    assert centroid_snap(query, c, metric="cosine") == 1


def test_centroid_snap_euclidean_picks_correct(cfg):
    c = np.zeros((3, cfg.dim), dtype=np.complex128)
    c[2, 0] = 1.0
    query = np.zeros(cfg.dim, dtype=np.complex128)
    query[0] = 0.9
    assert centroid_snap(query, c, metric="euclidean") == 2


def test_centroid_snap_bad_metric(cfg):
    c = np.zeros((2, cfg.dim), dtype=np.complex128)
    with pytest.raises(ValueError, match="unknown metric"):
        centroid_snap(np.zeros(cfg.dim, dtype=np.complex128), c, metric="manhattan")


def test_envelope_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unknown action kind"):
        make_envelope("teleport", {}, 0.9, True)


def test_envelope_clamps_confidence():
    env = make_envelope("translate", {"dx": 1.0}, 5.0, True)
    assert env.confidence == 1.0
    env2 = make_envelope("translate", {"dx": 1.0}, -3.0, True)
    assert env2.confidence == 0.0


def test_action_kinds_are_typed_not_a_token_vocabulary():
    """No 32k token head. Six typed families only (constraint C3)."""
    assert len(ACTION_KINDS) == 6
    assert "abstain" in ACTION_KINDS


def test_envelope_dict_roundtrip():
    env = make_envelope("segment", {"c": 3}, 0.8, True)
    d = env.as_dict()
    assert d["kind"] == "segment"
    assert d["verified"] is True
