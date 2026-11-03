"""Dataset contracts for the three tasks."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from henri.holarch.config import HolarchConfig
from henri.holarch.tasks import build_tasks


@pytest.fixture(scope="module")
def tasks():
    return build_tasks(HolarchConfig.small(), seed=11)


def test_vision_splits_are_disjoint_and_shaped(tasks):
    task = tasks["vision"]
    assert task.size("train") == 1200
    assert task.size("val") == 300
    assert task.size("test") == 297
    batch = task.sample(7, "train")
    assert batch.x.shape == (7, 1, 8, 8)
    assert batch.y.shape == (7,)
    assert int(batch.y.max()) < 10


def test_language_vocabulary_and_windows(tasks):
    task = tasks["language"]
    assert len(task.vocab) == 96
    assert task.vocab[0] == "<unk>"
    batch = task.sample(5, "val")
    assert batch.x.shape == (5, 24)
    assert int(batch.x.max()) < len(task.vocab)
    assert "\n" not in task.vocab[0]


def test_language_train_and_val_windows_do_not_overlap(tasks):
    task = tasks["language"]
    train = set(task.index["train"].tolist())
    val = set(task.index["val"].tolist())
    assert not (train & val)


def test_action_pairs_are_balanced_and_labels_are_optimal(tasks):
    task = tasks["action"]
    batch = task.sample(400, "train")
    counts = torch.bincount(batch.y, minlength=4)
    assert int(counts.min()) > 60
    observation, action = task.train_pairs[0]
    walls, agent, goal = task.train_layouts[0]
    assert action == task.world.optimal_action(walls, agent, goal)
    assert observation.shape == (79,)


def test_action_labels_are_the_first_step_of_the_bfs_path():
    cfg = HolarchConfig.small()
    task = build_tasks(cfg, seed=2)["action"]
    checked = 0
    for walls, agent, goal in task.val_layouts:
        path = task.world._bfs(walls, agent, goal)
        assert path is not None
        assert task.world.optimal_action(walls, agent, goal) == path[0]
        assert task.world.rollout(walls, agent, goal, path)
        checked += 1
    assert checked == len(task.val_layouts)


def test_sampling_is_deterministic_for_a_fixed_seed():
    cfg = HolarchConfig.small()
    a = build_tasks(cfg, seed=5)["action"].sample(16, "train")
    b = build_tasks(cfg, seed=5)["action"].sample(16, "train")
    assert torch.equal(a.x, b.x)
    assert torch.equal(a.y, b.y)
