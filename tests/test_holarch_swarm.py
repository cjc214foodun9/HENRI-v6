"""Zone A routing, fast weights, and replay consolidation."""

from __future__ import annotations

import torch

from henri.holarch.config import HolarchConfig
from henri.holarch.zone_a import FastWeightExpert, LeanSwarm, ReplayBuffer


def _blobs(cfg, count=32, classes=5, seed=0):
    generator = torch.Generator().manual_seed(seed)
    centers = torch.randn(classes, cfg.d_feat, generator=generator) * 2.0
    labels = torch.randint(0, classes, (count,), generator=generator)
    features = centers[labels] + 0.6 * torch.randn(count, cfg.d_feat, generator=generator)
    return features, labels


def test_fast_weights_learn_without_backprop():
    cfg = HolarchConfig.small()
    torch.manual_seed(0)
    swarm = LeanSwarm(cfg)
    features, labels = _blobs(cfg, 512)
    train_x, train_y = features[:256], labels[:256]
    test_x, test_y = features[256:], labels[256:]

    def accuracy(x, y):
        output = swarm(x, 0)
        return float((swarm.logits_for(output.logits, "vision").argmax(-1) == y).float().mean())

    before = accuracy(test_x, test_y)
    for step in range(120):
        start = (step * 32) % (len(train_x) - 32)
        swarm.online_update(train_x[start : start + 32], train_y[start : start + 32], 0, lr=0.3)
    after = accuracy(test_x, test_y)
    assert after > before + 0.3
    assert after > 0.7


def test_label_blocks_are_disjoint():
    cfg = HolarchConfig.small()
    swarm = LeanSwarm(cfg)
    vision = swarm.task_slice("vision")
    language = swarm.task_slice("language")
    action = swarm.task_slice("action")
    relational = swarm.task_slice("relational")
    assert vision[1] <= language[0]
    assert language[1] <= action[0]
    assert action[1] <= relational[0]
    assert relational[1] == swarm.d_out


def test_routing_is_finite_and_balanced():
    cfg = HolarchConfig.small()
    swarm = LeanSwarm(cfg)
    features, _ = _blobs(cfg)
    output = swarm(features, 1, training=True)
    assert bool(torch.isfinite(output.logits).all())
    assert output.selected.shape == (features.shape[0], cfg.top_k)
    assert float(output.load_balance.detach()) >= 0.9
    assert torch.allclose(output.weights.sum(-1), torch.ones(features.shape[0]), atol=1e-5)


def test_replay_consolidation_runs_and_reduces_loss():
    cfg = HolarchConfig.small()
    swarm = LeanSwarm(cfg)
    buffer = ReplayBuffer(capacity=256)
    features, labels = _blobs(cfg, 128)
    buffer.add(features, labels, 0)
    optimizer = torch.optim.Adam([p for p in swarm.parameters() if p.requires_grad], lr=5e-3)
    first = swarm.consolidate(features, labels, 0, optimizer, steps=1)
    for _ in range(20):
        batch_x, batch_y = buffer.sample(0, 32)
        swarm.consolidate(batch_x, batch_y, 0, optimizer, steps=1)
    last = swarm.consolidate(features, labels, 0, optimizer, steps=1)
    assert last < first


def test_exploration_returns_a_finite_reward():
    cfg = HolarchConfig.small()
    swarm = LeanSwarm(cfg)
    features, _ = _blobs(cfg)
    perturbation, reward = swarm.explore(features, 0, steps=3)
    assert perturbation.shape == features.shape
    assert reward >= 0.0


def test_expert_novelty_is_bounded():
    expert = FastWeightExpert(d_in=16, hidden=8, d_out=5, rank=4)
    x = torch.randn(6, 16)
    novelty = expert.novelty(x)
    assert novelty.shape == (6,)
    assert bool(((novelty >= 0.0) & (novelty <= 2.0)).all())
