"""Zone A: the lean continuous-learning expert swarm.

Evidence label: DERIVED for the routing algebra; OBSERVED for every
retention and accuracy number the holarch tests record.

Why a swarm, and what is actually lean here
-------------------------------------------
Each expert is a two-layer MLP with a rank-r associative fast weight
memory. The fast path is trained by the delta rule on every example,
with no backpropagation and no stored gradient. The slow path is trained
by the optimizer on a bounded replay buffer. That split is the whole
point: the system keeps learning after training stops, at a cost of
O(rank * d_out) per example.

Routing is a top-k mixture with a load-balancing auxiliary loss. A
novelty bonus, measured as the distance from an expert's own centroid,
biases exploration toward latent regions no expert has claimed. The
bonus is a training-time signal only; at inference the router is
deterministic.

Three anti-patterns are refused here, each from the v6 constraint list:

* C11: no bare top-k on a tie plateau. Expert selection is top-k, but
  ties break on expert index, and the *reported* accuracy never reads a
  rank order.
* C12: no readout floor from the random-wave distribution. Confidence is
  the softmax margin of the routed mixture on the task slice.
* C1: no resonator substitute for a linear readout. The readout is
  linear, and the photonic path feeds it features rather than replacing
  it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F

_EPS = 1e-8


@dataclass
class SwarmOutput:
    """Routing audit for one batch.

    Attributes:
        logits: mixed logits, shape (batch, d_out).
        expert_logits: per-expert logits, shape (n_experts, batch, d_out).
        router_probs: full softmax over experts, shape (batch, n_experts).
        weights: normalised top-k mixing weights, shape (batch, top_k).
        selected: expert indices, shape (batch, top_k).
        load_balance: switch-style auxiliary loss (scalar tensor).
        novelty: per-expert novelty score, shape (batch, n_experts).
    """

    logits: torch.Tensor
    expert_logits: torch.Tensor
    router_probs: torch.Tensor
    weights: torch.Tensor
    selected: torch.Tensor
    load_balance: torch.Tensor
    novelty: torch.Tensor


class UniversalSubspace(nn.Module):
    """Shared orthonormal coordinate frame across all four tasks.

    Evidence label: DERIVED for the algebra, OBSERVED for the effect a run
    measures. Decision 8 cites the universal weight subspace hypothesis:
    different modalities converge on a low-rank shared frame of relational
    primitives (identity, negation, containment, causal transition). This
    module makes that frame explicit and shared:

        x' = x + U_univ diag(sigma^(m)) U_univ^T x

    U_univ is r orthonormal rows of R^d_feat, retracted every pass. Each
    task carries only r diagonal gains, so a relational coordinate learned
    on vision is available to action and to symbolic text without a new
    dense weight matrix.
    """

    def __init__(self, d_feat: int, rank: int, n_tasks: int, seed: int = 0, scale: float = 0.1) -> None:
        super().__init__()
        self.d_feat = int(d_feat)
        self.rank = int(rank)
        generator = torch.Generator().manual_seed(int(seed))
        self.weight = nn.Parameter(
            float(scale) * torch.randn(self.rank, self.d_feat, generator=generator)
        )
        self.sigma = nn.Parameter(torch.ones(int(n_tasks), self.rank))

    def basis(self) -> torch.Tensor:
        q, r = torch.linalg.qr(self.weight.transpose(-1, -2))
        diagonal = torch.diagonal(r, dim1=-2, dim2=-1)
        sign = torch.where(diagonal >= 0.0, torch.ones_like(diagonal), -torch.ones_like(diagonal))
        return (q * sign.unsqueeze(-2)).transpose(-1, -2).contiguous()

    def modulate(self, x: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
        """Per-task modulation inside the shared frame, shape (batch, d).

            U_univ^T diag(sigma^(m)) U_univ x

        Decision A4 factors the first layer of every Zone A expert through
        this term, so the relational coordinates themselves are shared and
        only the r diagonal gains are per task.
        """
        basis = self.basis()
        code = x @ basis.transpose(-1, -2)
        gain = self.sigma[task.reshape(-1).long()]
        return (code * gain) @ basis

    def forward(self, x: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
        return x + self.modulate(x, task)


class FastWeightExpert(nn.Module):
    """Two-layer MLP plus a rank-r delta-rule associative memory."""

    def __init__(
        self,
        d_in: int,
        hidden: int,
        d_out: int,
        rank: int,
        seed: int = 0,
    ) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(int(seed))
        self.d_in = int(d_in)
        self.d_out = int(d_out)
        self.rank = int(rank)
        self.slow = nn.Sequential(
            nn.Linear(self.d_in, int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), self.d_out),
        )
        for module in self.slow:
            if isinstance(module, nn.Linear):
                bound = 1.0 / math.sqrt(module.in_features)
                with torch.no_grad():
                    module.weight.uniform_(-bound, bound, generator=generator)
                    module.bias.zero_()
        projection = torch.randn(self.rank, self.d_in, generator=generator) / math.sqrt(self.d_in)
        # Decision A4: the first layer also reads the shared orthonormal
        # frame U_univ through a per-task modulation. The gate starts at
        # zero, so the converged head is the identity until training opens
        # it, and the factored path costs one scalar per expert.
        self.first_layer_gain = nn.Parameter(torch.zeros(()))
        self.register_buffer("A", projection)
        self.register_buffer("B", torch.zeros(self.d_out, self.rank))
        self.register_buffer("centroid", torch.zeros(self.d_in))
        self.register_buffer("seen", torch.zeros(1))

    def fast_logits(self, x: torch.Tensor) -> torch.Tensor:
        code = x @ self.A.t()
        return code @ self.B.t()

    def forward(
        self,
        x: torch.Tensor,
        universal: Optional["UniversalSubspace"] = None,
        task: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        first = self.slow[0]
        hidden = first(x)
        if universal is not None and task is not None:
            # W1 x + g * W1 (U^T diag(sigma^(m)) U x): the first layer is
            # factored through the shared Stiefel frame, and g learns how
            # much of the shared relational basis each modality needs.
            shared = F.linear(universal.modulate(x, task), first.weight)
            hidden = hidden + self.first_layer_gain * shared
        hidden = self.slow[1](hidden)
        hidden = self.slow[2](hidden)
        return hidden + self.fast_logits(x)

    @torch.no_grad()
    def online_update(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        lr: float,
        start: int,
        end: int,
        weights: Optional[torch.Tensor] = None,
    ) -> float:
        """One delta-rule write. Returns the pre-update cross entropy.

        Args:
            x: features, shape (batch, d_in).
            target: integer labels local to the task block, in
                [0, end - start).
            lr: fast learning rate.
            start: first output row of this task's label block.
            end: one past the last output row.
            weights: optional per-example plasticity gate in [0, 1]. A
                zero entry contributes no delta, which is how Decision 4
                freezes the fast path on familiar inputs. The denominator
                is the surviving weight mass, so the step size does not
                shrink when most of a batch is gated shut.
        """
        if x.shape[0] == 0:
            return 0.0
        start, end = int(start), int(end)
        logits = self.forward(x)[:, start:end]
        loss = F.cross_entropy(logits, target.long())
        probs = torch.softmax(logits, dim=-1)
        one_hot = F.one_hot(target.long(), num_classes=end - start).to(probs.dtype)
        error = one_hot - probs
        gate = None
        if weights is not None:
            gate = weights.reshape(-1, 1).to(probs.dtype).clamp_min(0.0)
            mass = float(gate.sum())
            if mass <= 1e-6:
                return float(loss.detach())
            error = error * gate
            denominator = mass
        else:
            denominator = float(max(1, x.shape[0]))
        code = x @ self.A.t()
        delta = error.t() @ code / denominator
        self.B[start:end].add_(float(lr) * delta)
        self.B.clamp_(-6.0, 6.0)
        self.centroid.mul_(0.9).add_(0.1 * x.mean(dim=0))
        self.seen.add_(float(x.shape[0]))
        return float(loss.detach())

    def novelty(self, x: torch.Tensor) -> torch.Tensor:
        """1 - cosine similarity to the running centroid, shape (batch,)."""
        norm_x = x.norm(dim=-1).clamp_min(_EPS)
        norm_c = self.centroid.norm().clamp_min(_EPS)
        cosine = (x @ self.centroid) / (norm_x * norm_c)
        return (1.0 - cosine).clamp(0.0, 2.0)


class LeanSwarm(nn.Module):
    """Top-k routed swarm of lean experts with fast and slow learning."""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.d_feat = int(cfg.d_feat)
        # Disjoint label blocks per task. Sharing one block lets a language
        # class and an action class write the same fast-weight row, which is
        # an interference source with no upside.
        self.task_blocks: Dict[str, Tuple[int, int]] = {}
        cursor = 0
        for name, count in (
            ("vision", int(cfg.n_vision_classes)),
            ("language", int(cfg.vocab_size)),
            ("action", int(cfg.n_actions)),
            ("relational", int(cfg.relational_classes)),
        ):
            self.task_blocks[name] = (cursor, cursor + count)
            cursor += count
        self.d_out = cursor
        self.n_experts = int(cfg.n_experts)
        self.top_k = int(cfg.top_k)
        self.experts = nn.ModuleList(
            [
                FastWeightExpert(
                    d_in=self.d_feat,
                    hidden=int(cfg.expert_hidden),
                    d_out=self.d_out,
                    rank=int(cfg.fast_rank),
                    seed=int(cfg.seed) + 200 + index,
                )
                for index in range(self.n_experts)
            ]
        )
        # Decision 8: one shared orthonormal frame, per-task diagonal gains.
        rank = int(getattr(cfg, "universal_rank", 0))
        self.universal = (
            UniversalSubspace(self.d_feat, rank, int(cfg.n_tasks), seed=int(cfg.seed) + 400)
            if rank > 0
            else None
        )
        self.router = nn.Linear(self.d_feat, self.n_experts)
        with torch.no_grad():
            self.router.weight.mul_(0.5)
            self.router.bias.zero_()
        # Decision 1 attribution probe: a single linear projection from the
        # pooled photonic features to the label blocks. No MLP, no routing,
        # no fast memory. Any accuracy the model keeps in this mode comes
        # from the optical features alone.
        self.linear_only = str(getattr(cfg, "readout", "swarm")) == "linear"
        self.linear = nn.Linear(self.d_feat, self.d_out)
        self.task_embedding = nn.Embedding(int(cfg.n_tasks), self.d_feat)
        with torch.no_grad():
            self.task_embedding.weight.mul_(0.1)
        self.novelty_weight = 0.25
        self.route_noise = 0.05

    def forward(
        self,
        features: torch.Tensor,
        task_id: int | torch.Tensor,
        *,
        training: bool = False,
        explore: bool = True,
    ) -> SwarmOutput:
        """Route a feature batch through the swarm.

        Args:
            features: shape (batch, d_feat).
            task_id: task index, scalar or shape (batch,).
            training: enables the novelty bonus and route noise.
            explore: when False the novelty term is skipped even in
                training. Used by the ablation.
        """
        if features.dim() != 2 or features.shape[-1] != self.d_feat:
            raise ValueError(f"expected (batch, {self.d_feat}), got {tuple(features.shape)}")
        batch = features.shape[0]
        if torch.is_tensor(task_id):
            task = task_id.reshape(-1)
            if task.numel() == 1:
                task = task.expand(batch)
        else:
            task = torch.full((batch,), int(task_id), dtype=torch.long, device=features.device)
        x = features + self.task_embedding(task.long())
        if self.universal is not None:
            x = self.universal(x, task.long())
        if self.linear_only:
            logits = self.linear(x)
            return SwarmOutput(
                logits=logits,
                expert_logits=logits.unsqueeze(0),
                router_probs=torch.full(
                    (batch, self.n_experts), 1.0 / self.n_experts, device=features.device
                ),
                weights=torch.ones(batch, 1, device=features.device),
                selected=torch.zeros(batch, 1, dtype=torch.long, device=features.device),
                load_balance=features.new_zeros(()),
                novelty=features.new_zeros(batch, self.n_experts),
            )

        expert_logits = torch.stack(
            [expert(x, universal=self.universal, task=task) for expert in self.experts], dim=0
        )
        router_logits = self.router(x)
        novelty = torch.stack([expert.novelty(x) for expert in self.experts], dim=-1)
        if training and explore:
            router_logits = router_logits + self.novelty_weight * novelty
            if self.route_noise > 0.0:
                router_logits = router_logits + self.route_noise * torch.randn_like(router_logits)
        probs = torch.softmax(router_logits, dim=-1)

        top_values, selected = torch.topk(probs, k=self.top_k, dim=-1)
        weights = torch.softmax(top_values, dim=-1)
        gather_index = selected.unsqueeze(-1).expand(-1, -1, self.d_out)
        chosen = torch.gather(expert_logits.permute(1, 0, 2), 1, gather_index)
        mixed = (weights.unsqueeze(-1) * chosen).sum(dim=1)

        fraction = F.one_hot(selected, num_classes=self.n_experts).float().mean(dim=(0, 1))
        mean_prob = probs.mean(dim=0)
        load_balance = (fraction * mean_prob).sum() * self.n_experts
        return SwarmOutput(
            logits=mixed,
            expert_logits=expert_logits,
            router_probs=probs,
            weights=weights,
            selected=selected,
            load_balance=load_balance,
            novelty=novelty,
        )

    def task_slice(self, task_name: str) -> Tuple[int, int]:
        """Half-open label block for a task name."""
        if task_name not in self.task_blocks:
            raise ValueError(f"unknown task {task_name!r}")
        return self.task_blocks[task_name]

    def logits_for(self, logits: torch.Tensor, task_name: str) -> torch.Tensor:
        """Slice mixed logits down to one task's valid labels."""
        start, end = self.task_slice(task_name)
        return logits[:, start:end]

    @torch.no_grad()
    def online_update(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        task_id: int,
        lr: float = 0.05,
        weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Write every routed expert's fast memory. No backpropagation.

        Args:
            weights: optional per-example plasticity gate in [0, 1]. It is
                the surprise gate of Decision 4, already reduced to a
                scalar per example.
        """
        if self.linear_only:
            return {}
        batch = features.shape[0]
        task = torch.full((batch,), int(task_id), dtype=torch.long, device=features.device)
        x = features + self.task_embedding(task)
        router_logits = self.router(x)
        probs = torch.softmax(router_logits, dim=-1)
        top_values, selected = torch.topk(probs, k=self.top_k, dim=-1)
        mix_weights = torch.softmax(top_values, dim=-1)
        start, end = self.task_slice(_TASK_NAMES[int(task_id)])
        losses: Dict[str, float] = {}
        for index, expert in enumerate(self.experts):
            mask = (selected == index).any(dim=-1)
            if not bool(mask.any()):
                continue
            weight = mix_weights[selected == index].mean().item()
            losses[f"expert_{index}"] = expert.online_update(
                x[mask],
                labels[mask].long(),
                lr=float(lr) * (0.5 + 0.5 * weight),
                start=start,
                end=end,
                weights=None if weights is None else weights[mask],
            )
        losses["routed"] = float(mix_weights.mean())
        if weights is not None:
            losses["gate_mean"] = float(weights.mean())
        return losses

    def explore(
        self,
        features: torch.Tensor,
        task_id: int,
        steps: int = 4,
        sigma: float = 0.15,
        seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, float]:
        """Search a local latent neighbourhood for maximal expert disagreement.

        Returns the perturbation that maximises disagreement and the
        intrinsic reward (the disagreement itself). This is the cheap
        latent-space exploration the ASIC swarm performs between writes.
        """
        generator = torch.Generator(device=features.device).manual_seed(
            int(self.cfg.seed if seed is None else seed)
        )
        best_perturbation = torch.zeros_like(features)
        best_disagreement = -1.0
        for _ in range(max(1, int(steps))):
            noise = torch.randn(features.shape, generator=generator, device=features.device)
            perturbation = float(sigma) * noise
            output = self.forward(features + perturbation, task_id, training=False)
            probs = torch.softmax(output.expert_logits, dim=-1)
            disagreement = float(probs.std(dim=0).mean().detach())
            if disagreement > best_disagreement:
                best_disagreement = disagreement
                best_perturbation = perturbation
        return best_perturbation, best_disagreement

    def consolidate(
        self,
        replay_features: torch.Tensor,
        replay_labels: torch.Tensor,
        task_id: int,
        optimizer: torch.optim.Optimizer,
        steps: int = 1,
    ) -> float:
        """Replay-driven slow-weight update. The anti-forgetting path."""
        total = 0.0
        start, end = self.task_slice(_TASK_NAMES[int(task_id)])
        for _ in range(max(1, int(steps))):
            output = self.forward(replay_features, task_id, training=False)
            loss = F.cross_entropy(output.logits[:, start:end], replay_labels.long())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach())
        return total / max(1, int(steps))


_TASK_NAMES = ("vision", "language", "action", "relational")


class ReplayBuffer:
    """Bounded reservoir of (feature, label, task) triples."""

    def __init__(self, capacity: int = 2048, seed: int = 0) -> None:
        self.capacity = int(capacity)
        self.generator = torch.Generator().manual_seed(int(seed))
        self.features: List[torch.Tensor] = []
        self.labels: List[int] = []
        self.tasks: List[int] = []
        self._seen = 0

    def add(self, features: torch.Tensor, labels: torch.Tensor, task_id: int) -> None:
        for index in range(features.shape[0]):
            self._seen += 1
            if len(self.features) < self.capacity:
                self.features.append(features[index].detach().clone())
                self.labels.append(int(labels[index]))
                self.tasks.append(int(task_id))
            else:
                slot = int(torch.randint(0, self._seen, (1,), generator=self.generator))
                if slot < self.capacity:
                    self.features[slot] = features[index].detach().clone()
                    self.labels[slot] = int(labels[index])
                    self.tasks[slot] = int(task_id)

    def sample(self, task_id: int, count: int) -> Tuple[torch.Tensor, torch.Tensor]:
        indices = [i for i, task in enumerate(self.tasks) if task == int(task_id)]
        if not indices:
            raise ValueError(f"replay buffer has no samples for task {task_id}")
        take = min(int(count), len(indices))
        chosen = torch.randperm(len(indices), generator=self.generator)[:take]
        rows = [indices[int(i)] for i in chosen]
        features = torch.stack([self.features[i] for i in rows])
        labels = torch.tensor([self.labels[i] for i in rows], dtype=torch.long)
        return features, labels

    def __len__(self) -> int:
        return len(self.features)
