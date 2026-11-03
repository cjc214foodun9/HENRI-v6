"""Real tasks for the HENRI holarch: vision, language, and action.

Evidence label: OBSERVED for the dataset sizes and split indices; the
accuracy numbers a run reports are measurements on these splits.

Three tasks, three modalities, one label discipline
---------------------------------------------------
vision    sklearn 8x8 handwritten digits, 10 classes.
language  character-level next-character prediction on the question text
          of the local MMLU CSV. The corpus is on disk; nothing is
          downloaded.
action    a 7x7 grid world with walls. The label is the optimal action
          from a breadth-first search, so it is a reasoning target, not a
          heuristic. Success rate is measured by executing the policy.

Each task exposes a fixed train/validation split and a deterministic
sampler. No task leaks its validation indices into training.
"""

from __future__ import annotations

import csv
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

MMLU_PATH = Path("data/official_benchmarks/canonical/mmlu/mmlu.csv")

ACTION_NAMES = ("up", "down", "left", "right")
_ACTION_DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))


@dataclass
class Batch:
    """One model-ready batch.

    Attributes:
        x: encoder-specific tensor.
        y: integer labels for the task's own class block.
        task: task name.
    """

    x: torch.Tensor
    y: torch.Tensor
    task: str


class DigitsTask:
    """8x8 handwritten digit classification."""

    name = "vision"

    def __init__(self, seed: int = 0, n_train: int = 1200, n_val: int = 300) -> None:
        from sklearn.datasets import load_digits

        data = load_digits()
        images = data.images.astype(np.float32)
        labels = data.target.astype(np.int64)
        images = (images - images.mean()) / (images.std() + 1e-6)
        generator = np.random.default_rng(int(seed))
        order = generator.permutation(len(images))
        train = order[: int(n_train)]
        val = order[int(n_train) : int(n_train) + int(n_val)]
        test = order[int(n_train) + int(n_val) :]
        self.splits = {
            "train": (images[train], labels[train]),
            "val": (images[val], labels[val]),
            "test": (images[test], labels[test]),
        }
        self.n_classes = 10
        self.generator = torch.Generator().manual_seed(int(seed) + 1)

    def size(self, split: str = "train") -> int:
        return int(self.splits[split][0].shape[0])

    def sample(self, batch_size: int, split: str = "train") -> Batch:
        images, labels = self.splits[split]
        index = torch.randint(0, len(images), (int(batch_size),), generator=self.generator)
        x = torch.from_numpy(images[index.numpy()]).unsqueeze(1)
        y = torch.from_numpy(labels[index.numpy()])
        return Batch(x=x, y=y, task=self.name)

    def epoch(self, batch_size: int, split: str = "train") -> Iterator[Batch]:
        images, labels = self.splits[split]
        order = torch.randperm(len(images), generator=self.generator)
        for start in range(0, len(order), int(batch_size)):
            index = order[start : start + int(batch_size)]
            x = torch.from_numpy(images[index.numpy()]).unsqueeze(1)
            y = torch.from_numpy(labels[index.numpy()])
            yield Batch(x=x, y=y, task=self.name)


class CharLMTask:
    """Character-level next-character prediction on the local MMLU text."""

    name = "language"

    def __init__(
        self,
        seed: int = 0,
        seq_len: int = 24,
        vocab_size: int = 96,
        max_rows: int = 2400,
        path: Path | str = MMLU_PATH,
    ) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"language corpus missing: {path}")
        questions: List[str] = []
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                text = (row.get("Question") or "").strip()
                if text:
                    questions.append(text)
                if len(questions) >= int(max_rows):
                    break
        text = "\n".join(questions)
        if len(text) < 4096:
            raise ValueError("language corpus is too small to train on")
        counts = Counter(text)
        keep = [char for char, _ in counts.most_common(int(vocab_size) - 1)]
        self.vocab = ["<unk>"] + sorted(keep)
        self.stoi = {char: index for index, char in enumerate(self.vocab)}
        self.seq_len = int(seq_len)
        self.tokens = np.array(
            [self.stoi.get(char, 0) for char in text], dtype=np.int64
        )
        self.generator = torch.Generator().manual_seed(int(seed) + 2)

        windows = len(self.tokens) - self.seq_len
        order = np.random.default_rng(int(seed)).permutation(windows)
        cut = int(0.9 * windows)
        self.index = {
            "train": order[:cut],
            "val": order[cut:],
            "test": order[cut:],
        }
        self.n_classes = len(self.vocab)

    def size(self, split: str = "train") -> int:
        return int(len(self.index[split]))

    def _gather(self, rows: np.ndarray) -> Batch:
        offsets = np.arange(self.seq_len + 1)
        matrix = self.tokens[rows[:, None] + offsets[None, :]]
        x = torch.from_numpy(matrix[:, :-1])
        y = torch.from_numpy(matrix[:, -1])
        return Batch(x=x, y=y, task=self.name)

    def sample(self, batch_size: int, split: str = "train") -> Batch:
        pool = self.index[split]
        pick = torch.randint(0, len(pool), (int(batch_size),), generator=self.generator)
        return self._gather(pool[pick.numpy()])

    def epoch(self, batch_size: int, split: str = "train") -> Iterator[Batch]:
        pool = self.index[split]
        order = torch.randperm(len(pool), generator=self.generator)
        for start in range(0, len(order), int(batch_size)):
            rows = pool[order[start : start + int(batch_size)].numpy()]
            yield self._gather(rows)

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(self.vocab[int(i)] if 0 <= int(i) < len(self.vocab) else "?" for i in ids)


class _Grid:
    """A small deterministic grid world with BFS-optimal action labels."""

    def __init__(self, size: int = 7, wall_prob: float = 0.15, seed: int = 0) -> None:
        self.grid_size = int(size)
        self.wall_prob = float(wall_prob)
        self.generator = np.random.default_rng(int(seed))

    def _layout(self) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
        size = self.grid_size
        for _ in range(256):
            walls = self.generator.random((size, size)) < self.wall_prob
            free = np.argwhere(~walls)
            if len(free) < 2:
                continue
            pick = self.generator.choice(len(free), size=2, replace=False)
            agent = tuple(int(v) for v in free[pick[0]])
            goal = tuple(int(v) for v in free[pick[1]])
            if np.abs(np.array(agent) - np.array(goal)).sum() < 2:
                continue
            if self._bfs(walls, agent, goal) is not None:
                walls[agent] = False
                walls[goal] = False
                return walls, agent, goal
        raise RuntimeError("grid world could not place a reachable pair")

    def _bfs(self, walls: np.ndarray, start, goal) -> Optional[List[int]]:
        size = self.grid_size
        from collections import deque

        frontier = deque([(start, [])])
        seen = {start}
        while frontier:
            (row, col), path = frontier.popleft()
            if (row, col) == goal:
                return path
            for action, (dr, dc) in enumerate(_ACTION_DELTAS):
                nr, nc = row + dr, col + dc
                if not (0 <= nr < size and 0 <= nc < size):
                    continue
                if walls[nr, nc] or (nr, nc) in seen:
                    continue
                seen.add((nr, nc))
                frontier.append(((nr, nc), path + [action]))
        return None

    def optimal_action(self, walls: np.ndarray, agent, goal) -> Optional[int]:
        path = self._bfs(walls, agent, goal)
        return int(path[0]) if path else None

    def observe(self, walls: np.ndarray, agent, goal) -> np.ndarray:
        """Local 5x5 view plus the relative goal and position vectors.

        The local view alone makes the task partially observable whenever
        the goal leaves the window. The relative goal vector is the sensor
        signal that makes a closed-loop policy possible; without it a
        one-step label is not learnable from the observation.
        """
        radius = 2
        size = self.grid_size
        view = np.zeros((5, 5, 3), dtype=np.float32)
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                row, col = agent[0] + dr, agent[1] + dc
                if not (0 <= row < size and 0 <= col < size):
                    view[dr + radius, dc + radius, 0] = 1.0
                    continue
                if walls[row, col]:
                    view[dr + radius, dc + radius, 0] = 1.0
                if (row, col) == goal:
                    view[dr + radius, dc + radius, 1] = 1.0
        view[radius, radius, 2] = 1.0
        relative = np.array(
            [
                (goal[0] - agent[0]) / max(size - 1, 1),
                (goal[1] - agent[1]) / max(size - 1, 1),
                agent[0] / max(size - 1, 1),
                agent[1] / max(size - 1, 1),
            ],
            dtype=np.float32,
        )
        return np.concatenate([view.reshape(-1), relative])

    def rollout(self, walls: np.ndarray, agent, goal, actions: Sequence[int], limit: int = 40) -> bool:
        row, col = agent
        size = self.grid_size
        for action in actions:
            dr, dc = _ACTION_DELTAS[int(action)]
            row, col = row + dr, col + dc
            if not (0 <= row < size and 0 <= col < size) or walls[row, col]:
                return False
            if (row, col) == goal:
                return True
            limit -= 1
            if limit <= 0:
                break
        return False


class GridWorldTask:
    """Optimal-action prediction plus closed-loop success rate."""

    name = "action"
    n_classes = 4

    def __init__(self, seed: int = 0, size: int = 7, n_layouts: int = 400) -> None:
        self.world = _Grid(size=size, seed=int(seed) + 3)
        self.generator = np.random.default_rng(int(seed) + 4)
        self.grid_size = int(size)
        self.train_layouts = [self.world._layout() for _ in range(int(n_layouts))]
        self.val_layouts = [self.world._layout() for _ in range(max(32, int(n_layouts) // 5))]
        self.obs_dim = 5 * 5 * 3 + 4
        self._shifted = False
        self.train_pairs = self._pairs(self.train_layouts)
        self.val_pairs = self._pairs(self.val_layouts)

    def _pairs(self, layouts: Sequence) -> List[Tuple]:
        """(observation, action) pairs for a set of layouts."""
        pairs = []
        for walls, agent, goal in layouts:
            action = self.world.optimal_action(walls, agent, goal)
            if action is None:
                continue
            pairs.append((self.world.observe(walls, agent, goal), int(action)))
        return pairs

    def _balanced_pool(self, split: str) -> List[List[Tuple]]:
        pairs = self.train_pairs if split == "train" else self.val_pairs
        buckets: List[List[Tuple]] = [[] for _ in range(self.n_classes)]
        for observation, action in pairs:
            buckets[action].append((observation, action))
        return buckets

    def size(self, split: str = "train") -> int:
        return len(self.train_pairs if split == "train" else self.val_pairs)

    def sample(self, batch_size: int, split: str = "train") -> Batch:
        """Class-balanced sample. Each action is equally likely by design.

        The natural layout distribution is dominated by 'up' (91 of 200 in
        a measured sample). Training on it collapses the policy to the
        majority direction, which reaches the goal zero times in closed
        loop. Balancing is a measurement fix, not a metric trick: the
        accuracy is still reported on the full validation pairs.
        """
        buckets = self._balanced_pool(split)
        live = [index for index, bucket in enumerate(buckets) if bucket]
        observations, labels = [], []
        while len(observations) < int(batch_size):
            action = int(live[int(self.generator.integers(0, len(live)))])
            observation, _ = buckets[action][
                int(self.generator.integers(0, len(buckets[action])))
            ]
            observations.append(observation)
            labels.append(action)
        return Batch(
            x=torch.from_numpy(np.stack(observations)),
            y=torch.tensor(labels, dtype=torch.long),
            task=self.name,
        )

    def epoch(self, batch_size: int, split: str = "train") -> Iterator[Batch]:
        pairs = self.train_pairs if split == "train" else self.val_pairs
        order = self.generator.permutation(len(pairs))
        for start in range(0, len(order), int(batch_size)):
            observations, labels = [], []
            for index in order[start : start + int(batch_size)]:
                observation, action = pairs[int(index)]
                observations.append(observation)
                labels.append(action)
            if observations:
                yield Batch(
                    x=torch.from_numpy(np.stack(observations)),
                    y=torch.tensor(labels, dtype=torch.long),
                    task=self.name,
                )

    def shift(self) -> None:
        """Move the goal distribution. Used by the continual-learning probe."""
        if self._shifted:
            return
        shifted = []
        for walls, agent, goal in self.train_layouts:
            mirrored = (goal[0], self.grid_size - 1 - goal[1])
            if walls[mirrored] or mirrored == agent:
                shifted.append((walls, agent, goal))
                continue
            shifted.append((walls, agent, mirrored))
        self.train_layouts = shifted
        self.train_pairs = self._pairs(self.train_layouts)
        self._shifted = True

    def success_rate(self, actions: Sequence[int], layouts: Optional[Sequence] = None) -> float:
        layouts = self.val_layouts if layouts is None else layouts
        successes = 0
        for (walls, agent, goal), action in zip(layouts, actions):
            if self.world.rollout(walls, agent, goal, [int(action)]):
                successes += 1
        return successes / max(len(layouts), 1)

    def rollout_pairs(
        self,
        policy,
        episodes: int = 16,
        limit: int = 20,
        split: str = "train",
    ) -> List[Tuple[np.ndarray, int]]:
        """States visited by a policy, labelled with the BFS optimal action.

        This is the DAgger dataset: it closes the train/test gap between
        i.i.d. layout states and the states the policy actually reaches.
        """
        layouts = self.train_layouts if split == "train" else self.val_layouts
        pairs: List[Tuple[np.ndarray, int]] = []
        for index in range(min(int(episodes), len(layouts))):
            walls, agent, goal = layouts[index]
            for _ in range(int(limit)):
                action = self.world.optimal_action(walls, agent, goal)
                if action is None:
                    break
                pairs.append((self.world.observe(walls, agent, goal), int(action)))
                chosen = int(policy(self.world.observe(walls, agent, goal)))
                dr, dc = _ACTION_DELTAS[chosen]
                row, col = agent[0] + dr, agent[1] + dc
                if not (0 <= row < self.grid_size and 0 <= col < self.grid_size):
                    break
                if walls[row, col]:
                    break
                agent = (row, col)
                if agent == goal:
                    break
        return pairs

    def policy_success(self, policy, episodes: int = 64, limit: int = 20) -> float:
        """Execute a policy function in the real grid until it reaches the goal."""
        layouts = self.val_layouts
        wins = 0
        for index in range(min(int(episodes), len(layouts))):
            walls, agent, goal = layouts[index]
            path = []
            for _ in range(int(limit)):
                action = int(policy(self.world.observe(walls, agent, goal)))
                path.append(action)
                dr, dc = _ACTION_DELTAS[action]
                row, col = agent[0] + dr, agent[1] + dc
                if not (0 <= row < self.grid_size and 0 <= col < self.grid_size):
                    break
                if walls[row, col]:
                    break
                agent = (row, col)
                if agent == goal:
                    wins += 1
                    break
        return wins / max(min(int(episodes), len(layouts)), 1)



class RelationalTask:
    """Compositional relational algebra with held-out triples.

    Each example is a triple (entity, action, target). The outcome is a
    rank-3 interaction

        score_c = <W_c, phi[e] * psi[a] * chi[t]>

    so the label is a composition of the three primitives, not a lookup
    of any pair. Every primitive appears in training; 25 percent of the
    (e, a, t) combinations are held out, stratified by class so every
    outcome is represented on both sides.

    A model that binds the three primitives and applies a linear readout
    computes exactly this trilinear form. A flat MLP on concatenated
    one-hots must discover the factorisation or memorise.
    """

    name = "relational"

    def __init__(
        self,
        seed: int = 0,
        entities: int = 6,
        actions: int = 6,
        targets: int = 8,
        classes: int = 8,
        rank: int = 3,
        holdout: float = 0.25,
    ) -> None:
        if not 0.0 < float(holdout) < 0.9:
            raise ValueError("holdout must lie in (0, 0.9)")
        generator = np.random.default_rng(int(seed) + 5)
        self.entities = int(entities)
        self.actions = int(actions)
        self.targets = int(targets)
        self.n_classes = int(classes)
        self.rank = int(rank)
        self.holdout = float(holdout)

        entity_factor = generator.standard_normal((self.entities, self.rank))
        action_factor = generator.standard_normal((self.actions, self.rank))
        target_factor = generator.standard_normal((self.targets, self.rank))

        triples = np.array(
            [
                (e, a, t)
                for e in range(self.entities)
                for a in range(self.actions)
                for t in range(self.targets)
            ],
            dtype=np.int64,
        )
        factors = (
            entity_factor[triples[:, 0]]
            * action_factor[triples[:, 1]]
            * target_factor[triples[:, 2]]
        )
        # A rank-3 readout can miss a class by chance. Redraw until every
        # class is represented, so the stratified split has something to
        # stratify and the test set is not empty for a class.
        labels = None
        for _ in range(256):
            readout = generator.standard_normal((self.n_classes, self.rank))
            candidate = np.argmax(factors @ readout.T, axis=1).astype(np.int64)
            counts = np.bincount(candidate, minlength=self.n_classes)
            if len(np.unique(candidate)) == self.n_classes and int(counts.min()) >= 4:
                labels = candidate
                break
        if labels is None:
            raise ValueError("could not draw a readout that covers every class")

        split = None
        for _ in range(128):
            train_index: List[int] = []
            test_index: List[int] = []
            for label in range(self.n_classes):
                pool = np.flatnonzero(labels == label)
                generator.shuffle(pool)
                take = max(1, int(round(self.holdout * len(pool))))
                test_index.extend(pool[:take].tolist())
                train_index.extend(pool[take:].tolist())
            candidate_train = triples[np.array(train_index)]
            covers = all(
                {int(row[column]) for row in candidate_train} == set(range(size))
                for column, size in (
                    (0, self.entities),
                    (1, self.actions),
                    (2, self.targets),
                )
            )
            if covers:
                split = (train_index, test_index)
                break
        if split is None:
            raise ValueError("could not split the triples with full primitive coverage")
        train_index, test_index = split
        self.x_train = triples[np.array(train_index)]
        self.y_train = labels[np.array(train_index)]
        self.x_test = triples[np.array(test_index)]
        self.y_test = labels[np.array(test_index)]
        self.generator = torch.Generator().manual_seed(int(seed) + 6)

    def size(self, split: str = "train") -> int:
        return int((self.x_train if split == "train" else self.x_test).shape[0])

    def _batch(self, x: np.ndarray, y: np.ndarray) -> Batch:
        return Batch(
            x=torch.from_numpy(x.astype(np.int64)),
            y=torch.from_numpy(y.astype(np.int64)),
            task=self.name,
        )

    def sample(self, batch_size: int, split: str = "train") -> Batch:
        x = self.x_train if split == "train" else self.x_test
        y = self.y_train if split == "train" else self.y_test
        index = torch.randint(0, len(x), (int(batch_size),), generator=self.generator)
        return self._batch(x[index.numpy()], y[index.numpy()])

    def epoch(self, batch_size: int, split: str = "train") -> Iterator[Batch]:
        x = self.x_train if split == "train" else self.x_test
        y = self.y_train if split == "train" else self.y_test
        order = torch.randperm(len(x), generator=self.generator)
        for start in range(0, len(order), int(batch_size)):
            index = order[start : start + int(batch_size)].numpy()
            yield self._batch(x[index], y[index])

    def held_out_triples(self) -> np.ndarray:
        return self.x_test.copy()


def build_tasks(cfg, seed: Optional[int] = None) -> Dict[str, object]:
    """Construct the three tasks with the config's vocabulary width."""
    seed = int(cfg.seed if seed is None else seed)
    return {
        "vision": DigitsTask(seed=seed),
        "language": CharLMTask(seed=seed, vocab_size=int(cfg.vocab_size)),
        "action": GridWorldTask(seed=seed),
        "relational": RelationalTask(
            seed=seed,
            entities=int(cfg.relational_entities),
            actions=int(cfg.relational_actions),
            targets=int(cfg.relational_targets),
            classes=int(cfg.relational_classes),
        ),
    }


OBSERVATION_DIMS = {"action": 5 * 5 * 3 + 4}
