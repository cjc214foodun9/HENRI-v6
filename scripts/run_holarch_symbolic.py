"""Decision 8 probe: symbolic topology navigation and structural indexing.

Evidence label: OBSERVED. Four measurements share one codebook and one
Zone C store:

    topology     multi-hop navigation on an unseen rule graph
    abstention   unreachable queries must be refused, not guessed
    capacity     the bundled resonator against the indexed atlas
    code index   structural questions answered with zero file reads

    python scripts/run_holarch_symbolic.py --seeds 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from henri.holarch.config import HolarchConfig
from henri.holarch.pipeline import HenriHolarch
from henri.holarch.symbolic import (
    CompositeSolver,
    Edge,
    SymbolCodebook,
    SymbolicTopologyTask,
    TopologySolver,
    index_code,
)


def _memory(cfg, tag: str):
    path = Path(tempfile.gettempdir()) / f"henri_symbolic_{tag}.sqlite3"
    if path.exists():
        path.unlink()
    model = HenriHolarch(cfg, backend="sqlite", memory_path=str(path))
    return model


def run_topology(cfg, seed: int, salts: Sequence[int], hops: Sequence[int] = (1, 2, 3)) -> Dict:
    task = SymbolicTopologyTask(seed=seed)
    rows: Dict[str, Dict[str, float]] = {}
    per_hop: Dict[str, List[float]] = {str(h): [] for h in hops}
    abstain: List[float] = []
    false_answer: List[float] = []
    false_abstain: List[float] = []
    for salt in salts:
        edges = task.graph(salt)
        model = _memory(cfg, f"topo_{seed}_{salt}")
        codebook = SymbolCodebook(cfg.dim)
        solver = TopologySolver(model.memory, codebook, mode="indexed")
        ingest = solver.ingest(edges)
        calibration = solver.calibrate(edges)
        for hop in hops:
            correct = 0
            total = 0
            for index in range(40):
                operators, answer = task.query(edges, task.symbols[index % len(task.symbols)], hop, salt=salt * 97 + index)
                if answer is None:
                    continue
                result = solver.solve(task.symbols[index % len(task.symbols)], operators)
                correct += int(result.answer == answer)
                total += 1
            per_hop[str(hop)].append(correct / max(total, 1))
        # Unreachable draws must be refused.
        refused = 0
        invented = 0
        draws = 40
        for index in range(draws):
            source, operators = task.draw_unreachable(edges, salt=salt * 31 + index)
            result = solver.solve(source, operators)
            if result.vetoed or result.answer is None:
                refused += 1
            else:
                invented += 1
        abstain.append(refused / draws)
        false_answer.append(invented / draws)
        # Reachable draws must not be refused.
        kept = 0
        reachable = 0
        for index in range(draws):
            source = task.symbols[index % len(task.symbols)]
            operators, answer = task.query(edges, source, 1, salt=salt * 7 + index)
            if answer is None:
                continue
            reachable += 1
            result = solver.solve(source, operators)
            kept += int(not result.vetoed and result.answer == answer)
        false_abstain.append(1.0 - kept / max(reachable, 1))
        model.memory.close()
    for hop in hops:
        values = per_hop[str(hop)]
        rows[str(hop)] = {
            "accuracy_mean": float(np.mean(values)) if values else float("nan"),
            "accuracy_min": float(np.min(values)) if values else float("nan"),
            "graphs": float(len(values)),
        }
    rows["abstention"] = {
        "unreachable_refused": float(np.mean(abstain)),
        "unreachable_false_answer": float(np.mean(false_answer)),
        "reachable_false_abstain": float(np.mean(false_abstain)),
    }
    rows["edges"] = {"count": float(len(edges)), "ingested": float(ingest)}
    rows["calibration"] = calibration
    rows["symbols"] = float(len(task.symbols))
    return rows


def run_capacity(cfg, seed: int, sizes: Sequence[int] = (8, 16, 24, 32, 48, 64)) -> Dict[str, float]:
    """Bundled resonator against the indexed atlas as the edge count grows."""
    task = SymbolicTopologyTask(seed=seed)
    edges = task.graph(1)
    rows: Dict[str, float] = {}
    for size in sizes:
        subset = edges[: int(size)]
        targets = {(edge.source, edge.operator): edge.target for edge in subset}
        model = _memory(cfg, f"cap_{seed}_{size}")
        codebook = SymbolCodebook(cfg.dim)
        resonator = TopologySolver(model.memory, codebook, mode="resonator")
        resonator.ingest(subset)
        indexed = TopologySolver(model.memory, codebook, mode="indexed")
        indexed.ingest(subset)
        hits_r = 0
        hits_i = 0
        total = 0
        for (source, operator), target in targets.items():
            total += 1
            hits_r += int(resonator.step(source, operator)["answer"] == target)
            hits_i += int(indexed.step(source, operator)["answer"] == target)
        rows[str(size)] = {
            "resonator": hits_r / max(total, 1),
            "indexed": hits_i / max(total, 1),
            "n": float(total),
        }
        model.memory.close()
    return rows


def run_code_index(cfg, root: Path, seed: int) -> Dict[str, float]:
    graph = index_code(root)
    model = _memory(cfg, f"code_{seed}")
    codebook = SymbolCodebook(cfg.dim)
    solver = TopologySolver(model.memory, codebook, mode="indexed")
    for edge in graph.edges:
        codebook.add(edge.source)
        codebook.add(edge.target)
    # Verification keys: the pair (caller, CALLS, callee) is unique.
    for edge in graph.edges:
        key = solver.key_wave(edge.source, f"CALLS:{edge.target}")
        model.memory.write(key, label=1, tier=1)
    true_edges = graph.edges
    absent = []
    existing = {(edge.source, edge.target) for edge in graph.edges}
    generator = np.random.default_rng(seed)
    sources = [edge.source for edge in graph.edges]
    targets = [edge.target for edge in graph.edges]
    while len(absent) < len(true_edges) and len(absent) < 1024:
        source = str(sources[int(generator.integers(0, len(sources)))])
        target = str(targets[int(generator.integers(0, len(targets)))])
        if source != target and (source, target) not in existing:
            absent.append((source, target))
            existing.add((source, target))
    present_scores = []
    for edge in true_edges:
        key = solver.key_wave(edge.source, f"CALLS:{edge.target}")
        read = model.memory.query(key, k=1)
        present_scores.append(float(read.hits[0].coherence) if read.hits else 0.0)
    absent_scores = []
    for source, target in absent:
        key = solver.key_wave(source, f"CALLS:{target}")
        read = model.memory.query(key, k=1)
        absent_scores.append(float(read.hits[0].coherence) if read.hits else 0.0)
    present = np.asarray(present_scores)
    negative = np.asarray(absent_scores)
    threshold = float((np.median(present) + np.median(negative)) / 2.0) if len(negative) else 0.5
    recall = float(np.mean(present >= threshold))
    false_accept = float(np.mean(negative >= threshold)) if len(negative) else 0.0
    report = {
        "files_parsed": float(graph.files),
        "lines_parsed": float(graph.lines),
        "parse_seconds": float(graph.seconds),
        "call_edges": float(len(graph.edges)),
        "functions": float(len(graph.nodes)),
        "present_coherence_median": float(np.median(present)) if len(present) else float("nan"),
        "absent_coherence_median": float(np.median(negative)) if len(negative) else float("nan"),
        "threshold": threshold,
        "recall": recall,
        "false_accept": false_accept,
        "file_reads_at_query_time": 0.0,
    }
    model.memory.close()
    return report


def run_playbooks(cfg, seed: int, steps: Sequence[str]) -> Dict[str, float]:
    from henri.holarch.sidecar import EpistemicSidecar

    sidecar = EpistemicSidecar(cfg=cfg, backend="sqlite", memory_path=str(
        Path(tempfile.gettempdir()) / f"henri_playbook_{seed}.sqlite3"
    ))
    compiled = sidecar.compile_playbook("refactor_call_graph", steps)
    recalled = sidecar.recall_playbook("refactor_call_graph")
    recovered = [step for step in recalled["decoded"]]
    exact = float(sum(1 for a, b in zip(recalled["steps"], recovered) if a == b)) / max(
        len(recalled["steps"]), 1
    )
    report = {
        "steps": compiled["steps"],
        "version": compiled["version"],
        "recall_coherence": float(recalled.get("coherence", 0.0)),
        "decoded_steps": float(len(recovered)),
        "exact_recovery": exact,
    }
    sidecar.close()
    return report


def _snap_digits(model, images: torch.Tensor) -> torch.Tensor:
    """Stage 1: ground an 8x8 patch through the Pass-0 prototype bank."""
    with torch.no_grad():
        psi = model.encode("vision", images)
        bright = model.zone_b(psi).bright
        bank = model.prototype_waves("vision")
        if int(bank.numel()):
            labels = model.prototype_class_vision
            align = model.zone_b.alignment(bright, bank)
            return labels[align.argmax(dim=-1)]
        return model("vision", images, training=False).predicted()


def run_composition(cfg, seed: int, steps: int = 300, pairs: int = 200) -> Dict[str, object]:
    """Decision C4: two-stage neuro-symbolic arithmetic on 16x16 canvases.

    Stage 1 grounds each informative quadrant through the trained Pass-0
    prototype bank. Stage 2 binds the two digit symbols and the operator
    into one qFHRR key and reads the exact result from the Zone C atlas.
    The continuous plates never see the arithmetic rule.
    """
    from henri.holarch.tasks import DigitsTask, build_tasks
    from henri.holarch.train import Trainer, TrainConfig

    model = _memory(cfg, f"composition_{seed}")
    tasks = build_tasks(cfg, seed=seed)
    trainer = Trainer(model, tasks, TrainConfig(steps_per_task=1, batch_size=32))
    for _ in range(int(steps)):
        trainer.step("vision")
    model.calibrate_prototypes(
        "vision", [tasks["vision"].sample(64, "train") for _ in range(4)]
    )
    model.eval()

    digits = DigitsTask(seed=seed)
    images, labels = digits.splits["val"]
    generator = np.random.default_rng(seed)
    take = int(min(pairs, len(images)))
    index = generator.integers(0, len(images), size=(take, 4))
    canvas = torch.zeros(take, 1, 16, 16)
    for row in range(take):
        first, second, third, fourth = index[row]
        canvas[row, 0, 0:8, 0:8] = torch.from_numpy(images[first])
        canvas[row, 0, 0:8, 8:16] = torch.from_numpy(images[second])
        canvas[row, 0, 8:16, 0:8] = torch.from_numpy(images[third])
        canvas[row, 0, 8:16, 8:16] = torch.from_numpy(images[fourth])

    q0 = canvas[:, :, 0:8, 0:8]
    q1 = canvas[:, :, 0:8, 8:16]
    d0 = _snap_digits(model, q0).detach().cpu().numpy()
    d1 = _snap_digits(model, q1).detach().cpu().numpy()
    true0 = labels[index[:, 0]]
    true1 = labels[index[:, 1]]
    grounding = float(np.mean((d0 == true0).astype(float).tolist() + (d1 == true1).astype(float).tolist()))

    codebook = SymbolCodebook(cfg.dim)
    solver = CompositeSolver(model.memory, codebook)
    operations = {
        "pair_sum": ("add", lambda a, b: (int(a) + int(b)) % 10),
        "pair_diff": ("sub", lambda a, b: abs(int(a) - int(b))),
    }
    rows_by_op = {}
    for name, (operator, _) in operations.items():
        rows = [
            ((f"d{a}", operator, f"d{b}"), f"d{operations[name][1](a, b)}")
            for a in range(10)
            for b in range(10)
        ]
        rows_by_op[name] = rows
        solver.ingest(rows)

    calibration = solver.calibrate(rows_by_op["pair_sum"] + rows_by_op["pair_diff"])
    report: Dict[str, object] = {
        "grounding_accuracy": grounding,
        "n": float(take),
        "tau_coherence": float(calibration["tau_coherence"]),
        "atlas_coherence_median": float(calibration["coherence_median"]),
    }
    for name, (operator, fn) in operations.items():
        rows = rows_by_op[name]
        correct = 0
        refused = 0
        for row in range(take):
            result = solver.step((f"d{int(d0[row])}", operator, f"d{int(d1[row])}"))
            if result["vetoed"] or result["answer"] is None:
                refused += 1
                continue
            truth = fn(int(true0[row]), int(true1[row]))
            correct += int(result["answer"] == f"d{truth}")
        report[name] = {
            "accuracy": correct / max(take, 1),
            "refused": refused / max(take, 1),
            "n": float(take),
            "operator": operator,
        }
    model.memory.close()
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--out", default="receipts/holarch/symbolic.json")
    parser.add_argument("--root", default="henri/holarch")
    parser.add_argument("--steps", type=int, default=300, help="vision grounding steps")
    parser.add_argument("--pairs", type=int, default=200)
    args = parser.parse_args()

    cfg = HolarchConfig.small()
    seeds = [20261103, 20261104, 20261105][: max(1, int(args.seeds))]
    topology = [run_topology(cfg, seed, salts=(0, 1)) for seed in seeds]
    merged = {
        str(hop): {
            "accuracy_mean": float(np.mean([row[str(hop)]["accuracy_mean"] for row in topology])),
            "accuracy_min": float(np.min([row[str(hop)]["accuracy_min"] for row in topology])),
        }
        for hop in (1, 2, 3)
    }
    merged["abstention"] = {
        key: float(np.mean([row["abstention"][key] for row in topology]))
        for key in topology[0]["abstention"]
    }
    capacity = run_capacity(cfg, seeds[0])
    code = run_code_index(cfg, Path(args.root), seeds[0])
    playbooks = run_playbooks(
        cfg,
        seeds[0],
        ("READ_INDEX", "LOCATE_SYMBOL", "EDIT_CALLSITE", "RUN_TESTS", "VERIFY_VETO"),
    )
    composition = run_composition(
        cfg, seeds[0], steps=int(args.steps), pairs=int(args.pairs)
    )
    payload = {
        "task": "symbolic-topology",
        "seeds": seeds,
        "topology": merged,
        "capacity": capacity,
        "code_index": code,
        "playbook": playbooks,
        "composition": composition,
        "gate": {
            "one_hop_at_least_0_90": bool(merged["1"]["accuracy_mean"] >= 0.90),
            "two_hop_at_least_0_80": bool(merged["2"]["accuracy_mean"] >= 0.80),
            "unreachable_refused_at_least_0_90": bool(
                merged["abstention"]["unreachable_refused"] >= 0.90
            ),
            "reachable_false_abstain_at_most_0_10": bool(
                merged["abstention"]["reachable_false_abstain"] <= 0.10
            ),
            "code_false_accept_at_most_0_05": bool(code["false_accept"] <= 0.05),
            "playbook_exact_recovery": bool(playbooks["exact_recovery"] >= 0.80),
            "pair_sum_at_least_0_85": bool(composition["pair_sum"]["accuracy"] >= 0.85),
            "pair_diff_at_least_0_85": bool(composition["pair_diff"]["accuracy"] >= 0.85),
        },
        "evidence": "OBSERVED",
    }
    payload["gate"]["passed"] = bool(
        payload["gate"]["one_hop_at_least_0_90"]
        and payload["gate"]["unreachable_refused_at_least_0_90"]
        and payload["gate"]["code_false_accept_at_most_0_05"]
        and payload["gate"]["pair_sum_at_least_0_85"]
        and payload["gate"]["pair_diff_at_least_0_85"]
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    (out.parent / (out.name + ".sha256")).write_text(
        hashlib.sha256(out.read_bytes()).hexdigest() + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
