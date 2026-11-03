"""End-to-end demonstration of the HENRI agentic sidecar.

Evidence label: OBSERVED. One process, one checkpoint, five questions an
agent actually asks:

    is this input inside the manifold?
    can a structural index answer without reading files?
    can an unseen rule system be solved at zero weight cost?
    does the rule context survive a shift?
    what is the compiled strategy for this task?

    python scripts/run_holarch_sidecar_demo.py --steps 120
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from henri.holarch.config import HolarchConfig
from henri.holarch.sidecar import EpistemicSidecar
from henri.holarch.symbolic import SymbolicTopologyTask
from henri.holarch.tasks import build_tasks
from henri.holarch.train import Trainer, TrainConfig


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--out", default="receipts/holarch/sidecar.json")
    args = parser.parse_args()

    cfg = HolarchConfig.small(seed=20261103, fast_gate=True)
    torch.manual_seed(cfg.seed)
    tasks = build_tasks(cfg)
    path = Path(tempfile.gettempdir()) / "henri_sidecar_demo.sqlite3"
    if path.exists():
        path.unlink()
    sidecar = EpistemicSidecar(cfg=cfg, backend="sqlite", memory_path=str(path))
    trainer = Trainer(sidecar.model, tasks, TrainConfig(steps_per_task=1, batch_size=32))
    for _ in range(int(args.steps)):
        trainer.step("vision")
    sidecar.model.calibrate_chance(
        "vision", [tasks["vision"].sample(32, "val").x for _ in range(4)], percentile=1.0
    )
    sidecar.model.calibrate_veto(
        "vision", [tasks["vision"].sample(32, "val").x for _ in range(4)], ceiling=0.98
    )
    sidecar.model.calibrate_prototypes(
        "vision", [tasks["vision"].sample(64, "train") for _ in range(2)]
    )
    sidecar.model.calibrate_gate(
        "vision", [tasks["vision"].sample(32, "val").x for _ in range(4)]
    )
    sidecar.model.eval()

    # 1. Fail-closed veto on in-distribution and on uniform noise.
    clean = sidecar.veto_check("vision", tasks["vision"].sample(32, "val").x)
    generator = torch.Generator().manual_seed(cfg.seed + 5)
    noisy = sidecar.veto_check("vision", torch.rand(32, 1, 8, 8, generator=generator))

    # 2. Structural index of this package, then zero-file-read queries.
    index = sidecar.ingest_code("henri/holarch")

    # 3. An unseen rule system, ingested at test time with no weight change.
    task = SymbolicTopologyTask(seed=cfg.seed)
    edges = task.graph(2)
    ingest = sidecar.ingest_edges(edges)
    solved = 0
    total = 0
    for index_i in range(40):
        source = task.symbols[index_i % len(task.symbols)]
        operators, answer = task.query(edges, source, 2, salt=index_i)
        if answer is None:
            continue
        result = sidecar.solve(source, operators)
        solved += int(result["answer"] == answer)
        total += 1
    source, operators = task.draw_unreachable(edges, salt=9)
    refused = sidecar.solve(source, operators)

    # 4. Rule-context adaptation under a shift.
    permutation = torch.randperm(64, generator=torch.Generator().manual_seed(99))
    batch = tasks["vision"].sample(32, "train")
    shifted = batch.x.reshape(batch.x.shape[0], -1)[:, permutation].reshape(batch.x.shape)
    adaptation = sidecar.adapt_rule("vision", shifted, batch.y, rule_id=3)
    familiar = sidecar.adapt_rule("vision", batch.x, batch.y, rule_id=0)

    # 5. ACE playbook: compile, then recall the exact steps from the wave.
    steps = ["READ_INDEX", "LOCATE_SYMBOL", "EDIT_CALLSITE", "RUN_TESTS", "VERIFY_VETO"]
    compiled = sidecar.compile_playbook("refactor", steps)
    recalled = sidecar.recall_playbook("refactor")
    exact = sum(1 for a, b in zip(recalled["steps"], recalled["decoded"]) if a == b)

    prototype = sidecar.recall_prototype(
        "vision", tasks["vision"].sample(8, "val").x
    )
    payload = {
        "config_digest": cfg.digest(),
        "steps": int(args.steps),
        "veto": {
            "clean_veto_rate": clean["veto_rate"],
            "ood_veto_rate": noisy["veto_rate"],
            "clean_s_max": clean["s_max"],
            "ood_s_max": noisy["s_max"],
        },
        "code_index": index,
        "topology": {
            "edges": ingest["edges_ingested"],
            "tau_coherence": ingest.get("tau_coherence", 0.0),
            "two_hop_accuracy": solved / max(total, 1),
            "two_hop_n": float(total),
            "unreachable_refused": float(bool(refused["vetoed"])),
            "unreachable_text": refused["text"],
        },
        "adaptation": {"shifted": adaptation, "familiar": familiar},
        "playbook": {
            "compiled": compiled,
            "recall_coherence": recalled.get("coherence", 0.0),
            "decoded": recalled["decoded"],
            "exact_steps": float(exact),
            "steps": float(len(steps)),
            "text": recalled["text"],
        },
        "prototype_recall": {
            "alignment": prototype.get("alignment", float("nan")),
            "labels": prototype.get("labels", [])[:8],
        },
        "report": sidecar.report(),
        "gate": {
            "ood_veto_at_least_0_95": bool(noisy["veto_rate"] >= 0.95),
            "false_veto_at_most_0_05": bool(clean["veto_rate"] <= 0.05),
            "code_index_edges_positive": bool(index["edges"] > 0),
            "topology_two_hop_at_least_0_80": bool(solved / max(total, 1) >= 0.80),
            "unreachable_refused": bool(refused["vetoed"]),
            "playbook_exact": bool(exact == len(steps)),
        },
        "evidence": "OBSERVED",
    }
    payload["gate"]["passed"] = bool(
        payload["gate"]["ood_veto_at_least_0_95"]
        and payload["gate"]["topology_two_hop_at_least_0_80"]
        and payload["gate"]["unreachable_refused"]
        and payload["gate"]["playbook_exact"]
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    (out.parent / (out.name + ".sha256")).write_text(
        hashlib.sha256(out.read_bytes()).hexdigest() + "\n", encoding="utf-8"
    )
    print(json.dumps(payload["gate"], indent=2, default=float))
    print(payload["playbook"]["text"])
    print(payload["topology"]["unreachable_text"])
    sidecar.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
