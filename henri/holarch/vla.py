"""Unified VLA token space and the 12-episode end-to-end battery (K3).

The token ranges partition one address space into four modality blocks, so a
vision patch, a language token, an action and a relational binding can be
read by the same decoder without an adapter. The battery grounds a 16x16
visual canvas, derives a clamp bound from the grounded digit, navigates a 7x7
Cul-de-Sac layout with the depth-2 rotor lookahead, and synthesizes and
executes a parameterized three-stage Python function inside the same episode.
"""

from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

VLA_RANGES: Dict[str, Tuple[int, int]] = {
    "language": (0, 2048),
    "tier2": (2048, 4096),
    "vision": (4096, 4352),
    "action": (4352, 4608),
    "relational": (4608, 4864),
    "control": (4864, 5120),
}
"""Disjoint token ranges of the unified VLA address space."""


def vision_token(digit: int) -> int:
    return int(VLA_RANGES["vision"][0]) + (int(digit) % 256)


def action_token(action: int) -> int:
    return int(VLA_RANGES["action"][0]) + (int(action) % 256)


def relational_token(value: int) -> int:
    return int(VLA_RANGES["relational"][0]) + (int(value) % 256)


def control_token(value: int) -> int:
    return int(VLA_RANGES["control"][0]) + (int(value) % 256)


def token_modality(token: int) -> str:
    for name, (low, high) in VLA_RANGES.items():
        if int(low) <= int(token) < int(high):
            return name
    return "out-of-range"


def verify_chain(receipts: Sequence[Dict[str, object]]) -> bool:
    """A receipt chain is contiguous when every parent is the previous hash."""
    if not receipts:
        return False
    if str(receipts[0].get("parent")) != "genesis":
        return False
    for index in range(1, len(receipts)):
        if str(receipts[index].get("parent")) != str(receipts[index - 1].get("sha256")):
            return False
    return True


def _canvas_16(x: torch.Tensor) -> np.ndarray:
    image = x.reshape(8, 8) if x.numel() == 64 else x.reshape(16, 16)
    if image.shape == (8, 8):
        image = image.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)
    return image.detach().cpu().numpy().astype(np.float32)


def run_vla_battery(
    model,
    engine,
    tasks,
    episodes: int = 12,
    seed: int = 20261105,
    *,
    max_steps: int = 20,
) -> Dict[str, object]:
    """Execute the 12 end-to-end multi-modal VLA episodes."""
    generator = torch.Generator().manual_seed(int(seed))
    sample = tasks["vision"].sample(int(episodes), "test")
    traps = tasks.get("_traps") if isinstance(tasks, dict) else None
    if traps is None:
        from henri.holarch.model import build_cul_de_sac_traps

        traps = build_cul_de_sac_traps()
    layouts: List[Tuple[object, Tuple[int, int], Tuple[int, int]]] = [
        (trap["walls"], tuple(int(v) for v in trap["agent"]), tuple(int(v) for v in trap["goal"]))
        for trap in traps
    ]
    for layout in tasks["action"].val_layouts:
        layouts.append((layout[0], tuple(int(v) for v in layout[1]), tuple(int(v) for v in layout[2])))
        if len(layouts) >= int(episodes):
            break
    if engine is not None:
        generator_model = getattr(engine, "generator", None)
        if generator_model is not None and hasattr(generator_model, "mount_holarch"):
            generator_model.mount_holarch(model)

    rows: List[Dict[str, object]] = []
    for index in range(int(episodes)):
        canvas = _canvas_16(sample.x[index])
        walls, agent, goal = layouts[index % len(layouts)]
        name = f"vla_edge_scale_{index:02d}"
        intent = (
            f"Write a function {name} that takes argument edges and first returns the "
            "sorted unique source names of a list of edges then returns the number of "
            "tokens in that result then clips the result into the closed interval "
            "0.0 to {bound}."
        )
        test = (
            "class _E:\n"
            "    def __init__(self, source):\n"
            "        self.source = source\n"
            f"result = {name}([_E('b'), _E('a'), _E('b')])\n"
            "assert 0.0 <= result <= {bound}\n"
        )
        episode = model.generate_vla_episode(
            canvas,
            intent,
            task=tasks["action"],
            layout=(walls, agent, goal),
            engine=engine,
            test_source=test,
            max_steps=int(max_steps),
        )
        rows.append(
            {
                "index": int(index),
                "name": name,
                "digit": int(episode["vision"]["digit"]),
                "bound": float(episode["bound"]),
                "actions": list(episode["planning"].get("actions") or []),
                "reached_goal": bool(episode["planning"].get("reached_goal", False)),
                "code_status": str(episode["code"].get("status", "")),
                "code_verified": bool(episode["code"].get("verified", False)),
                "code_executed": bool(episode["code"].get("executed", False)),
                "receipt_chain_valid": bool(episode["receipt_chain_valid"]),
                "receipt_links": len(episode["receipts"]),
                "tokens": list(episode["tokens"]),
                "success": bool(episode["success"]),
            }
        )

    completed = sum(1 for row in rows if row["success"])
    hallucinated = sum(
        1 for row in rows if row["code_status"] == "verified" and not row["code_executed"]
    )
    payload = {
        "episodes": len(rows),
        "completed": int(completed),
        "completion_rate": float(completed / max(len(rows), 1)),
        "hallucination_rate": float(hallucinated / max(len(rows), 1)),
        "goals_reached": int(sum(1 for row in rows if row["reached_goal"])),
        "receipt_chain_valid": bool(all(row["receipt_chain_valid"] for row in rows)),
        "total_receipts": int(sum(row["receipt_links"] for row in rows)),
        "rows": rows,
        "generator_photonic_contexts": int(
            getattr(getattr(engine, "generator", None), "photonic_contexts", 0) or 0
        ),
        "gate": {
            "twelve_episodes": bool(len(rows) == 12),
            "twelve_of_twelve_completed": bool(len(rows) == 12 and completed == 12),
            "zero_hallucination": bool(hallucinated == 0),
            "receipt_chain_contiguous": bool(all(row["receipt_chain_valid"] for row in rows)),
            "every_episode_has_four_modalities": bool(
                all(
                    {"vision", "relational", "action", "code"}
                    <= {str(token).split(":")[0] for token in row["tokens"]}
                    for row in rows
                )
            ),
        },
        "evidence": "OBSERVED",
    }
    payload["gate"]["passed"] = bool(all(payload["gate"].values()))
    return payload
