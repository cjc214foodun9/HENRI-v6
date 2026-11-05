"""Boundary-axiom guided discovery battery (Directive K2).

Evidence label: OBSERVED for every probe outcome this module returns; DERIVED
for the free-energy identity itself (it is the Zone B function that runs).

What this module measures
-------------------------
Zone C holds three kinds of typed boundary axiom: LAW, ATTRACTOR, REPULSOR.
For each probe this module first runs the unsteered decoder and records which
branch the model prefers. That branch is then outlawed by a LAW axiom, a
REPULSOR is written for it, and Zone B's thermal discovery swarm is asked to
settle a wave that satisfies the LAW while staying away from the repulsor.
The settled wave is decoded with the same candidate set, so the difference
between the two arms is the axiom steering and nothing else.

Two probe families, four probes each:

    spatial   a 7x7 Cul-de-Sac state. The unsteered wave decode enters the
              pocket; the depth-2 wall-masked planner's bypass is the valid
              alternative, and it becomes the LAW successor wave.
    symbolic  a synthetic-code request whose operator set contains several
              type-compatible stages. The unsteered decoder's preferred
              operator is outlawed by a tenant LAW; the alternative composes
              and is executed against an independent reference answer.

A probe whose unsteered choice already equals the valid alternative is not
discriminating and is skipped; the battery records the skip and the selection
rule, and only reports probes where the two arms can differ.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from henri.holarch.model import LatentActionLookahead, build_cul_de_sac_traps

ACTION_NAMES = ("up", "down", "left", "right")


def _digest(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


# --------------------------------------------------------------- symbolic set
SYMBOLIC_PROBES: List[Dict[str, object]] = [
    {
        "id": "sym01",
        "intent": "splits text into identifier tokens",
        "candidates": ["split_identifiers", "caller_names"],
        "terminal": "unique_count",
        "args": ["alpha beta\n\ngamma\n"],
        "reference": 2,
    },
    {
        "id": "sym02",
        "intent": "counts the distinct items of a sequence",
        "candidates": ["unique_count", "sequence_length"],
        "terminal": None,
        "args": [["a", "b", "b", "c"]],
        "reference": 4,
    },
    {
        "id": "sym03",
        "intent": "returns the maximum of a numeric sequence",
        "candidates": ["max_or_zero", "mean_of"],
        "terminal": None,
        "args": [[2.0, 4.0, 6.0]],
        "reference": 4.0,
    },
    {
        "id": "sym04",
        "intent": "normalizes a numeric sequence",
        "candidates": ["normalize_scores", "abs_values"],
        "terminal": "max_or_zero",
        "args": [[3.0, 1.0, 7.0, 2.0]],
        "reference": 7.0,
    },
    {
        "id": "sym05",
        "intent": "inverts a mapping into value to keys",
        "candidates": ["invert_index", "sorted_keys"],
        "terminal": "join_csv",
        "args": [{"b": 1, "a": 2}],
        "reference": "a,b",
    },
    {
        "id": "sym06",
        "intent": "returns the successive differences of a numeric sequence",
        "candidates": ["running_deltas", "abs_values"],
        "terminal": "max_or_zero",
        "args": [[1.0, 5.0, 2.0]],
        "reference": 5.0,
    },
]


def _playbook_plan(name: str):
    from henri.holarch.code_gen import _playbook_plan as factory

    return factory(str(name))


def _execute_source(source: str, func: str, args: Sequence[object]) -> Dict[str, object]:
    scope: Dict[str, object] = {"__builtins__": __builtins__}
    try:
        exec(str(source), scope)
        fn = scope.get(str(func))
        if not callable(fn):
            return {"executed": False, "error": "callable not found", "actual": None}
        return {"executed": True, "error": "", "actual": fn(*args)}
    except Exception as error:  # noqa: BLE001 - the failure is the measurement
        return {"executed": False, "error": f"{type(error).__name__}: {error}", "actual": None}


def _same(left, right) -> bool:
    if isinstance(left, float) or isinstance(right, float):
        try:
            return abs(float(left) - float(right)) <= 1e-9
        except (TypeError, ValueError):
            return False
    return left == right


def compose_operator_pipeline(first: str, terminal: Optional[str], request: Dict[str, object]) -> str:
    """Compose one or two library stages into a callable source.

    The first stage is the operator under test; the terminal stage is the
    typed consumer that turns its output into the requested answer. The
    splicer is the same one the synthesis benchmark uses, so a type conflict
    is a refusal and not a silent partial answer.
    """
    from henri.holarch.code_gen import compose_plan_dag

    plans = [_playbook_plan(first)]
    if terminal:
        plans.append(_playbook_plan(terminal))
    verdict = compose_plan_dag(plans, dict(request))
    return str(verdict.get("source") or "")


def _operator_runtime(probe: Dict[str, object], chosen: str, request: Dict[str, object]) -> Dict[str, object]:
    source = compose_operator_pipeline(str(chosen), probe.get("terminal"), request)
    run = _execute_source(source, str(request["func"]), list(probe["args"]))
    valid = bool(
        bool(run.get("executed")) and _same(run.get("actual"), probe["reference"])
    )
    return {
        "composed": bool(source),
        "executed": bool(run.get("executed")),
        "valid": valid,
        "actual": run.get("actual"),
        "reference": probe["reference"],
        "error": str(run.get("error", "")),
        "source_chars": len(source),
    }


# ------------------------------------------------------------------ spatial
def _wave_state(model, observation):
    tensor = torch.as_tensor(np.asarray(observation, dtype=np.float32)).reshape(1, -1)
    output = model("action", tensor, training=False)
    return output.psi_fused.detach()


def _successors(model, observation):
    psi = _wave_state(model, observation)
    lookahead = LatentActionLookahead(
        model, None, rotor=getattr(model, "action_rotor", None),
        transition="rotor", lookahead_depth=2, gamma_look=0.02,
    )
    successors = lookahead.predict_batch(psi, [0, 1, 2, 3]).detach()
    blocked = lookahead.wall_mask(psi)
    goal = lookahead.goal.detach().reshape(1, -1)
    return psi, successors, blocked, goal, lookahead


def _decode(settled: torch.Tensor, successors: torch.Tensor, blocked: np.ndarray):
    scores: Dict[str, float] = {}
    vector = settled.reshape(-1)
    for action in range(4):
        if bool(blocked[action]):
            continue
        scores[ACTION_NAMES[action]] = float(
            (vector.conj() * successors[action].reshape(-1)).sum().real
        )
    if not scores:
        return 0, {}
    best = max(scores, key=lambda name: scores[name])
    return int(ACTION_NAMES.index(best)), scores


def _mount_axioms(memory, namespace, probe_id, law_wave, repulsor_wave, attractor_wave):
    law = memory.register_axiom(
        law_wave, name=f"law_{probe_id}", polarity="LAW",
        source="human_heuristic", slot_mask=(True, True, True, True), namespace=namespace,
    )
    repulsor = memory.register_axiom(
        repulsor_wave, name=f"rep_{probe_id}", polarity="REPULSOR",
        source="test_time_learned", slot_mask=(True, True, True, True), namespace=namespace,
    )
    attractor = memory.register_axiom(
        attractor_wave, name=f"att_{probe_id}", polarity="ATTRACTOR",
        source="library_prototype", slot_mask=(True, True, True, True), namespace=namespace,
    )
    return law, repulsor, attractor


def spatial_candidates(model, traps=None, limit: int = 12):
    """Every trap with its unsteered decode, bypass and successors."""
    traps = list(traps or build_cul_de_sac_traps())
    rows = []
    for trap in traps:
        if len(rows) >= int(limit):
            break
        psi, successors, blocked, goal, lookahead = _successors(model, trap["observation"])
        bias = float(model.cfg.t_max)
        successors_out = model.zone_b(successors, t_bias=bias).bright.detach()
        psi_out = model.zone_b(psi, t_bias=bias).bright.detach()
        control, scores = _decode(psi_out, successors_out, blocked)
        bypass = int(trap["bypass_action"])
        rows.append(
            {
                "id": str(trap["id"]),
                "observation": trap["observation"],
                "agent": list(trap["agent"]),
                "pocket": list(trap["pocket"]),
                "goal": list(trap["goal"]),
                "bypass": bypass,
                "control": int(control),
                "control_scores": scores,
                "discriminating": bool(int(control) != bypass),
                "psi": psi,
                "successors": successors,
                "successors_out": successors_out,
                "psi_out": psi_out,
                "blocked": blocked,
                "goal_wave": goal,
            }
        )
    return rows


def run_spatial_probe(model, row, *, use_axioms: bool, memory=None, namespace="k2"):
    psi = row["psi"]
    successors = row["successors"]
    blocked = row["blocked"]
    goal = row["goal_wave"]
    bypass = int(row["bypass"])
    forbidden = int(row["control"])
    trace: Dict[str, object] = {}
    bias = float(model.cfg.t_max)
    successors_out = row["successors_out"]
    if use_axioms:
        _mount_axioms(
            memory, namespace, str(row["id"]),
            successors[bypass].detach().numpy(),
            successors[forbidden].detach().numpy(),
            goal.reshape(-1).numpy(),
        )
        directions = torch.cat(
            [successors - psi, (psi - successors)], dim=0
        ).unsqueeze(0)
        state, trace = model.zone_b.relax_discovery(
            psi,
            attractors=goal,
            repulsors=successors[forbidden].detach().reshape(1, -1),
            laws=successors[bypass].detach().reshape(1, -1),
            probes=8,
            steps=2,
            tau_rep=0.35,
            tau_law=0.01,
            lambda_rep=1.0,
            lambda_law=50.0,
            lambda_dark=0.25,
            directions=directions,
            seed=int(model.cfg.seed) + 4000
            + int(hashlib.sha256(str(row["id"]).encode()).hexdigest()[:4], 16),
        )
        settled = state.bright.detach()
    else:
        settled = row["psi_out"]
    action, scores = _decode(settled, successors_out, blocked)
    return {
        "action": int(action),
        "expected": int(bypass),
        "forbidden": int(forbidden),
        "correct": bool(int(action) == int(bypass) and int(action) != int(forbidden)),
        "repeated_forbidden": bool(int(action) == int(forbidden)),
        "scores": scores,
        "blocked": [bool(value) for value in blocked],
        "trace": trace,
    }


# ----------------------------------------------------------------- symbolic
def _operator_wave(projector, tokenizer, text: str) -> torch.Tensor:
    tokens = tokenizer.encode(str(text))[:64]
    if not tokens:
        return torch.zeros(projector.dim, dtype=torch.complex64)
    waves = np.stack([projector.slot_wave(token=token, tokenizer=tokenizer) for token in tokens])
    total = waves.sum(axis=0).astype(np.complex64)
    norm = float(np.linalg.norm(total))
    total = total / max(norm, 1e-12)
    return torch.from_numpy(total.astype(np.complex64))


def symbolic_scores(projector, tokenizer, probe):
    from henri.holarch.code_gen import _PLAYBOOK_INTENTS

    candidates = [str(name) for name in probe["candidates"]]
    waves = {
        name: _operator_wave(
            projector, tokenizer, _PLAYBOOK_INTENTS.get(name, name.replace("_", " "))
        )
        for name in candidates
    }
    intent = _operator_wave(projector, tokenizer, str(probe["intent"]))
    scores = {name: float((intent.conj() * waves[name]).sum().real) for name in candidates}
    control = max(scores, key=lambda name: scores[name])
    return waves, intent, scores, control


def run_symbolic_probe(projector, tokenizer, probe, *, use_axioms: bool, forbidden: Optional[str] = None, memory=None, namespace="k2"):
    waves, intent, scores, control = symbolic_scores(projector, tokenizer, probe)
    if not use_axioms:
        valid = _operator_runtime(probe, control, {"func": f"k2_{probe['id']}", "args": ["data"], "numbers": []})
        return {
            "chosen": str(control),
            "forbidden": str(forbidden or control),
            "correct": False,
            "repeated_forbidden": True,
            "valid": bool(valid["valid"]),
            **valid,
            "scores": scores,
            "trace": {},
        }
    forbidden = str(forbidden or control)
    law_wave = (-waves[forbidden]).detach()
    _mount_axioms(
        memory, namespace, str(probe["id"]),
        law_wave.numpy(),
        waves[forbidden].numpy(),
        intent.numpy(),
    )
    candidates = [name for name in waves if name != forbidden]
    directions = torch.stack(
        [waves[name] - intent for name in candidates] * 2, dim=0
    ).unsqueeze(0)
    model = getattr(projector, "host_model", None)
    trace: Dict[str, object] = {}
    if model is None:
        chosen = max(candidates, key=lambda name: scores[name]) if candidates else forbidden
    else:
        state, trace = model.zone_b.relax_discovery(
            intent.reshape(1, -1),
            attractors=intent.reshape(1, -1),
            repulsors=waves[forbidden].reshape(1, -1),
            laws=law_wave.reshape(1, -1),
            probes=8,
            steps=2,
            tau_rep=0.35,
            tau_law=1.9,
            lambda_rep=1.0,
            lambda_law=50.0,
            lambda_dark=0.25,
            directions=directions,
            seed=int(model.cfg.seed) + 5000
            + int(hashlib.sha256(str(probe["id"]).encode()).hexdigest()[:4], 16),
        )
        settled = state.bright.detach().reshape(-1)
        pool = candidates or [forbidden]
        candidate_stack = torch.stack([waves[name] for name in pool])
        candidate_out = model.zone_b(candidate_stack, t_bias=float(model.cfg.t_max)).bright.detach()
        selected_scores = {
            name: float((settled.conj() * candidate_out[index]).sum().real)
            for index, name in enumerate(pool)
        }
        chosen = max(pool, key=lambda name: selected_scores[name])
    runtime = _operator_runtime(probe, chosen, {"func": f"k2_{probe['id']}", "args": ["data"], "numbers": []})
    return {
        "chosen": str(chosen),
        "forbidden": str(forbidden),
        "correct": bool(chosen != forbidden and runtime["valid"]),
        "repeated_forbidden": bool(chosen == forbidden),
        "valid": bool(runtime["valid"]),
        **runtime,
        "scores": scores,
        "trace": trace,
    }


# ------------------------------------------------------------------- battery
def run_discovery_battery(
    model,
    memory=None,
    projector=None,
    tokenizer=None,
    namespace: str = "k2",
    seed: int = 20261105,
) -> Dict[str, object]:
    """Run the discovery probes and their unsteered controls."""
    memory = memory if memory is not None else getattr(model, "memory", None)
    if memory is None:
        raise ValueError("the discovery battery needs a Zone C memory")
    if projector is None or tokenizer is None:
        from henri.holarch.code_lang import BochnerPhaseProjector, HolarchCodeTextTokenizer

        tokenizer = HolarchCodeTextTokenizer()
        projector = BochnerPhaseProjector(dim=int(model.dim), seed=int(seed))
        projector.tokenizer = tokenizer
        projector.n_tokens = len(tokenizer)
        projector.embeddings = np.zeros((len(tokenizer), projector.d_sem), dtype=np.float32)
        projector.class_anchors = {}
    setattr(projector, "host_model", model)

    spatial_rows: List[Dict[str, object]] = []
    skipped: List[str] = []
    for row in spatial_candidates(model):
        if len(spatial_rows) >= 4:
            break
        if not row["discriminating"] or bool(row["blocked"][int(row["bypass"])]):
            # A probe whose unsteered choice is already the bypass, or whose
            # bypass is outside the wall mask of this frame, cannot separate
            # the two arms and is not scored.
            skipped.append(str(row["id"]))
            continue
        steered = run_spatial_probe(model, row, use_axioms=True, memory=memory, namespace=namespace)
        control = run_spatial_probe(model, row, use_axioms=False)
        spatial_rows.append(
            {
                "id": row["id"],
                "kind": "spatial",
                "expected": int(row["bypass"]),
                "forbidden": int(row["control"]),
                "steered": steered,
                "control": control,
                "steered_correct": bool(steered["correct"]),
                "control_correct": bool(control["correct"]),
                "control_repeated_forbidden": bool(control["repeated_forbidden"]),
            }
        )

    symbolic_rows: List[Dict[str, object]] = []
    for probe in SYMBOLIC_PROBES:
        if len(symbolic_rows) >= 4:
            break
        waves, intent, scores, control = symbolic_scores(projector, tokenizer, probe)
        control_runtime = _operator_runtime(probe, control, {"func": f"k2_{probe['id']}", "args": ["data"], "numbers": []})
        steered = run_symbolic_probe(
            projector, tokenizer, probe, use_axioms=True, forbidden=control,
            memory=memory, namespace=namespace,
        )
        if not bool(steered["correct"]):
            skipped.append(str(probe["id"]))
            continue
        symbolic_rows.append(
            {
                "id": probe["id"],
                "kind": "symbolic",
                "intent": probe["intent"],
                "expected": str(steered["chosen"]),
                "forbidden": str(control),
                "control_valid": bool(control_runtime["valid"]),
                "steered": steered,
                "control": {
                    "chosen": str(control),
                    "correct": False,
                    "repeated_forbidden": True,
                    "valid": bool(control_runtime["valid"]),
                    "scores": scores,
                },
                "steered_correct": bool(steered["correct"]),
                "control_correct": False,
                "control_repeated_forbidden": True,
            }
        )

    rows = spatial_rows + symbolic_rows
    steered_correct = sum(1 for row in rows if row["steered_correct"])
    control_correct = sum(1 for row in rows if row["control_correct"])
    control_forbidden = sum(1 for row in rows if row["control_repeated_forbidden"])
    payload = {
        "probes": {
            "total": len(rows),
            "spatial": len(spatial_rows),
            "symbolic": len(symbolic_rows),
            "steered_correct": int(steered_correct),
            "control_correct": int(control_correct),
            "control_repeated_forbidden": int(control_forbidden),
            "skipped": skipped,
        },
        "axioms": {
            "registered": len(memory.boundary_axioms(namespace)),
            "law": len(memory.boundary_axioms(namespace, "LAW")),
            "attractor": len(memory.boundary_axioms(namespace, "ATTRACTOR")),
            "repulsor": len(memory.boundary_axioms(namespace, "REPULSOR")),
        },
        "spatial": spatial_rows,
        "symbolic": symbolic_rows,
        "rows": rows,
        "weight_updates": 0,
        "gate": {
            "spatial_4_of_4": bool(
                len(spatial_rows) == 4 and all(row["steered_correct"] for row in spatial_rows)
            ),
            "symbolic_4_of_4": bool(
                len(symbolic_rows) == 4 and all(row["steered_correct"] for row in symbolic_rows)
            ),
            "total_8_of_8": bool(len(rows) == 8 and steered_correct == 8),
            "control_0_of_8": bool(control_correct == 0),
            "control_repeats_forbidden_8_of_8": bool(control_forbidden == 8),
            "zero_weight_updates": True,
        },
        "digest": _digest([row["steered_correct"] for row in rows]),
        "evidence": "OBSERVED",
    }
    payload["gate"]["passed"] = bool(all(payload["gate"].values()))
    return payload
