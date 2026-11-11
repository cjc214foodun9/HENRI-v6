"""henri_holon: one real holonic discovery tick, as a CLI plugin.

    python scripts/henri_holon.py --json
    python scripts/henri_holon.py --spec my_tasks.json --receipt receipts/holarch/p4_henri_holon.json

Directive P4. The Cycle O plugin ran a synthetic dry-run worker (3 seeds,
0 accepted). This runner calls the REAL SynthesisEngine.holonic_discover:

    Ideation (wave coherence over the primitive library)
      -> Tier-1 JEV/AST/type gate
      -> Tier-2 sandbox execution critic
      -> Tier-3 PlanStore compilation, or a revertible ctx.effect rollback.

It exits 0 only when every valid seed is accepted into the PlanStore and
every trap is refused with zero residual state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HOLON_SEEDS: List[Dict[str, Any]] = [
    {
        "id": "holon_span",
        "kind": "seed",
        "intent": (
            "Write a function holon_span that takes argument text and returns the "
            "identifier tokens of the text, then returns the number of distinct "
            "items in that list, then clips the result into the closed interval "
            "0.0 to 2.5."
        ),
        "tests": "assert holon_span('a b a') == 2.0\nassert holon_span('') == 0.0\n",
        "request": {"func": "holon_span", "args": ["text"], "numbers": [0.0, 2.5]},
    },
    {
        "id": "holon_index_csv",
        "kind": "seed",
        "intent": (
            "Write a function holon_index_csv that takes argument index and inverts "
            "a mapping into value to keys, then returns the sorted keys of that "
            "mapping, then joins the sequence into comma separated text."
        ),
        "tests": "assert holon_index_csv({'b': 1, 'a': 2}) == 'a,b'\n",
        "request": {"func": "holon_index_csv", "args": ["index"], "numbers": []},
    },
    {
        "id": "holon_peak",
        "kind": "seed",
        "intent": (
            "Write a function holon_peak that takes argument values and returns the "
            "successive differences of a numeric sequence, then returns the absolute "
            "values of that sequence, then returns the maximum of the sequence above "
            "a floor."
        ),
        "tests": "assert holon_peak([1.0, 3.0, 2.0]) == 2.0\nassert holon_peak([2.0]) == 0.0\n",
        "request": {"func": "holon_peak", "args": ["values"], "numbers": []},
    },
    {
        "id": "holon_gate",
        "kind": "seed",
        "intent": (
            "Write a function holon_gate that takes argument values and normalizes a "
            "numeric sequence so it sums to one, then returns the arithmetic mean of "
            "the numeric sequence, then returns whether the number exceeds a threshold."
        ),
        "tests": "assert holon_gate([3.0]) == True\nassert holon_gate([1.0, 1.0, 1.0]) == False\n",
        "request": {"func": "holon_gate", "args": ["values"], "numbers": []},
    },
]

HOLON_TRAPS: List[Dict[str, Any]] = [
    {
        "id": "trap_cycle_callable",
        "kind": "trap",
        "intent": (
            "Write a function trap_cycle_callable that inverts a mapping, then returns "
            "the sorted keys, then joins them into a callable object."
        ),
        "tests": "assert callable(trap_cycle_callable({'a': 1}))\n",
        "request": {"func": "trap_cycle_callable", "args": ["index"], "numbers": []},
    },
    {
        "id": "trap_arity_pair",
        "kind": "trap",
        "intent": (
            "Write a function trap_arity_pair that takes two numeric sequences and "
            "returns the element-wise product of the two sequences."
        ),
        "tests": "assert trap_arity_pair([1.0], [2.0]) == [2.0]\n",
        "request": {
            "func": "trap_arity_pair",
            "args": ["left", "right"],
            "numbers": [],
        },
    },
    {
        "id": "trap_type_literal",
        "kind": "trap",
        "intent": (
            "Write a function trap_type_literal that strips comments from source text "
            "and then returns the exact string impossible_token."
        ),
        "tests": "assert trap_type_literal('x = 1  # note') == 'impossible_token'\n",
        "request": {"func": "trap_type_literal", "args": ["text"], "numbers": []},
    },
    {
        "id": "trap_guard_contradiction",
        "kind": "trap",
        "intent": (
            "Write a function trap_guard_contradiction that clips a number into the "
            "closed interval 0.0 to 2.5 and then returns a value greater than 2.5 for "
            "every input."
        ),
        "tests": "assert trap_guard_contradiction(0.0) > 2.5\n",
        "request": {"func": "trap_guard_contradiction", "args": ["value"], "numbers": [0.0, 2.5]},
    },
]


# Directive Q4: four new warm-daemon discoveries and four new
# rollback/fact-veto traps. They are separate from the P4 lists so the P4
# 100/100 battery keeps its exact membership; Cycle Q runs them through a
# process that mounts the engine ONCE.
HOLON_Q_SEEDS: List[Dict[str, Any]] = [
    {
        "id": "holon_q_peak",
        "kind": "seed",
        "intent": (
            "Write a function holon_q_peak that takes argument values and returns the "
            "absolute values of a numeric sequence, then returns the maximum value, then "
            "clips the result into the closed interval 0.0 to 2.5."
        ),
        "tests": "assert holon_q_peak([-1.0, -3.0]) == 2.5\nassert holon_q_peak([0.0]) == 0.0\n",
        "request": {"func": "holon_q_peak", "args": ["values"], "numbers": [0.0, 2.5]},
    },
    {
        "id": "holon_q_comments",
        "kind": "seed",
        "intent": (
            "Write a function holon_q_comments that takes argument text and strips "
            "comments from the source text, then returns the number of tokens in that "
            "text, then clips the result into the closed interval 0.0 to 2.5."
        ),
        "tests": "assert holon_q_comments('x = 1  # note') == 2.5\nassert holon_q_comments('') == 0.0\n",
        "request": {"func": "holon_q_comments", "args": ["text"], "numbers": [0.0, 2.5]},
    },
    {
        "id": "holon_q_words",
        "kind": "seed",
        "intent": (
            "Write a function holon_q_words that takes argument text and returns the "
            "identifier tokens of the text, then returns the number of distinct items "
            "in that list, then clips the result into the closed interval 0.0 to 2.5."
        ),
        "tests": "assert holon_q_words('a b a') == 2.0\nassert holon_q_words('') == 0.0\n",
        "request": {"func": "holon_q_words", "args": ["text"], "numbers": [0.0, 2.5]},
    },
    {
        "id": "holon_q_mean",
        "kind": "seed",
        "intent": (
            "Write a function holon_q_mean that takes argument values and returns the "
            "absolute values of a numeric sequence, then returns the arithmetic mean of "
            "the numeric sequence, then clips the result into the closed interval 0.0 to 2.5."
        ),
        "tests": "assert holon_q_mean([-1.0, -3.0]) == 2.0\nassert holon_q_mean([0.0]) == 0.0\n",
        "request": {"func": "holon_q_mean", "args": ["values"], "numbers": [0.0, 2.5]},
    },
]

HOLON_Q_TRAPS: List[Dict[str, Any]] = [
    {
        "id": "trap_q_fact",
        "kind": "trap",
        "intent": (
            "Write a function trap_q_fact that strips comments from source text and "
            "then returns the exact string impossible_token."
        ),
        "tests": "assert trap_q_fact('x = 1  # note') == 'impossible_token'\n",
        "request": {"func": "trap_q_fact", "args": ["text"], "numbers": []},
    },
    {
        "id": "trap_q_value",
        "kind": "trap",
        "intent": "Write a function trap_q_value that returns the number 12345 for every input.",
        "tests": "assert trap_q_value('a') == 12345\n",
        "request": {"func": "trap_q_value", "args": ["text"], "numbers": []},
    },
    {
        "id": "trap_q_arity",
        "kind": "trap",
        "intent": (
            "Write a function trap_q_arity that takes two numbers and returns their product."
        ),
        "tests": "assert trap_q_arity(2.0, 3.0) == 6.0\n",
        "request": {"func": "trap_q_arity", "args": ["left", "right"], "numbers": []},
    },
    {
        "id": "trap_q_callable",
        "kind": "trap",
        "intent": "Write a function trap_q_callable that turns a mapping into a callable object.",
        "tests": "assert callable(trap_q_callable({'a': 1}))\n",
        "request": {"func": "trap_q_callable", "args": ["index"], "numbers": []},
    },
]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_receipt(path: Path, payload: Dict[str, Any]) -> Dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    digest = _digest(path)
    path.with_name(path.name + ".sha256").write_text(digest + chr(10), encoding="utf-8")
    return {"path": str(path), "sha256": digest}


def run_tick(
    tasks: Sequence[Dict[str, Any]],
    *,
    root: str | Path = ROOT,
    checkpoint: Optional[str] = None,
) -> Dict[str, Any]:
    """One discovery tick over the seed and trap lists."""
    import torch

    from henri.holarch.code_gen import build_synthesis_engine
    from henri.holarch.config import HolarchConfig
    from henri.holarch.model import find_checkpoint, open_clean_room_memory

    base = Path(root).resolve()
    # The plugin mounts the repository's pinned checkpoint by name. Asking
    # find_checkpoint with a SMALL config returns nothing, because the pinned
    # checkpoint is the medium one; the explicit path is the same file the
    # sidecar mounts.
    default_checkpoint = base / "receipts" / "holarch" / "holarch_medium.pt"
    path = (
        checkpoint
        or (str(default_checkpoint) if default_checkpoint.exists() else None)
        or find_checkpoint(HolarchConfig.small(), explicit=None)
    )
    payload = torch.load(path, map_location="cpu", weights_only=True) if path else {}
    cfg = HolarchConfig(**payload.get("config", {})) if payload else HolarchConfig.small()
    workdir = base / ".pwork" / "_holon"
    workdir.mkdir(parents=True, exist_ok=True)
    memory = None
    model = None
    if path:
        from henri.holarch.pipeline import HenriHolarch
        from henri.holarch.model import load_checkpoint

        memory, _info = open_clean_room_memory(cfg, path, workdir, prefix="henri_holon")
        model = HenriHolarch(cfg, memory=memory)
        load_checkpoint(str(path), model=model)
        model.eval()
    engine, stats = build_synthesis_engine(str(base), memory)
    rows: List[Dict[str, Any]] = []
    for task in tasks:
        report = engine.holonic_discover(
            str(task["intent"]),
            str(task.get("tests", "")),
            dict(task.get("request") or {}),
        )
        rows.append(
            {
                "id": str(task["id"]),
                "kind": str(task.get("kind", "seed")),
                "verified": bool(report.get("verified")),
                "status": str(report.get("status")),
                "accepted": list(report.get("accepted") or []),
                "attempts": int(report.get("attempts", 0)),
                "candidates": int(report.get("candidates", 0)),
                "effects": int(len(report.get("effects") or [])),
                "rolled_back": bool(report.get("reverted", 0)),
                "residual_state": int(report.get("residual_state", 0)),
                "plans_before": int(report.get("plans_before", 0)),
                "plans_after": int(report.get("plans_after", 0)),
                "plan": str(report.get("plan", "")),
                "plan_digest": str(report.get("plan_digest", "")),
                "reason": str(report.get("reason", "")),
            }
        )
    seeds = [row for row in rows if row["kind"] == "seed"]
    traps = [row for row in rows if row["kind"] == "trap"]
    accepted = sum(1 for row in seeds if row["verified"])
    refused = sum(1 for row in traps if not row["verified"])
    residual = sum(abs(int(row["residual_state"])) for row in rows)
    harness = {
        "seeds": len(seeds),
        "accepted": int(accepted),
        "traps": len(traps),
        "refused": int(refused),
        "residual_state": int(residual),
        "candidates": int(sum(row["candidates"] for row in seeds)),
        "plan_store_size": int(len(engine.plans.plans)),
        "engine": dict(stats),
        "checkpoint": str(path or ""),
        "passed": bool(
            len(seeds) >= 4 and accepted == len(seeds) and refused == len(traps) and residual == 0
        ),
    }
    if memory is not None:
        memory.close()
    return {"rows": rows, "harness": harness, "evidence": "OBSERVED"}


class HolonDaemon:
    """A resident holon: mount the checkpoint and engine ONCE, tick many times.

    Directive Q4. run_tick rebuilds the tokenizer, projector, count model and
    plan store on every call, so a tick costs a corpus fit. The daemon keeps
    the mounted objects alive, which is what makes a warm discovery tick a
    millisecond-scale operation instead of a second-scale one.
    """

    def __init__(
        self,
        *,
        root: str | Path = ROOT,
        checkpoint: Optional[str] = None,
    ) -> None:
        import torch

        from henri.holarch.code_gen import build_synthesis_engine
        from henri.holarch.config import HolarchConfig
        from henri.holarch.model import find_checkpoint, load_checkpoint, open_clean_room_memory
        from henri.holarch.pipeline import HenriHolarch

        base = Path(root).resolve()
        self.base = base
        default_checkpoint = base / "receipts" / "holarch" / "holarch_medium.pt"
        path = (
            checkpoint
            or (str(default_checkpoint) if default_checkpoint.exists() else None)
            or find_checkpoint(HolarchConfig.small(), explicit=None)
        )
        payload = torch.load(path, map_location="cpu", weights_only=True) if path else {}
        cfg = HolarchConfig(**payload.get("config", {})) if payload else HolarchConfig.small()
        workdir = base / ".pwork" / "_holon_daemon"
        workdir.mkdir(parents=True, exist_ok=True)
        self.memory = None
        self.model = None
        if path:
            self.memory, _info = open_clean_room_memory(
                cfg, path, workdir, prefix="henri_holon_daemon"
            )
            self.model = HenriHolarch(cfg, memory=self.memory)
            load_checkpoint(str(path), model=self.model)
            self.model.eval()
        self.engine, self.stats = build_synthesis_engine(str(base), self.memory)
        self.checkpoint = str(path or "")
        self.ticks = 0

    def tick(self, tasks: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        """One warm tick over the task list; returns rows and per-task ms."""
        import time as _time

        rows: List[Dict[str, Any]] = []
        task_ms: List[float] = []
        for task in tasks:
            started = _time.perf_counter()
            report = self.engine.holonic_discover(
                str(task["intent"]),
                str(task.get("tests", "")),
                dict(task.get("request") or {}),
            )
            elapsed = (_time.perf_counter() - started) * 1e3
            task_ms.append(float(elapsed))
            rows.append(
                {
                    "id": str(task["id"]),
                    "kind": str(task.get("kind", "seed")),
                    "verified": bool(report.get("verified")),
                    "status": str(report.get("status")),
                    "accepted": list(report.get("accepted") or []),
                    "attempts": int(report.get("attempts", 0)),
                    "candidates": int(report.get("candidates", 0)),
                    "effects": int(len(report.get("effects") or [])),
                    "rolled_back": bool(report.get("reverted", 0)),
                    "residual_state": int(report.get("residual_state", 0)),
                    "plans_before": int(report.get("plans_before", 0)),
                    "plans_after": int(report.get("plans_after", 0)),
                    "plan": str(report.get("plan", "")),
                    "plan_digest": str(report.get("plan_digest", "")),
                    "reason": str(report.get("reason", "")),
                    "elapsed_ms": float(elapsed),
                }
            )
        self.ticks += 1
        return {"rows": rows, "task_ms": task_ms}

    def close(self) -> None:
        if self.memory is not None:
            self.memory.close()


def run_daemon(
    tasks: Sequence[Dict[str, Any]],
    *,
    ticks: int = 3,
    root: str | Path = ROOT,
    checkpoint: Optional[str] = None,
    warmup: bool = True,
) -> Dict[str, Any]:
    """Run a resident daemon for N ticks and report the warm latency."""
    import statistics

    daemon = HolonDaemon(root=root, checkpoint=checkpoint)
    if bool(warmup) and tasks:
        daemon.tick(list(tasks)[:1])
    all_ms: List[float] = []
    tick_medians: List[float] = []
    report: Dict[str, Any] = {"rows": [], "task_ms": []}
    for _ in range(max(1, int(ticks))):
        report = daemon.tick(tasks)
        all_ms.extend(report["task_ms"])
        tick_medians.append(
            float(statistics.median(report["task_ms"])) if report["task_ms"] else 0.0
        )
    rows = list(report["rows"])
    seeds = [row for row in rows if row["kind"] == "seed"]
    traps = [row for row in rows if row["kind"] == "trap"]
    accepted = sum(1 for row in seeds if row["verified"])
    refused = sum(1 for row in traps if not row["verified"])
    residual = sum(abs(int(row["residual_state"])) for row in rows)
    median_ms = float(statistics.median(all_ms)) if all_ms else 0.0
    harness = {
        "seeds": len(seeds),
        "accepted": int(accepted),
        "traps": len(traps),
        "refused": int(refused),
        "residual_state": int(residual),
        "candidates": int(sum(row["candidates"] for row in seeds)),
        "plan_store_size": int(len(daemon.engine.plans.plans)),
        "engine": dict(daemon.stats),
        "checkpoint": str(daemon.checkpoint),
        "daemon": True,
        "ticks": int(daemon.ticks),
        "warm_ticks": int(max(1, int(ticks))),
        "task_ticks": int(len(all_ms)),
        "median_tick_ms": float(median_ms),
        "first_tick_median_ms": float(tick_medians[0]) if tick_medians else 0.0,
        "warm_repeat_median_ms": float(
            statistics.median(tick_medians[1:]) if len(tick_medians) > 1 else median_ms
        ),
        "tick_medians_ms": [float(value) for value in tick_medians],
        "p95_tick_ms": float(
            sorted(all_ms)[max(0, int(0.95 * len(all_ms)) - 1)] if all_ms else 0.0
        ),
        "max_tick_ms": float(max(all_ms)) if all_ms else 0.0,
        "passed": bool(
            len(seeds) >= 4
            and accepted == len(seeds)
            and refused == len(traps)
            and residual == 0
            and median_ms <= 15.0
        ),
    }
    daemon.close()
    return {"rows": rows, "harness": harness, "evidence": "OBSERVED"}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="henri_holon")
    parser.add_argument("--spec", default="", help="JSON file with a list of tasks")
    parser.add_argument("--receipt", default="", help="write the JSON receipt here")
    parser.add_argument("--json", action="store_true", help="print the full report")
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="mount the engine once and run warm resident ticks",
    )
    parser.add_argument("--ticks", type=int, default=3, help="warm daemon ticks")
    args = parser.parse_args(argv)
    tasks: List[Dict[str, Any]] = list(HOLON_SEEDS) + list(HOLON_TRAPS)
    if args.spec:
        tasks = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    report = (
        run_daemon(tasks, ticks=int(args.ticks))
        if bool(args.daemon)
        else run_tick(tasks)
    )
    if args.receipt:
        written = _write_receipt(Path(args.receipt), report)
        report["receipt"] = written
    harness = report["harness"]
    print(
        "henri_holon seeds={seeds} accepted={accepted} traps={traps} refused={refused} "
        "residual={residual_state} plan_store={plan_store_size}".format(**harness)
    )
    for row in report["rows"]:
        print(
            "  {id:<26} {kind:<5} verified={verified!s:<5} effects={effects:<3} "
            "residual={residual_state}".format(**row)
        )
    if args.json:
        print(json.dumps(report, indent=2, default=float))
    return 0 if bool(harness["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
