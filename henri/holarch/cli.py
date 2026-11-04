"""Command line entry point for the HENRI holarch.

    python -m henri.holarch selftest
    python -m henri.holarch physics
    python -m henri.holarch memory --n 2000 --backend auto
    python -m henri.holarch budget
    python -m henri.holarch train --steps 300 --name demo

Every subcommand accepts --json. Exit code 1 means a gate failed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from henri.holarch.config import HolarchConfig, TASK_NAMES, VRAM_BUDGET_GB
from henri.holarch.zone_b import PhotonicCore, curie_weiss_gain


def _emit(payload: Dict, as_json: bool, title: str = "") -> None:
    if as_json:
        print(json.dumps(payload, indent=2, default=float))
        return
    if title:
        print(title)
    for key, value in payload.items():
        if isinstance(value, float):
            print(f"  {key:36s} {value:.6g}")
        elif isinstance(value, (dict, list)):
            print(f"  {key:36s} {json.dumps(value, default=float)}")
        else:
            print(f"  {key:36s} {value}")


def cmd_physics(args) -> int:
    cfg = HolarchConfig.small() if args.small else HolarchConfig(dim=int(args.dim))
    torch.manual_seed(cfg.seed)
    core = PhotonicCore(cfg)
    generator = torch.Generator().manual_seed(cfg.seed)
    wave = torch.randn(4, cfg.dim, generator=generator, dtype=torch.complex64)
    wave = wave / wave.norm(dim=-1, keepdim=True)
    axiom = torch.roll(wave, shifts=3, dims=-1)
    state = core(wave, axiom=axiom)
    shift_cold = core.reachable_shift(wave[0], 0.1, 0.1)
    shift_hot = core.reachable_shift(wave[0], 0.1, cfg.t_max - 1e-3)
    calibration = core.calibrate([wave[i] for i in range(4)])
    payload = {
        "dim": float(cfg.dim),
        "slots": float(cfg.n_slots),
        "layers": float(cfg.photonic_layers),
        "norm_drift": float(state.norm_drift),
        "delta_memory_mean": float(state.delta_memory.mean().detach()),
        "delta_reciprocal_mean": float(state.delta_reciprocal.mean().detach()),
        "gain_at_zero": float(curie_weiss_gain(torch.tensor([0.0]))),
        "gain_at_max": float(curie_weiss_gain(torch.tensor([cfg.t_max]))),
        "reachable_shift_cold": float(shift_cold),
        "reachable_shift_hot": float(shift_hot),
        "authority_ratio": float(shift_hot / max(shift_cold, 1e-9)),
        "epsilon": float(calibration["epsilon"]),
        "calibration": calibration,
    }
    _emit(payload, args.json, "Zone B photonic kernel")
    return 0 if payload["norm_drift"] < 1e-4 and shift_hot > shift_cold else 1


def cmd_memory(args) -> int:
    from henri.holarch.zone_c import ZoneCMemory

    cfg = HolarchConfig.small() if args.small else HolarchConfig(dim=int(args.dim))
    path = args.path or str(Path(tempfile.gettempdir()) / "henri_holarch_memory.sqlite3")
    if args.fresh and Path(path).exists():
        Path(path).unlink()
    memory = ZoneCMemory(cfg, backend=args.backend, path=path)
    if args.fresh:
        memory.store.reset()
        memory._size = 0
    generator = np.random.default_rng(cfg.seed)
    count = int(args.n)
    start = time.perf_counter()
    for index in range(count):
        wave = generator.standard_normal(cfg.dim) + 1j * generator.standard_normal(cfg.dim)
        wave = (wave / np.linalg.norm(wave)).astype(np.complex64)
        memory.write(wave, label=index % 10)
    write_seconds = time.perf_counter() - start

    queries = max(1, count // 10)
    hits = 0
    latencies: List[float] = []
    for index in range(queries):
        wave = generator.standard_normal(cfg.dim) + 1j * generator.standard_normal(cfg.dim)
        wave = (wave / np.linalg.norm(wave)).astype(np.complex64)
        memory.write(wave, label=index % 10)
    for index in range(queries):
        wave = memory.fetch_wave(index + 1)
        read = memory.query(wave, k=4)
        latencies.append(read.latency_ms)
        if read.hits and read.hits[0].engram_id == index + 1:
            hits += 1

    prefetch = memory.prefetch(memory.fetch_wave(1), k=4)
    consolidation = memory.consolidate(chart_size=min(64, memory.stats()["count"]))
    stats = memory.stats()
    prefix, read = memory.context_tensor(memory.fetch_wave(1), k=4)
    payload = {
        "backend": memory.backend,
        "backend_reason": memory.backend_reason,
        "count": stats["count"],
        "write_seconds": float(write_seconds),
        "writes_per_second": float(stats["count"] / max(write_seconds, 1e-9)),
        "recall_at_1": float(hits / max(queries, 1)),
        "mean_query_ms": float(np.mean(latencies)),
        "p95_query_ms": float(np.percentile(latencies, 95)),
        "prefetch": prefetch,
        "consolidation": consolidation,
        "context_shape": list(prefix.shape),
        "context_bytes": int(prefix.nbytes),
        "resident_index_bytes": int(stats["gram_bytes"]),
        "store_bytes": int(stats["wave_bytes"]),
    }
    _emit(payload, args.json, "Zone C holographic memory")
    memory.close()
    ok = payload["recall_at_1"] > 0.9 and payload["context_shape"] == [cfg.context_tokens, cfg.context_dim]
    return 0 if ok else 1


def _detect_accelerators() -> List[str]:
    """Decision D3: name the silicon, not only the torch device count."""
    import subprocess

    found: List[str] = []
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=6
        )
        if result.returncode == 0:
            found.extend(line.strip() for line in result.stdout.splitlines() if line.strip())
    except Exception:
        pass
    if not found:
        try:
            result = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "(Get-CimInstance Win32_VideoController).Name",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                found.extend(
                    line.strip() for line in result.stdout.splitlines() if line.strip()
                )
        except Exception:
            pass
    return found


def _step_latency(model, batch: int) -> Tuple[float, float]:
    """Median one-step forward and backward latency in milliseconds."""
    device = next(model.parameters()).device
    x = torch.zeros(int(batch), 1, 8, 8, device=device)
    model.train()
    forward_ms: List[float] = []
    backward_ms: List[float] = []

    def one_step():
        return model("vision", x, training=True, use_memory=False).task_logits.sum()

    for _ in range(2):
        loss = one_step()
        loss.backward()
        model.zero_grad(set_to_none=True)
    for _ in range(3):
        start = time.perf_counter()
        loss = one_step()
        forward_ms.append((time.perf_counter() - start) * 1e3)
        start = time.perf_counter()
        loss.backward()
        backward_ms.append((time.perf_counter() - start) * 1e3)
        model.zero_grad(set_to_none=True)
    return float(np.median(forward_ms)), float(np.median(backward_ms))


def cmd_budget(args) -> int:
    import hashlib
    import platform

    from henri.holarch.budget import budget_report
    from henri.holarch.pipeline import HenriHolarch
    from henri.holarch.zone_c import ZoneCMemory

    # Decision D3: measure all three profiles on the host that is running,
    # and record which silicon the host actually has. The CUDA verdict stays
    # BLOCKED unless a device answers; a CPU number is never called a GPU one.
    builders = {
        "small": HolarchConfig.small,
        "medium": HolarchConfig.medium,
        "production": lambda: HolarchConfig.production(device="auto"),
    }
    if args.small:
        builders = {"small": builders["small"]}

    profiles = {}
    for name, builder in builders.items():
        cfg = builder()
        path = args.path or str(
            Path(tempfile.gettempdir()) / f"henri_budget_{name}.sqlite3"
        )
        if Path(path).exists() and args.path is None:
            Path(path).unlink()
        model = HenriHolarch(cfg, backend="sqlite", memory_path=path)
        device = cfg.resolved_device()
        if device == "cuda":
            model.to(device)
        report = budget_report(model, batch_size=int(args.batch), cfg=cfg)
        forward_ms, backward_ms = _step_latency(model, int(args.batch))
        profiles[name] = {
            "dim": float(cfg.dim),
            "d_slot": float(cfg.d_slot),
            "photonic_layers": float(cfg.photonic_layers),
            "parameters": float(report["parameters"]),
            "training_gib": float(report["training_gib"]),
            "inference_gib": float(report["inference_gib"]),
            "headroom_gib": float(report["headroom_gib"]),
            "fits": bool(report["fits"]),
            "activation_bytes_training": float(report["activation_bytes_training"]),
            "measured_device": str(report.get("measured_device", "cpu")),
            "cuda_peak_allocated_gib": float(report.get("cuda_peak_allocated_gib", 0.0)),
            "forward_ms": forward_ms,
            "backward_ms": backward_ms,
        }
        model.memory.close()

    probe = ZoneCMemory(HolarchConfig.small(), backend="auto")
    storage = {
        "backend": str(probe.backend),
        "backend_reason": str(probe.backend_reason),
        "timescale_dsn_env_set": bool(os.environ.get("HENRI_TSDB_DSN")),
        "evidence": "OBSERVED" if str(probe.backend) == "timescaledb" else "BLOCKED",
    }
    probe.close()

    cuda_available = bool(torch.cuda.is_available())
    accelerators = _detect_accelerators()
    hardware = {
        "host_platform": platform.platform(),
        "hostname": platform.node(),
        "torch_version": str(torch.__version__),
        "cuda_compiled_version": str(torch.version.cuda),
        "cuda_available": cuda_available,
        "device_count": int(torch.cuda.device_count()) if cuda_available else 0,
        "device_name": torch.cuda.get_device_name(0) if cuda_available else "",
        "detected_accelerators": accelerators,
        "host_has_nvidia_gpu": bool(cuda_available)
        or any(
            token in accelerator.lower()
            for accelerator in accelerators
            for token in ("nvidia", "geforce", "rtx", "quadro", "tesla")
        ),
        "evidence": "OBSERVED" if cuda_available else "BLOCKED",
        "reason": ""
        if cuda_available
        else "no CUDA device is visible to torch; see detected_accelerators",
    }
    hardware["cuda_peak_allocated_gib"] = max(
        (profile["cuda_peak_allocated_gib"] for profile in profiles.values()),
        default=0.0,
    )
    primary = "production" if "production" in profiles else next(iter(profiles))
    all_fit = bool(all(profile["fits"] for profile in profiles.values()))
    payload = {
        "profiles": profiles,
        "profile": primary,
        "config_digest": HolarchConfig.production().digest()
        if primary == "production"
        else (HolarchConfig.medium().digest() if primary == "medium" else HolarchConfig.small().digest()),
        "dim": profiles[primary]["dim"],
        "parameters": profiles[primary]["parameters"],
        "training_gib": profiles[primary]["training_gib"],
        "inference_gib": profiles[primary]["inference_gib"],
        "budget_gib": VRAM_BUDGET_GB,
        "headroom_gib": profiles[primary]["headroom_gib"],
        "fits": profiles[primary]["fits"],
        "all_profiles_fit": all_fit,
        "method": "counted params + measured saved tensors + declared reserve",
        "device": profiles[primary]["measured_device"],
        "measured_device": profiles[primary]["measured_device"],
        "cuda_peak_allocated_gib": hardware["cuda_peak_allocated_gib"],
        "activation_bytes_training": profiles[primary]["activation_bytes_training"],
        "hardware": hardware,
        "storage": storage,
        "evidence": "DERIVED",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    (out.parent / (out.name + ".sha256")).write_text(
        hashlib.sha256(out.read_bytes()).hexdigest() + "\n", encoding="utf-8"
    )
    payload["receipt"] = str(out)
    payload["sha256"] = hashlib.sha256(out.read_bytes()).hexdigest()
    _emit(payload, args.json, "VRAM budget")
    return 0 if all_fit else 1


def cmd_serve(args) -> int:
    """Serve the sidecar over REST, MCP stdio, or MCP SSE (Decision A5)."""
    if args.transport == "rest":
        from henri.holarch.server import serve_rest

        serve_rest(
            host=str(args.host),
            port=int(args.port),
            backend=str(args.backend),
            memory_path=args.path,
        )
        return 0
    from henri.holarch import mcp_server

    transport = "stdio" if args.transport == "mcp-stdio" else "sse"
    return int(
        mcp_server.main(
            ["--transport", transport, "--host", str(args.host), "--port", str(args.port)]
        )
        or 0
    )


def cmd_train(args) -> int:
    from henri.holarch.tasks import build_tasks
    from henri.holarch.train import TrainConfig, run_experiment

    # Decision C2: the profile selects the wave dimension. "auto" keeps the
    # CI profile for ordinary runs, uses an explicit --dim when one is given,
    # and selects the scaled medium profile for a run named *production*.
    profile = str(getattr(args, "profile", "auto")).lower()
    if profile == "small" or args.small:
        cfg = HolarchConfig.small()
    elif profile == "medium":
        cfg = HolarchConfig.medium()
    elif profile == "production":
        cfg = HolarchConfig.production(device="cpu")
    elif int(args.dim) != 256:
        cfg = HolarchConfig(dim=int(args.dim))
    elif "production" in str(args.name).lower():
        cfg = HolarchConfig.medium()
    else:
        cfg = HolarchConfig.small()
    tasks = build_tasks(cfg)
    receipt = run_experiment(
        cfg,
        output_dir=args.out,
        name=args.name,
        steps_per_task=int(args.steps),
        backend=args.backend,
        memory_path=args.path,
        tasks=tasks,
        train_config=TrainConfig(
            steps_per_task=int(args.steps),
            batch_size=int(args.batch),
            ood_weight=float(args.ood_weight),
            ood_margin=float(args.ood_margin),
            relational_bonus_coupling=float(getattr(args, "coupling", 0.5)),
        ),
        calibrate=True,
        reset_memory=bool(args.fresh),
        dagger_rounds=int(args.dagger),
    )
    payload = {
        "name": receipt["name"],
        "parameters": receipt["parameters"],
        "training_gib": receipt["budget"]["training_gib"],
        "fit": bool(receipt["budget"]["fits"]),
        "history": receipt["history"],
        "calibration": {name: round(value["epsilon"], 4) for name, value in receipt["calibration"].items()},
        "metrics": {k: round(v, 4) for k, v in receipt["metrics"].items() if "accuracy" in k or "veto" in k},
        "test": {k: round(v, 4) for k, v in receipt["test"].items() if "accuracy" in k},
        "memory_recall": receipt["memory"],
        "action_success": receipt["action_success"],
        "receipt": receipt.get("receipt_path", ""),
        "sha256": receipt.get("receipt_sha256", ""),
    }
    _emit(payload, args.json, "Holarch training run")
    return 0 if receipt["budget"]["fits"] else 1


def cmd_selftest(args) -> int:
    from henri.holarch.zone_a import LeanSwarm
    from henri.holarch.zone_c import ZoneCMemory

    cfg = HolarchConfig.small()
    checks: Dict[str, bool] = {}
    core = PhotonicCore(cfg)
    generator = torch.Generator().manual_seed(1)
    wave = torch.randn(10, cfg.dim, generator=generator, dtype=torch.complex64)
    wave = wave / wave.norm(dim=-1, keepdim=True)
    state = core(wave, axiom=torch.roll(wave, 2, dims=-1))
    checks["norm_conserved"] = state.norm_drift < 1e-4
    checks["reciprocal_null"] = float(state.delta_reciprocal.mean().detach()) < 1e-3
    cold = core.reachable_shift(wave[0], 0.1, 0.1)
    hot = core.reachable_shift(wave[0], 0.1, cfg.t_max - 1e-3)
    checks["thermal_authority"] = hot > cold

    path = Path(tempfile.gettempdir()) / "henri_holarch_selftest.sqlite3"
    if path.exists():
        path.unlink()
    memory = ZoneCMemory(cfg, backend="sqlite", path=str(path))
    for index in range(8):
        memory.write(wave[index].numpy(), label=index % 3)
    read = memory.query(wave[0].numpy(), k=2)
    checks["memory_recall"] = bool(read.hits) and read.hits[0].engram_id == 1
    prefix, _ = memory.context_tensor(wave[0].numpy(), k=2)
    checks["context_fixed_shape"] = prefix.shape == (cfg.context_tokens, cfg.context_dim)
    memory.close()

    swarm = LeanSwarm(cfg)
    features = torch.randn(32, cfg.d_feat)
    labels = torch.randint(0, 5, (32,))
    swarm.online_update(features, labels, 0, lr=0.2)
    output = swarm(features, 0)
    checks["swarm_routes"] = bool(torch.isfinite(output.logits).all())
    checks["task_blocks_disjoint"] = (
        swarm.task_slice("vision")[1] <= swarm.task_slice("language")[0]
        and swarm.task_slice("language")[1] <= swarm.task_slice("action")[0]
    )
    payload = {"checks": checks, "passed": bool(all(checks.values())), "n": len(checks)}
    _emit(payload, args.json, "Holarch selftest")
    return 0 if payload["passed"] else 1


def cmd_demo(args) -> int:
    from henri.holarch.tasks import build_tasks
    from henri.holarch.train import TrainConfig, run_experiment

    cfg = HolarchConfig.small()
    receipt = run_experiment(
        cfg,
        output_dir=args.out,
        name=args.name,
        steps_per_task=int(args.steps),
        backend=args.backend,
        memory_path=args.path,
        tasks=build_tasks(cfg),
        train_config=TrainConfig(steps_per_task=int(args.steps), batch_size=32),
    )
    payload = {
        "parameters": receipt["parameters"],
        "wall_seconds": round(receipt["wall_seconds"], 2),
        "training_gib": receipt["budget"]["training_gib"],
        "vision_accuracy": receipt["metrics"]["vision_accuracy"],
        "language_accuracy": receipt["metrics"]["language_accuracy"],
        "language_perplexity": receipt["metrics"]["language_perplexity"],
        "action_accuracy": receipt["metrics"]["action_accuracy"],
        "action_success": receipt["action_success"],
        "memory_recall_at_1": receipt["memory"]["recall_at_1"],
        "receipt": receipt.get("receipt_path", ""),
    }
    _emit(payload, args.json, "Holarch demo")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m henri.holarch", description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", help="emit one JSON document")
    sub = parser.add_subparsers(dest="command", required=True)

    physics = sub.add_parser("physics", parents=[common], help="measure the Zone B invariants")
    physics.add_argument("--dim", type=int, default=256)
    physics.add_argument("--small", action="store_true")
    physics.set_defaults(func=cmd_physics)

    memory = sub.add_parser("memory", parents=[common], help="benchmark Zone C")
    memory.add_argument("--n", type=int, default=200)
    memory.add_argument("--dim", type=int, default=256)
    memory.add_argument("--small", action="store_true")
    memory.add_argument("--backend", default="auto", choices=["auto", "sqlite", "timescale"])
    memory.add_argument("--path", default=None)
    memory.add_argument("--fresh", action="store_true")
    memory.set_defaults(func=cmd_memory)

    budget = sub.add_parser("budget", parents=[common], help="estimate VRAM for the 32 GiB target")
    budget.add_argument("--batch", type=int, default=8)
    budget.add_argument("--small", action="store_true")
    budget.add_argument("--path", default=None)
    budget.add_argument("--out", default="receipts/holarch/budget.json")
    budget.set_defaults(func=cmd_budget)

    serve = sub.add_parser("serve", parents=[common], help="serve the sidecar over REST or MCP")
    serve.add_argument(
        "--transport", default="rest", choices=["rest", "mcp-stdio", "mcp-sse"]
    )
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument("--backend", default="sqlite", choices=["auto", "sqlite", "timescale"])
    serve.add_argument("--path", default=None)
    serve.set_defaults(func=cmd_serve)

    train = sub.add_parser("train", parents=[common], help="train and write a receipt")
    train.add_argument("--steps", type=int, default=150)
    train.add_argument("--batch", type=int, default=32)
    train.add_argument("--dim", type=int, default=256)
    train.add_argument(
        "--profile",
        default="auto",
        choices=["auto", "small", "medium", "production"],
        help="wave profile; auto uses medium for a *production* run name",
    )
    train.add_argument("--small", action="store_true")
    train.add_argument("--backend", default="auto", choices=["auto", "sqlite", "timescale"])
    train.add_argument("--path", default=None)
    train.add_argument("--out", default="receipts/holarch")
    train.add_argument("--name", default="holarch")
    train.add_argument("--fresh", action="store_true", help="erase Zone C before training")
    train.add_argument("--dagger", type=int, default=8, help="on-policy action correction rounds")
    train.add_argument("--ood-weight", type=float, default=3.0, help="outlier-exposure weight")
    train.add_argument("--ood-margin", type=float, default=0.0, help="outlier-exposure coherence margin")
    train.add_argument(
        "--coupling",
        type=float,
        default=0.5,
        help="bonus relational gradient coupling (Decision D2 sweeps 0.5 and 0.65)",
    )
    train.set_defaults(func=cmd_train)

    selftest = sub.add_parser("selftest", parents=[common], help="run the fast integration checks")
    selftest.set_defaults(func=cmd_selftest)

    demo = sub.add_parser("demo", parents=[common], help="short end-to-end run")
    demo.add_argument("--steps", type=int, default=40)
    demo.add_argument("--backend", default="sqlite", choices=["auto", "sqlite", "timescale"])
    demo.add_argument("--path", default=None)
    demo.add_argument("--out", default="receipts/holarch")
    demo.add_argument("--name", default="demo")
    demo.set_defaults(func=cmd_demo)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
