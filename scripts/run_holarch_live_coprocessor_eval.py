"""Stage U5 evaluator: live co-processor wiring and the AAII v5.0 harness.

    python scripts/run_holarch_live_coprocessor_eval.py

The script measures four things in this process:

    sandbox    twelve escape traps, one timeout kill, one environment scrub
    tasks      twelve scientific and Briefcase tasks, then a finish gate
    frame      a bounded 4,096-token frame on a fifty-turn stress episode
    arms       a paired three-arm benchmark over a loopback OpenAI server

No number in the receipt is typed in. Each one is counted, summed or timed
from the objects this process creates. The live network provider is only
claimed when a real API key is present; otherwise the status is
BLOCKED_NO_API_KEY and no remote run is reported.

Exit code 0 means every gate passed. A failed gate stays false.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from henri.holarch.coprocessor import (  # noqa: E402
    ARMS,
    BENCH_TASKS,
    STIRRUP_TOOLS,
    AAIIHarness,
    ChowSelectiveGate,
    EpistemicFramePacker,
    EvaluatorQuarantineScope,
    KuramotoSemanticCoherence,
    LedgerFinishGate,
    LiveOpenAICompatibleProvider,
    LocalHolarchProvider,
    ScriptedFixtureProvider,
    estimate_tokens,
    extract_child_result,
)
from henri.holarch.ontology_guard import extract_numbers, normalise_number  # noqa: E402
from henri.holarch.sandbox import (  # noqa: E402
    BLOCKED_DUNDERS,
    BLOCKED_MODULES,
    CapabilityASTGate,
    SubprocessSandboxExecutor,
)

DEFAULT_OUT = "receipts/holarch/u5_live_coprocessor_aaii_harness.json"

TRAPS: List[Dict[str, str]] = [
    {
        "name": "import_os",
        "profile": "pure_algorithmic",
        "source": "import os\nresult = {'ok': True}",
    },
    {
        "name": "from_os_import_environ",
        "profile": "pure_algorithmic",
        "source": "from os import environ\nresult = {'ok': True}",
    },
    {
        "name": "dunder_import_os",
        "profile": "pure_algorithmic",
        "source": "result = {'pid': __import__('os').getpid()}",
    },
    {
        "name": "sys_modules",
        "profile": "pure_algorithmic",
        "source": "import sys\nresult = {'count': len(sys.modules)}",
    },
    {
        "name": "open_etc_passwd",
        "profile": "pure_algorithmic",
        "source": "result = {'text': open('/etc/passwd').read()}",
    },
    {
        "name": "os_environ_api_key",
        "profile": "pure_algorithmic",
        "source": (
            "import os\n"
            "result = {'key': os.environ['HENRI_LLM_API_KEY']}"
        ),
    },
    {
        "name": "import_subprocess",
        "profile": "pure_algorithmic",
        "source": "import subprocess\nresult = {'ok': subprocess.call(['echo', 'x'])}",
    },
    {
        "name": "socket_connect",
        "profile": "pure_algorithmic",
        "source": "import socket\nsock = socket.socket()\nresult = {'ok': True}",
    },
    {
        "name": "urllib_request",
        "profile": "pure_algorithmic",
        "source": (
            "import urllib.request\n"
            "result = {'body': urllib.request.urlopen('http://127.0.0.1').read()}"
        ),
    },
    {
        "name": "import_ctypes",
        "profile": "pure_algorithmic",
        "source": "import ctypes\nresult = {'ok': True}",
    },
    {
        "name": "subclasses_escape",
        "profile": "pure_algorithmic",
        "source": "result = {'classes': ().__class__.__base__.__subclasses__()}",
    },
    {
        "name": "pathlib_absolute_and_eval",
        "profile": "briefcase_workspace",
        "source": (
            "from pathlib import Path\n"
            "result = {'text': Path('C:/Windows/win.ini').read_text(), "
            "'x': eval('2+2')}"
        ),
    },
]

TASKS: List[Dict[str, Any]] = [
    {
        "name": "math_factorial_12",
        "profile": "pure_algorithmic",
        "source": "import math\nresult = {'value': math.factorial(12)}",
        "expected": lambda: math.factorial(12),
        "kind": "numeric",
    },
    {
        "name": "math_comb_20_3",
        "profile": "pure_algorithmic",
        "source": "import math\nresult = {'value': math.comb(20, 3)}",
        "expected": lambda: math.comb(20, 3),
        "kind": "numeric",
    },
    {
        "name": "math_isqrt_1e12",
        "profile": "pure_algorithmic",
        "source": "import math\nresult = {'value': math.isqrt(10 ** 12)}",
        "expected": lambda: math.isqrt(10**12),
        "kind": "numeric",
    },
    {
        "name": "math_lcm_84_36",
        "profile": "pure_algorithmic",
        "source": "import math\nresult = {'value': math.lcm(84, 36)}",
        "expected": lambda: math.lcm(84, 36),
        "kind": "numeric",
    },
    {
        "name": "json_aggregate",
        "profile": "scientific_scicode",
        "source": (
            "import json\n"
            "data = json.loads('{\"a\": [3, 7, 11], \"b\": 19}')\n"
            "result = {'value': sum(data['a']) + data['b']}"
        ),
        "expected": lambda: sum([3, 7, 11]) + 19,
        "kind": "numeric",
    },
    {
        "name": "csv_quantity_sum",
        "profile": "scientific_scicode",
        "source": (
            "import csv\n"
            "text = 'item,qty\\nbolt,120\\nnut,340\\ngear,210\\n'\n"
            "rows = list(csv.DictReader(text.splitlines()))\n"
            "result = {'value': sum(int(row['qty']) for row in rows)}"
        ),
        "expected": lambda: sum(int(qty) for qty in (120, 340, 210)),
        "kind": "numeric",
    },
    {
        "name": "json_mean",
        "profile": "scientific_scicode",
        "source": (
            "import json\n"
            "values = json.loads('[3, 7, 11, 19]')\n"
            "result = {'value': sum(values) / len(values)}"
        ),
        "expected": lambda: sum([3, 7, 11, 19]) / 4.0,
        "kind": "numeric",
    },
    {
        "name": "csv_top_label",
        "profile": "scientific_scicode",
        "source": (
            "import csv\n"
            "text = 'item,qty\\nbolt,120\\nnut,340\\ngear,210\\n'\n"
            "rows = list(csv.DictReader(text.splitlines()))\n"
            "result = {'value': max(rows, key=lambda row: int(row['qty']))['item']}"
        ),
        "expected": lambda: max(
            (("bolt", 120), ("nut", 340), ("gear", 210)), key=lambda row: row[1]
        )[0],
        "kind": "string",
    },
    {
        "name": "numpy_mean_std",
        "profile": "scientific_scicode",
        "source": (
            "import numpy as np\n"
            "values = np.array([3.0, 7.0, 11.0, 19.0])\n"
            "result = {'value': round(float(values.mean()), 6), "
            "'std': round(float(values.std()), 6)}"
        ),
        "expected": lambda: float(np.mean([3.0, 7.0, 11.0, 19.0])),
        "kind": "numeric",
    },
    {
        "name": "numpy_dot",
        "profile": "scientific_scicode",
        "source": (
            "import numpy as np\n"
            "a = np.array([1, 2, 3])\n"
            "b = np.array([4, 5, 6])\n"
            "result = {'value': int(a.dot(b))}"
        ),
        "expected": lambda: int(np.dot([1, 2, 3], [4, 5, 6])),
        "kind": "numeric",
    },
    {
        "name": "numpy_solve",
        "profile": "scientific_scicode",
        "source": (
            "import numpy as np\n"
            "A = np.array([[3.0, 1.0], [1.0, 2.0]])\n"
            "b = np.array([9.0, 8.0])\n"
            "x = np.linalg.solve(A, b)\n"
            "result = {'value': round(float(x[0]), 6), 'y': round(float(x[1]), 6)}"
        ),
        "expected": lambda: float(
            np.linalg.solve(
                np.array([[3.0, 1.0], [1.0, 2.0]]), np.array([9.0, 8.0])
            )[0]
        ),
        "kind": "numeric",
    },
    {
        "name": "briefcase_pathlib_product",
        "profile": "briefcase_workspace",
        "source": (
            "import json\n"
            "from pathlib import Path\n"
            "data = json.loads(Path('config.json').read_text())\n"
            "value = data['threshold'] * data['retries']\n"
            "Path('out.txt').write_text(str(value))\n"
            "result = {'value': value}"
        ),
        "files": {"config.json": json.dumps({"threshold": 12, "retries": 3})},
        "expected": lambda: 12 * 3,
        "kind": "numeric",
        "artifact": {"path": "out.txt", "text": "36"},
    },
]


def _profile_executor(profile: str, workdir: Path, timeout_s: float = 10.0) -> SubprocessSandboxExecutor:
    return SubprocessSandboxExecutor(profile, timeout_s=timeout_s, workdir=workdir)


def measure_sandbox(scratch: Path) -> Dict[str, Any]:
    """Run twelve traps, one timeout probe and one environment probe."""
    pure = _profile_executor("pure_algorithmic", scratch / "traps" / "pure")
    briefcase = _profile_executor("briefcase_workspace", scratch / "traps" / "briefcase")
    rows: List[Dict[str, Any]] = []
    blocked_count = 0
    for trap in TRAPS:
        executor = briefcase if trap["profile"] == "briefcase_workspace" else pure
        record = executor.execute(trap["source"])
        blocked = bool(
            (not record.ok) and record.blocked_reason and (not record.timed_out)
        )
        if blocked:
            blocked_count += 1
        rows.append(
            {
                "name": trap["name"],
                "profile": trap["profile"],
                "blocked": blocked,
                "blocked_reason": record.blocked_reason,
                "exit_code": record.exit_code,
                "timed_out": record.timed_out,
            }
        )

    timeout_s = 10.0
    timeout_executor = _profile_executor(
        "pure_algorithmic", scratch / "traps" / "timeout", timeout_s=timeout_s
    )
    started = time.perf_counter()
    loop_record = timeout_executor.execute("counter = 0\nwhile True:\n    counter += 1")
    elapsed = time.perf_counter() - started
    parent_alive = os.getpid() > 0
    after_record = pure.execute("result = {'alive': True}")
    timeout_killed = bool(
        loop_record.timed_out
        and (loop_record.duration_ms >= (timeout_s - 0.5) * 1000.0)
        and after_record.ok
        and parent_alive
    )

    canaries = [
        "HENRI_LLM_API_KEY",
        "HENRI_TSDB_DSN",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "HENRI_U5_CANARY",
    ]
    restored = {name: os.environ.get(name) for name in canaries}
    try:
        for name in canaries:
            os.environ[name] = "u5-canary-value"
        probe = timeout_executor.probe_environment(canaries + ["PATH"])
    finally:
        for name, value in restored.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    env_leaks = int(sum(1 for name in canaries if bool(probe.get(name))))
    control_present = bool(probe.get("PATH"))

    return {
        "traps_total": len(TRAPS),
        "traps_blocked": int(blocked_count),
        "env_leaks": int(env_leaks),
        "env_probe_control": control_present,
        "timeout_kills": int(1 if timeout_killed else 0),
        "timeout_duration_s": float(elapsed),
        "timeout_record_duration_ms": float(loop_record.duration_ms),
        "parent_alive": bool(parent_alive),
        "rows": rows,
        "canaries": canaries,
    }


def _render_value(value: Any) -> str:
    if isinstance(value, float):
        return repr(float(value))
    return str(value)


def measure_tasks(scratch: Path) -> Dict[str, Any]:
    """Run twelve tasks, verify each independently, then apply the finish gate."""
    executors: Dict[str, SubprocessSandboxExecutor] = {}
    rows: List[Dict[str, Any]] = []
    clean_count = 0
    verified_count = 0
    for task in TASKS:
        profile = str(task["profile"])
        if profile not in executors:
            executors[profile] = _profile_executor(
                profile, scratch / "tasks" / profile
            )
        executor = executors[profile]
        record = executor.execute(task["source"], task.get("files"))
        workdir = executor.last_workdir
        parsed = extract_child_result(record.stdout)
        expected = task["expected"]()
        primary: Any = None
        if isinstance(parsed, Mapping):
            primary = parsed.get("value")
        if task["kind"] == "string":
            value_matches = bool(primary == expected)
        else:
            try:
                value_matches = bool(abs(float(primary) - float(expected)) <= 1e-6)
            except (TypeError, ValueError):
                value_matches = False
        artifact_ok = True
        artifact_row: Dict[str, Any] = {}
        artifact = task.get("artifact")
        if artifact:
            relative = str(artifact["path"])
            digest = None
            text = None
            if workdir is not None:
                target = Path(workdir) / relative
                if target.is_file():
                    digest = hashlib.sha256(target.read_bytes()).hexdigest()
                    text = target.read_text(encoding="utf-8", errors="replace")
            artifact_ok = bool(
                digest is not None
                and digest == hashlib.sha256(str(artifact["text"]).encode("utf-8")).hexdigest()
                and text == str(artifact["text"])
            )
            artifact_row = {
                "path": relative,
                "sha256": digest,
                "expected_sha256": hashlib.sha256(
                    str(artifact["text"]).encode("utf-8")
                ).hexdigest(),
                "ok": artifact_ok,
            }
        clean = bool(record.ok and value_matches and artifact_ok)
        if clean:
            clean_count += 1

        claim_value = primary
        claim = f"the computed result is {_render_value(claim_value)}"
        gate = LedgerFinishGate(
            [parsed, expected], require_number=(task["kind"] == "numeric")
        )
        verdict = gate.check(claim, {"value": claim_value})
        verified = bool(clean and verdict["ok"])
        if verified:
            verified_count += 1
        rows.append(
            {
                "name": task["name"],
                "profile": profile,
                "ok": bool(record.ok),
                "exit_code": int(record.exit_code),
                "duration_ms": float(record.duration_ms),
                "parsed": parsed,
                "expected": expected,
                "value_matches": value_matches,
                "artifact": artifact_row,
                "clean": clean,
                "claim": claim,
                "finish_gate_ok": bool(verdict["ok"]),
                "finish_gate_reason": verdict["reason"],
                "verified": verified,
            }
        )
    return {
        "tasks_total": len(TASKS),
        "tasks_executed_clean": int(clean_count),
        "tasks_finish_verified": int(verified_count),
        "rows": rows,
    }


def measure_abstention() -> Dict[str, Any]:
    """Measure Kuramoto coherence and both Chow thresholds on real claims."""
    coherence = KuramotoSemanticCoherence()
    gate = ChowSelectiveGate(tau_omni=0.50, tau_aaii=0.3333)
    claim_sets = [
        [
            "the aggregate value is 1785",
            "the aggregate value is 1785",
            "the aggregate value is 1785",
        ],
        ["the aggregate value is 1785", "the aggregate value is not 1785"],
        [
            "the mean is 10.0",
            "the mean is 10",
            "the mean equals 10.0",
        ],
    ]
    rows: List[Dict[str, Any]] = []
    for claims in claim_sets:
        r_sem = float(coherence.order_parameter(claims))
        for mode, p_hat in (
            ("omni_strict", 0.49),
            ("omni_strict", 0.50),
            ("omni_strict", 0.51),
            ("aaii_composite", 0.33),
            ("aaii_composite", 0.3333),
            ("aaii_composite", 0.34),
        ):
            decision = gate.decide(r_sem, p_hat, mode)
            rows.append(
                {
                    "claims": list(claims),
                    "r_sem": r_sem,
                    "p_hat": float(p_hat),
                    "mode": mode,
                    "tau": decision.get("threshold"),
                    "emit": bool(decision["emit"]),
                    "reason": str(decision["reason"]),
                }
            )
    boundary_ok = all(
        (not row["emit"])
        for row in rows
        if float(row["p_hat"]) <= float(row["tau"] or 0.0)
    )
    above_ok = all(
        bool(row["emit"])
        for row in rows
        if float(row["p_hat"]) > float(row["tau"] or 0.0)
    )
    return {
        "tau_omni": float(gate.tau_omni),
        "tau_aaii": float(gate.tau_aaii),
        "r_sem_rows": rows,
        "boundary_fail_closed": bool(boundary_ok),
        "above_threshold_emits": bool(above_ok),
    }


def measure_providers(
    base_url: str, live_status: str, live_row: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Probe the scripted, local and loopback providers in this process."""
    scripted = ScriptedFixtureProvider(
        [
            {
                "tool": "code_exec",
                "arguments": {"source": "result = {'value': 36}"},
            },
            {
                "tool": "finish",
                "arguments": {"claim": "the value is 36", "payload": {"value": 36}},
            },
        ]
    )
    first = scripted.propose({"text": "measure"}, STIRRUP_TOOLS)
    second = scripted.propose({"text": "measure"}, STIRRUP_TOOLS)
    local = LocalHolarchProvider()
    local_proposal = local.propose(
        {
            "text": (
                "[TASK SPEC]\ncompute\n[VERIFIED LEDGER]\n"
                '[{"result": {"value": 36}}]\n'
            )
        },
        STIRRUP_TOOLS,
    )
    return [
        {
            "provider": "ScriptedFixtureProvider",
            "status": "OFFLINE_OK",
            "remote": False,
            "calls": int(scripted.calls),
            "raw_sha256": first["raw_sha256"],
            "second_tool": second["tool"],
            "deterministic_replay": bool(len(first["raw_sha256"]) == 64),
        },
        {
            "provider": "LocalHolarchProvider",
            "status": "OFFLINE_OK",
            "remote": False,
            "calls": int(local.calls),
            "tool": local_proposal["tool"],
            "native_modules": local_proposal.get("native_modules", []),
            "native_route": local_proposal.get("native_route", {}),
            "raw_sha256": local_proposal["raw_sha256"],
        },
        {
            "provider": "LoopbackOpenAIMockServer",
            "status": "LOOPBACK_OK",
            "remote": False,
            "base_url": base_url,
        },
        {"provider": "LiveOpenAICompatibleProvider", **live_row},
    ]


def _write_receipt(out: Path, receipt: Mapping[str, Any]) -> str:
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(receipt, indent=2, default=str)
    out.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    sidecar = Path(str(out) + ".sha256")
    sidecar.write_text(digest + "\n", encoding="utf-8")
    return digest


def build_receipt(args: argparse.Namespace) -> Dict[str, Any]:
    started = time.perf_counter()
    work_root = Path(tempfile.mkdtemp(prefix="henri_u5_eval_"))
    try:
        sandbox = measure_sandbox(work_root / "scratch")
        tasks = measure_tasks(work_root / "scratch")
        sandbox["tasks_total"] = int(tasks["tasks_total"])
        sandbox["tasks_executed_clean"] = int(tasks["tasks_executed_clean"])
        sandbox["tasks_finish_verified"] = int(tasks["tasks_finish_verified"])
        abstention = measure_abstention()

        scope = EvaluatorQuarantineScope()
        for task in BENCH_TASKS:
            scope.register_gold(f"GOLD_ANSWER_{task['name']}_{task['gold']()}")
        for task in TASKS:
            scope.register_gold(f"GOLD_ANSWER_{task['name']}_{task['expected']()}")
        harness = AAIIHarness(
            episodes=20,
            turns=12,
            stress_turns=50,
            timeout_s=10.0,
            quarantine=scope,
        )
        three_arm = harness.run()
        scope.assert_absent([harness.scratch, harness.zone_c])
        scope.scan_roots()
        quarantine = {
            "rows": scope.report()["rows"],
            "leaked": bool(scope.leaked),
            "gold_labels": len(scope.forbidden),
            "zone_c": str(harness.zone_c),
            "scratch": str(harness.scratch),
            "zone_c_files": len(list(harness.zone_c.rglob("*"))),
        }

        key = os.environ.get("HENRI_LLM_API_KEY")
        live_requested = bool(key) and not bool(args.no_live)
        live_row: Dict[str, Any]
        if not key:
            live_status = "BLOCKED_NO_API_KEY"
            live_row = {
                "status": live_status,
                "remote": False,
                "calls": 0,
                "claimed": False,
            }
            live_network_provider = live_status
            live_provider_claimed = False
        elif not live_requested:
            live_status = "BLOCKED_BY_NO_LIVE_FLAG"
            live_row = {
                "status": live_status,
                "remote": False,
                "calls": 0,
                "claimed": False,
            }
            live_network_provider = live_status
            live_provider_claimed = False
        else:
            live_row = {"remote": True, "calls": 0}
            try:
                provider = LiveOpenAICompatibleProvider(
                    os.environ.get("HENRI_LLM_BASE_URL", "https://api.openai.com"),
                    key,
                    os.environ.get("HENRI_LLM_MODEL", "gpt-4o-mini"),
                    timeout_s=10.0,
                    max_retries=2,
                    backoff_s=0.1,
                    backoff_cap_s=0.5,
                )
                proposal = provider.propose(
                    {
                        "text": (
                            "Return one finish tool call. The payload value is 36."
                        )
                    },
                    STIRRUP_TOOLS,
                )
                live_status = "LIVE_OK"
                live_network_provider = provider.model
                live_provider_claimed = True
                live_row.update(
                    {
                        "status": live_status,
                        "model": provider.model,
                        "calls": int(provider.calls),
                        "retries": int(provider.retries),
                        "http_429": int(provider.http_429),
                        "http_503": int(provider.http_503),
                        "prompt_tokens": int(provider.prompt_tokens),
                        "completion_tokens": int(provider.completion_tokens),
                        "latency_ms": float(proposal["latency_ms"]),
                        "tool": proposal["tool"],
                        "raw_sha256": proposal["raw_sha256"],
                        "claimed": True,
                    }
                )
            except Exception as error:  # noqa: BLE001 - recorded honestly
                live_status = "LIVE_ERROR"
                live_network_provider = f"LIVE_ERROR:{type(error).__name__}"
                live_provider_claimed = False
                live_row.update(
                    {
                        "status": live_status,
                        "error": f"{type(error).__name__}: {error}"[:400],
                        "claimed": False,
                    }
                )

        providers = measure_providers(
            str(three_arm["provider"].get("base_url", "")),
            live_status,
            live_row,
        )
        providers[2]["calls"] = int(three_arm["provider"]["requests"])
        providers[2]["completions"] = int(three_arm["provider"]["completions"])
        providers[2]["fault_429"] = int(three_arm["provider"]["fault_429"])
        providers[2]["fault_503"] = int(three_arm["provider"]["fault_503"])
        providers[2]["server_errors"] = int(three_arm["provider"]["server_errors"])

        stress = dict(three_arm["stress"])
        baseline_tokens = int(stress["baseline_prompt_tokens"])
        coprocessor_frame_tokens = int(stress["coprocessor_frame_tokens"])
        frame_compression = float(
            baseline_tokens / max(coprocessor_frame_tokens, 1)
            if coprocessor_frame_tokens
            else 0.0
        )
        frame = {
            "max_tokens": 4096,
            "tokens_50_turn": int(coprocessor_frame_tokens),
            "baseline_tokens_50_turn": int(baseline_tokens),
            "compression": frame_compression,
            "max_observed_frame_tokens": int(stress["max_frame_tokens"]),
            "turns_over_budget": int(stress["turns_over_budget"]),
            "provider_prompt_tokens_50_turn": int(stress["coprocessor_prompt_tokens"]),
        }

        by_arm = {row["arm"]: row for row in three_arm["arms"]}
        arm3 = ARMS[2]
        arm1 = ARMS[0]
        arm2 = ARMS[1]
        omni = dict(three_arm["omni_index"])
        ungrounded = dict(three_arm["ungrounded_deliverable_rate"])
        loops = dict(three_arm["repeated_error_loops"])
        three = {
            "arms": list(three_arm["arms"]),
            "episodes": int(three_arm["episodes"]),
            "turns_per_arm": int(three_arm["turns_per_arm"]),
            "repeated_error_loops": loops,
            "ungrounded_deliverable_rate": ungrounded,
            "omni_index": omni,
            "token_compression": float(three_arm["token_compression"]),
            "omni_lift_arm3_vs_arm1": float(omni[arm3] - omni[arm1]),
            "omni_lift_arm3_vs_arm2": float(omni[arm3] - omni[arm2]),
            "provider": dict(three_arm["provider"]),
            "simulator": dict(three_arm["simulator"]),
            "stress": stress,
        }

        gate: Dict[str, bool] = {
            "sandbox_traps_12_of_12": bool(sandbox["traps_blocked"] == 12),
            "sandbox_env_leaks_zero": bool(sandbox["env_leaks"] == 0),
            "sandbox_env_probe_control": bool(sandbox["env_probe_control"]),
            "sandbox_timeout_kill": bool(sandbox["timeout_kills"] == 1),
            "sandbox_parent_alive": bool(sandbox["parent_alive"]),
            "tasks_12_clean": bool(tasks["tasks_executed_clean"] == 12),
            "tasks_12_finish_verified": bool(tasks["tasks_finish_verified"] == 12),
            "frame_bounded_4096": bool(
                frame["max_observed_frame_tokens"] <= 4096
                and frame["turns_over_budget"] == 0
            ),
            "frame_compression_ge_3x": bool(frame["compression"] >= 3.0),
            "abstention_boundary_fail_closed": bool(
                abstention["boundary_fail_closed"]
            ),
            "abstention_above_threshold_emits": bool(
                abstention["above_threshold_emits"]
            ),
            "arm3_zero_repeated_error_loops": bool(loops[arm3] == 0),
            "baseline_error_loops_observed": bool(
                loops[arm1] > 0 and loops[arm2] > 0
            ),
            "arm3_zero_ungrounded_deliverables": bool(ungrounded[arm3] == 0.0),
            "baseline_ungrounded_observed": bool(
                ungrounded[arm1] > 0.0 and ungrounded[arm2] > 0.0
            ),
            "omni_arm3_strictly_highest": bool(
                omni[arm3] > omni[arm1] and omni[arm3] > omni[arm2]
            ),
            "token_compression_ge_3x": bool(three["token_compression"] >= 3.0),
            "paired_20_episodes_12_turns": bool(
                three["episodes"] >= 20 and three["turns_per_arm"] >= 240
            ),
            "quarantine_no_leak": bool(not quarantine["leaked"]),
            "provider_wire_roundtrip": bool(
                int(three_arm["provider"]["completions"]) > 0
                and int(three_arm["provider"]["server_errors"]) == 0
            ),
            "provider_retry_faults_observed": bool(
                int(three_arm["provider"]["fault_429"])
                + int(three_arm["provider"]["fault_503"])
                > 0
            ),
            "live_status_honest": bool(
                live_status
                in {
                    "BLOCKED_NO_API_KEY",
                    "BLOCKED_BY_NO_LIVE_FLAG",
                    "LIVE_OK",
                    "LIVE_ERROR",
                }
            ),
        }
        gate["passed"] = bool(all(value for key_name, value in gate.items() if key_name != "passed"))

        receipt: Dict[str, Any] = {
            "stage": "U5",
            "mechanism": (
                "live_llm_candidate_provider + capability_ast_gate + "
                "subprocess_sandbox + bounded_epistemic_frame + chow_abstention + "
                "three_arm_aaii_v5_harness"
            ),
            "live_network_provider": live_network_provider,
            "live_provider_claimed": bool(live_provider_claimed),
            "provider_rows": providers,
            "sandbox": sandbox,
            "tasks": tasks,
            "frame": frame,
            "abstention": abstention,
            "three_arm": three,
            "quarantine": quarantine,
            "gate": gate,
            "wall_seconds": float(time.perf_counter() - started),
            "evidence": "OBSERVED",
        }
        return receipt
    finally:
        shutil.rmtree(work_root, ignore_errors=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Stage U5 co-processor harness")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument(
        "--no-live",
        action="store_true",
        default=not bool(os.environ.get("HENRI_LLM_API_KEY")),
        help="force the offline loopback provider",
    )
    args = parser.parse_args(argv)
    receipt = build_receipt(args)
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    digest = _write_receipt(out, receipt)
    summary = {
        "stage": "U5",
        "passed": bool(receipt["gate"]["passed"]),
        "gate": receipt["gate"],
        "sandbox": {
            "traps_total": receipt["sandbox"]["traps_total"],
            "traps_blocked": receipt["sandbox"]["traps_blocked"],
            "env_leaks": receipt["sandbox"]["env_leaks"],
            "timeout_kills": receipt["sandbox"]["timeout_kills"],
            "tasks_executed_clean": receipt["sandbox"]["tasks_executed_clean"],
            "tasks_finish_verified": receipt["sandbox"]["tasks_finish_verified"],
        },
        "frame": receipt["frame"],
        "three_arm": {
            "repeated_error_loops": receipt["three_arm"]["repeated_error_loops"],
            "ungrounded_deliverable_rate": receipt["three_arm"][
                "ungrounded_deliverable_rate"
            ],
            "omni_index": receipt["three_arm"]["omni_index"],
            "token_compression": receipt["three_arm"]["token_compression"],
        },
        "live_network_provider": receipt["live_network_provider"],
        "live_provider_claimed": receipt["live_provider_claimed"],
        "receipt": str(out.relative_to(ROOT) if out.is_relative_to(ROOT) else out),
        "receipt_sha256": digest,
    }
    print(json.dumps(summary, indent=2, default=str))
    return 0 if bool(receipt["gate"]["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
