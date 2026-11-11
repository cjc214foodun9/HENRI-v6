"""Stage U5 contracts: sandbox, frame, abstention, providers, quarantine."""

from __future__ import annotations

import hashlib
import math
import os

import pytest

from henri.holarch.coprocessor import (
    ARMS,
    STIRRUP_TOOLS,
    AAIIHarness,
    ChowSelectiveGate,
    EpistemicFramePacker,
    EvaluatorQuarantineScope,
    FaultInjectedSimulator,
    KuramotoSemanticCoherence,
    LedgerFinishGate,
    LiveOpenAICompatibleProvider,
    LocalHolarchProvider,
    LoopbackOpenAIMockServer,
    QuarantineViolation,
    ScriptedFixtureProvider,
    estimate_tokens,
)
from henri.holarch.sandbox import (
    BLOCKED_DUNDERS,
    BLOCKED_MODULES,
    BRIEFCASE_MODULES,
    PURE_ALGORITHMIC_MODULES,
    SCIENTIFIC_MODULES,
    PROFILE_ALLOWLISTS,
    CapabilityASTGate,
    SubprocessSandboxExecutor,
    extract_child_result,
)


def test_profile_allowlists_are_profile_scoped():
    assert PROFILE_ALLOWLISTS["pure_algorithmic"] == PURE_ALGORITHMIC_MODULES
    assert "numpy" not in PURE_ALGORITHMIC_MODULES
    assert "json" not in PURE_ALGORITHMIC_MODULES
    assert "numpy" in SCIENTIFIC_MODULES
    assert {"json", "csv", "numpy", "statistics"} <= SCIENTIFIC_MODULES
    assert {"pandas", "pathlib"} <= BRIEFCASE_MODULES
    assert {"os", "sys", "subprocess", "socket", "urllib", "ctypes"} <= BLOCKED_MODULES
    assert {"__subclasses__", "__globals__", "__builtins__"} <= BLOCKED_DUNDERS


def test_ast_gate_allows_the_profile_and_blocks_escapes():
    pure = CapabilityASTGate("pure_algorithmic")
    allowed = pure.check("import math\nresult = {'value': math.isqrt(81)}")
    assert allowed["ok"] is True
    assert allowed["blocked"] == []
    assert "math" in allowed["imports"]

    traps = [
        "import os",
        "from os import environ",
        "result = __import__('os')",
        "import sys\nresult = len(sys.modules)",
        "result = open('/etc/passwd').read()",
        "result = eval('2+2')",
        "result = exec('x = 1')",
        "result = compile('1', '<x>', 'eval')",
        "result = ().__class__.__base__.__subclasses__()",
        "import subprocess",
        "import socket",
        "import urllib.request",
        "import ctypes",
        "import inspect",
        "import gc",
    ]
    for source in traps:
        verdict = pure.check(source)
        assert verdict["ok"] is False, source
        assert verdict["blocked"], source
        assert verdict["reason"] != "ok"

    briefcase = CapabilityASTGate("briefcase_workspace")
    absolute = briefcase.check(
        "from pathlib import Path\nresult = Path('C:/Windows/win.ini').read_text()"
    )
    assert absolute["ok"] is False
    assert "absolute-path-literal" in absolute["blocked"]
    relative = briefcase.check(
        "from pathlib import Path\nresult = Path('notes.txt').read_text()"
    )
    assert relative["ok"] is True
    assert "pathlib" in relative["imports"]


def test_executor_blocks_before_spawn_and_records_sha(tmp_path):
    executor = SubprocessSandboxExecutor("pure_algorithmic", timeout_s=5.0, workdir=tmp_path)
    source = "import os\nresult = {'pid': os.getpid()}"
    record = executor.execute(source)
    assert record.ok is False
    assert record.timed_out is False
    assert record.exit_code == 126
    assert "module:os" in record.blocked_reason
    assert record.sha256 == hashlib.sha256(source.encode("utf-8")).hexdigest()
    assert executor.ledger and executor.ledger[-1] is record


def test_executor_runs_source_and_hashes_artifacts(tmp_path):
    executor = SubprocessSandboxExecutor("briefcase_workspace", timeout_s=5.0, workdir=tmp_path)
    source = (
        "from pathlib import Path\n"
        "Path('out.txt').write_text('36')\n"
        "result = {'value': 6 * 6}"
    )
    record = executor.execute(source)
    assert record.ok is True
    assert record.exit_code == 0
    assert extract_child_result(record.stdout) == {"value": 36}
    assert record.artifact_sha256["out.txt"] == hashlib.sha256(b"36").hexdigest()


def test_executor_kills_a_timeout_and_the_parent_survives(tmp_path):
    executor = SubprocessSandboxExecutor("pure_algorithmic", timeout_s=0.5, workdir=tmp_path)
    record = executor.execute("counter = 0\nwhile True:\n    counter += 1")
    assert record.timed_out is True
    assert record.ok is False
    assert record.duration_ms >= 400.0
    assert os.getpid() > 0
    assert executor.execute("result = {'alive': True}").ok is True


def test_child_environment_drops_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("HENRI_LLM_API_KEY", "secret-value")
    monkeypatch.setenv("OPENAI_API_KEY", "secret-value")
    monkeypatch.setenv("HENRI_TSDB_DSN", "postgres://secret")
    executor = SubprocessSandboxExecutor("pure_algorithmic", timeout_s=5.0, workdir=tmp_path)
    probe = executor.probe_environment(
        ["HENRI_LLM_API_KEY", "OPENAI_API_KEY", "HENRI_TSDB_DSN", "PATH"]
    )
    assert probe["HENRI_LLM_API_KEY"] is False
    assert probe["OPENAI_API_KEY"] is False
    assert probe["HENRI_TSDB_DSN"] is False
    assert probe["PATH"] is True


def test_frame_packer_is_bounded_and_keeps_every_section():
    packer = EpistemicFramePacker(max_tokens=4096)
    huge = " ".join(f"token-{index}-value-{index * 7}" for index in range(5000))
    frame = packer.pack(
        task_spec=huge,
        subgoal=huge,
        chart_hits=[huge[i : i + 400] for i in range(0, 4000, 400)],
        ledger=[{"turn": index, "result": {"value": index}} for index in range(80)],
        repulsor_bans=[huge[i : i + 200] for i in range(0, 2000, 200)],
        deltas=[{"turn": index, "status": "ok"} for index in range(20)],
    )
    assert frame["token_count"] <= 4096
    assert set(frame["sections"]) == {
        "task_spec",
        "subgoal",
        "chart_hits",
        "ledger",
        "repulsor_bans",
        "deltas",
    }
    for label in ("TASK SPEC", "SUB-GOAL", "TOP-K CHARTS", "VERIFIED LEDGER"):
        assert f"[{label}]" in frame["text"]


def test_frame_compression_beats_a_growing_transcript_by_three_times():
    packer = EpistemicFramePacker(max_tokens=4096)
    observation = " ".join(str(index * index) for index in range(1, 201))
    transcript = ""
    baseline_total = 0
    frame_total = 0
    for turn in range(1, 51):
        transcript += f"\nASSISTANT code_exec\nTOOL {observation}\n"
        baseline_total += estimate_tokens("task specification " + transcript)
        frame = packer.pack(
            "task specification",
            f"turn {turn}",
            ["source=task.md line=1: use a verified value"] * 6,
            [{"turn": index, "result": {"value": index}} for index in range(max(1, turn - 8), turn)],
            [{"sha256": "a" * 16, "reason": "failed"}] if turn > 1 else [],
            [{"turn": index, "status": "executed"} for index in range(max(1, turn - 3), turn)],
        )
        assert frame["token_count"] <= 4096
        frame_total += frame["token_count"]
    assert baseline_total / max(frame_total, 1) >= 3.0


def test_chow_gate_uses_strict_thresholds_and_fails_closed():
    gate = ChowSelectiveGate()
    assert gate.decide(0.9, 0.50, "omni_strict")["emit"] is False
    assert gate.decide(0.9, 0.5001, "omni_strict")["emit"] is True
    assert gate.decide(0.9, 0.3333, "aaii_composite")["emit"] is False
    assert gate.decide(0.9, 0.34, "aaii_composite")["emit"] is True
    assert gate.decide(0.9, 0.99, "unknown")["emit"] is False
    assert gate.decide(0.9, float("nan"), "omni_strict")["emit"] is False


def test_kuramoto_order_parameter_is_deterministic_and_negation_sensitive():
    coherence = KuramotoSemanticCoherence()
    same = ["the aggregate value is 1785"] * 3
    assert coherence.order_parameter(same) == pytest.approx(1.0)
    opposite = ["the aggregate value is 1785", "the aggregate value is not 1785"]
    assert coherence.order_parameter(opposite) == pytest.approx(0.0, abs=1e-9)
    assert coherence.order_parameter(same) == coherence.order_parameter(list(same))
    assert coherence.order_parameter([]) == 0.0


def test_finish_gate_refuses_an_unverified_quantity():
    gate = LedgerFinishGate([{"value": 36}])
    assert gate.check("the value is 36", {"value": 36})["ok"] is True
    refused = gate.check("the value is 37", {"value": 37})
    assert refused["ok"] is False
    assert refused["missing"] == ["37"]


def test_scripted_and_local_providers_return_structured_calls():
    scripted = ScriptedFixtureProvider(
        [{"tool": "finish", "arguments": {"payload": {"value": 1}}}]
    )
    first = scripted.propose({"text": "x"}, STIRRUP_TOOLS)
    second = scripted.propose({"text": "x"}, STIRRUP_TOOLS)
    assert first["tool"] == "finish"
    assert len(first["raw_sha256"]) == 64
    assert second["tool"] == "finish"
    local = LocalHolarchProvider()
    proposal = local.propose(
        {"text": '[VERIFIED LEDGER]\n[{"result": {"value": 36}}]'}, STIRRUP_TOOLS
    )
    assert proposal["tool"] == "finish"
    assert proposal["arguments"]["payload"]["value"] == 36


def test_loopback_provider_roundtrips_tool_calls_and_retries():
    simulator = FaultInjectedSimulator(seed=20261111, default_turns=12)
    server = LoopbackOpenAIMockServer(simulator, seed=20261111).start()
    try:
        provider = LiveOpenAICompatibleProvider(
            server.base_url,
            "offline-mock",
            "henri-u5-loopback-sim",
            timeout_s=5.0,
            max_retries=4,
            backoff_s=0.005,
            user=ARMS[2] + ":0",
        )
        for _ in range(40):
            proposal = provider.propose(
                {"text": "Compute the sum of the squares of one through seventeen."},
                STIRRUP_TOOLS,
            )
            assert proposal["tool"] in {
                "code_exec",
                "browse",
                "finish",
                "abandon",
            }
            assert len(proposal["raw_sha256"]) == 64
            assert proposal["tokens"]["prompt_tokens"] > 0
        assert provider.calls == 40
        assert server.completions == 40
        assert server.requests >= server.completions
        assert (server.fault_429 + server.fault_503) >= 1
        assert provider.retries >= 1
    finally:
        server.stop()


def test_quarantine_scope_raises_on_a_planted_gold_label(tmp_path):
    sandbox = tmp_path / "sandbox"
    zone_c = tmp_path / "zone_c"
    scratch = tmp_path / "scratch"
    for path in (sandbox, zone_c, scratch):
        path.mkdir()
    scope = EvaluatorQuarantineScope(roots=[sandbox, zone_c, scratch])
    scope.register_gold("GOLD_ANSWER_PLANT_42")
    clean = scope.inspect_text("no secret here", "frame:clean")
    assert clean["leaked"] is False
    scope.assert_absent([sandbox, zone_c, scratch])
    (sandbox / "leak.txt").write_text("GOLD_ANSWER_PLANT_42", encoding="utf-8")
    with pytest.raises(QuarantineViolation):
        scope.assert_absent([sandbox])
    assert scope.leaked is True


def test_quarantine_keeps_gold_out_of_the_frame_and_sandbox(tmp_path):
    gold = "GOLD_ANSWER_SUM_1785"
    scope = EvaluatorQuarantineScope()
    scope.register_gold(gold)
    packer = EpistemicFramePacker(max_tokens=4096)
    frame = packer.pack(
        "Compute a sum.", "turn 1", ["source=task.md"], [], [], []
    )
    assert scope.inspect_text(frame["text"], "frame")["leaked"] is False
    executor = SubprocessSandboxExecutor("pure_algorithmic", timeout_s=5.0, workdir=tmp_path)
    record = executor.execute(
        "print(' '.join(str(i * i) for i in range(1, 201)))\n"
        "result = {'value': sum(i * i for i in range(1, 18))}"
    )
    assert record.ok is True
    assert gold not in record.stdout
    assert scope.inspect_text(record.stdout, "stdout")["leaked"] is False


def test_mini_harness_measures_all_three_arms():
    harness = AAIIHarness(
        episodes=1,
        turns=12,
        stress_turns=12,
        timeout_s=5.0,
    )
    result = harness.run()
    assert [row["arm"] for row in result["arms"]] == list(ARMS)
    assert result["turns_per_arm"] == 12
    assert result["repeated_error_loops"][ARMS[2]] == 0
    assert result["repeated_error_loops"][ARMS[0]] > 0
    assert result["ungrounded_deliverable_rate"][ARMS[2]] == 0.0
    assert result["omni_index"][ARMS[2]] > result["omni_index"][ARMS[0]]
    assert result["stress"]["max_frame_tokens"] <= 4096
