"""Cycle T contracts: centered charts, induction, co-processor, doc hygiene."""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import numpy as np

from henri.holarch.config import HolarchConfig
from henri.holarch.coprocessor import HolarchCoProcessor, build_scenarios
from henri.holarch.induction import (
    InductionSpec,
    WaveProgramInducer,
    contradictory_specs,
    induction_tasks,
)
from henri.holarch.zone_c import ZoneCMemory

ROOT = Path(__file__).resolve().parents[1]


def test_t1_docs_sections_are_monotonic_and_unique():
    text = (ROOT / "docs" / "HOLARCH.md").read_text(encoding="utf-8")
    numbers = [int(value) for value in re.findall(r"^## (\d+)\.", text, flags=re.M)]
    assert numbers == sorted(numbers)
    assert len(numbers) == len(set(numbers))
    assert numbers[0] == 1
    assert numbers[-1] >= 34


def test_t2_centered_route_matches_flat_on_a_narrow_cone(tmp_path):
    cfg = dataclasses.replace(HolarchConfig(), cache_capacity=512)
    memory = ZoneCMemory(cfg, backend="sqlite", path=str(tmp_path / "centered.sqlite3"))
    generator = np.random.default_rng(5)
    dim = int(memory.dim)
    waves = []
    for domain in range(6):
        base = generator.standard_normal(dim) + 1j * generator.standard_normal(dim)
        base = base / np.linalg.norm(base)
        for _ in range(120):
            detail = generator.standard_normal(dim) + 1j * generator.standard_normal(dim)
            detail = detail / np.linalg.norm(detail)
            wave = (0.985 * base + 0.015 * detail).astype(np.complex64)
            wave = wave / np.linalg.norm(wave)
            record = memory.write(wave, label=domain, tier=2, namespace="cone")
            waves.append((int(record.engram_id), wave))
    stats = memory.build_centered_index(
        "cone", capacity=64, top_charts=3, top_charts_max=6
    )
    assert stats["synchronized"] is True
    picks = generator.choice(len(waves), size=40, replace=False)
    parity = 0
    scanned = 0
    for pick in picks:
        engram_id, wave = waves[int(pick)]
        noise = generator.standard_normal(dim) + 1j * generator.standard_normal(dim)
        noise = noise / np.linalg.norm(noise)
        query = (0.98 * wave + 0.02 * noise).astype(np.complex64)
        charted = memory.query(query, k=4, namespace="cone", use_charts=True)
        flat = memory.query(query, k=4, namespace="cone", use_charts=False)
        assert charted.centered is True
        parity += int(
            bool(charted.hits)
            and bool(flat.hits)
            and int(charted.hits[0].engram_id) == int(flat.hits[0].engram_id)
        )
        scanned += int(charted.scanned)
    assert parity == len(picks)
    assert scanned / float(len(picks) * 720) <= 0.40
    memory.close()


def test_t2_automatic_route_needs_a_synchronized_index(tmp_path):
    cfg = dataclasses.replace(HolarchConfig(), cache_capacity=256)
    memory = ZoneCMemory(cfg, backend="sqlite", path=str(tmp_path / "auto.sqlite3"))
    generator = np.random.default_rng(8)
    dim = int(memory.dim)
    wave = generator.standard_normal(dim) + 1j * generator.standard_normal(dim)
    wave = (wave / np.linalg.norm(wave)).astype(np.complex64)
    memory.write(wave, label=0, tier=2, namespace="auto")
    read = memory.query(wave, k=2, namespace="auto")
    assert read.centered is False
    memory.auto_centered_threshold = 1
    memory.build_centered_index("auto", top_charts=1, top_charts_max=2)
    again = memory.query(wave, k=2, namespace="auto")
    assert again.centered is True
    memory.close()


def test_t3_inducer_learns_bit_flip_and_abstains_on_contradiction():
    spec = InductionSpec(
        "bit_flip",
        ["0", "1"],
        [("01", "10"), ("110", "001"), ("0011", "1100"), ("10101", "01010")],
        [("101100", "010011"), ("1011001", "0100110")],
    )
    inducer = WaveProgramInducer(memory=None, node_budget=20000)
    outcome = inducer.induce(spec)
    assert outcome.status == "INDUCED"
    assert outcome.train_exact and outcome.test_exact
    assert outcome.text_tokens == 0
    assert outcome.repulsors >= 0
    contradiction = contradictory_specs()[0]
    rejected = inducer.induce(contradiction)
    assert rejected.status == "ABSTAIN"
    assert rejected.reason == "ABSTAIN: unsatisfiable specification"


def test_t3_task_battery_declares_ten_and_four_traps():
    assert len(induction_tasks()) == 10
    assert len(contradictory_specs()) == 4


def test_t4_co_processor_finishes_and_abandons(tmp_path):
    scenarios = {row["name"]: row for row in build_scenarios(tmp_path / "ws")}
    processor = HolarchCoProcessor(tmp_path / "ws", max_turns=16)
    finished = processor.run(scenarios["sales_sum"])
    assert finished["status"] == "finished"
    assert finished["deliverable"] == scenarios["sales_sum"]["expected"]
    abandoned = processor.run(scenarios["missing_audit"])
    assert abandoned["status"] == "abandoned"
    assert abandoned["deliverable"] is None
    assert abandoned["abandon_reason"] == "audit.csv"
    assert processor.chain_valid()


def test_t4_repeated_failing_command_is_blocked_by_a_repulsor(tmp_path):
    scenarios = {row["name"]: row for row in build_scenarios(tmp_path / "ws2")}
    processor = HolarchCoProcessor(tmp_path / "ws2", max_turns=16)
    outcome = processor.run(scenarios["fault_recovery"])
    assert outcome["status"] == "finished"
    assert outcome["deliverable"] == scenarios["fault_recovery"]["expected"]
    assert outcome["repulsor_blocks"] >= 1
    failures = [str(row["error"]) for row in outcome["failures"]]
    assert len(failures) == len(set(failures))
    assert processor.chain_valid()
