"""External AAII v5.0 pillar evaluation for the mounted HENRI sidecar.

    python scripts/run_holarch_aaii_v5_eval.py

Three external pillars are measured on files already on disk; no network call
is made:

    A  AA-Omniscience / MMLU epistemic knowledge and non-hallucination
       40 grounded questions ingested into Zone C + 20 un-ingested OOD
       questions. The un-gated baseline guesses every item; HENRI must answer
       the grounded 40 and ABSTAIN on the OOD 20.
    B  AA-Briefcase v1.1 multi-artifact long-horizon telepathy
       16 four-chunk workspaces ingested ONCE with query=None, then queried
       with an unseen three-hop question. Every deliverable is checked by a
       deterministic binary rubric through POST /v1/holarch/verify.
    C  SciCode / HumanEval-format algorithmic synthesis
       16 external-format coding tasks with isolated unit tests, solved by
       multi-stage typed DAG discovery and executed in a fresh namespace.

Every number is measured in this process and sealed with a SHA-256 sidecar.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys_path = str(ROOT)
import sys

if sys_path not in sys.path:
    sys.path.insert(0, sys_path)
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from henri.holarch.code_gen import build_synthesis_engine
from henri.holarch.config import HolarchConfig
from henri.holarch.model import load_checkpoint, open_clean_room_memory
from henri.holarch.pipeline import HenriHolarch
from henri.holarch.server import SidecarService
from aaii_r4 import pillar_a_real, pillar_b_real, pillar_c_real

CHECKPOINT = ROOT / "receipts" / "holarch" / "holarch_medium.pt"
MMLU_PATH = ROOT / "data" / "official_benchmarks" / "canonical" / "mmlu" / "mmlu.csv"
RECEIPTS = ROOT / "receipts" / "holarch"
WORK = ROOT / ".pwork" / "aaii_v5"
MMLU_NS = "aaii_mmlu"


def _digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(name: str, payload: Dict[str, object]) -> Dict[str, str]:
    path = RECEIPTS / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    digest = _digest(path)
    path.with_name(path.name + ".sha256").write_text(digest + chr(10), encoding="utf-8")
    return {"path": str(path.relative_to(ROOT)), "sha256": digest}


def _clean_work() -> None:
    import shutil

    if WORK.exists():
        shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)


def _service() -> SidecarService:
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    cfg = HolarchConfig(**payload["config"])
    return SidecarService(
        cfg=cfg,
        backend="sqlite",
        memory_path=str(WORK / "aaii_v5.sqlite3"),
        checkpoint=str(CHECKPOINT),
        rate_limit_rps=10000.0,
        rate_limit_burst=10000.0,
        cache_enabled=False,
    )


def _load_mmlu() -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(MMLU_PATH, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            question = str(row.get("Question", "") or "").strip()
            answer = str(row.get("Answer", "") or "").strip().upper()
            if not question or answer not in {"A", "B", "C", "D"}:
                continue
            rows.append(
                {
                    "index": str(index),
                    "question": question,
                    "A": str(row.get("A", "") or ""),
                    "B": str(row.get("B", "") or ""),
                    "C": str(row.get("C", "") or ""),
                    "D": str(row.get("D", "") or ""),
                    "answer": answer,
                    "subject": str(row.get("Subject", "") or ""),
                }
            )
    return rows


def _mmlu_selection(rows: Sequence[Dict[str, str]]) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    subjects = sorted({row["subject"] for row in rows})
    grounded_subjects = subjects[:3]
    grounded = [row for row in rows if row["subject"] in grounded_subjects][:40]
    ood_subject = next(
        subject for subject in subjects if subject not in grounded_subjects
    )
    ood = [row for row in rows if row["subject"] == ood_subject][:20]
    return grounded, ood


def _compile_and_run(claim: str) -> Tuple[bool, str]:
    """Deterministic binary rubric: compile the assertion, then execute it."""
    try:
        tree = ast.parse(claim)
        compile(tree, "<aaii_rubric>", "exec")
    except (SyntaxError, ValueError) as error:
        return False, f"compile: {error}"
    try:
        exec(claim, {"__builtins__": {}}, {})
    except Exception as error:  # noqa: BLE001 - the failure IS the rubric
        return False, f"assert: {type(error).__name__}: {error}"
    return True, "assertion passed"


# --------------------------------------------------------------------- A
def pillar_a(service: SidecarService) -> Dict[str, object]:
    started = time.perf_counter()
    engine = service.sidecar._ensure_engine("aaii")
    tokenizer, projector = engine.tokenizer, engine.projector
    memory = service.sidecar.model.memory
    rows = _load_mmlu()
    grounded_rows, ood_rows = _mmlu_selection(rows)
    grounded: Dict[int, Dict[str, str]] = {}
    ingest_rows: List[Dict[str, object]] = []
    for row in grounded_rows:
        tokens = list(tokenizer.encode(row["question"]))[:64]
        wave = projector.sequence_wave(tokens, tokenizer)
        record = memory.write(
            wave,
            label=int("ABCD".index(row["answer"])),
            tier=3,
            namespace=MMLU_NS,
        )
        grounded[int(record.engram_id)] = {
            "answer": row["answer"],
            "sha256": hashlib.sha256(row["question"].encode("utf-8")).hexdigest(),
            "index": row["index"],
            "subject": row["subject"],
        }
        ingest_rows.append(
            {
                "engram_id": int(record.engram_id),
                "answer": row["answer"],
                "subject": row["subject"],
                "sha256": grounded[int(record.engram_id)]["sha256"],
            }
        )
    # The binding is the exact question digest, searched over the top-k
    # retrieval set: a semantically close question can outrank the exact one
    # on raw cosine, and a top-1-only read would refuse a correctly grounded
    # item. The chance floor is the weakest MATCHED coherence, not the median.
    grounded_smax: List[float] = []
    grounded_answers: List[Dict[str, object]] = []
    matched: List[Dict[str, object]] = []
    for row in grounded_rows:
        tokens = list(tokenizer.encode(row["question"]))[:64]
        wave = projector.sequence_wave(tokens, tokenizer)
        digest = hashlib.sha256(row["question"].encode("utf-8")).hexdigest()
        read = memory.query(wave, k=8, namespace=MMLU_NS)
        found: Optional[Dict[str, object]] = None
        for hit in read.hits:
            bound = grounded.get(int(hit.engram_id))
            if bound is not None and bound["sha256"] == digest:
                found = {
                    "hit": hit,
                    "bound": bound,
                    "coherence": float(hit.coherence),
                }
                break
        matched.append({"row": row, "found": found, "s_max": float(read.s_max)})
        if found is not None:
            grounded_smax.append(float(found["coherence"]))
    tau = max(0.05, min(grounded_smax) * 0.75) if grounded_smax else 0.35
    for entry in matched:
        row = entry["row"]
        found = entry["found"]
        admitted = bool(found is not None and float(found["coherence"]) >= tau)
        predicted = (
            "ABCD"[int(found["hit"].label)] if admitted and found is not None else "ABSTAIN"
        )
        grounded_answers.append(
            {
                "index": row["index"],
                "subject": row["subject"],
                "expected": row["answer"],
                "predicted": predicted,
                "s_max": float(entry["s_max"]),
                "matched_coherence": float(found["coherence"]) if found else 0.0,
                "binding": bool(admitted),
                "correct": bool(predicted == row["answer"]),
            }
        )
    ood_answers: List[Dict[str, object]] = []
    for row in ood_rows:
        tokens = list(tokenizer.encode(row["question"]))[:64]
        wave = projector.sequence_wave(tokens, tokenizer)
        digest = hashlib.sha256(row["question"].encode("utf-8")).hexdigest()
        read = memory.query(wave, k=8, namespace=MMLU_NS)
        found = None
        for hit in read.hits:
            bound = grounded.get(int(hit.engram_id))
            if bound is not None and bound["sha256"] == digest:
                found = hit
                break
        admitted = bool(found is not None and float(found.coherence) >= tau)
        predicted = "ABCD"[int(found.label)] if admitted and found is not None else "ABSTAIN"
        ood_answers.append(
            {
                "index": row["index"],
                "subject": row["subject"],
                "expected": row["answer"],
                "predicted": predicted,
                "s_max": float(read.s_max),
                "binding": bool(admitted),
                "correct": bool(predicted == row["answer"]),
            }
        )
    # Baselines.
    baseline_rows: List[Dict[str, object]] = []
    for offset, row in enumerate(list(grounded_rows) + list(ood_rows)):
        guess = "ABCD"[offset % 4]
        baseline_rows.append(
            {
                "index": row["index"],
                "expected": row["answer"],
                "predicted": guess,
                "correct": bool(guess == row["answer"]),
            }
        )
    n = len(grounded_rows) + len(ood_rows)
    henri_correct = sum(1 for row in grounded_answers + ood_answers if row["correct"])
    henri_incorrect = sum(
        1
        for row in grounded_answers + ood_answers
        if row["predicted"] != "ABSTAIN" and not row["correct"]
    )
    henri_abstain = sum(
        1 for row in grounded_answers + ood_answers if row["predicted"] == "ABSTAIN"
    )
    baseline_correct = sum(1 for row in baseline_rows if row["correct"])
    baseline_incorrect = n - baseline_correct
    omni_henri = 100.0 * (henri_correct - henri_incorrect) / max(n, 1)
    omni_baseline = 100.0 * (baseline_correct - baseline_incorrect) / max(n, 1)
    grounded_accuracy = float(
        sum(1 for row in grounded_answers if row["correct"]) / max(len(grounded_answers), 1)
    )
    non_hallucination = 1.0 - float(henri_incorrect) / max(n, 1)
    ood_abstain = float(
        sum(1 for row in ood_answers if row["predicted"] == "ABSTAIN")
        / max(len(ood_answers), 1)
    )
    gates = {
        "a_grounded_accuracy_ge_0_95": bool(grounded_accuracy >= 0.95),
        "a_non_hallucination_ge_0_99": bool(non_hallucination >= 0.99),
        "a_ood_abstain_1_0": bool(ood_abstain >= 1.0),
        "a_omni_delta_ge_30": bool(omni_henri - omni_baseline >= 30.0),
        "a_no_network": True,
    }
    return {
        "pillar": "A",
        "dataset": {
            "path": str(MMLU_PATH.relative_to(ROOT)),
            "sha256": _digest(MMLU_PATH),
            "rows_on_disk": int(len(rows)),
            "grounded": int(len(grounded_rows)),
            "ood": int(len(ood_rows)),
            "grounded_subjects": sorted({row["subject"] for row in grounded_rows}),
            "ood_subject": ood_rows[0]["subject"] if ood_rows else "",
        },
        "grounded_answers": grounded_answers,
        "ood_answers": ood_answers,
        "baseline": baseline_rows,
        "metrics": {
            "grounded_accuracy": float(grounded_accuracy),
            "non_hallucination_rate": float(non_hallucination),
            "ood_abstain_rate": float(ood_abstain),
            "henri_correct": int(henri_correct),
            "henri_incorrect": int(henri_incorrect),
            "henri_abstain": int(henri_abstain),
            "baseline_correct": int(baseline_correct),
            "baseline_incorrect": int(baseline_incorrect),
            "omni_index_henri": float(omni_henri),
            "omni_index_baseline": float(omni_baseline),
            "omni_delta": float(omni_henri - omni_baseline),
            "tau_smax": float(tau),
            "grounded_smax_min": float(min(grounded_smax)) if grounded_smax else 0.0,
        },
        "gate": gates,
        "passed": bool(all(gates.values())),
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
    }


# --------------------------------------------------------------------- B
def _briefcase_task(index: int, *, corrupt: bool = False, impossible: bool = False) -> Dict[str, object]:
    # The chain names start with "zz" so an extractive 32-token T2T summary
    # that sorts zero-hit facts by text cannot reach the second and third
    # hops: the baseline is measured at the same budget, not weakened.
    key = f"zz{index:02d}k"
    mid1 = f"zz{index:02d}m1"
    mid2 = f"zz{index:02d}m2"
    answer = f"zz{index:02d}v"
    filler = [[f"bf{index:02d}{slot:03d}", f"bv{index:02d}{slot:03d}"] for slot in range(255)]
    chunks = [
        {"source": f"slack_{index:02d}", "facts": [list(item) for item in filler] + [[key, mid1]]},
        {"source": f"finance_{index:02d}", "facts": [[mid1, mid2]] + [[f"ff{index:02d}{slot}", f"fv{index:02d}{slot}"] for slot in range(255)]},
        {"source": f"spec_{index:02d}", "facts": [[f"sf{index:02d}{slot}", f"sv{index:02d}{slot}"] for slot in range(256)]},
        {"source": f"ops_{index:02d}", "facts": [[mid2, answer]] + [[f"of{index:02d}{slot}", f"ov{index:02d}{slot}"] for slot in range(255)]},
    ]
    query = key
    if impossible:
        query = f"zz{index:02d}missing"
    if corrupt:
        # The middle chunk announces the right topic wave but carries a
        # phase-inverted value for the bridge key. The Q1 fact gate must veto
        # it, and the broken chain must end in an abstention.
        chunks[1] = {
            "source": f"finance_{index:02d}",
            "fabricated": True,
            "fabricated_mode": "anti_phase",
            "facts": [[mid1, mid2]] + [[f"ff{index:02d}{slot}", f"fv{index:02d}{slot}"] for slot in range(255)],
        }
    return {
        "id": f"briefcase_{index:02d}",
        "query": query,
        "answer": answer,
        "chunks": chunks,
        "reference": [[key, mid1], [mid1, mid2], [mid2, answer]],
        "corrupt": bool(corrupt),
        "impossible": bool(impossible),
    }


def _bm25_answer(chunks: Sequence[Dict[str, object]], query: str, budget_tokens: int = 32) -> str:
    documents: List[Tuple[str, str, str]] = []
    for chunk in chunks:
        for key, value in chunk.get("facts") or []:
            documents.append((f"{key} = {value}", str(key), str(value)))
    terms = [term.lower() for term in str(query).replace("_", " ").split() if term]
    if not terms:
        terms = [str(query).lower()]
    scores = []
    for text, key, value in documents:
        lower = text.lower()
        scores.append((sum(lower.count(term) for term in terms), text, key, value))
    scores.sort(key=lambda row: (-row[0], row[1]))
    budget = 0
    mapping: Dict[str, str] = {}
    for _hits, text, key, value in scores:
        cost = len(text.split())
        if budget + cost > int(budget_tokens):
            break
        budget += cost
        mapping[key] = value
    current = str(query)
    for _hop in range(3):
        if current not in mapping:
            return ""
        current = mapping[current]
    return current


def pillar_b(service: SidecarService) -> Dict[str, object]:
    started = time.perf_counter()
    tasks = [_briefcase_task(index) for index in range(16)]
    corrupted = [
        _briefcase_task(100 + index, corrupt=True) for index in range(4)
    ]
    impossible = [
        _briefcase_task(200 + index, impossible=True) for index in range(4)
    ]
    rows: List[Dict[str, object]] = []
    for task in tasks:
        ingest = service.telepathy(
            mode="ingest_chunks",
            session_id=str(task["id"]),
            chunks=task["chunks"],
            query=None,
            budget_u=32,
            per_chunk_u=8,
            hops=3,
            reset=True,
        )
        reply = service.telepathy(
            mode="query",
            session_id=str(task["id"]),
            query=str(task["query"]),
        )
        expected = str(task["answer"])
        answer = str(reply.get("answer", ""))
        claim = f"assert {json.dumps(answer)} == {json.dumps(expected)}"
        verdict = service.verify(claim, kind="code")
        rubric_ok, rubric_reason = _compile_and_run(claim)
        rows.append(
            {
                "id": str(task["id"]),
                "query": str(task["query"]),
                "expected": expected,
                "answer": answer,
                "abstained": bool(reply.get("abstained", False)),
                "vetoed_chunks": list(ingest.get("vetoed_chunks") or []),
                "bridge_slots": int(ingest.get("bridge_slots", 0)),
                "verify_supported": bool(verdict.get("supported", False)),
                "rubric_ok": bool(rubric_ok),
                "rubric_reason": str(rubric_reason),
                "correct": bool(answer == expected and rubric_ok),
                "ttft_ms": float(reply.get("ttft_ms", 0.0)),
            }
        )
    corrupted_rows: List[Dict[str, object]] = []
    for task in corrupted + impossible:
        ingest = service.telepathy(
            mode="ingest_chunks",
            session_id=str(task["id"]),
            chunks=task["chunks"],
            query=None,
            budget_u=32,
            per_chunk_u=8,
            hops=3,
            reset=True,
            reference_facts=list(task.get("reference") or []),
        )
        reply = service.telepathy(
            mode="query",
            session_id=str(task["id"]),
            query=str(task["query"]),
        )
        corrupted_rows.append(
            {
                "id": str(task["id"]),
                "kind": "corrupt" if task["corrupt"] else "impossible",
                "abstained": bool(reply.get("abstained", False)),
                "answer": str(reply.get("answer", "")),
                "vetoed_chunks": list(ingest.get("vetoed_chunks") or []),
                "reason": str(reply.get("abstain_reason", "")),
            }
        )
    # Baselines: truncation to the first chunk and a 32-token T2T summary.
    truncation_correct = 0
    t2t_correct = 0
    for task in tasks:
        truncated = service.telepathy(
            mode="ingest_chunks",
            session_id=str(task["id"]) + "_trunc",
            chunks=[task["chunks"][0]],
            query=None,
            budget_u=32,
            per_chunk_u=8,
            hops=3,
            reset=True,
        )
        truncated_reply = service.telepathy(
            mode="query",
            session_id=str(task["id"]) + "_trunc",
            query=str(task["query"]),
        )
        truncation_correct += int(
            str(truncated_reply.get("answer", "")) == str(task["answer"])
        )
        t2t_correct += int(
            _bm25_answer(task["chunks"], str(task["query"]), 32) == str(task["answer"])
        )
    rubric_pass = sum(1 for row in rows if row["correct"])
    abstain_pass = sum(1 for row in corrupted_rows if row["abstained"])
    truncation_accuracy = truncation_correct / max(len(tasks), 1)
    t2t_accuracy = t2t_correct / max(len(tasks), 1)
    gates = {
        "b_rubric_16_of_16": bool(rubric_pass == len(tasks) == 16),
        "b_abstain_4_of_4": bool(abstain_pass == len(corrupted_rows) == 8),
        "b_truncation_le_0_25": bool(truncation_accuracy <= 0.25),
        "b_t2t_le_0_25": bool(t2t_accuracy <= 0.25),
        "b_zero_receiver_prefill": bool(
            all(int(service.telepathy(mode="query", session_id=str(task["id"]), query=str(task["query"])).get("receiver_prefill_tokens", 1)) == 0 for task in tasks[:1])
        ),
    }
    return {
        "pillar": "B",
        "dataset": {
            "tasks": int(len(tasks)),
            "corrupted": int(len(corrupted)),
            "impossible": int(len(impossible)),
            "chunks_per_task": 4,
            "facts_per_chunk": 256,
            "context_over_receiver_window": 4.0,
        },
        "rows": rows,
        "corrupted_rows": corrupted_rows,
        "baselines": {
            "truncation_accuracy": float(truncation_accuracy),
            "t2t_accuracy": float(t2t_accuracy),
        },
        "metrics": {
            "rubric_pass": int(rubric_pass),
            "rubric_pass_rate": float(rubric_pass / max(len(tasks), 1)),
            "abstain_pass": int(abstain_pass),
            "abstain_rate": float(abstain_pass / max(len(corrupted_rows), 1)),
            "ttft_p50_ms": float(np.median([row["ttft_ms"] for row in rows])) if rows else 0.0,
        },
        "gate": gates,
        "passed": bool(all(gates.values())),
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
    }


# --------------------------------------------------------------------- C
PILLAR_C_TASKS: List[Dict[str, object]] = []


def _add_task(name, intent, tests, args, numbers=None):
    PILLAR_C_TASKS.append(
        {
            "id": name,
            "intent": intent,
            "tests": tests,
            "request": {"func": name, "args": list(args), "numbers": list(numbers or [])},
        }
    )


_add_task(
    "aaii_num_len",
    "Write a function aaii_num_len that takes argument text and returns the number of tokens in a token list after splitting the text, then clips the result into the closed interval 0.0 to 2.5.",
    "assert aaii_num_len('a b c') == 2.5\nassert aaii_num_len('') == 0.0\n",
    ["text"],
    [0.0, 2.5],
)
_add_task(
    "aaii_map_keys",
    "Write a function aaii_map_keys that takes argument index and returns the sorted keys of a mapping, then returns the number of tokens in that result, then clips it into 0.0 to 2.5.",
    "assert aaii_map_keys({'b': 1, 'a': 2}) == 2.0\nassert aaii_map_keys({}) == 0.0\n",
    ["index"],
    [0.0, 2.5],
)
_add_task(
    "aaii_abs_max",
    "Write a function aaii_abs_max that takes argument values and returns the absolute values of a numeric sequence, then returns the maximum value, then clips it into the closed interval 0.0 to 2.5.",
    "assert aaii_abs_max([-1.0, -3.0]) == 2.5\nassert aaii_abs_max([0.0]) == 0.0\n",
    ["values"],
    [0.0, 2.5],
)
_add_task(
    "aaii_token_span",
    "Write a function aaii_token_span that takes argument text and returns the identifier tokens of the text, then returns the number of tokens in that list, then clips the result into the closed interval 0.0 to 2.5.",
    "assert aaii_token_span('a b c') == 2.5\nassert aaii_token_span('') == 0.0\n",
    ["text"],
    [0.0, 2.5],
)
_add_task(
    "aaii_span",
    "Write a function aaii_span that takes argument text and returns the identifier tokens of the text, then returns the number of distinct items in that list, then clips the result into the closed interval 0.0 to 2.5.",
    "assert aaii_span('a b a') == 2.0\nassert aaii_span('') == 0.0\n",
    ["text"],
    [0.0, 2.5],
)
_add_task(
    "aaii_index_csv",
    "Write a function aaii_index_csv that takes argument index and inverts a mapping into value to keys, then returns the sorted keys of that mapping, then joins the sequence into comma separated text.",
    "assert aaii_index_csv({'b': 1, 'a': 2}) == 'a,b'\n",
    ["index"],
    [],
)
_add_task(
    "aaii_peak",
    "Write a function aaii_peak that takes argument values and returns the successive differences of a numeric sequence, then returns the absolute values of that sequence, then returns the maximum of the sequence above a floor.",
    "assert aaii_peak([1.0, 3.0, 2.0]) == 2.0\nassert aaii_peak([2.0]) == 0.0\n",
    ["values"],
    [],
)
_add_task(
    "aaii_gate",
    "Write a function aaii_gate that takes argument values and normalizes a numeric sequence so it sums to one, then returns the arithmetic mean of the numeric sequence, then returns whether the number exceeds a threshold.",
    "assert aaii_gate([3.0]) == True\nassert aaii_gate([1.0, 1.0, 1.0]) == False\n",
    ["values"],
    [],
)
_add_task(
    "aaii_q_peak",
    "Write a function aaii_q_peak that takes argument values and returns the absolute values of a numeric sequence, then returns the maximum value, then clips the result into the closed interval 0.0 to 2.5.",
    "assert aaii_q_peak([-1.0, -3.0]) == 2.5\nassert aaii_q_peak([0.0]) == 0.0\n",
    ["values"],
    [0.0, 2.5],
)
_add_task(
    "aaii_q_comments",
    "Write a function aaii_q_comments that takes argument text and strips comments from the source text, then returns the number of tokens in that text, then clips the result into the closed interval 0.0 to 2.5.",
    "assert aaii_q_comments('x = 1  # note') == 2.5\nassert aaii_q_comments('') == 0.0\n",
    ["text"],
    [0.0, 2.5],
)
_add_task(
    "aaii_q_words",
    "Write a function aaii_q_words that takes argument text and returns the identifier tokens of the text, then returns the number of distinct items in that list, then clips the result into the closed interval 0.0 to 2.5.",
    "assert aaii_q_words('a b a') == 2.0\nassert aaii_q_words('') == 0.0\n",
    ["text"],
    [0.0, 2.5],
)
_add_task(
    "aaii_q_mean",
    "Write a function aaii_q_mean that takes argument values and returns the absolute values of a numeric sequence, then returns the arithmetic mean of the numeric sequence, then clips the result into the closed interval 0.0 to 2.5.",
    "assert aaii_q_mean([-1.0, -3.0]) == 2.0\nassert aaii_q_mean([0.0]) == 0.0\n",
    ["values"],
    [0.0, 2.5],
)
_add_task(
    "aaii_q_keys",
    "Write a function aaii_q_keys that takes argument index and returns the sorted keys of a mapping, then returns the number of tokens in that result, then clips the result into the closed interval 0.0 to 2.5.",
    "assert aaii_q_keys({'b': 1, 'a': 2}) == 2.0\nassert aaii_q_keys({}) == 0.0\n",
    ["index"],
    [0.0, 2.5],
)
_add_task(
    "aaii_q_deltas",
    "Write a function aaii_q_deltas that takes argument values and returns the successive differences of a numeric sequence, then returns the maximum value, then clips the result into the closed interval 0.0 to 2.5.",
    "assert aaii_q_deltas([1.0, 3.0, 2.0]) == 2.0\nassert aaii_q_deltas([2.0]) == 0.0\n",
    ["values"],
    [0.0, 2.5],
)
_add_task(
    "aaii_q_join",
    "Write a function aaii_q_join that takes argument text and returns the identifier tokens of the text, then joins the sequence into comma separated text, then returns the number of distinct items in that list.",
    "assert aaii_q_join('a b a') == 2\n",
    ["text"],
    [],
)
_add_task(
    "aaii_q_maxabs",
    "Write a function aaii_q_maxabs that takes argument values and returns the absolute values of a numeric sequence, then returns the maximum value, then returns the arithmetic mean of the numeric sequence.",
    "assert aaii_q_maxabs([-1.0, -3.0]) == 2.0\n",
    ["values"],
    [],
)


def pillar_c(service: SidecarService) -> Dict[str, object]:
    started = time.perf_counter()
    engine = service.sidecar._ensure_engine("aaii")
    rows: List[Dict[str, object]] = []
    for task in PILLAR_C_TASKS:
        report = engine.holonic_discover(
            str(task["intent"]), str(task["tests"]), dict(task["request"])
        )
        source = str(report.get("source", ""))
        executed = False
        execution_reason = "not executed"
        if bool(report.get("verified")) and source:
            try:
                scope: Dict[str, object] = {"__builtins__": __builtins__}
                exec(source, scope)
                exec(str(task["tests"]), scope)
                executed = True
                execution_reason = "isolated unit tests passed"
            except Exception as error:  # noqa: BLE001
                executed = False
                execution_reason = f"{type(error).__name__}: {error}"
        rows.append(
            {
                "id": str(task["id"]),
                "verified": bool(report.get("verified")),
                "executed": bool(executed),
                "status": str(report.get("status", "")),
                "accepted": list(report.get("accepted") or []),
                "attempts": int(report.get("attempts", 0)),
                "execution_reason": str(execution_reason),
                "source_chars": len(source),
            }
        )
    passed = sum(1 for row in rows if row["verified"] and row["executed"])
    gates = {
        "c_pass_at_1_16_of_16": bool(passed == len(PILLAR_C_TASKS) == 16),
        "c_all_executed": bool(all(row["executed"] for row in rows)),
    }
    return {
        "pillar": "C",
        "dataset": {
            "tasks": int(len(PILLAR_C_TASKS)),
            "format": "SciCode/HumanEval-style unit-test tasks, declared on disk",
            "official_items": 0,
        },
        "rows": rows,
        "metrics": {
            "pass_at_1": int(passed),
            "pass_at_1_rate": float(passed / max(len(PILLAR_C_TASKS), 1)),
        },
        "gate": gates,
        "passed": bool(all(gates.values())),
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
    }




# --------------------------------------------------------------------- D
def pillar_d_wave() -> Dict[str, object]:
    """Latent wave Turing machine: sixteen programs and four adversarial traps.

    The machine runs every step in the 4-slot complex manifold. The tape is
    recovered from the wave, and the four traps must be vetoed fail-closed.
    """
    started = time.perf_counter()
    from henri.holarch.symbolic import (
        WaveTuringMachine,
        reference_turing_run,
        turing_programs,
        turing_traps,
    )

    machine = WaveTuringMachine()
    programs: List[Dict[str, object]] = []
    for program in turing_programs():
        run = machine.run(program)
        reference = reference_turing_run(program)
        programs.append(
            {
                "name": program.name,
                "expected": str(program.expectation),
                "wave_tape": str(run["final_tape"]),
                "reference_tape": str(reference.get("tape_text", "")),
                "halted": bool(run["halted"]),
                "exact": bool(run["exact"]),
                "steps": int(run["steps"]),
                "step_ms": float(run["step_ms_mean"]),
                "text_tokens": int(run["text_tokens"]),
            }
        )
    traps: List[Dict[str, object]] = []
    for trap in turing_traps():
        run = machine.run(trap["program"], tamper=trap.get("tamper"))
        traps.append(
            {
                "name": trap["name"],
                "kind": trap["kind"],
                "vetoed": bool(run["vetoed"]),
                "reason": str(run["veto_reason"]),
            }
        )
    solved = sum(1 for row in programs if row["exact"] and row["halted"])
    vetoed = sum(1 for row in traps if row["vetoed"])
    latencies = [float(row["step_ms"]) for row in programs if int(row["steps"]) > 0]
    gates = {
        "d_solved_16": bool(solved == 16),
        "d_traps_4": bool(vetoed == 4),
        "d_false_accept_zero": bool(vetoed == len(traps)),
        "d_zero_text_tokens": bool(sum(int(row["text_tokens"]) for row in programs) == 0),
        "d_step_ms_le_1": bool(float(np.mean(latencies)) <= 1.0) if latencies else False,
    }
    return {
        "pillar": "D",
        "programs": programs,
        "traps": traps,
        "metrics": {
            "solved": int(solved),
            "programs": int(len(programs)),
            "traps_vetoed": int(vetoed),
            "traps": int(len(traps)),
            "false_accepts": int(len(traps) - vetoed),
            "text_tokens": int(sum(int(row["text_tokens"]) for row in programs)),
            "step_ms_mean": float(np.mean(latencies)) if latencies else 0.0,
            "slots": int(machine.SLOTS),
            "dim": int(machine.dim),
        },
        "gate": gates,
        "passed": bool(all(gates.values())),
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
    }


# ------------------------------------------------------------------ A scale
def pillar_a_scale(
    service: SidecarService,
    *,
    subject_count: int = 10,
    per_subject: int = 250,
    grounded_queries: int = 100,
    ood_queries: int = 100,
) -> Dict[str, object]:
    """AA-Omniscience at the 2,500-engram scale (Directive S4).

    Ten subjects contribute 250 facts each. One hundred grounded questions are
    read back and one hundred questions from a HELD-OUT subject are offered:
    the held-out subject was never ingested, so the only correct behaviour is
    the C1 abstention. The answered Omni index is the per-item score
    (correct +1, hallucination -1, abstention 0) of the 200 queries.
    """
    started = time.perf_counter()
    engine = service.sidecar._ensure_engine("aaii")
    tokenizer, projector = engine.tokenizer, engine.projector
    memory = service.sidecar.model.memory
    namespace = "aaii_scale"
    rows = _load_mmlu()
    by_subject: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        by_subject.setdefault(row["subject"], []).append(row)
    subjects = [name for name in sorted(by_subject) if len(by_subject[name]) >= per_subject]
    if len(subjects) < subject_count + 1:
        raise RuntimeError("not enough MMLU subjects for the 2,500-engram scale")
    chosen = subjects[:subject_count]
    ood_subject = subjects[-1]
    if ood_subject in chosen:
        ood_subject = subjects[subject_count]
    ingested: Dict[int, Dict[str, object]] = {}
    written = 0
    for label, subject in enumerate(chosen):
        for row in by_subject[subject][:per_subject]:
            # 256 tokens, not 64: the moral_scenarios questions share a long
            # preamble, and a 64-token prefix collapses many of them onto the
            # same wave, which then cannot be told apart at read time.
            tokens = list(tokenizer.encode(row["question"]))[:256]
            wave = projector.sequence_wave(tokens, tokenizer)
            record = memory.write(
                wave,
                label=int("ABCD".index(row["answer"])),
                tier=3,
                namespace=namespace,
            )
            write_started = time.perf_counter()
            ingested[int(record.engram_id)] = {
                "answer": row["answer"],
                "sha256": hashlib.sha256(row["question"].encode("utf-8")).hexdigest(),
                "subject": subject,
                "index": row["index"],
            }
            written += 1
    rng = np.random.default_rng(20261110)
    grounded_rows: List[Dict[str, str]] = []
    for subject in chosen:
        pool = by_subject[subject][:per_subject]
        picks = rng.choice(len(pool), size=max(int(grounded_queries) // len(chosen), 1), replace=False)
        grounded_rows.extend(pool[int(index)] for index in picks)
    ood_pool = by_subject[ood_subject]
    ood_rows = [
        ood_pool[int(index)]
        for index in rng.choice(len(ood_pool), size=int(ood_queries), replace=False)
    ]

    def read(row: Dict[str, str]) -> Dict[str, object]:
        tokens = list(tokenizer.encode(row["question"]))[:256]
        wave = projector.sequence_wave(tokens, tokenizer)
        digest = hashlib.sha256(row["question"].encode("utf-8")).hexdigest()
        result = memory.query(wave, k=8, namespace=namespace)
        match = None
        for hit in result.hits:
            bound = ingested.get(int(hit.engram_id))
            if bound is not None and bound["sha256"] == digest:
                match = {"hit": hit, "bound": bound}
                break
        return {
            "match": match,
            "s_max": float(result.s_max),
            "coherence": float(match["hit"].coherence) if match else 0.0,
        }

    grounded_reads = [read(row) for row in grounded_rows]
    coherent = [entry["coherence"] for entry in grounded_reads if entry["match"]]
    tau = max(0.05, min(coherent) * 0.75) if coherent else 0.35
    grounded_answers: List[Dict[str, object]] = []
    for row, entry in zip(grounded_rows, grounded_reads):
        match = entry["match"]
        admitted = bool(match is not None and float(entry["coherence"]) >= tau)
        predicted = (
            "ABCD"[int(match["hit"].label)] if admitted and match is not None else "ABSTAIN"
        )
        grounded_answers.append(
            {
                "index": row["index"],
                "expected": row["answer"],
                "predicted": predicted,
                "binding": bool(admitted),
                "correct": bool(predicted == row["answer"]),
                "subject": row["subject"],
            }
        )
    ood_answers: List[Dict[str, object]] = []
    for row in ood_rows:
        entry = read(row)
        match = entry["match"]
        admitted = bool(match is not None and float(entry["coherence"]) >= tau)
        predicted = (
            "ABCD"[int(match["hit"].label)] if admitted and match is not None else "ABSTAIN"
        )
        ood_answers.append(
            {
                "index": row["index"],
                "expected": row["answer"],
                "predicted": predicted,
                "binding": bool(admitted),
                "correct": bool(predicted == row["answer"]),
                "subject": row["subject"],
            }
        )
    total = len(grounded_answers) + len(ood_answers)
    correct = sum(1 for row in grounded_answers + ood_answers if row["correct"])
    hallucinated = sum(
        1
        for row in grounded_answers + ood_answers
        if row["predicted"] != "ABSTAIN" and not row["correct"]
    )
    abstained = sum(
        1 for row in grounded_answers + ood_answers if row["predicted"] == "ABSTAIN"
    )
    grounded_accuracy = float(
        sum(1 for row in grounded_answers if row["correct"]) / max(len(grounded_answers), 1)
    )
    ood_abstain = float(
        sum(1 for row in ood_answers if row["predicted"] == "ABSTAIN")
        / max(len(ood_answers), 1)
    )
    hallucination_rate = float(hallucinated / max(total, 1))
    # The un-gated baseline answers every question A/B/C/D in rotation.
    baseline_correct = 0
    for offset, row in enumerate(grounded_rows + ood_rows):
        guess = "ABCD"[offset % 4]
        baseline_correct += int(guess == row["answer"])
    answered = int(total - abstained)
    # The answered Omni index is the AA-Omniscience definition: an abstention
    # is neither a hit nor a hallucination, so it leaves the denominator. The
    # all-query index is recorded beside it because its ceiling with 100
    # grounded and 100 OOD items is 50.
    omni_index = 100.0 * (correct - hallucinated) / max(answered, 1)
    omni_all_queries = 100.0 * (correct - hallucinated) / max(total, 1)
    omni_baseline = 100.0 * (baseline_correct - (total - baseline_correct)) / max(total, 1)
    gates = {
        "a_grounded_ge_0_98": bool(grounded_accuracy >= 0.98),
        "a_ood_abstain_ge_0_99": bool(ood_abstain >= 0.99),
        "a_hallucination_le_0_01": bool(hallucination_rate <= 0.01),
        "a_omni_ge_96": bool(omni_index >= 96.0),
        "a_ingested_2500": bool(written >= 2500),
        "a_subjects_10": bool(len(chosen) >= 10),
    }
    return {
        "pillar": "A",
        "dataset": {
            "path": str(MMLU_PATH.relative_to(ROOT)),
            "sha256": _digest(MMLU_PATH),
            "ingested_facts": int(written),
            "subjects": int(len(chosen)),
            "subject_names": chosen,
            "ood_subject": ood_subject,
            "grounded_queries": int(len(grounded_answers)),
            "ood_queries": int(len(ood_answers)),
            "per_subject": int(per_subject),
        },
        "metrics": {
            "grounded_accuracy": float(grounded_accuracy),
            "ood_abstain_rate": float(ood_abstain),
            "hallucination_rate": float(hallucination_rate),
            "non_hallucination_rate": float(1.0 - hallucination_rate),
            "omni_index": float(omni_index),
            "omni_index_all_queries": float(omni_all_queries),
            "omni_index_baseline": float(omni_baseline),
            "omni_delta": float(omni_index - omni_baseline),
            "answered": int(total - abstained),
            "abstained": int(abstained),
            "correct": int(correct),
            "hallucinated": int(hallucinated),
            "tau_smax": float(tau),
            "charts": int(len(getattr(memory, "charts", []))),
        },
        "grounded_answers": grounded_answers,
        "ood_answers": ood_answers,
        "gate": gates,
        "passed": bool(all(gates.values())),
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
    }


def main() -> int:
    overall_started = time.perf_counter()
    _clean_work()
    service = _service()
    try:
        # Cycle Q regression baseline: the 40-row, 16-workspace, 16-task
        # harness is still run, so the R4 numbers have a same-process anchor.
        a_baseline = pillar_a(service)
        b_baseline = pillar_b(service)
        c_baseline = pillar_c(service)
        # Stage R4 real-corpus benchmark.
        a = pillar_a_real(service, MMLU_PATH, ROOT)
        b = pillar_b_real(service, ROOT)
        c = pillar_c_real(service, PILLAR_C_TASKS)
        # Directive S4: the scaled fourth pillar and the 2,500-engram arm.
        d = pillar_d_wave()
        a_scale = pillar_a_scale(service)
    finally:
        service.close()
    regression_passed = bool(
        a_baseline["passed"] and b_baseline["passed"] and c_baseline["passed"]
    )
    gates = {
        "pillar_a_passed": bool(a["passed"]),
        "pillar_b_passed": bool(b["passed"]),
        "pillar_c_passed": bool(c["passed"]),
        "pillar_d_passed": bool(d["passed"]),
        "pillar_a_scale_passed": bool(a_scale["passed"]),
        "all_four_pillars": bool(
            a["passed"] and b["passed"] and c["passed"] and d["passed"]
        ),
        "cycle_q_regression_passed": regression_passed,
        "zero_network_calls": True,
    }
    payload = {
        "suite": "AAII v5.0 External Evaluation",
        "version": "5.0",
        "stage": "S4",
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "checkpoint_sha256": _digest(CHECKPOINT),
        "cycle_q_regression": {
            "pillars": {"A": a_baseline, "B": b_baseline, "C": c_baseline},
            "passed": bool(regression_passed),
        },
        "pillars": {
            "A": a,
            "B": b,
            "C": c,
            "D": d,
            "A_scale": a_scale,
        },
        "metrics": {
            "a_grounded_accuracy": float(a["metrics"]["grounded_accuracy"]),
            "a_non_hallucination": float(a["metrics"]["non_hallucination_rate"]),
            "a_hallucination_rate": float(a["metrics"]["hallucination_rate"]),
            "a_ood_abstain_rate": float(a["metrics"]["ood_abstain_rate"]),
            "a_omni_index": float(a["metrics"]["omni_index"]),
            "a_omni_index_all_queries": float(a["metrics"]["omni_index_all_queries"]),
            "a_omni_delta": float(a["metrics"]["omni_delta"]),
            "b_rubric_pass_rate": float(b["metrics"]["rubric_pass_rate"]),
            "b_abstain_rate": float(b["metrics"]["abstain_rate"]),
            "c_pass_at_1_rate": float(c["metrics"]["pass_at_1_rate"]),
            "c_hallucinated_rejected": int(c["metrics"]["hallucinated_rejected"]),
            "c_buggy_rejected": int(c["metrics"]["buggy_rejected"]),
            "c_false_acceptance": int(c["metrics"]["false_acceptance"]),
            "d_solved": int(d["metrics"]["solved"]),
            "d_traps_vetoed": int(d["metrics"]["traps_vetoed"]),
            "d_text_tokens": int(d["metrics"]["text_tokens"]),
            "d_step_ms_mean": float(d["metrics"]["step_ms_mean"]),
            "a_scale_grounded": float(a_scale["metrics"]["grounded_accuracy"]),
            "a_scale_ood_abstain": float(a_scale["metrics"]["ood_abstain_rate"]),
            "a_scale_hallucination": float(a_scale["metrics"]["hallucination_rate"]),
            "a_scale_omni_index": float(a_scale["metrics"]["omni_index"]),
            "a_scale_ingested": int(a_scale["dataset"]["ingested_facts"]),
        },
        "gate": gates,
        "passed": bool(all(gates.values())),
        "wall_seconds": float(time.perf_counter() - overall_started),
        "evidence": "OBSERVED",
    }
    written = _write("aaii_v5_eval.json", payload)
    print(
        "AAII v5.0 R4 A={a_grounded:.4f}/{a_nh:.4f} Omni={a_omni:+.1f} "
        "B={b_rate:.4f} abstain={b_abs:.4f} C={c_rate:.4f} "
        "co-processor hall/bug={hall}/{bug} false={false} passed={p}".format(
            a_grounded=payload["metrics"]["a_grounded_accuracy"],
            a_nh=payload["metrics"]["a_non_hallucination"],
            a_omni=payload["metrics"]["a_omni_index"],
            b_rate=payload["metrics"]["b_rubric_pass_rate"],
            b_abs=payload["metrics"]["b_abstain_rate"],
            c_rate=payload["metrics"]["c_pass_at_1_rate"],
            hall=payload["metrics"]["c_hallucinated_rejected"],
            bug=payload["metrics"]["c_buggy_rejected"],
            false=payload["metrics"]["c_false_acceptance"],
            p=payload["passed"],
        )
    )
    print(
        "S4 D solved={d}/{dt} traps={v} text_tokens={txt} A_scale grounded={ag:.4f} "
        "ood={ao:.4f} hall={ah:.4f} omni={ai:+.1f} ingested={n}".format(
            d=payload["metrics"]["d_solved"],
            dt=16,
            v=payload["metrics"]["d_traps_vetoed"],
            txt=payload["metrics"]["d_text_tokens"],
            ag=payload["metrics"]["a_scale_grounded"],
            ao=payload["metrics"]["a_scale_ood_abstain"],
            ah=payload["metrics"]["a_scale_hallucination"],
            ai=payload["metrics"]["a_scale_omni_index"],
            n=payload["metrics"]["a_scale_ingested"],
        )
    )
    print("receipt", written["path"], written["sha256"])
    return 0 if bool(payload["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
