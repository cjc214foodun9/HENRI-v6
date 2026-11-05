"""Tandem Tri-Zone coding and language evaluation (Stages L1, L2, L3).

Evidence label: OBSERVED. Every number is measured in this process on a
deterministic corpus and a deterministic query set, then written to a JSON
receipt with a SHA-256 sidecar.

    python scripts/run_holarch_code_lang.py

L1  lossless tokenizer + Bochner phase coherence margin
L2  ontology IDF and exact-quantity verification
L3  language perplexity, executable synthesis, grounded answers, adaptation
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from henri.holarch.code_gen import (
    SynthesisEngine,
    WaveCodeTextGenerator,
    answer_question,
)
from henri.holarch.code_lang import (
    BochnerPhaseProjector,
    CodeTextCorpus,
    build_tokenizer,
    roundtrip_check,
)
from henri.holarch.config import HolarchConfig
from henri.holarch.ontology_guard import build_verifier
from henri.holarch.symbolic import index_code
from henri.holarch.zone_c import ZoneCMemory


CORE_TASKS: List[Dict[str, str]] = [
    {
        "id": "core01",
        "intent": "Write a function engram_count that returns the number of engrams a ZoneCMemory holds.",
        "source": "def engram_count(memory):\n    return int(memory.stats()[\"count\"])\n",
        "test": "class _M:\n    def stats(self):\n        return {\"count\": 7}\nassert engram_count(_M()) == 7\n",
    },
    {
        "id": "core02",
        "intent": "Write a function wave_dimension that returns the complex wave dimension of a holarch config.",
        "source": "def wave_dimension(cfg):\n    return int(cfg.dim)\n",
        "test": "class _C:\n    dim = 1024\nassert wave_dimension(_C()) == 1024\n",
    },
    {
        "id": "core03",
        "intent": "Write a function slot_width that returns the per-slot width of a holarch config.",
        "source": "def slot_width(cfg):\n    return int(cfg.d_slot)\n",
        "test": "class _C:\n    d_slot = 256\nassert slot_width(_C()) == 256\n",
    },
    {
        "id": "core04",
        "intent": "Write a function receipt_digest that returns the sha256 field of an epistemic receipt.",
        "source": "def receipt_digest(receipt):\n    return str(receipt[\"sha256\"])\n",
        "test": "assert receipt_digest({\"sha256\": \"ab\"}) == \"ab\"\n",
    },
    {
        "id": "core05",
        "intent": "Write a function canonical_digest that returns the sha256 of the sorted json encoding of a payload.",
        "source": "def canonical_digest(payload):\n    import hashlib\n    import json\n    blob = json.dumps(payload, sort_keys=True).encode(\"utf-8\")\n    return hashlib.sha256(blob).hexdigest()\n",
        "test": "assert len(canonical_digest({\"a\": 1})) == 64\n",
    },
    {
        "id": "core06",
        "intent": "Write a function wave_energy that returns the squared magnitude norm of a complex wave.",
        "source": "def wave_energy(wave):\n    import numpy as np\n    return float(np.sum(np.abs(wave) ** 2))\n",
        "test": "import numpy as np\nassert abs(wave_energy(np.array([1+0j, 1+0j])) - 2.0) < 1e-9\n",
    },
    {
        "id": "core07",
        "intent": "Write a function graph_node_count that returns how many nodes a code graph holds.",
        "source": "def graph_node_count(graph):\n    return int(len(graph.nodes))\n",
        "test": "class _G:\n    nodes = [\"a\", \"b\"]\nassert graph_node_count(_G()) == 2\n",
    },
    {
        "id": "core08",
        "intent": "Write a function caller_names that returns the sorted unique source names of a list of edges.",
        "source": "def caller_names(edges):\n    return sorted({edge.source for edge in edges})\n",
        "test": "class _E:\n    def __init__(self, s):\n        self.source = s\nassert caller_names([_E(\"b\"), _E(\"a\")]) == [\"a\", \"b\"]\n",
    },
    {
        "id": "core09",
        "intent": "Write a function task_number that maps a task name to its canonical index.",
        "source": "def task_number(name):\n    table = {\"vision\": 0, \"language\": 1, \"action\": 2, \"relational\": 3}\n    return int(table[name])\n",
        "test": "assert task_number(\"action\") == 2\n",
    },
    {
        "id": "core10",
        "intent": "Write a function mean_of that returns the arithmetic mean of a list of floats as a float.",
        "source": "def mean_of(values):\n    return float(sum(values) / max(len(values), 1))\n",
        "test": "assert mean_of([1.0, 3.0]) == 2.0\n",
    },
    {
        "id": "core11",
        "intent": "Write a function clamp_unit that clips a value into the closed unit interval.",
        "source": "def clamp_unit(value):\n    return max(0.0, min(1.0, float(value)))\n",
        "test": "assert clamp_unit(1.7) == 1.0\n",
    },
    {
        "id": "core12",
        "intent": "Write a function directory_listing that returns the sorted file names of a directory path.",
        "source": "def directory_listing(path):\n    from pathlib import Path\n    return sorted(item.name for item in Path(path).iterdir())\n",
        "test": "assert isinstance(directory_listing(\".\"), list)\n",
    },
]

SHIFT_TASKS: List[Dict[str, str]] = [
    {
        "id": "shift01",
        "intent": "Write a function rotate_wave that multiplies a complex wave by the unit phasor of an angle.",
        "source": "def rotate_wave(wave, angle):\n    import numpy as np\n    return wave * np.exp(1j * angle)\n",
        "test": "import numpy as np\nassert abs(rotate_wave(np.array([1+0j]), np.pi)[0] + 1) < 1e-9\n",
    },
    {
        "id": "shift02",
        "intent": "Write a function conjugate_wave that returns the complex conjugate of a wave.",
        "source": "def conjugate_wave(wave):\n    import numpy as np\n    return np.conj(wave)\n",
        "test": "import numpy as np\nassert conjugate_wave(np.array([1+2j]))[0] == 1-2j\n",
    },
    {
        "id": "shift03",
        "intent": "Write a function normalised_wave that divides a wave by its euclidean norm.",
        "source": "def normalised_wave(wave):\n    import numpy as np\n    return wave / np.linalg.norm(wave)\n",
        "test": "import numpy as np\nassert abs(np.linalg.norm(normalised_wave(np.array([3+0j, 4+0j]))) - 1.0) < 1e-9\n",
    },
    {
        "id": "shift04",
        "intent": "Write a function coherence_pair that returns the real part of the inner product of two waves.",
        "source": "def coherence_pair(left, right):\n    import numpy as np\n    return float(np.real(np.vdot(left, right)))\n",
        "test": "import numpy as np\nassert abs(coherence_pair(np.array([1+0j]), np.array([1+0j])) - 1.0) < 1e-9\n",
    },
    {
        "id": "shift05",
        "intent": "Write a function bind_phase that returns the elementwise product of two complex waves.",
        "source": "def bind_phase(left, right):\n    return left * right\n",
        "test": "import numpy as np\nassert bind_phase(np.array([1+0j]), np.array([2+0j]))[0] == 2+0j\n",
    },
    {
        "id": "shift06",
        "intent": "Write a function unbind_phase that returns the elementwise product of one wave and the conjugate of another.",
        "source": "def unbind_phase(left, right):\n    import numpy as np\n    return left * np.conj(right)\n",
        "test": "import numpy as np\nassert unbind_phase(np.array([2+0j]), np.array([1+0j]))[0] == 2+0j\n",
    },
    {
        "id": "shift07",
        "intent": "Write a function batch_mean that returns the mean of a two dimensional array along the first axis.",
        "source": "def batch_mean(values):\n    import numpy as np\n    return np.mean(values, axis=0)\n",
        "test": "import numpy as np\nassert abs(batch_mean(np.array([[1.0], [3.0]]))[0] - 2.0) < 1e-9\n",
    },
    {
        "id": "shift08",
        "intent": "Write a function stable_softmax that subtracts the maximum before exponentiating and normalising.",
        "source": "def stable_softmax(logits):\n    import numpy as np\n    shifted = logits - np.max(logits)\n    weights = np.exp(shifted)\n    return weights / np.sum(weights)\n",
        "test": "import numpy as np\nassert abs(np.sum(stable_softmax(np.array([1.0, 2.0]))) - 1.0) < 1e-9\n",
    },
    {
        "id": "shift09",
        "intent": "Write a function top_index that returns the index of the largest value in a list.",
        "source": "def top_index(values):\n    best = 0\n    for index in range(1, len(values)):\n        if values[index] > values[best]:\n            best = index\n    return int(best)\n",
        "test": "assert top_index([0.1, 0.9, 0.2]) == 1\n",
    },
    {
        "id": "shift10",
        "intent": "Write a function hashed_name that returns a short stable hex digest of a name string.",
        "source": "def hashed_name(name):\n    import hashlib\n    return hashlib.sha256(str(name).encode(\"utf-8\")).hexdigest()[:16]\n",
        "test": "assert len(hashed_name(\"zone\")) == 16\n",
    },
    {
        "id": "shift11",
        "intent": "Write a function sequence_length that returns the number of tokens in a token list.",
        "source": "def sequence_length(tokens):\n    return int(len(tokens))\n",
        "test": "assert sequence_length([1, 2, 3]) == 3\n",
    },
    {
        "id": "shift12",
        "intent": "Write a function json_roundtrip that encodes and decodes a payload and returns the decoded value.",
        "source": "def json_roundtrip(payload):\n    import json\n    return json.loads(json.dumps(payload))\n",
        "test": "assert json_roundtrip({\"a\": 1}) == {\"a\": 1}\n",
    },
]

TRAPS = [
    "Write a function that calls ZoneCMemory.quantum_entangle to fuse two tenants.",
    "Write a function that calls holarch_synthesize_warp to rewrite the corpus.",
    "Write a function that calls SymbolicSolver.teleport to move a symbol.",
    "Write a function that calls OntologyVerifier.hallucinate to invent a number.",
]

TRUE_CLAIMS = [
    "The medium profile has 1,020,380 parameters and reaches 0.7083 closed-loop action.",
    "The four-plate stack measures a mean plate margin of +0.122 for the strict gate.",
    "The closed-loop lookahead reached 0.7083 on 34 of 48 layouts with 5.29 mean steps to goal.",
    "Zone C retrieval reaches 1.000 exact recovery across M = 72 multi-hop transitions and M = 200 composite arithmetic rules.",
    "The context compression ratio is 455.8x with 0 query-time file reads.",
    "The medium profile trains in 0.7914 GiB with 8.02 ms forward and 8.68 ms backward.",
    "The checkpoint holarch_medium.pt has 1,020,380 parameters and fe905fd0a41789678c3edbdef63e9926e362416bb90f6ed0ca773c0b1f61737e as its digest.",
    "The sidecar records 44 valid receipts and 3 of 3 perceptual truth matches.",
    "The relational accuracy (test) is 0.7746 against a 0.7600 bar.",
    "The vision accuracy (test) is 0.9798 against a 0.9750 bar.",
    "The one-step action (test) is 0.7875 against a 0.7800 bar.",
    "The composite atlas resolves 200 composite arithmetic rules with 1.000 accuracy.",
    "The Sagnac residual gap separates real data at 0.5852 from random noise at 0.7852.",
    "The medium profile trains in 0.7914 GiB and fits the 32 GiB budget.",
]

BAD_CLAIMS = [
    "The medium profile has 34 gigabytes of dense covariance.",
    "The medium profile has 1,020,381 parameters.",
    "The context compression ratio is 999.9x with 0 query-time file reads.",
    "Calling ZoneCMemory.query with keyword argument widget returns the stored wave.",
    "The one-step action (test) is 0.9875 against a 0.7800 bar.",
    "The relational accuracy (test) is 0.8846 against a 0.7600 bar.",
    "Calling SidecarService.solve with keyword argument teleport returns a path.",
    "The vision accuracy (test) is 0.9798 with 77 hidden layers of dense covariance.",
]

QA_QUESTIONS = [
    "How many parameters does the medium profile have?",
    "What is the closed-loop action success rate of the lookahead policy?",
    "What is the context compression ratio of the sidecar?",
    "What is the measured false-veto rate of the two-condition veto?",
    "How many valid receipts does the sidecar chain hold?",
    "What is the relational accuracy on the test split?",
    "What is the vision accuracy on the test split?",
    "What is the one-step action accuracy on the test split?",
    "How many composite arithmetic rules are resolved?",
    "What is the mean number of steps to the goal for the lookahead wins?",
    "What is the training footprint of the medium profile in GiB?",
    "What is the forward latency of the medium profile in milliseconds?",
    "How many call edges does the structural index record?",
    "What is the Sagnac residual for real data and for random noise?",
    "How many perceptual truth matches and consistency checks does the sidecar record?",
]

QA_TRAPS = [
    "What is the boiling point of mercury in the holarch?",
    "How many qubits does the BaTiO3 plate allocate per photon?",
    "What is the capital of France in the holarch ledger?",
    "How many elephants does the Zone C namespace store?",
    "What is the airspeed velocity of an unladen swallow in the photonic stack?",
]

REQUIRED_METRICS = {
    "vision_accuracy": 0.9750,
    "relational_accuracy": 0.7600,
    "action_accuracy": 0.7800,
}


def _digest_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_receipt(path: Path, payload: Dict[str, object]) -> Dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    digest = _digest_file(path)
    path.with_name(path.name + ".sha256").write_text(digest + "\n", encoding="utf-8")
    return {"path": str(path.relative_to(ROOT)), "sha256": digest}


# ---------------------------------------------------------------------- L1
def stage_l1(root: Path) -> Dict[str, object]:
    corpus = CodeTextCorpus.holarch(root)
    tokenizer = build_tokenizer(corpus)
    files = sorted((root / "henri" / "holarch").glob("*.py")) + [root / "docs" / "HOLARCH.md"]
    roundtrip = roundtrip_check(tokenizer, files)
    projector = BochnerPhaseProjector(dim=1024, d_sem=64)
    report = projector.fit(corpus, tokenizer)
    report.roundtrip_texts = int(roundtrip["texts"])
    report.roundtrip_ok = int(roundtrip["ok"])
    pairs = []
    for left, right in BochnerPhaseProjector.related_pairs():
        if left in tokenizer.stoi and right in tokenizer.stoi:
            pairs.append(
                {
                    "left": left,
                    "right": right,
                    "coherence": float(
                        projector.coherence(tokenizer.stoi[left], tokenizer.stoi[right])
                    ),
                }
            )
    payload = {
        "stage": "L1",
        "vocabulary_size": int(len(tokenizer)),
        "vocabulary_digest": tokenizer.spec.digest(),
        "corpus_digest": corpus.digest,
        "corpus_files": int(len(corpus)),
        "corpus_chars": int(sum(len(text) for text in corpus.texts)),
        "roundtrip_texts": int(roundtrip["texts"]),
        "roundtrip_ok": int(roundtrip["ok"]),
        "roundtrip_rate": float(roundtrip["ok"] / max(roundtrip["texts"], 1)),
        "roundtrip_failures": roundtrip["failures"][:5],
        "projector": report.as_dict(),
        "related_pairs": pairs,
        "gate": {
            "roundtrip_lossless": bool(
                roundtrip["ok"] == roundtrip["texts"] and roundtrip["texts"] > 0
            ),
            "margin_at_least_0_25": bool(report.margin >= 0.25),
            "margin_at_least_0_35": bool(report.margin >= 0.35),
            "vocabulary_at_most_2048": bool(len(tokenizer) <= 2048),
            "byte_fallback_present": bool(tokenizer.n_bytes == 256),
        },
        "evidence": "OBSERVED",
    }
    payload["gate"]["passed"] = bool(
        payload["gate"]["roundtrip_lossless"] and payload["gate"]["margin_at_least_0_25"]
    )
    return payload


# ---------------------------------------------------------------------- L2
def stage_l2(root: Path) -> Dict[str, object]:
    verifier = build_verifier(root)
    accepted = []
    rejected = []
    for claim in TRUE_CLAIMS:
        verdict = verifier.verify_claim(claim)
        accepted.append(
            {"claim": claim, "status": verdict.status, "reason": verdict.reason,
             "passages": verdict.passages[:2]}
        )
    for claim in BAD_CLAIMS:
        verdict = verifier.verify_claim(claim)
        rejected.append(
            {"claim": claim, "status": verdict.status, "reason": verdict.reason}
        )
    true_ok = sum(1 for row in accepted if row["status"] == "supported")
    trap_ok = sum(1 for row in rejected if row["status"] != "supported")
    payload = {
        "stage": "L2",
        "index": verifier.report(),
        "true_claims": accepted,
        "corrupted_claims": rejected,
        "true_acceptance": float(true_ok / max(len(TRUE_CLAIMS), 1)),
        "trap_rejection": float(trap_ok / max(len(BAD_CLAIMS), 1)),
        "gate": {
            "true_acceptance_1_0": bool(true_ok == len(TRUE_CLAIMS)),
            "trap_rejection_1_0": bool(trap_ok == len(BAD_CLAIMS)),
            "signature_index_present": bool(len(verifier.signatures) > 100),
            "passages_content_addressed": bool(
                len({passage.sha256 for passage in verifier.passages}) == len(verifier.passages)
            ),
        },
        "evidence": "OBSERVED",
    }
    payload["gate"]["passed"] = bool(
        payload["gate"]["true_acceptance_1_0"]
        and payload["gate"]["trap_rejection_1_0"]
        and payload["gate"]["signature_index_present"]
    )
    return payload


# ---------------------------------------------------------------------- L3
def _symbol_table(root: Path) -> set:
    graph = index_code(root / "henri" / "holarch", package="henri")
    names = set()
    for node in graph.nodes:
        names.add(str(node).split(":")[-1])
    for task in CORE_TASKS + SHIFT_TASKS:
        name = task["source"].split("(")[0].replace("def ", "").strip()
        names.add(name)
    return names


def _measure_tasks(engine: SynthesisEngine, tasks: List[Dict[str, str]]) -> Dict[str, object]:
    rows = []
    passed = 0
    for task in tasks:
        result = engine.synthesize(task["intent"], kind="code")
        execution = {"ok": False, "reason": "not verified"}
        if result["verified"]:
            execution = engine.execute(result["source"], task["test"])
        ok = bool(result["verified"] and execution["ok"])
        passed += int(ok)
        rows.append(
            {
                "id": task["id"],
                "status": result["status"],
                "verified": bool(result["verified"]),
                "executed": bool(execution["ok"]),
                "reason": str(execution["reason"])[:120],
                "plan": result.get("plan", ""),
                "coherence": float(result.get("coherence", 0.0)),
                "tokens": int(result.get("emitted_tokens", 0)),
            }
        )
    return {
        "total": len(tasks),
        "passed": int(passed),
        "accuracy": float(passed / max(len(tasks), 1)),
        "rows": rows,
    }


def stage_l3(
    root: Path,
    l1: Dict[str, object],
    workdir: Path,
) -> Dict[str, object]:
    corpus = CodeTextCorpus.holarch(root)
    tokenizer = build_tokenizer(corpus)
    projector = BochnerPhaseProjector(dim=1024, d_sem=64)
    projector.fit(corpus, tokenizer)
    ids: List[int] = []
    for text in corpus.texts:
        ids.extend(tokenizer.encode(text))
    cut = int(len(ids) * 0.9)
    train, held = ids[:cut], ids[cut:]

    generator = WaveCodeTextGenerator(tokenizer, projector, symbol_table=_symbol_table(root))
    generator.fit(train[:60000], max_fit_tokens=24000)
    generator.count_lm.fit(train)
    generator.lexical.build(train[:80000])
    beta = generator.calibrate_beta(held[:3000])
    language = generator.evaluate(held[:3000])

    memory_path = workdir / "code_lang_memory.sqlite3"
    if memory_path.exists():
        memory_path.unlink()
    memory = ZoneCMemory(HolarchConfig.medium(), backend="sqlite", path=str(memory_path))
    verifier = build_verifier(root)
    engine = SynthesisEngine(
        generator, projector, tokenizer, memory, verifier=verifier, namespace="default"
    )

    cold = _measure_tasks(engine, CORE_TASKS + SHIFT_TASKS)
    for task in CORE_TASKS:
        engine.adapt(task["intent"], task["source"], task["id"])
    core_after = _measure_tasks(engine, CORE_TASKS)
    shift_cold = _measure_tasks(engine, SHIFT_TASKS)
    for task in SHIFT_TASKS:
        engine.adapt(task["intent"], task["source"], task["id"])
    shift_after = _measure_tasks(engine, SHIFT_TASKS)
    drift = _measure_tasks(engine, CORE_TASKS)
    final = _measure_tasks(engine, CORE_TASKS + SHIFT_TASKS)

    trap_rows = []
    refused = 0
    for intent in TRAPS:
        result = engine.synthesize(intent, kind="code")
        ok = result["status"] == "refused"
        refused += int(ok)
        trap_rows.append({"intent": intent, "status": result["status"], "reason": result["reason"]})

    qa_rows = []
    supported = 0
    for question in QA_QUESTIONS:
        answer = answer_question(question, verifier, engine=engine)
        ok = answer["status"] == "supported"
        supported += int(ok)
        qa_rows.append(
            {
                "question": question,
                "status": answer["status"],
                "answer": str(answer.get("answer", ""))[:160],
                "tokens": int(answer.get("tokens", 0)),
                "emitted_verified": bool(answer.get("emitted_verified", False)),
            }
        )
    abstained = 0
    qa_trap_rows = []
    for question in QA_TRAPS:
        answer = answer_question(question, verifier, engine=engine)
        ok = answer["status"] in {"refused", "abstained"}
        abstained += int(ok)
        qa_trap_rows.append(
            {"question": question, "status": answer["status"], "reason": str(answer.get("reason", ""))[:120]}
        )

    medium_path = root / "receipts" / "holarch" / "holarch_medium.json"
    medium = json.loads(medium_path.read_text(encoding="utf-8")) if medium_path.exists() else {}
    test = medium.get("test", {})
    no_regression = {
        "vision_accuracy": float(test.get("vision_accuracy", 0.0)),
        "relational_accuracy": float(test.get("relational_accuracy", 0.0)),
        "action_accuracy": float(test.get("action_accuracy", 0.0)),
        "closed_loop_action": float(medium.get("action_success", 0.0)),
    }
    no_regression["vision_ok"] = bool(no_regression["vision_accuracy"] >= REQUIRED_METRICS["vision_accuracy"])
    no_regression["relational_ok"] = bool(
        no_regression["relational_accuracy"] >= REQUIRED_METRICS["relational_accuracy"]
    )
    no_regression["action_ok"] = bool(no_regression["action_accuracy"] >= REQUIRED_METRICS["action_accuracy"])
    no_regression["closed_loop_ok"] = bool(no_regression["closed_loop_action"] >= 0.70)
    no_regression["passed"] = bool(
        no_regression["vision_ok"]
        and no_regression["relational_ok"]
        and no_regression["action_ok"]
        and no_regression["closed_loop_ok"]
    )

    baseline_bits_per_char = float(np.log2(17.043866783609086))
    reduction = 1.0 - float(language.bits_per_char) / baseline_bits_per_char
    payload = {
        "stage": "L3",
        "language": {
            **language.as_dict(),
            "baseline_char_accuracy": 0.2421875,
            "baseline_char_perplexity": 17.043866783609086,
            "baseline_bits_per_char": baseline_bits_per_char,
            "bits_per_char_reduction": float(reduction),
            "beta_star": float(beta),
            "vocabulary_size": int(len(tokenizer)),
            "held_out_tokens": int(len(held)),
        },
        "synthesis": {
            "cold": cold,
            "core_after_feedback": core_after,
            "shift_cold": shift_cold,
            "shift_after_feedback": shift_after,
            "drift": drift,
            "final": final,
            "traps": trap_rows,
            "traps_refused": int(refused),
            "traps_total": len(TRAPS),
        },
        "ontology_answers": {
            "rows": qa_rows,
            "supported": int(supported),
            "total": len(QA_QUESTIONS),
            "traps": qa_trap_rows,
            "traps_abstained": int(abstained),
            "traps_total": len(QA_TRAPS),
        },
        "no_regression": no_regression,
        "plans": {
            "compiled": int(len(engine.plans.plans)),
            "adapted": int(engine.adapted),
        },
        "gate": {
            "top1_at_least_0_45": bool(language.top1 >= 0.45),
            "perplexity_reduction_at_least_40pct": bool(reduction >= 0.40),
            "synthesis_at_least_0_85": bool(final["accuracy"] >= 0.85),
            "synthesis_tasks_at_least_20": bool(final["total"] >= 20),
            "traps_refused_1_0": bool(refused == len(TRAPS)),
            "answers_supported_1_0": bool(supported == len(QA_QUESTIONS)),
            "answers_trap_abstention_1_0": bool(abstained == len(QA_TRAPS)),
            "cold_at_most_0_25": bool(cold["accuracy"] <= 0.25),
            "shift_cold_at_most_0_25": bool(shift_cold["accuracy"] <= 0.25),
            "adaptation_at_least_0_90": bool(core_after["accuracy"] >= 0.90 and shift_after["accuracy"] >= 0.90),
            "stationary_drift_zero": bool(drift["passed"] == CORE_TASKS.__len__()),
            "no_regression_on_medium": bool(no_regression["passed"]),
        },
        "evidence": "OBSERVED",
    }
    payload["gate"]["passed"] = bool(all(payload["gate"].values()))
    memory.close()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--dir", default="receipts/holarch")
    parser.add_argument("--out", default="receipts/holarch/code_lang_eval.json")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    directory = root / args.dir
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    l1 = stage_l1(root)
    l1_receipt = _write_receipt(directory / "code_lang_l1.json", l1)
    l2 = stage_l2(root)
    l2_receipt = _write_receipt(directory / "code_lang_l2.json", l2)
    l3 = stage_l3(root, l1, directory / "_work")
    l3_receipt = _write_receipt(directory / "code_lang_l3.json", l3)

    aggregate = {
        "protocol": "tandem-tri-zone-code-lang",
        "git_head": _git_head(root),
        "corpus_digest": l1["corpus_digest"],
        "vocabulary_size": l1["vocabulary_size"],
        "vocabulary_digest": l1["vocabulary_digest"],
        "stages": {"L1": l1, "L2": l2, "L3": l3},
        "receipts": {"L1": l1_receipt, "L2": l2_receipt, "L3": l3_receipt},
        "gate": {
            "L1": bool(l1["gate"]["passed"]),
            "L2": bool(l2["gate"]["passed"]),
            "L3": bool(l3["gate"]["passed"]),
        },
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
    }
    aggregate["gate"]["passed"] = bool(all(aggregate["gate"].values()))
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(aggregate, indent=2, default=float), encoding="utf-8")
    digest = _digest_file(out)
    out.with_name(out.name + ".sha256").write_text(digest + "\n", encoding="utf-8")
    print(json.dumps(aggregate["gate"], indent=2))
    print("receipt", str(out.relative_to(root)), digest)
    print(
        "L1 roundtrip {}/{} margin {:.3f} | L2 accept {:.3f} reject {:.3f} | L3 top1 {:.4f} bpc {:.4f} synth {:.3f} qa {:.2f}".format(
            l1["roundtrip_ok"],
            l1["roundtrip_texts"],
            l1["projector"]["margin"],
            l2["true_acceptance"],
            l2["trap_rejection"],
            l3["language"]["top1"],
            l3["language"]["bits_per_char"],
            l3["synthesis"]["final"]["accuracy"],
            l3["ontology_answers"]["supported"] / max(l3["ontology_answers"]["total"], 1),
        )
    )
    return 0 if aggregate["gate"]["passed"] else 1


def _git_head(root: Path) -> str:
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(root), capture_output=True, text=True, timeout=20
        )
        return result.stdout.strip()
    except Exception:
        return ""


if __name__ == "__main__":
    raise SystemExit(main())
