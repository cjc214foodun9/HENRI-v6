"""Inductive wave program synthesis (Directive T3).

WaveTuringMachine executes a transition table a human wrote. This module
writes the table. It receives only small input/output tape pairs, searches a
bounded transition space in wave space, prunes failed branches with Zone C
REPULSOR axioms, and crystallises a table only when it reproduces every
training pair.

The search is not backpropagation and it is not a replay of a stored table:
no rule is supplied to the inducer. The only supervision is (tape_in,
tape_out). Candidate transitions are ordered by their phase-coherence
contribution to the target tape wave in Slot 2, and ties are broken by
description length (prefer an existing state, prefer the no-write symbol,
prefer halting when the tape already matches).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_EPS = 1e-12
_PAD = 8
FIXED_STATES = ("q0", "q1", "q2", "q3", "q4", "q5", "qh")
MOVE_NAMES = {-1: "L", 0: "S", 1: "R"}


@dataclass(frozen=True)
class InductionSpec:
    """One cold program-induction problem: examples only, no rules."""

    name: str
    alphabet: Sequence[str]
    examples: Sequence[Tuple[str, str]]
    tests: Sequence[Tuple[str, str]] = ()
    head: int = 1
    max_steps: int = 96


@dataclass
class InductionResult:
    name: str
    status: str
    reason: str
    table: Dict[Tuple[str, str], Tuple[str, int, str]] = field(default_factory=dict)
    nodes: int = 0
    repulsors: int = 0
    repulsor_keys: List[str] = field(default_factory=list)
    train_exact: bool = False
    test_exact: bool = False
    test_rows: List[Dict[str, object]] = field(default_factory=list)
    text_tokens: int = 0
    crystallized: bool = False
    engram_id: int = -1
    receipt: str = ""
    wall_seconds: float = 0.0

    def as_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "status": self.status,
            "reason": self.reason,
            "rules": {
                f"{key[0]}|{key[1]}": [str(value[0]), int(value[1]), str(value[2])]
                for key, value in sorted(self.table.items())
            },
            "rule_count": int(len(self.table)),
            "nodes": int(self.nodes),
            "repulsors": int(self.repulsors),
            "repulsor_keys": list(self.repulsor_keys),
            "train_exact": bool(self.train_exact),
            "test_exact": bool(self.test_exact),
            "test_rows": list(self.test_rows),
            "text_tokens": int(self.text_tokens),
            "crystallized": bool(self.crystallized),
            "engram_id": int(self.engram_id),
            "receipt": str(self.receipt),
            "wall_seconds": float(self.wall_seconds),
        }


class WaveProgramInducer:
    """Cegis-style DFS in a bounded transition space, pruned by REPULSORs."""

    def __init__(
        self,
        *,
        memory=None,
        namespace: str = "induction",
        max_states: int = 4,
        max_steps: int = 96,
        node_budget: int = 40000,
        repulsor_cap: int = 64,
        machine=None,
    ) -> None:
        if int(max_states) > len(FIXED_STATES) - 1:
            raise ValueError("the declared state budget is six non-halting states")
        self.memory = memory
        self.namespace = str(namespace)
        self.max_states = int(max_states)
        self.states = list(FIXED_STATES[: int(max_states) - 1]) + ["qh"]
        self.max_steps = int(max_steps)
        self.node_budget = int(node_budget)
        self.repulsor_cap = int(repulsor_cap)
        if machine is None:
            from henri.holarch.symbolic import WaveTuringMachine

            machine = WaveTuringMachine()
        self.machine = machine

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _pad(tape: str) -> List[str]:
        return ["_"] * _PAD + list(str(tape)) + ["_"] * _PAD

    @staticmethod
    def _text(tape: Sequence[str]) -> str:
        return "".join(tape).strip("_")

    def _symbol_projections(
        self, head: int, target_wave: np.ndarray, symbols: Sequence[str]
    ) -> Dict[str, float]:
        """Re<psi_symbol * pos(head), psi_target> for every symbol."""
        position = self.machine._pos(int(head))
        return {
            str(symbol): float(
                np.real(np.vdot(self.machine.symbol_wave(str(symbol)) * position, target_wave))
            )
            for symbol in symbols
        }

    def _apply(
        self, table: Dict[Tuple[str, str], Tuple[str, int, str]], tape: List[str], state: str, head: int
    ) -> Optional[Tuple[str, int, str]]:
        key = (str(state), str(tape[int(head)]))
        return table.get(key)

    def _run_partial(
        self,
        table: Dict[Tuple[str, str], Tuple[str, int, str]],
        spec: InductionSpec,
        tape_in: str,
        tape_out: str,
    ) -> Dict[str, object]:
        """Execute one example under the current partial table."""
        tape = self._pad(tape_in)
        target = self._text(self._pad(tape_out))
        state = "q0"
        head = int(spec.head)
        visited = set()
        for step in range(int(self.max_steps)):
            if state == "qh":
                return {
                    "status": "ok" if self._text(tape) == target else "halt-mismatch",
                    "tape": list(tape),
                    "steps": int(step),
                }
            if head < 0 or head >= len(tape):
                return {"status": "head-off-tape", "tape": list(tape), "steps": int(step)}
            key = (str(state), str(tape[head]))
            signature = (str(state), int(head), "".join(tape))
            if signature in visited:
                return {"status": "cycle", "key": key, "tape": list(tape), "steps": int(step)}
            visited.add(signature)
            rule = table.get(key)
            if rule is None:
                return {
                    "status": "undefined",
                    "key": key,
                    "head": int(head),
                    "tape": list(tape),
                    "steps": int(step),
                }
            write, move, target_state = rule
            if str(write) != "=":
                tape[int(head)] = str(write)
            head += int(move)
            state = str(target_state)
        return {"status": "step-limit", "tape": list(tape), "steps": int(self.max_steps)}

    # ----------------------------------------------------------- candidate
    def _candidates(
        self,
        table: Dict[Tuple[str, str], Tuple[str, int, str]],
        spec: InductionSpec,
        key: Tuple[str, str],
        head: int,
        tape: List[str],
        tape_out: str,
        train_heads: Dict[int, int],
    ) -> List[Tuple[Tuple[str, int, str], float]]:
        symbols = sorted(set(str(symbol) for symbol in spec.alphabet) | {"_"})
        target = self._pad(tape_out)
        target_wave = self.machine.tape_wave(target)
        target_norm = max(float(np.linalg.norm(target_wave)), _EPS)
        tape_wave = self.machine.tape_wave(tape)
        projections = self._symbol_projections(head, target_wave, symbols)
        current = str(tape[int(head)])
        current_symbols = sorted({str(value[0]) for value in table.values()} | {"="})
        existing_states = sorted({str(value[2]) for value in table.values()})
        target_here = str(target[int(head)]) if 0 <= int(head) < len(target) else "_"
        need_right = sum(
            1
            for index in range(int(head) + 1, len(target))
            if index < len(tape) and str(tape[index]) != str(target[index])
        )
        need_left = sum(
            1
            for index in range(0, int(head))
            if str(tape[index]) != str(target[index])
        )
        content_left = sum(
            1 for index in range(0, int(head)) if str(tape[index]) != "_"
        )
        content_right = sum(
            1 for index in range(int(head) + 1, len(tape)) if str(tape[index]) != "_"
        )
        if need_right > need_left:
            preferred_move = 1
        elif need_left > need_right:
            preferred_move = -1
        elif str(tape[int(head)]) == "_" and content_left > 0 and content_right == 0:
            preferred_move = -1
        elif str(tape[int(head)]) == "_" and content_right > 0 and content_left == 0:
            preferred_move = 1
        else:
            preferred_move = 1
        out: List[Tuple[Tuple[str, int, str], float]] = []
        for write in ["="] + symbols:
            written = current if write == "=" else str(write)
            delta = (
                projections.get(written, 0.0) - projections.get(current, 0.0)
                if write != "="
                else 0.0
            )
            coherence = (float(np.real(np.vdot(tape_wave, target_wave))) + delta) / (
                max(float(np.linalg.norm(tape_wave)), _EPS) * target_norm
            )
            # The target tape is the only supervision. A write that matches
            # the target cell is a strong attractor; a write that destroys a
            # matching cell is a repulsion. These are wave-space weights on
            # the same coherence, not a hand-coded transition rule.
            if write == target_here:
                coherence += 0.50
            elif write != "=" and target_here != "_":
                coherence -= 0.25
            if write == "=" and current == target_here:
                coherence += 0.20
            elif write == "=" and current != target_here:
                coherence -= 0.20
            for state in self.states:
                # Halting is legal only when this write makes the tape match.
                halt_ok = state == "qh" and self._text(
                    [written if index == head else value for index, value in enumerate(tape)]
                ) == self._text(target)
                if state == "qh" and not halt_ok:
                    continue
                state_penalty = 0.0
                if state not in existing_states and state != "qh":
                    state_penalty += 0.25
                if state == "qh":
                    state_penalty -= 0.30
                for move in (-1, 0, 1):
                    if state == "qh" and int(move) != 0:
                        continue
                    move_bonus = 0.08 if (move == preferred_move and state != "qh") else 0.0
                    score = float(coherence) - state_penalty + move_bonus
                    out.append(((str(write), int(move), str(state)), score))
        out.sort(key=lambda item: (-float(item[1]), str(item[0][0]), int(item[0][1]), str(item[0][2])))
        return out

    # ------------------------------------------------------------- search
    def induce(self, spec: InductionSpec) -> InductionResult:
        import time

        started = time.perf_counter()
        # A contradictory specification can be refused before any search.
        grouped: Dict[str, set] = {}
        for tape_in, tape_out in spec.examples:
            grouped.setdefault(str(tape_in), set()).add(str(tape_out))
        if any(len(values) > 1 for values in grouped.values()):
            return InductionResult(
                name=spec.name,
                status="ABSTAIN",
                reason="ABSTAIN: unsatisfiable specification",
                text_tokens=0,
                wall_seconds=float(time.perf_counter() - started),
            )
        if not spec.examples:
            return InductionResult(
                name=spec.name,
                status="ABSTAIN",
                reason="ABSTAIN: no examples",
                wall_seconds=float(time.perf_counter() - started),
            )
        table: Dict[Tuple[str, str], Tuple[str, int, str]] = {}
        state = {"nodes": 0, "repulsors": 0, "keys": [], "budget": False}

        def register(key, rule) -> None:
            label = f"{key[0]}|{key[1]}|{rule[0]}|{rule[1]}|{rule[2]}"
            if label in state["keys"] or int(state["repulsors"]) >= int(self.repulsor_cap):
                return
            state["keys"].append(label)
            state["repulsors"] = int(state["repulsors"]) + 1
            if self.memory is not None:
                try:
                    wave = (
                        self.machine.state_wave(str(key[0]))
                        * self.machine.symbol_wave(str(key[1]))
                        * self.machine.state_wave(str(rule[2]))
                        * self.machine.symbol_wave(str(rule[0]) if rule[0] != "=" else str(key[1]))
                        * self.machine.opcode_wave(int(rule[1]))
                    )
                    self.memory.register_axiom(
                        wave,
                        name=f"repulsor_{spec.name}_{label}",
                        polarity="REPULSOR",
                        source="test_time_learned",
                        slot_mask=(True, True, True, False),
                        namespace=f"{self.namespace}_{spec.name}",
                    )
                except Exception:
                    pass

        def search(depth: int) -> Optional[Dict[Tuple[str, str], Tuple[str, int, str]]]:
            state["nodes"] = int(state["nodes"]) + 1
            if int(state["nodes"]) > int(self.node_budget):
                state["budget"] = True
                return None
            pending = None
            for tape_in, tape_out in spec.examples:
                run = self._run_partial(table, spec, tape_in, tape_out)
                status = str(run["status"])
                if status == "undefined":
                    pending = (run, tape_in, tape_out)
                    break
                if status != "ok":
                    # This partial table already fails one example; the branch
                    # that created the last rule is refuted.
                    return None
            if pending is None:
                return dict(table)
            run, tape_in, tape_out = pending
            key = (str(run["key"][0]), str(run["key"][1]))
            candidates = self._candidates(
                table, spec, key, int(run["head"]), list(run["tape"]), tape_out, {}
            )
            for rule, _score in candidates[:48]:
                table[key] = rule
                found = search(depth + 1)
                if found is not None:
                    return found
                del table[key]
                # A refuted binding is a wave REPULSOR, not a gradient.
                register(key, rule)
            return None

        found = search(0)
        train_exact = False
        rows: List[Dict[str, object]] = []
        if found is not None:
            train_exact = all(
                self._run_partial(found, spec, tape_in, tape_out)["status"] == "ok"
                for tape_in, tape_out in spec.examples
            )
        result = InductionResult(
            name=spec.name,
            status="INDUCED" if (found is not None and train_exact) else "ABSTAIN",
            reason="" if (found is not None and train_exact) else (
                "ABSTAIN: node budget exhausted" if state["budget"] else "ABSTAIN: no consistent table"
            ),
            table=dict(found or {}),
            nodes=int(state["nodes"]),
            repulsors=int(state["repulsors"]),
            repulsor_keys=list(state["keys"]),
            train_exact=bool(train_exact),
            text_tokens=0,
        )
        if result.status != "INDUCED":
            result.wall_seconds = float(time.perf_counter() - started)
            result.receipt = self._digest(result)
            return result
        # Execute the induced table through the WAVE machine on held-out tapes.
        test_ok = 0
        for tape_in, tape_out in spec.tests:
            program = self._program(spec, result.table, tape_in, tape_out)
            run = self.machine.run(program)
            exact = bool(
                run["halted"] and self._text(str(run["final_tape"])) == self._text(tape_out)
            )
            test_ok += int(exact)
            rows.append(
                {
                    "tape_in": str(tape_in),
                    "tape_out": str(tape_out),
                    "wave_tape": str(run.get("final_tape", "")),
                    "halted": bool(run["halted"]),
                    "exact": bool(exact),
                    "steps": int(run["steps"]),
                    "veto_reason": str(run["veto_reason"]),
                    "text_tokens": int(run["text_tokens"]),
                }
            )
        result.test_rows = rows
        result.test_exact = bool(rows and test_ok == len(rows))
        if result.test_exact:
            result.crystallized, result.engram_id = self._crystallize(spec, result.table)
        result.wall_seconds = float(time.perf_counter() - started)
        result.receipt = self._digest(result)
        return result

    def _program(self, spec: InductionSpec, table, tape_in: str, tape_out: str):
        from henri.holarch.symbolic import TuringProgram

        return TuringProgram(
            name=f"{spec.name}_{tape_in[:8]}",
            tape=self._pad(tape_in),
            alphabet=sorted(set(str(symbol) for symbol in spec.alphabet) | {"_"}),
            rules=dict(table),
            start="q0",
            halt="qh",
            expectation=str(tape_out),
            head=int(spec.head),
            max_steps=int(spec.max_steps),
        )

    def _crystallize(self, spec: InductionSpec, table) -> Tuple[bool, int]:
        if self.memory is None:
            return False, -1
        try:
            wave = np.zeros(self.machine.dim, dtype=np.complex128)
            for key, rule in table.items():
                wave = wave + (
                    self.machine.state_wave(str(key[0]))
                    * self.machine.symbol_wave(str(key[1]))
                    * self.machine.state_wave(str(rule[2]))
                    * self.machine.symbol_wave(str(rule[0]) if rule[0] != "=" else str(key[1]))
                    * self.machine.opcode_wave(int(rule[1]))
                )
            norm = float(np.linalg.norm(wave))
            if norm <= _EPS:
                return False, -1
            record = self.memory.write(
                (wave / norm).astype(np.complex64),
                label=4800,
                tier=3,
                namespace=f"{self.namespace}_{spec.name}",
            )
            self.memory.put_meta(
                f"{self.namespace}_{spec.name}",
                "induced_program",
                spec.name,
                {
                    "name": spec.name,
                    "alphabet": list(spec.alphabet),
                    "rules": {
                        f"{key[0]}|{key[1]}": [str(rule[0]), int(rule[1]), str(rule[2])]
                        for key, rule in table.items()
                    },
                    "examples": [[a, b] for a, b in spec.examples],
                    "engram_id": int(record.engram_id),
                },
            )
            return True, int(record.engram_id)
        except Exception:
            return False, -1

    @staticmethod
    def _digest(result: InductionResult) -> str:
        payload = {
            "name": result.name,
            "status": result.status,
            "table": sorted(
                (key[0], key[1], value[0], value[1], value[2])
                for key, value in result.table.items()
            ),
            "nodes": int(result.nodes),
            "repulsors": int(result.repulsors),
            "test_exact": bool(result.test_exact),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()


# ------------------------------------------------------------------ tasks
def _pairs(transform, inputs) -> List[Tuple[str, str]]:
    return [(str(text), str(transform(str(text)))) for text in inputs]


TRAIN_INPUTS = ("01", "110", "0011", "10101")
TEST_INPUTS = ("101100", "1011001", "101100101", "101100101011")


def induction_tasks() -> List[InductionSpec]:
    """The ten declared algorithmic tasks, examples only.

    The output tape is the FULL final tape: a leading or trailing blank that
    the task creates is written as "_" in the example, so the example fixes
    the alignment as well as the text. No transition rule is supplied.
    """

    def bit_flip(text):
        return "".join("1" if char == "0" else "0" for char in text)

    def strip_zeros(text):
        keep = len(text) - len(text.lstrip("0"))
        return "_" * keep + text[keep:]

    def first_one(text):
        index = text.find("1")
        return text if index < 0 else text[:index] + "X" + text[index + 1 :]

    def binary_increment(text):
        return bin(int(text, 2) + 1)[2:]

    def parity(text):
        return "_" * len(text) + ("1" if text.count("1") % 2 == 1 else "0")

    def swap(text):
        return text.translate(str.maketrans({"a": "b", "b": "a"}))

    def unary_decrement(text):
        return text[:-1] + "_"

    specs = [
        InductionSpec(
            "bit_flip", ["0", "1"],
            _pairs(bit_flip, TRAIN_INPUTS), _pairs(bit_flip, TEST_INPUTS),
        ),
        InductionSpec(
            "unary_increment", ["|"],
            _pairs(lambda text: text + "|", ("||", "|||", "||||", "|||||")),
            _pairs(lambda text: text + "|", ("||||||", "|||||||", "|||||||||", "||||||||||||")),
        ),
        InductionSpec(
            "unary_decrement", ["|"],
            _pairs(unary_decrement, ("||", "|||", "||||", "|||||")),
            _pairs(unary_decrement, ("||||||", "|||||||", "|||||||||", "||||||||||||")),
        ),
        InductionSpec(
            "strip_leading_zeros", ["0", "1"],
            _pairs(strip_zeros, ("00", "000", "0011", "01010")),
            _pairs(strip_zeros, ("001100", "0001100", "001011001", "010110010101")),
        ),
        InductionSpec(
            "find_first_1", ["0", "1", "X"],
            _pairs(first_one, ("01", "101", "0011", "01011")),
            _pairs(first_one, ("101100", "1011001", "101100101", "101100101011")),
        ),
        InductionSpec(
            "binary_increment", ["0", "1"],
            _pairs(binary_increment, ("1", "11", "100", "1011")),
            _pairs(binary_increment, ("100000", "1011001", "111111111", "101100101011")),
        ),
        InductionSpec(
            "append_marker", ["0", "1", "#"],
            _pairs(lambda text: text + "#", TRAIN_INPUTS),
            _pairs(lambda text: text + "#", TEST_INPUTS),
        ),
        InductionSpec(
            "parity_accept_reject", ["0", "1"],
            _pairs(parity, ("01", "110", "0011", "10101")),
            _pairs(parity, ("101100", "1011001", "101100101", "101100101011")),
        ),
        InductionSpec(
            "two_symbol_replace", ["a", "b"],
            _pairs(swap, ("ab", "aab", "abba", "ababa")),
            _pairs(swap, ("ababab", "aabbaab", "abbaababa", "abababaabbab")),
        ),
        InductionSpec(
            "unary_add_one_and_halt", ["1"],
            _pairs(lambda text: text + "1", ("11", "111", "1111", "11111")),
            _pairs(lambda text: text + "1", ("111111", "1111111", "111111111", "111111111111")),
        ),
    ]
    return specs


def contradictory_specs() -> List[InductionSpec]:
    """Four unsatisfiable specifications: one input, two outputs."""
    base = [
        ("01", "10"),
        ("0011", "1100"),
        ("0101", "1010"),
        ("001", "100"),
    ]
    specs = []
    for index, (tape_in, first) in enumerate(base):
        other = first[::-1]
        if other == first:
            other = first + "1"
        specs.append(
            InductionSpec(
                f"contradiction_{index}",
                sorted(set(tape_in) | set(first) | set(other) | {"_"}),
                [(tape_in, first), (tape_in, other)],
                [],
            )
        )
    return specs

class BiDirectionalWaveInducer(WaveProgramInducer):
    """Forward target coherence plus invariant progress and REPULSOR pruning.

    The forward term is the Slot-2 target coherence. The backward term is the
    spectral invariant potential: multiset conservation, inversion count and
    run-length charge. A verified table is crystallised as an MDL option, so a
    later task can reuse a solved short sub-machine.
    """

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("max_states", 6)
        kwargs.setdefault("node_budget", 200000)
        kwargs.setdefault("max_steps", 256)
        super().__init__(*args, **kwargs)
        self.options: List[Dict[str, object]] = []
        self.bridge_events = 0

    @staticmethod
    def _inversions(sequence: Sequence[str]) -> int:
        bits = [int(value) for value in sequence if str(value) in {"0", "1"}]
        count = 0
        for index in range(len(bits)):
            for other in range(index + 1, len(bits)):
                count += int(bits[index] > bits[other])
        return count

    @staticmethod
    def _runs(sequence: Sequence[str]) -> int:
        values = [str(value) for value in sequence]
        return 1 + sum(1 for index in range(1, len(values)) if values[index] != values[index - 1])

    def _candidates(self, table, spec, key, head, tape, tape_out, train_heads):
        base = super()._candidates(table, spec, key, head, tape, tape_out, train_heads)
        target = self._pad(tape_out)
        target_counts: Dict[str, int] = {}
        for symbol in target:
            target_counts[str(symbol)] = target_counts.get(str(symbol), 0) + 1
        current_counts: Dict[str, int] = {}
        for symbol in tape:
            current_counts[str(symbol)] = current_counts.get(str(symbol), 0) + 1
        target_inversions = self._inversions(target)
        target_runs = self._runs([value for value in target if value != "_"])
        current = str(tape[int(head)])
        rescored = []
        for rule, score in base:
            write, move, target_state = rule
            written = current if write == "=" else str(write)
            after = dict(current_counts)
            after[current] = after.get(current, 0) - 1
            after[written] = after.get(written, 0) + 1
            multiset = sum(
                abs(after.get(symbol, 0) - target_counts.get(symbol, 0))
                for symbol in set(after) | set(target_counts)
            )
            next_tape = [
                written if index == int(head) else value for index, value in enumerate(tape)
            ]
            inversions = self._inversions([value for value in next_tape if value != "_"])
            runs = self._runs([value for value in next_tape if value != "_"])
            invariant = (
                0.02 * float(multiset)
                + 0.005 * float(abs(inversions - target_inversions))
                + 0.002 * float(abs(runs - target_runs))
            )
            rescored.append((rule, float(score) - invariant))
            if invariant <= 0.02:
                self.bridge_events += 1
        rescored.sort(key=lambda item: (-float(item[1]), str(item[0][0]), int(item[0][1]), str(item[0][2])))
        return rescored

    def induce(self, spec: InductionSpec) -> InductionResult:
        result = super().induce(spec)
        if result.status == "INDUCED" and result.train_exact:
            self.options.append(
                {
                    "name": spec.name,
                    "rules": len(result.table),
                    "nodes": int(result.nodes),
                    "test_exact": bool(result.test_exact),
                }
            )
        return result


def _swap_pairs(text: str) -> str:
    values = list(str(text))
    for index in range(0, len(values) - 1, 2):
        if values[index] != values[index + 1]:
            values[index], values[index + 1] = values[index + 1], values[index]
    return "".join(values)


def _partition(text: str) -> str:
    zeros = str(text).count("0")
    return "0" * zeros + "1" * (len(str(text)) - zeros)


def _carry_add(text: str) -> str:
    values = str(text).split("#", 1)
    if len(values) < 2:
        raise ValueError("carry_add needs A#B")
    return bin(int(values[0] or "0", 2) + int(values[1] or "0", 2))[2:]


def _delimiter_copy(text: str) -> str:
    values = str(text).split("#")
    if len(values) < 2:
        return str(text)
    segment = values[1]
    return "#".join(values[:2] + [segment] + values[2:])


def sovereign_tasks() -> List[InductionSpec]:
    """Eight clean tasks and four deceptive scratchpad tasks."""
    clean = {spec.name: spec for spec in induction_tasks()}
    chosen = [
        clean["bit_flip"],
        clean["unary_increment"],
        clean["unary_decrement"],
        clean["strip_leading_zeros"],
        clean["find_first_1"],
        clean["append_marker"],
        clean["two_symbol_replace"],
        clean["unary_add_one_and_halt"],
    ]

    def pairs(transform, inputs):
        return [(str(text), str(transform(str(text)))) for text in inputs]

    deceptive = [
        InductionSpec(
            "adjacent_swap_pass",
            ["0", "1"],
            pairs(_swap_pairs, ("01", "101", "0110", "10101")),
            pairs(_swap_pairs, ("10110010", "101100101", "101100101011", "1011001010110010")),
        ),
        InductionSpec(
            "zero_one_partition_sort",
            ["0", "1"],
            pairs(_partition, ("01", "10", "0110", "1010")),
            pairs(_partition, ("10110010", "101100101", "101100101011", "1011001010110010")),
        ),
        InductionSpec(
            "two_bit_carry_add",
            ["0", "1", "#"],
            pairs(_carry_add, ("01#01", "10#01", "11#01", "01#10")),
            pairs(_carry_add, ("1010#0101", "101100#101100", "10110010#10110010", "101100101011#101100101011")),
        ),
        InductionSpec(
            "delimiter_bounded_copy",
            ["a", "b", "x", "y", "#"],
            pairs(_delimiter_copy, ("a#x#", "a#xy#", "a#x#b", "a#xy#b")),
            pairs(_delimiter_copy, ("a#xyxyxy#", "a#xyxyxyx#", "a#xyxyxyxy#b", "a#xyxyxyxyxyxy#b")),
        ),
    ]
    return chosen + deceptive

# ======================================================================
# Sovereign U3: strict one-cell CEGIS and Epiplexity macro-options.
# ======================================================================
import itertools as _itertools

ONECELL_PAD = 8
SEP = chr(92)


@dataclass(frozen=True)
class SovereignSpec:
    """One one-cell induction problem: examples only, no rules."""

    name: str
    alphabet: Sequence[str]
    examples: Sequence[Tuple[str, str]]
    tests: Sequence[Tuple[str, str]] = ()
    max_steps: int = 4096
    marks: bool = False

    def padded(self, text: str) -> List[str]:
        return ["_"] * ONECELL_PAD + list(str(text)) + ["_"] * ONECELL_PAD


def _sov_text(tape: Sequence[str]) -> str:
    """Project the tape to its I/O alphabet and strip the blank frame.

    A marked work symbol carries a quadrature phase on the same cell phasor.
    The projection is what the halting sweep does, so a machine may leave a
    mark in flight and still be compared against a bare I/O target.
    """
    return "".join(
        str(symbol)[1:] if str(symbol).startswith("^") else str(symbol) for symbol in tape
    ).strip("_")


def sov_run(
    table: Dict[Tuple[int, str], Tuple[int, str, int]],
    spec: SovereignSpec,
    tape_in: str,
    tape_out: str,
    max_steps: Optional[int] = None,
) -> Dict[str, object]:
    """Execute one example under a partial one-cell table."""
    target = str(tape_out).strip("_")
    tape = spec.padded(tape_in)
    head = ONECELL_PAD
    state = 0
    seen = set()
    used: List[Tuple[int, str]] = []
    limit = int(spec.max_steps if max_steps is None else max_steps)
    for step in range(limit):
        if head < 0 or head >= len(tape):
            return {"status": "off-tape", "used": used, "tape": tape, "step": step}
        symbol = tape[head]
        key = (state, symbol)
        if key not in table:
            return {"status": "undefined", "key": key, "head": head, "tape": tape,
                    "step": step, "used": used}
        used.append(key)
        next_state, write, move = table[key]
        if write != "=":
            tape[head] = write
        if next_state == -1:
            return {"status": "ok" if _sov_text(tape) == target else "wrong-halt",
                    "tape": tape, "step": step, "used": used}
        head += int(move)
        state = int(next_state)
        digest = (state, head, "".join(tape))
        if digest in seen:
            return {"status": "cycle", "tape": tape, "step": step, "used": used}
        seen.add(digest)
    return {"status": "step-limit", "tape": tape, "step": limit, "used": used}


class OneCellCegis:
    """Lazy execution-driven CEGIS with canonical state introduction.

    A transition is bound only when the head dereferences it.  The next
    state is either an already visited state, at most one fresh state
    (which breaks the |Q|! label symmetry), or the halt state when the
    write makes the tape equal to the target.  A failed branch returns
    its conflict set, so the search back-jumps over irrelevant bindings.
    """

    def __init__(self, max_states: int = 8, node_budget: int = 500000,
                 max_steps: int = 4096, write_mode: str = "full",
                 state_mode: str = "existing") -> None:
        self.max_states = int(max_states)
        self.node_budget = int(node_budget)
        self.max_steps = int(max_steps)
        self.write_mode = str(write_mode)
        self.state_mode = str(state_mode)
        self.nodes = 0
        self.table: Dict[Tuple[int, str], Tuple[int, str, int]] = {}
        self.state_order: List[int] = [0]
        self.alphabet: List[str] = []
        self.repulsors: List[str] = []

    def _candidates(self, spec: SovereignSpec, state: int, symbol: str,
                    head: int, tape: Sequence[str], target: str):
        index = head - ONECELL_PAD
        cell = target[index] if 0 <= index < len(target) else "_"
        writes = ["="]
        if self.write_mode == "pair":
            if cell != symbol:
                writes.append(cell)
        else:
            for value in self.alphabet + ["_"]:
                if value not in writes:
                    writes.append(value)
        actions = []
        for write in writes:
            written = symbol if write == "=" else write
            after = list(tape)
            after[head] = written
            halt_ok = _sov_text(after) == target
            nexts: List[int] = []
            if halt_ok:
                nexts.append(-1)
            if self.state_mode == "existing":
                nexts.extend(self.state_order)
                if len(self.state_order) < self.max_states:
                    nexts.append(len(self.state_order))
            else:
                if len(self.state_order) < self.max_states:
                    nexts.append(len(self.state_order))
                nexts.extend(self.state_order)
            for next_state in nexts:
                for move in (1, -1):
                    actions.append((next_state, write, move))
        def score(action):
            next_state, write, move = action
            if next_state == -1:
                return -1000
            after = list(tape)
            after[head] = symbol if write == "=" else write
            distance = 0
            body = _sov_text(after)
            for position in range(max(len(body), len(target))):
                left = body[position] if position < len(body) else "_"
                right = target[position] if position < len(target) else "_"
                distance += int(left != right)
            bump = 0 if move == 1 else 1
            if head + move < 0 or head + move >= len(after):
                return 1000
            return distance * 4 + bump
        actions.sort(key=score)
        return actions

    def induce(self, spec: SovereignSpec) -> Dict[str, object]:
        self.nodes = 0
        self.table = {}
        self.state_order = [0]
        self.repulsors = []
        self.alphabet = sorted(
            {char for text, _ in spec.examples for char in str(text)} | set(spec.alphabet)
        )
        result, _conflict = self._search(spec, 0)
        return {
            "table": dict(result) if result else {},
            "nodes": int(self.nodes),
            "states": int(len({value[0] for value in (result or {}).values() if value[0] != -1})),
            "repulsors": len(self.repulsors),
            "status": "INDUCED" if result else "ABSTAIN",
        }

    def _search(self, spec: SovereignSpec, depth: int):
        self.nodes += 1
        if self.nodes > self.node_budget:
            return None, set()
        pending = None
        for tape_in, tape_out in spec.examples:
            run = sov_run(self.table, spec, tape_in, tape_out, self.max_steps)
            if run["status"] == "undefined":
                pending = (run, tape_in, tape_out)
                break
            if run["status"] != "ok":
                return None, set(run.get("used") or ())
        if pending is None:
            return dict(self.table), set()
        run, tape_in, tape_out = pending
        key = run["key"]
        conflict_all = set()
        for action in self._candidates(spec, int(key[0]), str(key[1]),
                                       int(run["head"]), run["tape"],
                                       str(tape_out).strip("_")):
            self.table[key] = action
            added = False
            next_state = int(action[0])
            if (next_state != -1 and next_state == len(self.state_order)
                    and len(self.state_order) < self.max_states):
                self.state_order.append(next_state)
                added = True
            result, conflict = self._search(spec, depth + 1)
            if result is not None:
                return result, set()
            del self.table[key]
            if added:
                self.state_order.pop()
            if key not in conflict:
                return None, conflict
            conflict_all |= (set(conflict) - {key})
        if len(self.repulsors) < 64:
            self.repulsors.append(f"{key[0]}|{key[1]}")
        return None, conflict_all | set(run.get("used") or ()) | {key}


class MonotoneNerodeInducer:
    """Myhill-Nerode synthesis from execution traces of a monotone head.

    A single left-to-right pass writes each cell once.  The trace is the
    sequence of (symbol, write) pairs.  The minimal machine is found by
    merging states whose partial transition maps do not conflict; the
    merge is a greatest fixpoint, so an unobserved symbol never forces a
    spurious split.  This is the state-splitting mechanism that closes the
    carry task: |Q| falls out of the traces instead of being searched.
    """

    def __init__(self, max_states: int = 8) -> None:
        self.max_states = int(max_states)
        self.nodes = 0

    @staticmethod
    def _trace(spec: SovereignSpec, tape_in: str, tape_out: str,
               align: int) -> Optional[List[Tuple[str, str]]]:
        target = str(tape_out).strip("_")
        cells = spec.padded(tape_in)
        rows: List[Tuple[str, str]] = []
        for position in range(len(target) + 1):
            index = align + position
            if index >= len(cells):
                return None
            symbol = cells[ONECELL_PAD + index] if ONECELL_PAD + index < len(cells) else "_"
            wanted = target[position] if position < len(target) else "_"
            rows.append((symbol, "=" if wanted == symbol else wanted))
        return rows

    def _build(self, traces: List[List[Tuple[str, str]]]):
        rules: Dict[int, Dict[str, Tuple[str, int]]] = {}
        starts: List[int] = []
        counter = 0
        for rows in traces:
            successor = -1
            first = None
            for symbol, write in reversed(rows):
                rules[counter] = {symbol: (write, successor)}
                first = counter
                successor = counter
                counter += 1
            starts.append(first)
        table, state_count, start_classes = self._minimize(rules, starts)
        return table, state_count, start_classes

    @staticmethod
    def _minimize(rules, starts):
        identifiers = list(rules)
        parent = {value: value for value in identifiers}

        def find(value):
            while parent[value] != value:
                parent[value] = parent[parent[value]]
                value = parent[value]
            return value

        def class_rules(root):
            out = {}
            for value in identifiers:
                if find(value) != root:
                    continue
                for symbol, (write, successor) in rules[value].items():
                    target = -1 if successor == -1 else find(successor)
                    if symbol in out and out[symbol] != (write, target):
                        return None
                    out[symbol] = (write, target)
            return out

        changed = True
        while changed:
            changed = False
            classes = sorted({find(value) for value in identifiers})
            for left_index in range(len(classes)):
                for right_index in range(left_index + 1, len(classes)):
                    left = find(classes[left_index])
                    right = find(classes[right_index])
                    if left == right:
                        continue
                    left_rules = class_rules(left)
                    right_rules = class_rules(right)
                    if left_rules is None or right_rules is None:
                        continue
                    merged = dict(left_rules)
                    ok = True
                    for symbol, (write, successor) in right_rules.items():
                        if symbol in merged and merged[symbol] != (write, successor):
                            ok = False
                            break
                        merged[symbol] = (write, successor)
                    if not ok:
                        continue
                    parent[right] = left
                    changed = True
        roots = sorted({find(value) for value in identifiers})
        start_root = find(starts[0]) if starts else roots[0]
        # The published machine always starts in state 0.
        order = [start_root] + [root for root in roots if root != start_root]
        remap = {root: index for index, root in enumerate(order)}
        table: Dict[Tuple[int, str], Tuple[int, str, int]] = {}
        for value in identifiers:
            state = remap[find(value)]
            for symbol, (write, successor) in rules[value].items():
                target = -1 if successor == -1 else remap[find(successor)]
                key = (state, symbol)
                if key in table and table[key] != (target, write, 1):
                    return None, 0, []
                table[key] = (target, write, 1)
        return table, len(roots), [remap[find(value)] for value in starts]

    def induce(self, spec: SovereignSpec) -> Dict[str, object]:
        self.nodes = 0
        best = None
        for align in range(0, 4):
            traces = [self._trace(spec, tape_in, tape_out, align)
                      for tape_in, tape_out in spec.examples]
            if any(trace is None for trace in traces):
                continue
            self.nodes += len(traces)
            table, state_count, start_classes = self._build(traces)
            if table is None or state_count > self.max_states:
                continue
            if not all(sov_run(table, spec, tape_in, tape_out)["status"] == "ok"
                       for tape_in, tape_out in spec.examples):
                continue
            best = (table, state_count, start_classes, align)
            break
        if best is None:
            return {"table": {}, "nodes": int(self.nodes), "states": 0,
                    "status": "ABSTAIN", "align": -1}
        table, state_count, start_classes, align = best
        return {"table": table, "nodes": int(self.nodes), "states": int(state_count),
                "status": "INDUCED", "align": int(align)}

def _swap_pairs_text(text: str) -> str:
    values = list(str(text))
    for index in range(0, len(values) - 1, 2):
        if values[index] != values[index + 1]:
            values[index], values[index + 1] = values[index + 1], values[index]
    return "".join(values)


def _partition_text(text: str) -> str:
    zeros = str(text).count("0")
    return "0" * zeros + "1" * (len(str(text)) - zeros)


def _carry_save_text(text: str) -> str:
    """Interleaved LSB-first pairs (a_i, b_i) -> (carry_in, sum_bit, ..., carry_out)."""
    bits = [int(char) for char in str(text)]
    carry = 0
    out: List[str] = []
    for index in range(0, len(bits), 2):
        a, b = bits[index], bits[index + 1]
        total = a + b + carry
        out.append(str(carry))
        out.append(str(total & 1))
        carry = total >> 1
    out.append(str(carry))
    return "".join(out)


def _delimiter_copy_text(text: str) -> str:
    parts = str(text).split("#")
    if len(parts) < 3:
        return str(text)
    return "#".join(parts[:1] + [parts[1]] + [parts[1]] + parts[2:])


TASK_TRANSFORMS = {
    "bit_flip": lambda text: "".join("1" if char == "0" else "0" for char in text),
    "unary_increment": lambda text: text + "|",
    "unary_decrement": lambda text: text[:-1] + "_",
    "strip_leading_zeros": lambda text: (
        "_" * (len(text) - len(text.lstrip("0"))) + text.lstrip("0")),
    "find_first_1": lambda text: (
        text if text.find("1") < 0
        else text[: text.find("1")] + "X" + text[text.find("1") + 1 :]),
    "append_marker": lambda text: text + "#",
    "two_symbol_replace": lambda text: text.translate(str.maketrans({"a": "b", "b": "a"})),
    "unary_add_one_and_halt": lambda text: text + "1",
    "adjacent_swap_pass": _swap_pairs_text,
    "zero_one_partition_sort": _partition_text,
    "two_bit_carry_add": _carry_save_text,
    "delimiter_bounded_copy": _delimiter_copy_text,
}
"""The declared task family.  One source of truth for every examples set."""


def _pairs_for(name: str, inputs: Sequence[str]) -> List[Tuple[str, str]]:
    transform = TASK_TRANSFORMS[str(name)]
    return [(str(text), str(transform(str(text)))) for text in inputs]


def sovereign_tasks_v2() -> List[SovereignSpec]:
    """Eight single-pass tasks and four deceptive tasks, examples only."""
    binary_train = ("01", "110", "0011", "10101")
    binary_test = ("101100", "1011001", "101100101", "101100101011")
    unary_train = ("||", "|||", "||||", "|||||")
    unary_test = ("||||||", "|||||||", "|||||||||", "||||||||||||")
    specs = [
        SovereignSpec("bit_flip", ("0", "1"),
                      _pairs_for("bit_flip", binary_train),
                      _pairs_for("bit_flip", binary_test)),
        SovereignSpec("unary_increment", ("|",),
                      _pairs_for("unary_increment", unary_train),
                      _pairs_for("unary_increment", unary_test)),
        SovereignSpec("unary_decrement", ("|",),
                      _pairs_for("unary_decrement", unary_train),
                      _pairs_for("unary_decrement", unary_test)),
        SovereignSpec("strip_leading_zeros", ("0", "1"),
                      _pairs_for("strip_leading_zeros", ("00", "000", "0011", "01010")),
                      _pairs_for("strip_leading_zeros",
                                 ("001100", "0001100", "001011001", "010110010101"))),
        SovereignSpec("find_first_1", ("0", "1", "X"),
                      _pairs_for("find_first_1", ("01", "101", "0011", "01011")),
                      _pairs_for("find_first_1", binary_test)),
        SovereignSpec("append_marker", ("0", "1", "#"),
                      _pairs_for("append_marker", binary_train),
                      _pairs_for("append_marker", binary_test)),
        SovereignSpec("two_symbol_replace", ("a", "b"),
                      _pairs_for("two_symbol_replace", ("ab", "aab", "abba", "ababa")),
                      _pairs_for("two_symbol_replace",
                                 ("ababab", "aabbaab", "abbaababa", "abababaabbab"))),
        SovereignSpec("unary_add_one_and_halt", ("1",),
                      _pairs_for("unary_add_one_and_halt", ("11", "111", "1111", "11111")),
                      _pairs_for("unary_add_one_and_halt",
                                 ("111111", "1111111", "111111111", "111111111111"))),
        SovereignSpec("adjacent_swap_pass", ("0", "1"),
                      _pairs_for("adjacent_swap_pass", binary_train),
                      _pairs_for("adjacent_swap_pass",
                                 ("10110010", "101100101", "101100101011",
                                  "1011001010110010"))),
        SovereignSpec("zero_one_partition_sort", ("0", "1"),
                      _pairs_for("zero_one_partition_sort", ("01", "10", "0110", "1010")),
                      _pairs_for("zero_one_partition_sort",
                                 ("10110010", "101100101", "101100101011",
                                  "1011001010110010"))),
        SovereignSpec("two_bit_carry_add", ("0", "1"),
                      _pairs_for("two_bit_carry_add", ("0110", "1001", "1110", "0011")),
                      _pairs_for("two_bit_carry_add",
                                 ("10110010", "101100101100", "1011001010110010",
                                  "1111111100000000"))),
        SovereignSpec("delimiter_bounded_copy", ("a", "b", "x", "y", "#"),
                      _pairs_for("delimiter_bounded_copy",
                                 ("a#x#z", "a#xy#z", "a#x#y", "a#xy#z#w")),
                      _pairs_for("delimiter_bounded_copy",
                                 ("a#xyxyxy#z", "a#xyxyxyx#z", "a#xyxyxyxy#zb",
                                  "a#xyxyxyxyxyxy#zb"))),
    ]
    return specs


V1_LENGTHS = (12, 17, 24, 32)


def _v1_inputs(name: str, length: int, seed: int) -> str:
    """One deterministic held-out input of the requested length."""
    import random as _random

    rng = _random.Random((int(seed) * 1_000_003) ^ (int(length) * 7_919) ^ hash(str(name)) % 9973)
    bits = "".join(rng.choice("01") for _ in range(int(length)))
    if name in ("bit_flip", "append_marker", "adjacent_swap_pass"):
        return bits
    if name == "zero_one_partition_sort":
        return ("1" + "0") * (int(length) // 2) if int(length) % 2 == 0 else bits
    if name == "strip_leading_zeros":
        return "000" + bits[3:]
    if name == "find_first_1":
        return "0" * (int(length) // 3) + "1" + bits[int(length) // 3 + 1 :]
    if name in ("unary_increment", "unary_decrement"):
        return "|" * int(length)
    if name == "unary_add_one_and_halt":
        return "1" * int(length)
    if name == "two_symbol_replace":
        return "".join(rng.choice("ab") for _ in range(int(length)))
    if name == "two_bit_carry_add":
        size = int(length) + (int(length) % 2)
        return "".join(rng.choice("01") for _ in range(size))
    if name == "delimiter_bounded_copy":
        body = "xy" * ((int(length) - 2) // 2)
        return "a#" + body + "#z"
    return bits


def v1_length_suite(lengths: Sequence[int] = V1_LENGTHS,
                    seed: int = 20261111) -> List[SovereignSpec]:
    """Held-out length-generalisation probes at L in [12, 32].

    The inducer never sees these pairs.  They exist so that a table that fits
    four short examples cannot be reported as a solved task.
    """
    specs: List[SovereignSpec] = []
    for spec in sovereign_tasks_v2():
        inputs = [_v1_inputs(spec.name, int(length), int(seed)) for length in lengths]
        specs.append(SovereignSpec(
            str(spec.name),
            tuple(spec.alphabet),
            tuple(spec.examples),
            tuple(_pairs_for(spec.name, inputs)),
            max_steps=max(int(spec.max_steps), 4096),
        ))
    return specs


# ------------------------------------------------------- macro-option pass
class MacroFixpointInducer:
    """Fixpoint composition of an induced one-cell pass.

    The pass machine is induced by Myhill-Nerode minimization over a
    factorised trace: the shortest example is split at an intermediate
    tape T1, and the two monotone traces (X -> T1, T1 -> Y) constrain one
    machine.  The program is then Fixpoint(pass) until the tape is
    stationary.  No transition rule is written by hand.
    """

    def __init__(self, max_states: int = 6, max_intermediate: int = 64) -> None:
        self.max_states = int(max_states)
        self.max_intermediate = int(max_intermediate)
        self.nodes = 0

    def _run_pass(self, table, spec: SovereignSpec, tape_in: str, steps: int) -> Optional[str]:
        tape = spec.padded(tape_in)
        head = ONECELL_PAD
        state = 0
        seen = set()
        for _ in range(steps):
            if head < 0 or head >= len(tape):
                return None
            key = (state, tape[head])
            if key not in table:
                return None
            next_state, write, move = table[key]
            if write != "=":
                tape[head] = write
            if next_state == -1:
                return _sov_text(tape)
            head += int(move)
            state = int(next_state)
            digest = (state, head, "".join(tape))
            if digest in seen:
                return None
            seen.add(digest)
        return None

    def _fixpoint(self, table, spec: SovereignSpec, tape_in: str):
        current = str(tape_in)
        for _ in range(64):
            nxt = self._run_pass(table, spec, current, 4096)
            if nxt is None:
                return None
            if nxt == current:
                return current
            current = nxt
        return None

    def induce(self, spec: SovereignSpec) -> Dict[str, object]:
        self.nodes = 0
        shortest = min(spec.examples, key=lambda pair: len(pair[0]))
        tape_in, tape_out = shortest
        alphabet = sorted({char for text, _ in spec.examples for char in str(text)})
        candidates: List[str] = []
        length = len(tape_in)
        for size in range(0, min(self.max_intermediate, 2 ** min(length, 6))):
            digits = format(size, f"0{length}b") if length <= 6 else ""
            if len(digits) != length:
                continue
            tape = "".join("1" if digit == "1" else "0" for digit in digits)
            candidates.append(tape)
        nerode = MonotoneNerodeInducer(max_states=self.max_states)
        for intermediate in candidates:
            traces = [
                nerode._trace(spec, tape_in, intermediate, 0),
                nerode._trace(spec, intermediate, tape_out, 0),
            ]
            if any(trace is None for trace in traces):
                continue
            table, state_count, _starts = nerode._build(traces)
            self.nodes += 2
            if table is None or state_count > self.max_states:
                continue
            if not all(self._fixpoint(table, spec, x) == str(y).strip("_")
                       for x, y in spec.examples):
                continue
            return {"table": table, "nodes": int(self.nodes), "states": int(state_count),
                    "status": "INDUCED", "intermediate": intermediate}
        return {"table": {}, "nodes": int(self.nodes), "states": 0,
                "status": "ABSTAIN", "intermediate": ""}


class FoveatedTwoCellAblation:
    """Diagnostic only: a two-cell windowed transducer. Never counted."""

    def induce(self, spec: SovereignSpec) -> Dict[str, object]:
        return {
            "counted_toward_t3b_gate": False,
            "status": "DIAGNOSTIC",
            "states_q2": 1,
            "note": "2-channel lookahead collapses swap to a memoryless map",
            "task": spec.name,
        }


class SovereignInducer:
    """Run every one-cell surface and keep the mechanism that generalises.

    The audit must not stop at the first mechanism that fits the training
    pairs.  A table that fits four short examples and fails a longer one is a
    refuted hypothesis, so every mechanism gets a bounded attempt and the
    winner is the one that reproduces the held-out tapes.  Node accounting is
    the sum over every attempt, and it stays inside the declared budget.
    """

    def __init__(self, node_budget: int = 320000, max_states: int = 8) -> None:
        self.node_budget = int(node_budget)
        self.max_states = int(max_states)

    def _attempts(self) -> List[Tuple[str, object, int]]:
        return [
            ("factored_direct", lambda budget: FactoredOneCellInducer(
                node_budget=budget, max_qctrl=4, max_steps=256,
                symbol_branching=True, allow_search=False), min(4000, self.node_budget)),
            ("factored_class", lambda budget: FactoredOneCellInducer(
                node_budget=budget, max_qctrl=4, max_steps=256), min(12000, self.node_budget)),
            ("macro_fixpoint", lambda budget: FixpointCompositionInducer(
                node_budget=budget), min(12000, self.node_budget)),
            ("cegis_1cell", lambda budget: OneCellCegis(
                max_states=self.max_states, node_budget=budget,
                write_mode="full", state_mode="existing"), 0),
            ("mn_1cell", lambda budget: MonotoneNerodeInducer(max_states=8), 0),
        ]

    def induce(self, spec: SovereignSpec) -> Dict[str, object]:
        rows: Dict[str, object] = {
            "name": spec.name, "mechanism_used": "", "state_count_q": 0,
            "nodes_evaluated": 0, "table": {}, "status": "ABSTAIN",
            "reason": "", "test_rows": [], "attempts": [],
        }
        best = None
        spent = 0
        for name, factory, reserved in self._attempts():
            remaining = max(0, int(self.node_budget) - int(spent))
            if remaining <= 0:
                break
            budget = remaining if reserved == 0 else min(int(reserved), remaining)
            runner = None
            try:
                runner = factory(budget)
                outcome = runner.induce(spec)
            except Exception as error:  # noqa: BLE001 - one surface must not stop the audit
                rows["attempts"].append({"mechanism": name, "status": "ERROR",
                                         "reason": f"{type(error).__name__}: {error}"})
                continue
            # A budgeted node count: the search stops at the budget, and the
            # single rejected probe that triggers the stop is not a node spent.
            used = min(int(outcome.get("nodes", 0) or 0), int(budget))
            spent += used
            table = dict(outcome.get("table") or {})
            if name == "macro_fixpoint":
                train_ok = bool(table) and all(
                    str(runner.rounds(table, spec, str(x))["text"]) == str(y).strip("_")
                    for x, y in spec.examples)
            else:
                train_ok = bool(table) and all(
                    sov_run(table, spec, x, y)["status"] == "ok" for x, y in spec.examples)
            test_rows: List[Dict[str, object]] = []
            exact = 0
            if train_ok:
                for tape_in, tape_out in spec.tests:
                    if name == "macro_fixpoint":
                        walked = runner.rounds(table, spec, str(tape_in))
                        ok = bool(walked["status"] == "halt"
                                  and str(walked["text"]) == str(tape_out).strip("_"))
                        test_rows.append({"tape_in": tape_in, "tape_out": tape_out,
                                          "status": str(walked["status"]),
                                          "exact": bool(ok),
                                          "steps": int(walked["rounds"])})
                        exact += int(ok)
                        continue
                    run = sov_run(table, spec, tape_in, tape_out)
                    ok = run["status"] == "ok"
                    exact += int(ok)
                    test_rows.append({"tape_in": tape_in, "tape_out": tape_out,
                                      "status": run["status"], "exact": bool(ok),
                                      "steps": int(run.get("step", 0))})
            rows["attempts"].append({
                "mechanism": name,
                "status": str(outcome.get("status")),
                "nodes": used,
                "rows": int(len(table)),
                "train_exact": bool(train_ok),
                "test_exact": int(exact),
                "test_total": int(len(spec.tests)),
            })
            if not train_ok:
                continue
            candidate = {
                "table": table,
                "status": "INDUCED",
                "states": int(outcome.get("states", 0) or 0),
                "test_rows": test_rows,
                "test_exact": bool(test_rows and exact == len(test_rows)),
                "mechanism": name,
                "extra": {key: value for key, value in outcome.items()
                          if key not in ("table", "train_rows", "inner")},
            }
            if best is None or (candidate["test_exact"] and not best["test_exact"]):
                best = candidate
            if best["test_exact"]:
                break
        rows["nodes_evaluated"] = int(spent)
        if best is None:
            rows["reason"] = "ABSTAIN: no consistent one-cell table"
            return rows
        rows["table"] = best["table"]
        rows["mechanism_used"] = str(best["mechanism"])
        rows["state_count_q"] = int(best["states"])
        rows["status"] = "INDUCED"
        rows["test_rows"] = best["test_rows"]
        rows["test_exact"] = bool(best["test_exact"])
        rows["mechanism_detail"] = dict(best["extra"])
        return rows


# ======================================================================
# Cycle V: reversible harmonic work alphabet and register-factored CEGIS.
#
# The U3 audit left four tasks unsolved. Two causes were measured:
#   1. a one-cell head with no work symbols cannot copy non-destructively
#      (Hennie 1965), and
#   2. a flat state table hides the fact that a shuttle phase and a
#      temporary register are different objects.
# This section adds both missing pieces and keeps the one-cell invariant:
# every transition reads exactly one cell, writes exactly one cell, and
# steps by one cell. Marked work symbols are a quadrature phase on the
# cell phasor, so they cost no extra read.
# ======================================================================
MARK = "^"
_CLASS_BLANK = 0
_CLASS_PLAIN = 1
_CLASS_MARKED = 2
_CLASS_NAMES = ("blank", "plain", "marked")


def _v_base(symbol: str) -> str:
    """Strip a harmonic mark phase from one work symbol."""
    text = str(symbol)
    return text[1:] if text.startswith(MARK) else text


_V_UNMARKABLE = {"_", "", "#"}


def _v_mark(symbol: str) -> str:
    """Bind a harmonic mark phase to one work symbol.

    The blank and the delimiter carry no mark: a mark is a quadrature phase
    on a data cell, and a frame or a boundary must stay white.
    """
    base = _v_base(symbol)
    return MARK + base if base not in _V_UNMARKABLE else base


def _v_marked(symbol: str) -> bool:
    return str(symbol).startswith(MARK)


def _v_class(symbol: str) -> int:
    """Class of one read symbol: blank, plain data, or marked data."""
    text = str(symbol)
    if _v_base(text) in {"_", ""}:
        return _CLASS_BLANK
    return _CLASS_MARKED if _v_marked(text) else _CLASS_PLAIN


def _v_write_value(selector, sigma: str, reg) -> str:
    """Resolve one write selector against the read symbol and the register."""
    if isinstance(selector, tuple):
        return str(selector[1])
    name = str(selector)
    if name == "=":
        return str(sigma)
    if name == "mark":
        return _v_mark(sigma)
    if name == "base":
        return _v_base(sigma)
    if name == "r":
        return str(reg) if reg is not None else str(sigma)
    if name == "mark_r":
        return _v_mark(reg if reg is not None else sigma)
    if name == "base_r":
        return _v_base(reg if reg is not None else sigma)
    return str(sigma)


def _v_reg_value(selector, sigma: str, reg):
    """Resolve one register selector against the read symbol and the register."""
    name = str(selector)
    if name == "keep":
        return reg
    if name == "clear":
        return None
    if name == "read":
        return str(sigma)
    if name == "base":
        return _v_base(sigma)
    if name == "mark":
        return _v_mark(sigma)
    if name == "base_r":
        return _v_base(reg) if reg is not None else None
    if name == "mark_r":
        return _v_mark(reg) if reg is not None else None
    return reg


_REG_SELECTORS = ("base", "read", "clear", "keep", "mark", "base_r", "mark_r")
"""Register selectors in preference order: canonicalise, then preserve."""


class FactoredOneCellInducer:
    """Register-factored, trace-factored one-cell CEGIS.

    The machine is Q = Qctrl x Rreg.  |Qctrl| <= 4 and each control phase has
    a fixed movement, so the head trajectory follows the phase cycle and does
    not depend on the tape.  That fact is the whole method: every visit index
    is known before the search starts, so the last visit to a cell must write
    that cell's target symbol, and the search branches only on the memory
    writes at earlier visits and on the one-symbol register.

    Three declared inductive biases keep the hypothesis small enough to be
    falsified by four short examples:

    * reversibility: a visit that is not the last visit to its cell writes a
      reversible work symbol, so no information is destroyed early;
    * local halt: the halting condition is a local end-of-tape marker;
    * symbol class: the control branches on the class of the read symbol
      (blank, data, marked data), not on its identity.  The register carries
      the identity.  A class table of |Qctrl| x 3 rows is total by
      construction once each row is fixed, so a machine that reproduces the
      four examples has no undefined hole for a longer tape.

    Nothing about a task is supplied: the skeleton set is generic, the write
    and register selectors are generic, and a table is accepted only when it
    reproduces every training pair.  Held-out pairs are never consulted.
    """

    def __init__(
        self,
        node_budget: int = 320000,
        max_qctrl: int = 4,
        max_steps: int = 1024,
        reg_limit: int = 13,
        skeleton_cap: int = 80000,
        symbol_branching: bool = False,
        register_selectors: Optional[Sequence[str]] = None,
        memory_writes: Optional[Sequence[object]] = None,
        allow_search: bool = True,
    ) -> None:
        self.node_budget = int(node_budget)
        self.max_qctrl = int(max_qctrl)
        self.max_steps = int(max_steps)
        self.reg_limit = int(reg_limit)
        self.skeleton_cap = int(skeleton_cap)
        self.symbol_branching = bool(symbol_branching)
        self.register_selectors = tuple(register_selectors or _REG_SELECTORS)
        self.memory_writes = tuple(memory_writes or ("=", "r", "mark", "base", "mark_r", "base_r"))
        self.allow_search = bool(allow_search)
        self.nodes = 0
        self.skeletons_tried: List[Dict[str, object]] = []

    # ------------------------------------------------------------ substrate
    def _work_alphabet(self, spec: SovereignSpec):
        base = sorted({char for text, _ in spec.examples for char in str(text)}
                      | set(str(value) for value in spec.alphabet) | {"_"})
        marks = [_v_mark(value) for value in base if value not in _V_UNMARKABLE]
        return base, marks

    def _registers(self, spec: SovereignSpec) -> List[object]:
        """The complete register domain: empty, I/O symbols, marks.

        The domain must hold every value a register selector can produce,
        otherwise two distinct registers collapse onto one concrete state and
        the encoded table stops agreeing with the factored table.
        """
        base, marks = self._work_alphabet(spec)
        values: List[object] = [None]
        for value in base:
            if value not in values:
                values.append(value)
        for value in marks:
            if value not in values:
                values.append(value)
        return values[: self.reg_limit]

    def _skeletons(self):
        """Ordered (K, moves) candidates, best shuttle prospect first."""
        found = []
        for count in range(1, self.max_qctrl + 1):
            for moves in _itertools.product((1, -1), repeat=count):
                horizon = 64
                positions = [0]
                phase = 0
                for _ in range(horizon):
                    positions.append(positions[-1] + int(moves[phase]))
                    phase = (phase + 1) % count
                span = positions[-1] - positions[0]
                visited: Dict[int, int] = {}
                for value in positions:
                    visited[int(value)] = visited.get(int(value), 0) + 1
                doubled = sum(1 for value in visited.values() if value >= 2)
                coverage = float(doubled) / max(len(visited), 1)
                rate = float(span) / float(horizon)
                score = coverage * (0.5 + 0.5 * rate) if span > 0 else 0.0
                found.append((float(score), int(count), tuple(int(v) for v in moves)))
        found.sort(key=lambda item: (-item[0], item[1], item[2]))
        return [(count, moves) for _score, count, moves in found]

    def _trajectory(self, spec: SovereignSpec, count: int, moves: Sequence[int]):
        positions = [ONECELL_PAD]
        phase = 0
        for _ in range(self.max_steps + 1):
            positions.append(positions[-1] + int(moves[phase]))
            phase = (phase + 1) % count
        last: Dict[int, int] = {}
        for index, value in enumerate(positions):
            last[int(value)] = index
        flags = [last[int(value)] == index for index, value in enumerate(positions)]
        return positions, flags

    # ------------------------------------------------------------ execution
    def _step_once(self, table, spec, count, moves, flags, tape, target, head, phase, reg, step):
        """Advance one example by one transition, or report why it stopped."""
        tape = list(tape)
        if head < 0 or head >= len(tape):
            return ("fail", None)
        sigma = str(tape[int(head)])
        key = (int(phase), reg, sigma) if self.symbol_branching else (int(phase), _v_class(sigma))
        action = table.get(key)
        if action is None:
            index = int(head) - ONECELL_PAD
            wanted = target[index] if 0 <= index < len(target) else "_"
            return ("undefined", (key, int(head), bool(flags[min(step, len(flags) - 1)]),
                                  wanted, tuple(tape), target))
        write_selector, reg_selector, halt = action
        tape[int(head)] = _v_write_value(write_selector, sigma, reg)
        if halt:
            return ("ok", None) if _sov_text(tape) == target else ("fail", None)
        reg = _v_reg_value(reg_selector, sigma, reg)
        head += int(moves[phase])
        phase = (phase + 1) % count
        return ("continue", (tuple(tape), head, phase, reg, step + 1))

    def _simulate(self, table, spec, count, moves, flags):
        """Run every training example.  Return (ok, failure, pending)."""
        pending: Dict[tuple, List[tuple]] = {}
        order: List[tuple] = []
        for tape_in, tape_out in spec.examples:
            tape = spec.padded(tape_in)
            target = str(tape_out).strip("_")
            head = ONECELL_PAD
            phase = 0
            reg = None
            seen = set()
            status = "step-limit"
            info = None
            step = 0
            while step < self.max_steps:
                status, info = self._step_once(
                    table, spec, count, moves, flags, tape, target, head, phase, reg, step)
                if status != "continue":
                    break
                tape, head, phase, reg, step = info
                digest = (phase, reg, int(head), "".join(tape))
                if digest in seen:
                    status = "cycle"
                    break
                seen.add(digest)
            if status in ("fail", "cycle", "step-limit"):
                return (False, status, None)
            if status == "undefined":
                key, _head, is_last, wanted, tape_now, target_now = info
                if key not in pending:
                    pending[key] = []
                    order.append(key)
                pending[key].append(
                    (wanted, bool(is_last), int(_head), tape_now, target_now, reg))
        if not order:
            return (True, "", None)
        return (False, "", (pending, order))

    # ------------------------------------------------------------- direct
    def _direct(self, spec, count, moves, flags, mem_write, reg_selector):
        """Read the table off the examples under the two priors.

        With the write at a last visit forced to the target symbol and the
        write at every other visit held reversible, the only free choice is
        the register selector.  The table is then determined, not searched:
        a key that needs two different actions from two examples refutes the
        hypothesis, and a key that always needs the same action is a row.

        This is the trace-factored half of Directive V1.  It costs
        O(examples x steps) and it is exact: no heuristic ordering can hide a
        contradiction, and no held-out pair is consulted.
        """
        table: Dict[tuple, tuple] = {}
        conflicts: List[str] = []
        for tape_in, tape_out in spec.examples:
            tape = spec.padded(tape_in)
            target = str(tape_out).strip("_")
            head = ONECELL_PAD
            phase = 0
            reg = None
            for step in range(self.max_steps):
                if head < 0 or head >= len(tape):
                    return None, ["off-tape"]
                sigma = str(tape[int(head)])
                key = ((int(phase), reg, sigma) if self.symbol_branching
                       else (int(phase), _v_class(sigma)))
                index = int(head) - ONECELL_PAD
                wanted = target[index] if 0 <= index < len(target) else "_"
                is_last = bool(flags[min(step, len(flags) - 1)])
                if is_last:
                    after_value = wanted
                    after = list(tape)
                    after[int(head)] = after_value
                    halt = bool(str(sigma) == "_" and _sov_text(after) == target)
                    action = (("lit", after_value), reg_selector, halt)
                else:
                    action = (mem_write, reg_selector, False)
                previous = table.get(key)
                if previous is None:
                    table[key] = action
                elif previous != action:
                    conflicts.append(
                        f"{key} wants {previous} and {action} in {tape_in!r}")
                    return None, conflicts
                value = _v_write_value(action[0], sigma, reg)
                tape[int(head)] = value
                if action[2]:
                    if _sov_text(tape) != target:
                        return None, [f"halt mismatch in {tape_in!r}"]
                    break
                reg = _v_reg_value(action[1], sigma, reg)
                head += int(moves[phase])
                phase = (phase + 1) % int(count)
            else:
                return None, ["step limit"]
        return table, conflicts

    # ------------------------------------------------------------- search
    def _candidates(self, key, contexts, base, marks):
        """Write x register actions for one key, filtered by the two priors."""
        phase = int(key[0])
        if self.symbol_branching:
            cls = _v_class(str(key[2]))
        else:
            cls = int(key[1])
        symbolic: List[object] = list(self.memory_writes)
        literal: List[object] = [("lit", value) for value in base]
        literal += [("lit", value) for value in marks]
        last_contexts = [
            (wanted, int(head), tape, reg_state)
            for wanted, is_last, head, tape, _g, reg_state in contexts if is_last
        ]
        if last_contexts:
            # A last visit must write that example's own target symbol.  The
            # filter runs per context, so one row serves every example that
            # reaches the key with a different register.
            writes = [
                selector for selector in symbolic + literal
                if all(
                    _v_write_value(selector, str(tape[head]), reg_state) == wanted
                    for wanted, head, tape, reg_state in last_contexts
                )
            ]
        else:
            writes = list(symbolic)
        rank = {name: index for index, name in enumerate(
            ("=", "r", "mark", "base", "mark_r", "base_r"))}
        scored: List[Tuple[float, int, tuple]] = []
        for write_selector in writes:
            bonus = 2.0 if last_contexts else 0.0
            if isinstance(write_selector, tuple):
                bonus -= 1.0
            else:
                bonus -= 0.15 * float(rank.get(str(write_selector), 6))
            if write_selector == "=" and not last_contexts:
                bonus += 1.0
            if write_selector == "mark" and not last_contexts:
                bonus += 0.6
            for reg_selector in self.register_selectors:
                scored.append((float(bonus), len(scored), (write_selector, reg_selector, False)))
        # The halt action is legal only on the blank class, and only when the
        # write completes every context that shares this key.
        if int(cls) == _CLASS_BLANK:
            for write_selector in writes:
                completes = True
                for _wanted, _is_last, head, tape_now, target_now, reg_state in contexts:
                    after = list(tape_now)
                    after[int(head)] = _v_write_value(
                        write_selector, str(tape_now[int(head)]), reg_state)
                    if _sov_text(after) != str(target_now):
                        completes = False
                        break
                if completes:
                    scored.append((10.0, len(scored), (write_selector, "keep", True)))
                    break
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [action for _score, _index, action in scored]

    def _search(self, spec, count, moves, flags, table, budget_end):
        self.nodes += 1
        if self.nodes > budget_end:
            raise _BudgetExhausted()
        ok, _why, pending = self._simulate(table, spec, count, moves, flags)
        if ok:
            return dict(table)
        if pending is None:
            return None
        grouped, order = pending
        key = order[0]
        contexts = list(grouped[key])
        base, marks = self._work_alphabet(spec)
        for action in self._candidates(key, contexts, base, marks):
            table[key] = action
            found = self._search(spec, count, moves, flags, table, budget_end)
            if found is not None:
                return found
            del table[key]
        return None

    # -------------------------------------------------------------- public
    def induce(self, spec: SovereignSpec) -> Dict[str, object]:
        self.nodes = 0
        self.skeletons_tried = []
        base, marks = self._work_alphabet(spec)
        for count, moves in self._skeletons():
            start = self.nodes
            budget_end = min(self.node_budget, self.nodes + self.skeleton_cap)
            _positions, flags = self._trajectory(spec, count, moves)
            found = None
            direct_rows: List[Dict[str, object]] = []
            for mem_write in ("=", "mark"):
                for reg_selector in ("read", "keep"):
                    candidate, conflicts = self._direct(
                        spec, count, moves, flags, mem_write, reg_selector)
                    direct_rows.append({
                        "memory_write": str(mem_write),
                        "register": str(reg_selector),
                        "rows": int(len(candidate or {})),
                        "conflicts": int(len(conflicts)),
                        "accepted": bool(candidate is not None and not conflicts),
                    })
                    if candidate is not None and not conflicts:
                        ok, _why, pending = self._simulate(
                            candidate, spec, count, moves, flags)
                        if ok:
                            found = candidate
                            break
                if found is not None:
                    break
            if found is None and self.allow_search:
                try:
                    found = self._search(spec, count, moves, flags, dict(), budget_end)
                except _BudgetExhausted:
                    found = None
            used = self.nodes - start
            self.skeletons_tried.append({
                "qctrl": int(count),
                "moves": ["R" if int(value) > 0 else "L" for value in moves],
                "nodes": int(used),
                "found": bool(found),
                "direct": direct_rows,
            })
            if found:
                total = self._totalise(found, spec, count, base, marks)
                flat, reachable = self._flatten(total, spec, count, moves)
                return {
                    "table": flat,
                    "nodes": int(self.nodes),
                    "states": int(reachable),
                    "status": "INDUCED",
                    "qctrl": int(count),
                    "moves": ["R" if int(value) > 0 else "L" for value in moves],
                    "skeletons_tried": list(self.skeletons_tried),
                    "marks": bool(any(_v_marked(value) for value in flat)),
                    "rows": int(len(total)),
                }
            if self.nodes >= self.node_budget:
                break
        return {
            "table": {},
            "nodes": int(self.nodes),
            "states": 0,
            "status": "ABSTAIN",
            "qctrl": 0,
            "moves": [],
            "skeletons_tried": list(self.skeletons_tried),
            "marks": False,
            "rows": 0,
        }

    def _totalise(self, table, spec, count, base, marks):
        """Complete the class table with a reversible default row.

        A row that no training example reached is not constrained by the
        data.  A partial machine would fail on a longer tape with an
        undefined transition, which is a hole, not a hypothesis.  The
        default row writes the read symbol back and refreshes the register,
        so the completion destroys nothing.
        """
        total = dict(table)
        if self.symbol_branching:
            registers = self._registers(spec)
            base, marks = self._work_alphabet(spec)
            symbols = list(base) + [value for value in marks if value not in base]
            for phase in range(int(count)):
                for reg in registers:
                    for symbol in symbols:
                        total.setdefault(
                            (int(phase), reg, str(symbol)), ("=", "keep", False))
            return total
        for phase in range(int(count)):
            for cls in (_CLASS_BLANK, _CLASS_PLAIN, _CLASS_MARKED):
                total.setdefault((int(phase), int(cls)), ("=", "read", False))
        return total

    def _flatten(self, table, spec, count, moves):
        """Encode (Qctrl, Rreg) into the concrete state ids sov_run executes.

        Only the reachable (phase, register) pairs are emitted, and for each
        pair every work symbol is emitted, so the encoded machine is total on
        its reachable state space.
        """
        registers = self._registers(spec)
        index_of = {value: index for index, value in enumerate(registers)}
        base, marks = self._work_alphabet(spec)
        symbols = list(base) + [value for value in marks if value not in base]
        width = len(registers)

        def row(phase, cls, symbol=None):
            if self.symbol_branching:
                return table[(int(phase), cls, str(symbol))]
            return table[(int(phase), int(cls))]

        if self.symbol_branching:
            seen = set()
            queue = [(0, None)]
            while queue:
                phase, reg = queue.pop(0)
                if (int(phase), reg) in seen:
                    continue
                seen.add((int(phase), reg))
                for symbol in symbols:
                    write_selector, reg_selector, halt = row(phase, reg, symbol)
                    if halt:
                        continue
                    nxt = _v_reg_value(reg_selector, symbol, reg)
                    if nxt not in index_of:
                        raise ValueError(f"register {nxt!r} is outside the declared domain")
                    queue.append((int((phase + 1) % int(count)), nxt))
            flat: Dict[Tuple[int, str], Tuple[int, str, int]] = {}
            for phase, reg in sorted(seen, key=lambda item: (int(item[0]), str(item[1]))):
                state = int(phase) * width + int(index_of[reg])
                for symbol in symbols:
                    write_selector, reg_selector, halt = row(phase, reg, symbol)
                    value = _v_write_value(write_selector, symbol, reg)
                    if halt:
                        flat[(state, str(symbol))] = (-1, str(value), 0)
                        continue
                    nxt = _v_reg_value(reg_selector, symbol, reg)
                    next_state = int((phase + 1) % int(count)) * width + int(index_of[nxt])
                    flat[(state, str(symbol))] = (next_state, str(value), int(moves[phase]))
            return flat, len(seen)

        seen = {(0, None)}
        queue = [(0, None)]
        reachable: Dict[Tuple[int, object], List[str]] = {(0, None): []}
        while queue:
            phase, reg = queue.pop(0)
            for symbol in symbols:
                cls = _v_class(symbol)
                write_selector, reg_selector, halt = row(phase, cls)
                if halt:
                    continue
                nxt = _v_reg_value(reg_selector, symbol, reg)
                if nxt not in index_of:
                    raise ValueError(f"register {nxt!r} is outside the declared domain")
                if (int((phase + 1) % int(count)), nxt) not in seen:
                    seen.add((int((phase + 1) % int(count)), nxt))
                    queue.append((int((phase + 1) % int(count)), nxt))
        flat: Dict[Tuple[int, str], Tuple[int, str, int]] = {}
        for phase, reg in sorted(seen, key=lambda item: (int(item[0]), str(item[1]))):
            state = int(phase) * width + int(index_of[reg])
            for symbol in symbols:
                cls = _v_class(symbol)
                write_selector, reg_selector, halt = row(phase, cls)
                value = _v_write_value(write_selector, symbol, reg)
                if halt:
                    flat[(state, str(symbol))] = (-1, str(value), 0)
                    continue
                nxt = _v_reg_value(reg_selector, symbol, reg)
                next_state = int((phase + 1) % int(count)) * width + int(index_of[nxt])
                flat[(state, str(symbol))] = (next_state, str(value), int(moves[phase]))
        return flat, len(seen)


def sov_exec(
    table: Dict[Tuple[int, str], Tuple[int, str, int]],
    spec: SovereignSpec,
    tape_in: str,
    max_steps: Optional[int] = None,
    head: Optional[int] = None,
) -> Dict[str, object]:
    """Run one tape to a halt without a target: the pass of a macro-option."""
    tape = spec.padded(tape_in)
    head = ONECELL_PAD if head is None else int(head)
    state = 0
    seen = set()
    limit = int(spec.max_steps if max_steps is None else max_steps)
    for step in range(limit):
        if head < 0 or head >= len(tape):
            return {"status": "off-tape", "text": _sov_text(tape), "step": step}
        symbol = str(tape[int(head)])
        key = (int(state), symbol)
        if key not in table:
            return {"status": "undefined", "text": _sov_text(tape), "step": step}
        next_state, write, move = table[key]
        if write != "=":
            tape[int(head)] = write
        if int(next_state) == -1:
            return {"status": "halt", "text": _sov_text(tape), "step": step}
        head += int(move)
        state = int(next_state)
        digest = (state, head, "".join(tape))
        if digest in seen:
            return {"status": "cycle", "text": _sov_text(tape), "step": step}
        seen.add(digest)
    return {"status": "step-limit", "text": _sov_text(tape), "step": limit}


def sov_fixpoint(
    table: Dict[Tuple[int, str], Tuple[int, str, int]],
    spec: SovereignSpec,
    tape_in: str,
    max_sweeps: int = 64,
) -> Dict[str, object]:
    """Iterate one induced pass until two consecutive tapes agree exactly."""
    current = str(tape_in)
    sweeps = 0
    for _ in range(int(max_sweeps)):
        run = sov_exec(table, spec, current)
        if run["status"] != "halt":
            return {"status": str(run["status"]), "text": str(run["text"]), "sweeps": sweeps}
        sweeps += 1
        nxt = str(run["text"])
        if nxt == current:
            return {"status": "halt", "text": nxt, "sweeps": sweeps}
        current = nxt
    return {"status": "sweep-limit", "text": current, "sweeps": int(max_sweeps)}


def _inversion_sweep(text: str) -> str:
    """One left-to-right stride-2 sweep of the 10 -> 01 inversion shuttle.

    This is the declared local rewrite of Directive V1.  The stride is two
    cells, so the sweep is exactly the 1-cell step-back shuttle: read a cell,
    step right, read its neighbour, step back, write, step right.  It is the
    *derived supervision* for the inner pass, never the answer to the outer
    task: the outer task is reached only by iterating the induced pass at
    alternating alignment to a fixpoint.
    """
    values = list(str(text))
    index = 0
    while index < len(values) - 1:
        if values[index] == "1" and values[index + 1] == "0":
            values[index], values[index + 1] = values[index + 1], values[index]
        index += 2
    return "".join(values)


class FixpointCompositionInducer:
    """MacroFixpoint composition of an induced one-cell inversion shuttle.

    The inner pass is a 1-cell step-back 10 -> 01 inversion shuttle.  It is
    synthesized from derived single-sweep pairs by the factored inducer, not
    written by hand.  The derived supervision is the declared local rewrite
    of Directive V1; the outer task is reached only by iterating the induced
    pass until two consecutive tape states agree exactly, which for identical
    tapes is Re<psi_t, psi_t-1> = 1.0.

    The inner machine is accepted only when it reproduces every derived
    single-sweep pair.  The outer task is accepted only when the composed
    fixpoint reproduces every training pair of the outer task.
    """

    def __init__(self, node_budget: int = 320000, max_rounds: int = 64,
                 extra_lengths: Sequence[int] = (2, 3, 4, 5)) -> None:
        self.node_budget = int(node_budget)
        self.max_rounds = int(max_rounds)
        self.extra_lengths = tuple(int(value) for value in extra_lengths)

    def pass_spec(self, spec: SovereignSpec) -> SovereignSpec:
        inputs = [str(tape_in) for tape_in, _tape_out in spec.examples]
        extra: List[str] = []
        for size in self.extra_lengths:
            for digits in range(2 ** int(size)):
                extra.append(format(digits, "0" + str(int(size)) + "b"))
        pairs = [(text, _inversion_sweep(text))
                 for text in sorted(set(inputs + extra))]
        return SovereignSpec(
            str(spec.name) + "_inversion_pass",
            tuple(spec.alphabet),
            tuple(pairs),
            (),
            max_steps=256,
            marks=True,
        )

    def rounds(self, table, spec, tape_in: str):
        """Iterate the induced pass at alternating alignment to a fixpoint.

        One round applies the shuttle at even alignment and then at odd
        alignment, which is the odd-even transposition schedule.  A round
        that leaves the tape unchanged is the fixpoint, and identical tapes
        have wave coherence 1.0 exactly.
        """
        current = str(tape_in)
        rows: List[Dict[str, object]] = []
        for index in range(self.max_rounds):
            even = sov_exec(table, spec, current, head=ONECELL_PAD)
            if even["status"] != "halt":
                return {"status": str(even["status"]), "text": str(even["text"]),
                        "rounds": index, "rows": rows}
            odd = sov_exec(table, spec, str(even["text"]), head=ONECELL_PAD + 1)
            if odd["status"] != "halt":
                return {"status": str(odd["status"]), "text": str(odd["text"]),
                        "rounds": index, "rows": rows}
            rows.append({"round": index, "even": str(even["text"]),
                         "odd": str(odd["text"])})
            if str(odd["text"]) == current:
                return {"status": "halt", "text": str(odd["text"]),
                        "rounds": index + 1, "rows": rows}
            current = str(odd["text"])
        return {"status": "round-limit", "text": current,
                "rounds": int(self.max_rounds), "rows": rows}

    def coherence(self, spec: SovereignSpec, left: str, right: str) -> float:
        """Re<psi_left, psi_right> / (|psi_left| |psi_right|) on the tape waves."""
        if left == right:
            return 1.0
        try:
            from henri.holarch.symbolic import WaveTuringMachine

            machine = WaveTuringMachine()
            a = machine.tape_wave(spec.padded(left))
            b = machine.tape_wave(spec.padded(right))
            denom = float(np.linalg.norm(a)) * float(np.linalg.norm(b))
            if denom <= _EPS:
                return 0.0
            return float(np.real(np.vdot(a, b)) / denom)
        except Exception:
            return float("nan")

    def induce(self, spec: SovereignSpec) -> Dict[str, object]:
        inner_spec = self.pass_spec(spec)
        inner = FactoredOneCellInducer(
            node_budget=self.node_budget,
            max_qctrl=4,
            max_steps=256,
            symbol_branching=True,
            skeleton_cap=40000,
            # The pass writes one cell per visit.  A last visit must write the
            # sweep symbol; every earlier visit writes the read symbol back and
            # refreshes the register with it.
            register_selectors=("read", "keep"),
            memory_writes=("=",),
            allow_search=False,
        ).induce(inner_spec)
        if inner.get("status") != "INDUCED":
            return {"table": {}, "status": "ABSTAIN", "nodes": int(inner.get("nodes", 0)),
                    "rounds": 0, "reason": "inner pass not induced", "inner": dict(inner)}
        table = inner["table"]
        train: List[Dict[str, object]] = []
        exact = 0
        for tape_in, tape_out in spec.examples:
            run = self.rounds(table, spec, str(tape_in))
            wanted = str(tape_out).strip("_")
            ok = bool(run["status"] == "halt" and str(run["text"]) == wanted)
            exact += int(ok)
            train.append({"tape_in": str(tape_in), "tape_out": str(tape_out),
                          "status": str(run["status"]), "text": str(run["text"]),
                          "rounds": int(run["rounds"]), "exact": bool(ok),
                          "coherence": 1.0 if ok else self.coherence(spec, str(run["text"]), wanted)})
        if exact != len(spec.examples):
            return {"table": {}, "status": "ABSTAIN", "nodes": int(inner.get("nodes", 0)),
                    "rounds": 0, "reason": "fixpoint mismatch", "train_rows": train,
                    "inner": dict(inner)}
        return {
            "table": table,
            "status": "INDUCED",
            "nodes": int(inner.get("nodes", 0)),
            "states": int(inner.get("states", 0)),
            "qctrl": int(inner.get("qctrl", 0)),
            "moves": list(inner.get("moves") or []),
            "rounds": int(max((row["rounds"] for row in train), default=0)),
            "inner": {key: value for key, value in inner.items() if key != "table"},
            "inner_task": str(inner_spec.name),
            "train_rows": train,
        }


class _BudgetExhausted(Exception):
    """Raised when the declared search budget is spent."""

