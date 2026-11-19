"""Cycle AD1: the deterministic sixteen-repository repair suite.

Every case is author-clean: flat modules at the workspace root, type
annotations on every public function, one planted defect, a public
reproduction suite and an unseen maintainer regression suite.  The defect
classes cover data pipelines, graph and routing helpers, numerical validators,
parser/serialiser helpers, and state machines.

The four traps are deliberately unsolvable: two hold contradictory public
assertions, one carries a stale issue whose reproduction suite already passes,
and one holds an architectural capability conflict that the pre-execution
RepoSheaf gate refuses.
"""

from __future__ import annotations

from typing import Dict, List

from .repo_patch import RepoCase


def _case(
    name: str,
    issue: str,
    files: Dict[str, str],
    public: str,
    hidden: str,
) -> RepoCase:
    return RepoCase(
        name=name,
        issue=issue,
        files=dict(files),
        public={"tests/test_public.py": public},
        hidden={"tests/test_hidden.py": hidden},
    )


# ---------------------------------------------------------------- pipelines
def pipeline_drop_tail() -> RepoCase:
    return _case(
        "pipeline_drop_tail",
        "The batch pipeline reports one fewer row than it receives; the final "
        "row of every batch disappears from the aggregate.",
        {
            "rows.py": '''"""Row helpers."""


def normalize(text: str) -> str:
    return text.strip().lower()
''',
            "stats.py": '''"""Aggregates."""


def total_score(rows: list) -> float:
    total = 0.0
    for row in rows:
        total += float(row["score"])
    return total
''',
            "service.py": '''"""Service entry points."""
from rows import normalize
from stats import total_score


def summarize(rows: list, label: str) -> float:
    _ = normalize(label)
    return total_score(rows)
''',
            "pipeline.py": '''"""Batch pipeline."""
from service import summarize


def select(rows: list) -> list:
    kept = []
    for item in rows:
        kept.append(item)
    return kept[:-1]


def run_batches(rows: list) -> float:
    return summarize(select(rows), "batch")
''',
        },
        '''import pipeline


def test_every_row_is_counted():
    rows = [{"score": 1}, {"score": 2}, {"score": 3}]
    assert pipeline.run_batches(rows) == 6.0


def test_single_row():
    assert pipeline.run_batches([{"score": 5}]) == 5.0
''',
        '''import pipeline


def test_final_row_is_included():
    assert pipeline.run_batches([{"score": 2}, {"score": 4}]) == 6.0


def test_empty_batch_is_zero():
    assert pipeline.run_batches([]) == 0.0
''',
    )


def ledger_accumulator() -> RepoCase:
    return _case(
        "ledger_accumulator",
        "The ledger counter is one too high for every entry set: its running "
        "total starts from the wrong value.",
        {
            "ledger.py": '''"""Ledger counting."""


def count_entries(entries: list) -> int:
    total = 1
    for _entry in entries:
        total += 1
    return total
''',
            "audit.py": '''"""Audit wrapper."""
from ledger import count_entries


def audited(entries: list) -> int:
    return count_entries(entries)
''',
            "report.py": '''"""Report."""
from audit import audited


def line_count(entries: list) -> int:
    return audited(entries)
''',
        },
        '''import report


def test_three_entries():
    assert report.line_count([10, 20, 30]) == 3


def test_one_entry():
    assert report.line_count([1]) == 1
''',
        '''import report


def test_no_entries():
    assert report.line_count([]) == 0


def test_five_entries():
    assert report.line_count([1, 2, 3, 4, 5]) == 5
''',
    )


def chunk_boundary() -> RepoCase:
    return _case(
        "chunk_boundary",
        "A chunk whose length equals the requested size is reported as too "
        "large, so an exact fit is rejected.",
        {
            "sizing.py": '''"""Chunk sizing."""


def fits(items: list, size: int) -> bool:
    return len(items) < size
''',
            "planner.py": '''"""Planner."""
from sizing import fits


def accepts(items: list, size: int) -> bool:
    return fits(items, size)
''',
            "service.py": '''"""Service."""
from planner import accepts


def enqueue(items: list, size: int) -> bool:
    return accepts(items, size)
''',
        },
        '''import service


def test_exact_fit_is_accepted():
    assert service.enqueue([1, 2], 2) is True


def test_under_fit_is_accepted():
    assert service.enqueue([1], 2) is True
''',
        '''import service


def test_over_fit_is_rejected():
    assert service.enqueue([1, 2, 3], 2) is False


def test_empty_chunk_is_accepted():
    assert service.enqueue([], 3) is True
''',
    )


def dedupe_none_guard() -> RepoCase:
    return _case(
        "dedupe_none_guard",
        "Deduplicating an absent sequence raises a type error instead of "
        "returning an empty sequence.",
        {
            "dedupe.py": '''"""Deduplication."""


def unique(items: list) -> list:
    seen = []
    for item in items:
        if item not in seen:
            seen.append(item)
    return seen
''',
            "collect.py": '''"""Collection helpers."""
from dedupe import unique


def distinct(items: list) -> list:
    return unique(items)
''',
            "service.py": '''"""Service."""
from collect import distinct


def normalise(items: list) -> list:
    return distinct(items)
''',
        },
        '''import service


def test_absent_sequence():
    assert service.normalise(None) == []


def test_repeated_values():
    assert service.normalise([1, 1, 2]) == [1, 2]
''',
        '''import service


def test_empty_sequence():
    assert service.normalise([]) == []


def test_order_is_preserved():
    assert service.normalise([3, 1, 3, 2]) == [3, 1, 2]
''',
    )


# ------------------------------------------------------------------- routing
def router_attribute() -> RepoCase:
    return _case(
        "router_attribute",
        "The router always answers with the fallback route; it reads the "
        "secondary attribute where the primary route is required.",
        {
            "settings.py": '''"""Routing settings."""


class Settings:
    def __init__(self, primary: str, secondary: str) -> None:
        self.primary = primary
        self.secondary = secondary
''',
            "router.py": '''"""Route selection."""
from settings import Settings


def describe(settings: Settings) -> str:
    return settings.primary + "/" + settings.secondary


def choose(settings: Settings) -> str:
    return settings.secondary
''',
            "service.py": '''"""Service."""
from router import choose
from settings import Settings


def plan(settings: Settings) -> str:
    return choose(settings)
''',
        },
        '''from service import plan
from settings import Settings


def test_primary_route_is_used():
    assert plan(Settings("alpha", "beta")) == "alpha"


def test_describe_pairs_routes():
    from router import describe

    assert describe(Settings("alpha", "beta")) == "alpha/beta"
''',
        '''from service import plan
from settings import Settings


def test_other_primary_route():
    assert plan(Settings("gamma", "delta")) == "gamma"


def test_secondary_is_not_the_answer():
    assert plan(Settings("north", "south")) != "south"
''',
    )


def router_callee() -> RepoCase:
    return _case(
        "router_callee",
        "Enqueuing one item stores it twice: the router calls the replay "
        "primitive instead of the record primitive.",
        {
            "queue.py": '''"""Queue primitives."""


def record(queue: list, item: str) -> list:
    queue.append(item)
    return queue


def replay(queue: list, item: str) -> list:
    queue.append(item)
    queue.append(item)
    return queue
''',
            "router.py": '''"""Queue router."""
from queue import record, replay


def push(queue: list, item: str) -> list:
    return replay(queue, item)
''',
            "service.py": '''"""Service."""
from router import push


def enqueue(queue: list, item: str) -> list:
    return push(queue, item)
''',
        },
        '''import service


def test_single_enqueue():
    assert service.enqueue([], "alpha") == ["alpha"]


def test_enqueue_keeps_prior_items():
    assert service.enqueue(["beta"], "alpha") == ["beta", "alpha"]
''',
        '''import service


def test_no_duplicate_is_recorded():
    assert service.enqueue([], "gamma") == ["gamma"]


def test_length_after_enqueue():
    assert len(service.enqueue([], "delta")) == 1
''',
    )


# ------------------------------------------------------------------- numbers
def numeric_sign() -> RepoCase:
    return _case(
        "numeric_sign",
        "The upper tolerance is below the mean instead of above it; the "
        "validator subtracts the tolerance where it must add it.",
        {
            "tolerance.py": '''"""Tolerance limits."""


def upper_bound(mean: float, delta: float) -> float:
    return mean - delta
''',
            "report.py": '''"""Report."""
from tolerance import upper_bound


def limit(mean: float, delta: float) -> float:
    return upper_bound(mean, delta)
''',
            "service.py": '''"""Service."""
from report import limit


def acceptance_limit(mean: float, delta: float) -> float:
    return limit(mean, delta)
''',
        },
        '''import service


def test_upper_bound_is_above_the_mean():
    assert service.acceptance_limit(10.0, 2.0) == 12.0


def test_zero_tolerance():
    assert service.acceptance_limit(3.5, 0.0) == 3.5
''',
        '''import service


def test_negative_tolerance():
    assert service.acceptance_limit(10.0, -1.0) == 9.0


def test_large_tolerance():
    assert service.acceptance_limit(1.0, 4.0) == 5.0
''',
    )


def numeric_compare() -> RepoCase:
    return _case(
        "numeric_compare",
        "A value exactly at the limit is reported as over the limit; the "
        "validator uses an inclusive comparison where a strict one is required.",
        {
            "validator.py": '''"""Tolerance validator."""


def exceeds(value: float, limit: float) -> bool:
    return value >= limit
''',
            "review.py": '''"""Review."""
from validator import exceeds


def check(value: float, limit: float) -> bool:
    return exceeds(value, limit)
''',
            "service.py": '''"""Service."""
from review import check


def over_limit(value: float, limit: float) -> bool:
    return check(value, limit)
''',
        },
        '''import service


def test_value_at_the_limit_is_not_over():
    assert service.over_limit(5.0, 5.0) is False


def test_value_above_the_limit_is_over():
    assert service.over_limit(6.0, 5.0) is True
''',
        '''import service


def test_value_below_the_limit_is_not_over():
    assert service.over_limit(4.0, 5.0) is False


def test_negative_limit_is_strict():
    assert service.over_limit(-2.0, -2.0) is False
''',
    )


def numeric_scale() -> RepoCase:
    return _case(
        "numeric_scale",
        "Scaling divides by the factor instead of multiplying; every scaled "
        "quantity moves in the wrong direction.",
        {
            "scaling.py": '''"""Scaling."""


def scale(value: float, factor: float) -> float:
    return value / factor
''',
            "report.py": '''"""Report."""
from scaling import scale


def total(value: float, factor: float) -> float:
    return scale(value, factor)
''',
            "service.py": '''"""Service."""
from report import total


def projected(value: float, factor: float) -> float:
    return total(value, factor)
''',
        },
        '''import service


def test_doubling():
    assert service.projected(10.0, 2.0) == 20.0


def test_fractional_factor():
    assert service.projected(3.0, 0.5) == 1.5
''',
        '''import service


def test_identity_factor():
    assert service.projected(7.0, 1.0) == 7.0


def test_tripling():
    assert service.projected(2.0, 3.0) == 6.0
''',
    )


# -------------------------------------------------------------------- parser
def parser_index() -> RepoCase:
    return _case(
        "parser_index",
        "The parsed record names the quantity field instead of the name "
        "field; the wrong element of the split line is read.",
        {
            "fields.py": '''"""Field splitting."""


def split_fields(line: str) -> list:
    return line.split(",")
''',
            "record.py": '''"""Record parsing."""
from fields import split_fields


def parse_line(line: str) -> dict:
    parts = split_fields(line)
    return {"name": parts[1].strip(), "quantity": len(parts)}
''',
            "service.py": '''"""Service."""
from record import parse_line


def load_line(line: str) -> dict:
    return parse_line(line)
''',
        },
        '''import service


def test_name_field():
    assert service.load_line("widget,3")["name"] == "widget"


def test_quantity_field():
    assert service.load_line("widget,3")["quantity"] == 2
''',
        '''import service


def test_whitespace_is_trimmed():
    assert service.load_line("  bolt , 4")["name"] == "bolt"


def test_other_name():
    assert service.load_line("nut,9")["name"] == "nut"
''',
    )


def parser_empty_guard() -> RepoCase:
    return _case(
        "parser_empty_guard",
        "Decoding an empty payload raises a key error instead of returning an "
        "empty record.",
        {
            "codec.py": '''"""Codec."""


def decode(payload: dict) -> dict:
    return {"id": payload["id"], "size": len(payload)}
''',
            "record.py": '''"""Record."""
from codec import decode


def load(payload: dict) -> dict:
    return decode(payload)
''',
            "service.py": '''"""Service."""
from record import load


def ingest(payload: dict) -> dict:
    return load(payload)
''',
        },
        '''import service


def test_empty_payload():
    assert service.ingest({}) == {}


def test_named_payload():
    assert service.ingest({"id": "alpha"}) == {"id": "alpha", "size": 1}
''',
        '''import service


def test_two_field_payload():
    assert service.ingest({"id": "beta", "kind": "x"}) == {"id": "beta", "size": 2}


def test_empty_record_is_a_dict():
    assert isinstance(service.ingest({}), dict)
''',
    )


def parser_slice() -> RepoCase:
    return _case(
        "parser_slice",
        "Text normalisation silently drops the first character of every "
        "value because the slice starts one element late.",
        {
            "textnorm.py": '''"""Text normalisation."""


def clean(text: str) -> str:
    chars = list(text)
    return "".join(chars[1:]).strip()
''',
            "record.py": '''"""Record."""
from textnorm import clean


def tidy(text: str) -> str:
    return clean(text)
''',
            "service.py": '''"""Service."""
from record import tidy


def normalize(text: str) -> str:
    return tidy(text)
''',
        },
        '''import service


def test_first_character_is_kept():
    assert service.normalize("alpha") == "alpha"


def test_padding_is_removed():
    assert service.normalize("  beta ") == "beta"
''',
        '''import service


def test_single_character():
    assert service.normalize("b") == "b"


def test_two_characters():
    assert service.normalize("cd") == "cd"
''',
    )


# --------------------------------------------------------------------- state
def fsm_string() -> RepoCase:
    return _case(
        "fsm_string",
        "A freshly created machine reports the terminal state instead of the "
        "initial state.",
        {
            "states.py": '''"""State names."""


def initial() -> str:
    return "done"


def advance(state: str) -> str:
    if state == "idle":
        return "running"
    if state == "running":
        return "done"
    return "idle"
''',
            "machine.py": '''"""Machine."""
from states import advance, initial


def start() -> str:
    return initial()


def tick(state: str) -> str:
    return advance(state)
''',
            "service.py": '''"""Service."""
from machine import start, tick


def boot() -> str:
    return start()


def step(state: str) -> str:
    return tick(state)
''',
        },
        '''import service


def test_boot_reports_idle():
    assert service.boot() == "idle"


def test_first_step():
    assert service.step("idle") == "running"
''',
        '''import service


def test_second_step():
    assert service.step("running") == "done"


def test_reset_after_terminal():
    assert service.step("done") == "idle"
''',
    )


def machine_accumulator() -> RepoCase:
    return _case(
        "machine_accumulator",
        "The tick counter reports one extra step for every limit because its "
        "accumulator starts at the wrong value.",
        {
            "machine.py": '''"""Tick machine."""


def ticks(limit: int) -> int:
    count = 1
    for _step in range(limit):
        count += 1
    return count
''',
            "planner.py": '''"""Planner."""
from machine import ticks


def budget(limit: int) -> int:
    return ticks(limit)
''',
            "service.py": '''"""Service."""
from planner import budget


def steps(limit: int) -> int:
    return budget(limit)
''',
        },
        '''import service


def test_three_ticks():
    assert service.steps(3) == 3


def test_one_tick():
    assert service.steps(1) == 1
''',
        '''import service


def test_zero_ticks():
    assert service.steps(0) == 0


def test_five_ticks():
    assert service.steps(5) == 5
''',
    )


def queue_offbyone() -> RepoCase:
    return _case(
        "queue_offbyone",
        "Draining a queue leaves its final element behind; the loop stops one "
        "element early.",
        {
            "drain.py": '''"""Queue draining."""


def drain(items: list) -> list:
    kept = list(items)
    out = []
    while len(kept) > 1:
        out.append(kept[0])
        kept = kept[1:]
    return out
''',
            "batch.py": '''"""Batch."""
from drain import drain


def drain_all(items: list) -> list:
    return drain(items)
''',
            "service.py": '''"""Service."""
from batch import drain_all


def flush(items: list) -> list:
    return drain_all(items)
''',
        },
        '''import service


def test_three_elements_are_drained():
    assert service.flush([1, 2, 3]) == [1, 2, 3]


def test_two_elements_are_drained():
    assert service.flush([4, 5]) == [4, 5]
''',
        '''import service


def test_one_element_is_drained():
    assert service.flush([7]) == [7]


def test_empty_queue_is_drained():
    assert service.flush([]) == []
''',
    )


def counter_reset() -> RepoCase:
    return _case(
        "counter_reset",
        "Resetting a counter leaves it at one instead of zero.",
        {
            "counter.py": '''"""Counters."""


def reset(counter: dict) -> dict:
    counter["count"] = 1
    return counter
''',
            "store.py": '''"""Store."""
from counter import reset


def clear_store(counter: dict) -> dict:
    return reset(counter)
''',
            "service.py": '''"""Service."""
from store import clear_store


def clear(counter: dict) -> dict:
    return clear_store(counter)
''',
        },
        '''import service


def test_reset_zeroes_the_counter():
    assert service.clear({"count": 5}) == {"count": 0}


def test_reset_new_counter():
    assert service.clear({}) == {"count": 0}
''',
        '''import service


def test_reset_keeps_other_keys():
    assert service.clear({"count": 9, "kind": "x"}) == {"count": 0, "kind": "x"}


def test_reset_twice():
    value = service.clear({"count": 4})
    assert service.clear(value) == {"count": 0}
''',
    )


# --------------------------------------------------------------------- traps
def trap_conflicting_assertions() -> RepoCase:
    return _case(
        "trap_conflicting_assertions",
        "The aggregate is reported as wrong, but the published expectations "
        "contradict each other.",
        {
            "stats.py": '''"""Aggregates."""


def total(items: list) -> int:
    value = 0
    for item in items:
        value += item
    return value
''',
            "service.py": '''"""Service."""
from stats import total


def sum_total(items: list) -> int:
    return total(items)
''',
        },
        '''import service


def test_total_is_three():
    assert service.sum_total([1, 2]) == 3


def test_total_is_four():
    assert service.sum_total([1, 2]) == 4
''',
        '''import service


def test_total_is_five():
    assert service.sum_total([1, 2]) == 5
''',
    )


def trap_impossible_assert() -> RepoCase:
    return _case(
        "trap_impossible_assert",
        "One assertion demands two different values of the same expression.",
        {
            "stats.py": '''"""Aggregates."""


def total(items: list) -> int:
    value = 0
    for item in items:
        value += item
    return value
''',
            "service.py": '''"""Service."""
from stats import total


def sum_total(items: list) -> int:
    return total(items)
''',
        },
        '''import service


def test_total_is_both_one_and_two():
    assert service.sum_total([1]) == 1 and service.sum_total([1]) == 2
''',
        '''import service


def test_total_is_one():
    assert service.sum_total([1]) == 1
''',
    )


def trap_stale_issue() -> RepoCase:
    return _case(
        "trap_stale_issue",
        "The issue reports a broken average, but the published reproduction "
        "suite already passes on the current tree.",
        {
            "stats.py": '''"""Aggregates."""


def average(values: list) -> float:
    total = 0.0
    for value in values:
        total += float(value)
    return total / len(values)
''',
            "service.py": '''"""Service."""
from stats import average


def mean(values: list) -> float:
    return average(values)
''',
        },
        '''import service


def test_mean_of_two():
    assert service.mean([2.0, 4.0]) == 3.0


def test_mean_of_three():
    assert service.mean([1.0, 2.0, 3.0]) == 2.0
''',
        '''import service


def test_mean_of_one():
    assert service.mean([5.0]) == 5.0
''',
    )


def trap_sheaf_conflict() -> RepoCase:
    return _case(
        "trap_sheaf_conflict",
        "The issue asks the api to forward its text payload into the row "
        "store, which demands a mutable sequence the text cannot provide.",
        {
            "store.py": '''"""Storage."""


def push(rows: list) -> list:
    rows.append(1)
    return rows
''',
            "api.py": '''"""Api."""
from store import push


def run(text: str) -> list:
    return push(text)
''',
        },
        '''import api


def test_forwarding_appends_one():
    assert api.run("row") == ["row", 1]
''',
        '''import api


def test_forwarding_is_stable():
    assert api.run("line") == ["line", 1]
''',
    )


# -------------------------------------------------------------------- builders
def build_cases() -> List[RepoCase]:
    return [
        pipeline_drop_tail(),
        ledger_accumulator(),
        chunk_boundary(),
        dedupe_none_guard(),
        router_attribute(),
        router_callee(),
        numeric_sign(),
        numeric_compare(),
        numeric_scale(),
        parser_index(),
        parser_empty_guard(),
        parser_slice(),
        fsm_string(),
        machine_accumulator(),
        queue_offbyone(),
        counter_reset(),
    ]


def build_traps() -> List[RepoCase]:
    return [
        trap_conflicting_assertions(),
        trap_impossible_assert(),
        trap_stale_issue(),
        trap_sheaf_conflict(),
    ]


def build_all() -> List[RepoCase]:
    return build_cases() + build_traps()
