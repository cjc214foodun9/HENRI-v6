"""Build and seal clean-room multi-file RepoPatchEngine benchmark suite (Cycle AG3 Part A).

Materializes 12 multi-file (3-5 module) realistic Python repository defect cases
plus 4 cross-module RepoSheaf-incompatible traps into:
    receipts/holarch/external/repo_patch_multifile/
with per-file SHA-256 digests in MANIFEST.json.

Every case enforces strict import-graph indirection: tests import ONLY the
top-level Module A facade, while the defect lives 1 to 4 cross-module import hops
deep inside Module B/C/D/E.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "receipts" / "holarch" / "external" / "repo_patch_multifile"


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_multifile_cases() -> List[Dict[str, Any]]:
    """Return 12 author-clean multi-module (3-5 file) Python defect repositories."""
    return [
        # 1. 3-module financial ledger rolling cutoff (compare: < -> <=)
        {
            "id": "mf_01_ledger_window",
            "module_count": 3,
            "defect_family": "boundary_compare",
            "call_chain": "ledger.api:count_eligible -> ledger.policy:filter_entries -> ledger.window:within_limit",
            "issue": "Inclusive ceiling entries at exact threshold are dropped by the ledger window.",
            "files": {
                "ledger/__init__.py": "",
                "ledger/window.py": (
                    "def within_limit(amount: int, ceiling: int) -> bool:\n"
                    "    return amount < ceiling\n"
                ),
                "ledger/policy.py": (
                    "from ledger.window import within_limit\n\n"
                    "def filter_entries(amounts: list, ceiling: int) -> list:\n"
                    "    kept = []\n"
                    "    for value in amounts:\n"
                    "        if within_limit(value, ceiling):\n"
                    "            kept.append(value)\n"
                    "    return kept\n"
                ),
                "ledger/api.py": (
                    "from ledger.policy import filter_entries\n\n"
                    "def count_eligible(amounts: list, ceiling: int) -> int:\n"
                    "    return len(filter_entries(amounts, ceiling))\n"
                ),
            },
            "public": {
                "tests/test_public.py": (
                    "from ledger.api import count_eligible\n\n"
                    "def test_inclusive_ceiling():\n"
                    "    assert count_eligible([10, 20, 30, 40], 30) == 3\n\n"
                    "def test_strictly_below():\n"
                    "    assert count_eligible([5, 10], 20) == 2\n"
                ),
            },
            "hidden": {
                "tests/test_hidden.py": (
                    "from ledger.api import count_eligible\n\n"
                    "def test_exact_single():\n"
                    "    assert count_eligible([50], 50) == 1\n\n"
                    "def test_negative_threshold():\n"
                    "    assert count_eligible([-10, -5, 0, 5], -5) == 2\n\n"
                    "def test_empty():\n"
                    "    assert count_eligible([], 10) == 0\n"
                ),
            },
            "test_path": "tests/test_public.py",
        },
        # 2. 3-module telemetry mean rate (guard_empty on ZeroDivisionError)
        {
            "id": "mf_02_metrics_rate",
            "module_count": 3,
            "defect_family": "missing_empty_guard",
            "call_chain": "telemetry.api:average_latency -> telemetry.aggregator:summarize_window -> telemetry.math_ops:mean_value",
            "issue": "Empty telemetry window raises ZeroDivisionError inside deep math_ops kernel.",
            "files": {
                "telemetry/__init__.py": "",
                "telemetry/math_ops.py": (
                    "def mean_value(samples: list) -> float:\n"
                    "    total = 0.0\n"
                    "    for item in samples:\n"
                    "        total += item\n"
                    "    return total / len(samples)\n"
                ),
                "telemetry/aggregator.py": (
                    "from telemetry.math_ops import mean_value\n\n"
                    "def summarize_window(samples: list) -> float:\n"
                    "    return mean_value(samples)\n"
                ),
                "telemetry/api.py": (
                    "from telemetry.aggregator import summarize_window\n\n"
                    "def average_latency(samples: list) -> float:\n"
                    "    return summarize_window(samples)\n"
                ),
            },
            "public": {
                "tests/test_public.py": (
                    "from telemetry.api import average_latency\n\n"
                    "def test_non_empty():\n"
                    "    assert average_latency([2.0, 4.0, 6.0]) == 4.0\n\n"
                    "def test_empty_window():\n"
                    "    assert average_latency([]) == 0.0\n"
                ),
            },
            "hidden": {
                "tests/test_hidden.py": (
                    "from telemetry.api import average_latency\n\n"
                    "def test_single_sample():\n"
                    "    assert average_latency([7.5]) == 7.5\n\n"
                    "def test_empty_again():\n"
                    "    assert average_latency([]) == 0.0\n\n"
                    "def test_four_samples():\n"
                    "    assert average_latency([1.0, 2.0, 3.0, 4.0]) == 2.5\n"
                ),
            },
            "test_path": "tests/test_public.py",
        },
        # 3. 4-module backend route selector (call swap: max -> min)
        {
            "id": "mf_03_router_dispatch",
            "module_count": 4,
            "defect_family": "wrong_builtin_callee",
            "call_chain": "routing.gateway:pick_lowest_cost -> routing.planner:plan_route -> routing.selector:choose_best -> routing.cost:adjusted_costs",
            "issue": "Route selector chooses the highest-cost backend instead of the lowest-cost backend.",
            "files": {
                "routing/__init__.py": "",
                "routing/cost.py": (
                    "def adjusted_costs(latencies: list, overhead: int) -> list:\n"
                    "    out = []\n"
                    "    for ms in latencies:\n"
                    "        out.append(ms + overhead)\n"
                    "    return out\n"
                ),
                "routing/selector.py": (
                    "from routing.cost import adjusted_costs\n\n"
                    "def choose_best(latencies: list, overhead: int) -> int:\n"
                    "    scored = adjusted_costs(latencies, overhead)\n"
                    "    return max(scored)\n"
                ),
                "routing/planner.py": (
                    "from routing.selector import choose_best\n\n"
                    "def plan_route(latencies: list, overhead: int) -> int:\n"
                    "    return choose_best(latencies, overhead)\n"
                ),
                "routing/gateway.py": (
                    "from routing.planner import plan_route\n\n"
                    "def pick_lowest_cost(latencies: list, overhead: int) -> int:\n"
                    "    return plan_route(latencies, overhead)\n"
                ),
            },
            "public": {
                "tests/test_public.py": (
                    "from routing.gateway import pick_lowest_cost\n\n"
                    "def test_pick_cheapest():\n"
                    "    assert pick_lowest_cost([50, 12, 35], 5) == 17\n\n"
                    "def test_two_routes():\n"
                    "    assert pick_lowest_cost([9, 4], 1) == 5\n"
                ),
            },
            "hidden": {
                "tests/test_hidden.py": (
                    "from routing.gateway import pick_lowest_cost\n\n"
                    "def test_single_route():\n"
                    "    assert pick_lowest_cost([22], 3) == 25\n\n"
                    "def test_tied_routes():\n"
                    "    assert pick_lowest_cost([8, 8, 15], 2) == 10\n\n"
                    "def test_zero_overhead():\n"
                    "    assert pick_lowest_cost([100, 20, 40], 0) == 20\n"
                ),
            },
            "test_path": "tests/test_public.py",
        },
        # 4. 3-module cache eviction quota (int constant off-by-one)
        {
            "id": "mf_04_cache_eviction",
            "module_count": 3,
            "defect_family": "off_by_one_int",
            "call_chain": "cache.facade:evict_count -> cache.store:excess_items -> cache.bounds:compute_overflow",
            "issue": "Cache eviction computes one extra eviction past the configured capacity.",
            "files": {
                "cache/__init__.py": "",
                "cache/bounds.py": (
                    "def compute_overflow(current: int, capacity: int) -> int:\n"
                    "    if current <= capacity:\n"
                    "        return 0\n"
                    "    return current - capacity + 1\n"
                ),
                "cache/store.py": (
                    "from cache.bounds import compute_overflow\n\n"
                    "def excess_items(entries: list, capacity: int) -> int:\n"
                    "    return compute_overflow(len(entries), capacity)\n"
                ),
                "cache/facade.py": (
                    "from cache.store import excess_items\n\n"
                    "def evict_count(entries: list, capacity: int) -> int:\n"
                    "    return excess_items(entries, capacity)\n"
                ),
            },
            "public": {
                "tests/test_public.py": (
                    "from cache.facade import evict_count\n\n"
                    "def test_overflow_two():\n"
                    "    assert evict_count([1, 2, 3, 4, 5], 3) == 2\n\n"
                    "def test_overflow_one():\n"
                    "    assert evict_count([1, 2, 3, 4], 3) == 1\n\n"
                    "def test_under_capacity():\n"
                    "    assert evict_count([1, 2], 4) == 0\n"
                ),
            },
            "hidden": {
                "tests/test_hidden.py": (
                    "from cache.facade import evict_count\n\n"
                    "def test_exact_capacity():\n"
                    "    assert evict_count([1, 2, 3], 3) == 0\n\n"
                    "def test_overflow_four():\n"
                    "    assert evict_count([1, 2, 3, 4, 5, 6, 7], 3) == 4\n\n"
                    "def test_zero_capacity():\n"
                    "    assert evict_count([1, 2], 0) == 2\n"
                ),
            },
            "test_path": "tests/test_public.py",
        },
        # 5. 4-module billing discount pipeline (binop: + -> -)
        {
            "id": "mf_05_invoice_discount",
            "module_count": 4,
            "defect_family": "binop_sign_flip",
            "call_chain": "billing.api:final_total -> billing.invoice:build_total -> billing.discount:apply_discount -> billing.rounding:round_cents",
            "issue": "Discount module adds the discount amount to the subtotal instead of subtracting it.",
            "files": {
                "billing/__init__.py": "",
                "billing/rounding.py": (
                    "def round_cents(value: float) -> float:\n"
                    "    return round(value, 2)\n"
                ),
                "billing/discount.py": (
                    "from billing.rounding import round_cents\n\n"
                    "def apply_discount(subtotal: float, rate: float) -> float:\n"
                    "    deduction = subtotal * rate\n"
                    "    return round_cents(subtotal + deduction)\n"
                ),
                "billing/invoice.py": (
                    "from billing.discount import apply_discount\n\n"
                    "def build_total(subtotal: float, rate: float, shipping: float) -> float:\n"
                    "    discounted = apply_discount(subtotal, rate)\n"
                    "    return discounted + shipping\n"
                ),
                "billing/api.py": (
                    "from billing.invoice import build_total\n\n"
                    "def final_total(subtotal: float, rate: float, shipping: float) -> float:\n"
                    "    return build_total(subtotal, rate, shipping)\n"
                ),
            },
            "public": {
                "tests/test_public.py": (
                    "from billing.api import final_total\n\n"
                    "def test_ten_percent_discount():\n"
                    "    assert final_total(100.0, 0.1, 5.0) == 95.0\n\n"
                    "def test_zero_discount():\n"
                    "    assert final_total(50.0, 0.0, 10.0) == 60.0\n"
                ),
            },
            "hidden": {
                "tests/test_hidden.py": (
                    "from billing.api import final_total\n\n"
                    "def test_quarter_discount():\n"
                    "    assert final_total(200.0, 0.25, 0.0) == 150.0\n\n"
                    "def test_half_discount():\n"
                    "    assert final_total(80.0, 0.5, 4.5) == 44.5\n"
                ),
            },
            "test_path": "tests/test_public.py",
        },
        # 6. 3-module schedule interval overlap (boolop: or -> and)
        {
            "id": "mf_06_schedule_overlap",
            "module_count": 3,
            "defect_family": "boolop_inversion",
            "call_chain": "sched.api:has_conflict -> sched.planner:check_slots -> sched.intervals:intervals_overlap",
            "issue": "Disjoint time windows are falsely reported as conflicting due to boolean disjunction.",
            "files": {
                "sched/__init__.py": "",
                "sched/intervals.py": (
                    "def intervals_overlap(s1: int, e1: int, s2: int, e2: int) -> bool:\n"
                    "    return s1 < e2 or s2 < e1\n"
                ),
                "sched/planner.py": (
                    "from sched.intervals import intervals_overlap\n\n"
                    "def check_slots(s1: int, e1: int, s2: int, e2: int) -> bool:\n"
                    "    return intervals_overlap(s1, e1, s2, e2)\n"
                ),
                "sched/api.py": (
                    "from sched.planner import check_slots\n\n"
                    "def has_conflict(s1: int, e1: int, s2: int, e2: int) -> bool:\n"
                    "    return check_slots(s1, e1, s2, e2)\n"
                ),
            },
            "public": {
                "tests/test_public.py": (
                    "from sched.api import has_conflict\n\n"
                    "def test_disjoint():\n"
                    "    assert has_conflict(1, 3, 5, 7) is False\n\n"
                    "def test_disjoint_reversed():\n"
                    "    assert has_conflict(8, 10, 1, 3) is False\n\n"
                    "def test_overlapping():\n"
                    "    assert has_conflict(1, 5, 4, 8) is True\n"
                ),
            },
            "hidden": {
                "tests/test_hidden.py": (
                    "from sched.api import has_conflict\n\n"
                    "def test_touching_edges():\n"
                    "    assert has_conflict(1, 4, 4, 6) is False\n\n"
                    "def test_nested():\n"
                    "    assert has_conflict(2, 10, 4, 6) is True\n\n"
                    "def test_reversed_disjoint():\n"
                    "    assert has_conflict(10, 12, 1, 5) is False\n"
                ),
            },
            "test_path": "tests/test_public.py",
        },
        # 7. 4-module packet payload slice repair (slice lower bound 1 -> None)
        {
            "id": "mf_07_frame_slice",
            "module_count": 4,
            "defect_family": "slice_lower_bound",
            "call_chain": "codec.api:extract_body -> codec.parser:parse_packet -> codec.frame:strip_header -> codec.payload:copy_payload",
            "issue": "Payload decoder drops the first byte of the already-stripped frame body.",
            "files": {
                "codec/__init__.py": "",
                "codec/payload.py": (
                    "def copy_payload(body: list) -> list:\n"
                    "    kept = []\n"
                    "    for item in body:\n"
                    "        kept.append(item)\n"
                    "    return kept[1:]\n"
                ),
                "codec/frame.py": (
                    "from codec.payload import copy_payload\n\n"
                    "def strip_header(packet: list, header_len: int) -> list:\n"
                    "    body = []\n"
                    "    for idx, val in enumerate(packet):\n"
                    "        if idx >= header_len:\n"
                    "            body.append(val)\n"
                    "    return copy_payload(list(body))\n"
                ),
                "codec/parser.py": (
                    "from codec.frame import strip_header\n\n"
                    "def parse_packet(packet: list, header_len: int) -> list:\n"
                    "    return strip_header(packet, header_len)\n"
                ),
                "codec/api.py": (
                    "from codec.parser import parse_packet\n\n"
                    "def extract_body(packet: list, header_len: int) -> list:\n"
                    "    return parse_packet(packet, header_len)\n"
                ),
            },
            "public": {
                "tests/test_public.py": (
                    "from codec.api import extract_body\n\n"
                    "def test_header_two():\n"
                    "    assert extract_body([9, 9, 10, 20, 30], 2) == [10, 20, 30]\n\n"
                    "def test_header_zero():\n"
                    "    assert extract_body([1, 2, 3], 0) == [1, 2, 3]\n"
                ),
            },
            "hidden": {
                "tests/test_hidden.py": (
                    "from codec.api import extract_body\n\n"
                    "def test_header_one():\n"
                    "    assert extract_body([7, 42], 1) == [42]\n\n"
                    "def test_header_three():\n"
                    "    assert extract_body([1, 2, 3, 40, 50], 3) == [40, 50]\n\n"
                    "def test_empty_body():\n"
                    "    assert extract_body([5, 6], 2) == []\n"
                ),
            },
            "test_path": "tests/test_public.py",
        },
        # 8. 5-module ML feature preprocessing pipeline (binop operand order swap)
        {
            "id": "mf_08_feature_norm",
            "module_count": 5,
            "defect_family": "binop_operand_order",
            "call_chain": "mlops.api:center_feature -> mlops.pipeline:run_stage -> mlops.preprocess:shift_values -> mlops.stats:apply_offset -> mlops.kernel:signed_delta",
            "issue": "Five-module feature pipeline inverts the sign of centered deviations (offset - value instead of value - offset).",
            "files": {
                "mlops/__init__.py": "",
                "mlops/kernel.py": (
                    "def signed_delta(value: float, offset: float) -> float:\n"
                    "    return offset - value\n"
                ),
                "mlops/stats.py": (
                    "from mlops.kernel import signed_delta\n\n"
                    "def apply_offset(values: list, offset: float) -> list:\n"
                    "    out = []\n"
                    "    for item in values:\n"
                    "        out.append(signed_delta(float(item), offset))\n"
                    "    return out\n"
                ),
                "mlops/preprocess.py": (
                    "from mlops.stats import apply_offset\n\n"
                    "def shift_values(values: list, offset: float) -> list:\n"
                    "    return apply_offset(values, offset)\n"
                ),
                "mlops/pipeline.py": (
                    "from mlops.preprocess import shift_values\n\n"
                    "def run_stage(values: list, offset: float) -> list:\n"
                    "    return shift_values(values, offset)\n"
                ),
                "mlops/api.py": (
                    "from mlops.pipeline import run_stage\n\n"
                    "def center_feature(values: list, offset: float) -> list:\n"
                    "    return run_stage(values, offset)\n"
                ),
            },
            "public": {
                "tests/test_public.py": (
                    "from mlops.api import center_feature\n\n"
                    "def test_positive_shift():\n"
                    "    assert center_feature([10.0, 15.0, 20.0], 10.0) == [0.0, 5.0, 10.0]\n\n"
                    "def test_negative_result():\n"
                    "    assert center_feature([2.0, 5.0], 4.0) == [-2.0, 1.0]\n"
                ),
            },
            "hidden": {
                "tests/test_hidden.py": (
                    "from mlops.api import center_feature\n\n"
                    "def test_zero_offset():\n"
                    "    assert center_feature([-3.0, 4.0], 0.0) == [-3.0, 4.0]\n\n"
                    "def test_single_item():\n"
                    "    assert center_feature([7.5], 2.5) == [5.0]\n"
                ),
            },
            "test_path": "tests/test_public.py",
        },
        # 9. 3-module audit log extractor (missing None guard)
        {
            "id": "mf_09_audit_guard",
            "module_count": 3,
            "defect_family": "missing_none_guard",
            "call_chain": "audit.api:normalize_tag -> audit.verify:sanitize_entry -> audit.entry:clean_label",
            "issue": "Audit log cleaner crashes with AttributeError when optional label is None.",
            "files": {
                "audit/__init__.py": "",
                "audit/entry.py": (
                    "def clean_label(label: str) -> str:\n"
                    "    return label.strip().lower()\n"
                ),
                "audit/verify.py": (
                    "from audit.entry import clean_label\n\n"
                    "def sanitize_entry(label: str) -> str:\n"
                    "    return clean_label(label)\n"
                ),
                "audit/api.py": (
                    "from audit.verify import sanitize_entry\n\n"
                    "def normalize_tag(label: str) -> str:\n"
                    "    return sanitize_entry(label)\n"
                ),
            },
            "public": {
                "tests/test_public.py": (
                    "from audit.api import normalize_tag\n\n"
                    "def test_standard_tag():\n"
                    "    assert normalize_tag('  Prod-A  ') == 'prod-a'\n\n"
                    "def test_none_tag():\n"
                    "    assert normalize_tag(None) == ''\n"
                ),
            },
            "hidden": {
                "tests/test_hidden.py": (
                    "from audit.api import normalize_tag\n\n"
                    "def test_empty_tag():\n"
                    "    assert normalize_tag('   ') == ''\n\n"
                    "def test_none_again():\n"
                    "    assert normalize_tag(None) == ''\n\n"
                    "def test_mixed_case():\n"
                    "    assert normalize_tag('SecOps') == 'secops'\n"
                ),
            },
            "test_path": "tests/test_public.py",
        },
        # 10. 4-module ACL bitmask accumulator (augassign: ^= -> |=)
        {
            "id": "mf_10_acl_bitmask",
            "module_count": 4,
            "defect_family": "augassign_bitwise",
            "call_chain": "acl.api:effective_mask -> acl.policy:resolve_roles -> acl.evaluator:combine_grants -> acl.bitmask:merge_bits",
            "issue": "Duplicate permission grants cancel each other out because bitmask merger uses XOR instead of OR.",
            "files": {
                "acl/__init__.py": "",
                "acl/bitmask.py": (
                    "def merge_bits(flags: list) -> int:\n"
                    "    mask = 0\n"
                    "    for bit in flags:\n"
                    "        mask ^= bit\n"
                    "    return mask\n"
                ),
                "acl/evaluator.py": (
                    "from acl.bitmask import merge_bits\n\n"
                    "def combine_grants(flags: list) -> int:\n"
                    "    return merge_bits(flags)\n"
                ),
                "acl/policy.py": (
                    "from acl.evaluator import combine_grants\n\n"
                    "def resolve_roles(flags: list) -> int:\n"
                    "    return combine_grants(flags)\n"
                ),
                "acl/api.py": (
                    "from acl.policy import resolve_roles\n\n"
                    "def effective_mask(flags: list) -> int:\n"
                    "    return resolve_roles(flags)\n"
                ),
            },
            "public": {
                "tests/test_public.py": (
                    "from acl.api import effective_mask\n\n"
                    "def test_duplicate_grants():\n"
                    "    assert effective_mask([1, 2, 2, 4]) == 7\n\n"
                    "def test_distinct_grants():\n"
                    "    assert effective_mask([1, 4]) == 5\n"
                ),
            },
            "hidden": {
                "tests/test_hidden.py": (
                    "from acl.api import effective_mask\n\n"
                    "def test_all_same():\n"
                    "    assert effective_mask([8, 8, 8, 8]) == 8\n\n"
                    "def test_empty_grants():\n"
                    "    assert effective_mask([]) == 0\n"
                ),
            },
            "test_path": "tests/test_public.py",
        },
        # 11. 3-module market spread quoter (cross-module call argument order swap)
        {
            "id": "mf_11_market_spread",
            "module_count": 3,
            "defect_family": "call_arg_order_swap",
            "call_chain": "market.api:quote_spread -> market.quoter:evaluate_quote -> market.spread:compute_spread",
            "issue": "Quoter passes (ask, bid) to compute_spread(bid, ask), producing negative spreads.",
            "files": {
                "market/__init__.py": "",
                "market/spread.py": (
                    "def compute_spread(bid: float, ask: float) -> float:\n"
                    "    return round(ask - bid, 4)\n"
                ),
                "market/quoter.py": (
                    "from market.spread import compute_spread\n\n"
                    "def evaluate_quote(bid: float, ask: float) -> float:\n"
                    "    return compute_spread(ask, bid)\n"
                ),
                "market/api.py": (
                    "from market.quoter import evaluate_quote\n\n"
                    "def quote_spread(bid: float, ask: float) -> float:\n"
                    "    return evaluate_quote(bid, ask)\n"
                ),
            },
            "public": {
                "tests/test_public.py": (
                    "from market.api import quote_spread\n\n"
                    "def test_positive_spread():\n"
                    "    assert quote_spread(100.0, 102.5) == 2.5\n\n"
                    "def test_tight_spread():\n"
                    "    assert quote_spread(9.95, 10.0) == 0.05\n"
                ),
            },
            "hidden": {
                "tests/test_hidden.py": (
                    "from market.api import quote_spread\n\n"
                    "def test_zero_spread():\n"
                    "    assert quote_spread(50.0, 50.0) == 0.0\n\n"
                    "def test_wide_spread():\n"
                    "    assert quote_spread(10.0, 17.25) == 7.25\n"
                ),
            },
            "test_path": "tests/test_public.py",
        },
        # 12. 5-module document normalization pipeline (wrong imported sibling callee)
        {
            "id": "mf_12_text_dispatch",
            "module_count": 5,
            "defect_family": "wrong_imported_callee",
            "call_chain": "docops.api:normalize_doc -> docops.router:route_text -> docops.formatter:format_body -> docops.cleaner:clean_body -> docops.rules:lower_text",
            "issue": "Five-module document pipeline invokes imported upper_text instead of lower_text in cleaner.",
            "files": {
                "docops/__init__.py": "",
                "docops/rules.py": (
                    "def lower_text(text: str) -> str:\n"
                    "    return text.strip().lower()\n\n"
                    "def upper_text(text: str) -> str:\n"
                    "    return text.strip().upper()\n"
                ),
                "docops/cleaner.py": (
                    "from docops.rules import lower_text, upper_text\n\n"
                    "def clean_body(text: str) -> str:\n"
                    "    return upper_text(text)\n"
                ),
                "docops/formatter.py": (
                    "from docops.cleaner import clean_body\n\n"
                    "def format_body(text: str) -> str:\n"
                    "    return clean_body(text)\n"
                ),
                "docops/router.py": (
                    "from docops.formatter import format_body\n\n"
                    "def route_text(text: str) -> str:\n"
                    "    return format_body(text)\n"
                ),
                "docops/api.py": (
                    "from docops.router import route_text\n\n"
                    "def normalize_doc(text: str) -> str:\n"
                    "    return route_text(text)\n"
                ),
            },
            "public": {
                "tests/test_public.py": (
                    "from docops.api import normalize_doc\n\n"
                    "def test_lowercase_doc():\n"
                    "    assert normalize_doc('  Hello World  ') == 'hello world'\n\n"
                    "def test_already_lower():\n"
                    "    assert normalize_doc('alpha') == 'alpha'\n"
                ),
            },
            "hidden": {
                "tests/test_hidden.py": (
                    "from docops.api import normalize_doc\n\n"
                    "def test_all_caps():\n"
                    "    assert normalize_doc('HENRI V6') == 'henri v6'\n\n"
                    "def test_tabs():\n"
                    "    assert normalize_doc('\\tSpecDoc\\n') == 'specdoc'\n"
                ),
            },
            "test_path": "tests/test_public.py",
        },
    ]


def build_multifile_traps() -> List[Dict[str, Any]]:
    """Return 4 cross-module RepoSheaf-incompatible traps (E_sheaf > 0.0)."""
    return [
        {
            "id": "trap_mf_01_str_to_numeric_sum",
            "module_count": 3,
            "kind": "cross_module_capability_mismatch_str_to_sum",
            "expected_sheaf_violation_kind": "capability",
            "issue": "API passes a string literal across service into numeric sum kernel.",
            "files": {
                "trap1/__init__.py": "",
                "trap1/stats.py": (
                    "def total_cost(items):\n"
                    "    return sum(items)\n"
                ),
                "trap1/service.py": (
                    "from trap1.stats import total_cost\n\n"
                    "def aggregate(items):\n"
                    "    return total_cost(items)\n"
                ),
                "trap1/api.py": (
                    "from trap1.service import aggregate\n\n"
                    "def run_report():\n"
                    "    return aggregate('1234')\n"
                ),
            },
            "public": {
                "tests/test_public.py": (
                    "from trap1.api import run_report\n\n"
                    "def test_report():\n"
                    "    assert run_report() == 10\n"
                ),
            },
            "hidden": {
                "tests/test_hidden.py": (
                    "from trap1.api import run_report\n\n"
                    "def test_report_hidden():\n"
                    "    assert run_report() == 10\n"
                ),
            },
            "test_path": "tests/test_public.py",
        },
        {
            "id": "trap_mf_02_int_to_sized_len",
            "module_count": 3,
            "kind": "cross_module_capability_mismatch_int_to_len",
            "expected_sheaf_violation_kind": "capability",
            "issue": "API passes an integer across validator into sized len kernel.",
            "files": {
                "trap2/__init__.py": "",
                "trap2/sizing.py": (
                    "def measure_depth(payload):\n"
                    "    return len(payload) + 1\n"
                ),
                "trap2/validator.py": (
                    "from trap2.sizing import measure_depth\n\n"
                    "def check_depth(payload):\n"
                    "    return measure_depth(payload)\n"
                ),
                "trap2/api.py": (
                    "from trap2.validator import check_depth\n\n"
                    "def run_check():\n"
                    "    return check_depth(42)\n"
                ),
            },
            "public": {
                "tests/test_public.py": (
                    "from trap2.api import run_check\n\n"
                    "def test_depth():\n"
                    "    assert run_check() == 3\n"
                ),
            },
            "hidden": {
                "tests/test_hidden.py": (
                    "from trap2.api import run_check\n\n"
                    "def test_depth_hidden():\n"
                    "    assert run_check() == 3\n"
                ),
            },
            "test_path": "tests/test_public.py",
        },
        {
            "id": "trap_mf_03_cross_module_arity",
            "module_count": 3,
            "kind": "cross_module_arity_mismatch",
            "expected_sheaf_violation_kind": "arity",
            "issue": "Dispatcher invokes 3-argument worker with only 2 positional arguments.",
            "files": {
                "trap3/__init__.py": "",
                "trap3/worker.py": (
                    "def compute_score(base: int, weight: int, bonus: int) -> int:\n"
                    "    return base * weight + bonus\n"
                ),
                "trap3/dispatch.py": (
                    "from trap3.worker import compute_score\n\n"
                    "def dispatch_score(base: int, weight: int) -> int:\n"
                    "    return compute_score(base, weight)\n"
                ),
                "trap3/api.py": (
                    "from trap3.dispatch import dispatch_score\n\n"
                    "def score_entry(base: int, weight: int) -> int:\n"
                    "    return dispatch_score(base, weight)\n"
                ),
            },
            "public": {
                "tests/test_public.py": (
                    "from trap3.api import score_entry\n\n"
                    "def test_score():\n"
                    "    assert score_entry(3, 4) == 12\n"
                ),
            },
            "hidden": {
                "tests/test_hidden.py": (
                    "from trap3.api import score_entry\n\n"
                    "def test_score_hidden():\n"
                    "    assert score_entry(5, 2) == 10\n"
                ),
            },
            "test_path": "tests/test_public.py",
        },
        {
            "id": "trap_mf_04_tuple_to_seq_mut",
            "module_count": 3,
            "kind": "cross_module_transitive_tuple_to_append",
            "expected_sheaf_violation_kind": "capability",
            "issue": "API passes an immutable tuple through pipeline into mutable append sink.",
            "files": {
                "trap4/__init__.py": "",
                "trap4/sink.py": (
                    "def append_event(buf, item: int) -> int:\n"
                    "    buf.append(item)\n"
                    "    return len(buf)\n"
                ),
                "trap4/pipeline.py": (
                    "from trap4.sink import append_event\n\n"
                    "def forward_event(buf, item: int) -> int:\n"
                    "    return append_event(buf, item)\n"
                ),
                "trap4/api.py": (
                    "from trap4.pipeline import forward_event\n\n"
                    "def record_fixed() -> int:\n"
                    "    return forward_event((1, 2), 3)\n"
                ),
            },
            "public": {
                "tests/test_public.py": (
                    "from trap4.api import record_fixed\n\n"
                    "def test_record():\n"
                    "    assert record_fixed() == 3\n"
                ),
            },
            "hidden": {
                "tests/test_hidden.py": (
                    "from trap4.api import record_fixed\n\n"
                    "def test_record_hidden():\n"
                    "    assert record_fixed() == 3\n"
                ),
            },
            "test_path": "tests/test_public.py",
        },
    ]


def materialize_manifest(out_dir: Path = OUT_DIR) -> Path:
    cases_dir = out_dir / "cases"
    traps_dir = out_dir / "traps"
    cases_dir.mkdir(parents=True, exist_ok=True)
    traps_dir.mkdir(parents=True, exist_ok=True)

    case_entries: List[Dict[str, Any]] = []
    for case in build_multifile_cases():
        rel = f"cases/{case['id']}.json"
        target = out_dir / rel
        payload = json.dumps(case, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        target.write_bytes(payload)
        case_entries.append(
            {
                "id": case["id"],
                "file": rel,
                "sha256": _sha_bytes(payload),
                "module_count": int(case["module_count"]),
                "defect_family": case["defect_family"],
                "call_chain": case["call_chain"],
            }
        )

    trap_entries: List[Dict[str, Any]] = []
    for trap in build_multifile_traps():
        rel = f"traps/{trap['id']}.json"
        target = out_dir / rel
        payload = json.dumps(trap, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        target.write_bytes(payload)
        trap_entries.append(
            {
                "id": trap["id"],
                "file": rel,
                "sha256": _sha_bytes(payload),
                "module_count": int(trap["module_count"]),
                "kind": trap["kind"],
                "expected_sheaf_violation_kind": trap["expected_sheaf_violation_kind"],
            }
        )

    slice_hasher = hashlib.sha256()
    for row in case_entries + trap_entries:
        slice_hasher.update(f"{row['id']}:{row['sha256']}\n".encode("utf-8"))

    manifest = {
        "slice_id": "ag3_repo_patch_multifile_12_plus_4_traps",
        "provenance": "clean_room_multifile_suite_cycle_ag3",
        "module_depth_range": [3, 5],
        "cases_count": len(case_entries),
        "traps_count": len(trap_entries),
        "slice_sha256": slice_hasher.hexdigest(),
        "cases": case_entries,
        "traps": trap_entries,
    }
    manifest_path = out_dir / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


if __name__ == "__main__":
    mpath = materialize_manifest()
    print(f"[ag3-suite] materialized {mpath}")
