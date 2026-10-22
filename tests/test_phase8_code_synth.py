"""Phase 8 ACTION 3: cartridge-driven code synthesis when the pool is omitted.

Pre-registered clauses, fixed BEFORE the first run of this file:

  A3-1  OMIT + NO BANK   synthesize_from_bank with no bank mounted must
                         return NO_CANDIDATES, never an empty success.
  A3-2  OMIT + NO FLAG   no candidates and no synthesis flag -> 422.
  A3-3  SUPPLIED POOL    the Phase 7 behaviour is unchanged: a supplied
                         pool still verifies. Regression guard.
  A3-4  SYNTH FROM BANK  pool built from the bank; receipt carries a
                         retrieved/mutated provenance split.
  A3-5  OPERATOR SWAP    `return a + b` -> a source whose AST carries
                         Sub, not Add. Parsed, not string-matched.
  A3-6  NAME REPAIR      a retrieved template defining another name still
                         executes, because the entry point is renamed from
                         the ASSERTIONS.
  A3-7  ENTRY FROM TESTS the asserted name is read from `tests`, not from
                         the docstring.
  A3-8  HONEST MISS      a bank holding nothing relevant must report
                         NO_CANDIDATES_PASS, not a fabricated pass.
  A3-9  MUTATION CREDIT  when only a MUTATED candidate passes, the winner
                         is labelled `mutated`, so retrieval and mutation
                         are not blended into one score.

Why A3-9 matters: without per-candidate origin, a pass in a synthesis arm
cannot be attributed. That is the difference between measuring synthesis and
measuring retrieval under a new name.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from henri import api_server                        # noqa: E402
from henri.code_synth import (                      # noqa: E402
    CodeTemplateBank,
    Template,
    operator_variants,
    rename_entry_function,
    spec_entry_name,
    synthesize_candidates,
)
from henri.code_verify import solve_verified_code   # noqa: E402


# -- fixtures -------------------------------------------------------------

ADD_DOC = "Write a function to add two numbers and return the sum."
ADD_TESTS = ["assert add(2, 3) == 5", "assert add(-1, 1) == 0"]

ADD_SRC = "def add_two(a, b):\n    return a + b"
SUB_SRC = "def sub_two(a, b):\n    return a - b"
MUL_SRC = "def mul_two(a, b):\n    return a * b"

# The SUPPLIED-POOL path executes candidates as given; it never renames.
# MEASURED DEFECT IN MY FIRST FIXTURE, repaired. I first passed
# (SUB_SRC, ADD_SRC) to the regression test. Both define `add_two`/
# `sub_two`, and the assertions call `add(...)`, so both raised NameError
# and the endpoint correctly returned NO_CANDIDATES_PASS. The failure was
# my fixture, not the code: the supplied path must receive already-callable
# source. These two fixtures define the asserted name directly.
ADD_CANON = "def add(a, b):\n    return a + b"
SUB_CANON = "def add(a, b):\n    return a - b"


@pytest.fixture(scope="module")
def backbone():
    return api_server.get_api().backbone


@pytest.fixture(scope="module")
def bank(backbone):
    """Small bank: the add template is stored with a CORRECT body."""
    b = CodeTemplateBank(dim=int(backbone.encode(ADD_DOC).psi.shape[0]),
                         rank=64)
    b.add(backbone, Template(docstring=ADD_DOC, source=ADD_SRC,
                             task_id="t-add"))
    b.add(backbone, Template(docstring="Write a function to subtract.",
                             source=SUB_SRC, task_id="t-sub"))
    b.add(backbone, Template(docstring="Write a function to multiply.",
                             source=MUL_SRC, task_id="t-mul"))
    b.finalize()
    return b


@pytest.fixture(scope="module")
def broken_bank(backbone):
    """The add template stores a SUBTRACT body. Only a mutation can pass.

    The docstring is identical to ADD_DOC, so retrieval is maximal and
    mutation is the only path to a passing candidate. This isolates the
    mutation channel instead of letting retrieval carry the test.
    """
    b = CodeTemplateBank(dim=int(backbone.encode(ADD_DOC).psi.shape[0]),
                         rank=64)
    b.add(backbone, Template(docstring=ADD_DOC, source=SUB_SRC,
                             task_id="t-broken"))
    b.finalize()
    return b


@pytest.fixture(scope="module")
def bank_client(bank):
    api_server.reset_banks()
    api_server._CODE_BANKS["t1"] = bank
    return TestClient(api_server.app)


@pytest.fixture(scope="module")
def broken_client(broken_bank):
    yield TestClient(api_server.app)


# -- A3-5 / A3-7 / A3-6: the synthesis primitives -------------------------

def test_operator_swap_changes_the_ast_not_the_string():
    """A3-5. The mutation must be verified by parsing, not by substring."""
    variants = operator_variants(ADD_SRC, limit=2)
    assert variants, "an add body must offer an operator variant"
    ops = set()
    for v in variants:
        tree = ast.parse(v)
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp):
                ops.add(type(node.op).__name__)
    assert "Sub" in ops, f"expected an Add->Sub mutation, saw {ops}"
    assert "Add" not in ops, (
        "every emitted variant must have replaced the operator; a variant "
        "that still parses to Add is not a mutation")


def test_operator_swap_returns_empty_when_no_operator_exists():
    """Control: a body with no swappable operator must yield nothing."""
    assert operator_variants("def f(x):\n    return x") == []


def test_operator_swap_rejects_unparseable_source():
    assert operator_variants("def f(:\n  return") == []


def test_entry_name_is_read_from_the_assertions():
    """A3-7. The asserted name comes from what will actually run."""
    assert spec_entry_name("\n".join(ADD_TESTS)) == "add"
    assert spec_entry_name("assert not empty(x)") == "empty"
    assert spec_entry_name("no callable here") is None


def test_rename_makes_a_retrieved_template_executable():
    """A3-6. A template defining `add_two` must become callable as `add`."""
    renamed = rename_entry_function(ADD_SRC, "add")
    assert renamed is not None
    tree = ast.parse(renamed)
    names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    assert names == ["add"], names


# -- A3-4 / A3-9: synthesis through the verifier --------------------------

def test_synthesis_builds_a_pool_with_provenance(backbone, bank):
    """A3-4. The pool comes from the bank and names each candidate's origin."""
    rows = synthesize_candidates(backbone, ADD_DOC, bank, tests=ADD_TESTS,
                                 k=8, max_variants=2)
    assert rows, "synthesis produced no candidate text"
    origins = {r["origin"] for r in rows}
    assert "retrieved" in origins
    assert all(r["mutation"] for r in rows)


def test_omitted_pool_is_synthesized_and_verified(backbone, bank):
    r = solve_verified_code(backbone, ADD_DOC, [], tests=ADD_TESTS,
                            top_k=3, bank=bank, synthesize_from_bank=True)
    assert r["verdict"] == "EXEC_VERIFIED", r["verdict"]
    assert r["synthesis"]["n_synthesized"] > 0
    assert r["synthesis"]["origins"]["retrieved"] >= 1
    assert r["n_candidates"] == r["synthesis"]["n_synthesized"]


def test_mutation_credit_is_attributed_separately(backbone, broken_bank):
    """A3-9. Only a MUTATED candidate can pass here, and it must say so.

    The stored body subtracts. Retrieval alone cannot pass. If the winner
    were labelled `retrieved`, the receipt would credit the wrong mechanism.
    """
    r = solve_verified_code(backbone, ADD_DOC, [], tests=ADD_TESTS,
                            top_k=5, bank=broken_bank,
                            synthesize_from_bank=True)
    assert r["verdict"] == "EXEC_VERIFIED", r
    assert r["winner"]["origin"] == "mutated", (
        f"expected mutation credit, got {r['winner']}")


# -- A3-8: an honest miss --------------------------------------------------

def test_irrelevant_bank_reports_a_real_failure(backbone):
    """A3-8. A bank with nothing relevant must not fabricate a pass."""
    b = CodeTemplateBank(dim=int(backbone.encode("x").psi.shape[0]),
                         rank=64)
    b.add(backbone, Template(docstring="Write a function to reverse a list.",
                             source="def rev(xs):\n    return xs[::-1]"))
    b.finalize()
    r = solve_verified_code(backbone, ADD_DOC, [], tests=ADD_TESTS,
                            top_k=3, bank=b, synthesize_from_bank=True)
    assert r["verdict"] in ("NO_CANDIDATES_PASS", "NO_CANDIDATES"), r["verdict"]
    assert r["verdict"] != "EXEC_VERIFIED"


# -- A3-1 / A3-2 / A3-3: the HTTP surface ---------------------------------

def test_synthesis_without_a_bank_is_no_candidates():
    """A3-1. Fail-closed with the reason named."""
    api_server.reset_banks()
    c = TestClient(api_server.app)
    r = c.post("/v1/code/solve",
               json={"spec": ADD_DOC, "candidates": [],
                     "tests": ADD_TESTS, "synthesize_from_bank": True,
                     "tenant_id": "absent"})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["verdict"] == "NO_CANDIDATES"
    assert b["executed"] == []
    assert "bank" in b["note"]


def test_omitted_pool_without_flag_is_422():
    """A3-2. Silence is not an empty pool."""
    api_server.reset_banks()
    c = TestClient(api_server.app)
    r = c.post("/v1/code/solve",
               json={"spec": ADD_DOC, "candidates": [], "tests": ADD_TESTS})
    assert r.status_code == 422


def test_supplied_pool_still_verifies():
    """A3-3. Regression: the Phase 7 path is unchanged."""
    api_server.reset_banks()
    c = TestClient(api_server.app)
    r = c.post("/v1/code/solve",
               json={"spec": ADD_DOC,
                     "candidates": [SUB_CANON, ADD_CANON],
                     "tests": ADD_TESTS, "top_k": 2})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["verdict"] == "EXEC_VERIFIED"
    assert "synthesis" not in b, "supplied-pool path must not claim synthesis"


def test_endpoint_synthesizes_from_the_mounted_bank(bank_client):
    """A3-4 over HTTP: pool omitted, bank mounted, verified."""
    r = bank_client.post("/v1/code/solve",
                         json={"spec": ADD_DOC, "candidates": [],
                               "tests": ADD_TESTS, "top_k": 3,
                               "synthesize_from_bank": True,
                               "tenant_id": "t1"})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["verdict"] == "EXEC_VERIFIED", b["verdict"]
    assert b["synthesis"]["bank_rows"] == 3
    assert all("origin" in e for e in b["executed"]), (
        "every executed candidate must carry provenance")


def test_ingest_is_fail_closed_on_a_zero_limit():
    api_server.reset_banks()
    c = TestClient(api_server.app)
    r = c.post("/v1/code/bank/ingest",
               json={"tenant_id": "x", "limit": 0})
    assert r.status_code == 422
