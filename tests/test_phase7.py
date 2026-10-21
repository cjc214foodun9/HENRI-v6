"""Phase 7 gates: daemon endpoints, fail-closed mounting, code verification.

Every test can fail. Three pin measured defects so a regression reopens the
exact bug rather than a paraphrase of it:

  * `test_cartridges_write_uses_wave_dim` pins the `.dim` defect: the first
    draft sized the bank from `backbone.dim`, an attribute that does not
    exist, so every write raised AttributeError.
  * `test_solve_without_tests_is_not_a_pass` pins the honesty gate: absent
    assertions must never report a pass.
  * `test_mount_before_write_is_404` pins fail-closed mounting.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from henri.api_server import app, reset_banks  # noqa: E402
from henri.code_verify import (executor_status, solve_verified_code)  # noqa: E402


@pytest.fixture(scope="module")
def client():
    reset_banks()
    return TestClient(app)


@pytest.fixture(scope="module")
def backbone():
    from henri.semantic_backbone import SemanticBackbone
    return SemanticBackbone()


# -- code verification (pure, no HTTP) -----------------------------------

def test_executor_loads_by_path():
    st = executor_status()
    assert st["available"] is True, st.get("error")
    assert st["has_run_candidate"] is True


def test_solve_without_tests_is_not_a_pass(backbone):
    """HONESTY GATE. No assertions means no execution and no verdict."""
    r = solve_verified_code(backbone, "add two numbers",
                            ["def add(a, b):\n    return a + b\n"])
    assert r["verdict"] == "NO_TESTS_UNVERIFIABLE", r["verdict"]
    assert r["executed"] == []


def test_solve_with_tests_verifies_correct_candidate(backbone):
    good = "def add(a, b):\n    return a + b\n"
    bad = "def add(a, b):\n    return a - b\n"
    r = solve_verified_code(backbone, "add two numbers", [bad, good],
                            tests=["assert add(2, 3) == 5"])
    assert r["verdict"] == "EXEC_VERIFIED", r["verdict"]
    assert r["winner"] is not None
    assert r["n_executed"] >= 1


def test_solve_all_wrong_reports_no_pass(backbone):
    r = solve_verified_code(
        backbone, "add two numbers",
        ["def add(a, b):\n    return a - b\n",
         "def add(a, b):\n    return 0\n"],
        tests=["assert add(2, 3) == 5"])
    assert r["verdict"] == "NO_CANDIDATES_PASS", r["verdict"]


def test_solve_empty_pool(backbone):
    r = solve_verified_code(backbone, "spec", [], tests=["assert True"])
    assert r["verdict"] == "NO_CANDIDATES"


def test_ranking_reports_tie_group(backbone):
    """Phase 6 measured a tie plateau on MBPP. The tie group must surface."""
    from henri.code_verify import rank_candidates
    rk = rank_candidates(backbone, "add two numbers",
                         ["def f(): pass", "def g(): pass"])
    assert rk["tie_group_size"] >= 1
    assert "ranking_well_posed" not in rk  # computed by the caller


# -- HTTP surface --------------------------------------------------------

def test_health_seal(client):
    r = client.get("/v1/health/seal")
    assert r.status_code == 200
    assert r.json().get("status") == "SEALED", r.json()


def test_health_executor(client):
    r = client.get("/v1/health/executor")
    assert r.status_code == 200
    assert r.json().get("available") is True


def test_reason_rank(client):
    r = client.post("/v1/reason/rank", json={
        "question": "the pressure increases with temperature",
        "options": ["the pressure decreases", "a recipe for bread"]})
    assert r.status_code == 200, r.text
    assert "selected_index" in r.json()


def test_rank_rejects_single_option(client):
    r = client.post("/v1/reason/rank", json={"question": "q", "options": ["x"]})
    assert r.status_code == 422


def test_mount_before_write_is_404(client):
    """FAIL-CLOSED. A mount for an unknown namespace must not return an
    empty bank dressed as success."""
    r = client.post("/v1/cartridges/mount", json={"tenant_id": "nope"})
    assert r.status_code == 404


def test_cartridges_write_uses_wave_dim(client):
    """MEASURED DEFECT. The first draft used `CartridgeBank(dim=backbone.dim)`
    but `SemanticBackbone` has no `.dim` attribute, so every write raised
    AttributeError. The bank must be sized from the encoded wave."""
    r = client.post("/v1/cartridges/write", json={
        "tenant_id": "dimcheck",
        "keys": ["the pressure increases with temperature"],
        "values": ["the pressure rises"]})
    assert r.status_code == 200, r.text
    assert r.json()["rows"] == 1


def test_write_rejects_unequal_lengths(client):
    r = client.post("/v1/cartridges/write", json={
        "tenant_id": "bad", "keys": ["a", "b"], "values": ["c"]})
    assert r.status_code == 422


def test_write_then_mount_roundtrip(client):
    w = client.post("/v1/cartridges/write", json={
        "tenant_id": "rt", "keys": ["acid plus base"],
        "values": ["salt and water"]})
    assert w.status_code == 200, w.text
    m = client.post("/v1/cartridges/mount", json={"tenant_id": "rt"})
    assert m.status_code == 200
    assert m.json()["cartridges"] == 1


def test_namespaces_isolated(client):
    client.post("/v1/cartridges/write", json={
        "tenant_id": "iso_a", "keys": ["alpha"], "values": ["one"]})
    client.post("/v1/cartridges/write", json={
        "tenant_id": "iso_b", "keys": ["beta"], "values": ["two"]})
    a = client.post("/v1/cartridges/mount", json={"tenant_id": "iso_a"})
    b = client.post("/v1/cartridges/mount", json={"tenant_id": "iso_b"})
    assert a.json()["cartridges"] == b.json()["cartridges"] == 1


def test_code_solve_endpoint(client):
    r = client.post("/v1/code/solve", json={
        "spec": "add two numbers",
        "candidates": ["def add(a, b):\n    return a + b\n"]})
    assert r.status_code == 200, r.text
    assert r.json()["verdict"] == "NO_TESTS_UNVERIFIABLE"


def test_code_solve_rejects_empty_candidates(client):
    r = client.post("/v1/code/solve", json={"spec": "s", "candidates": []})
    assert r.status_code == 422
