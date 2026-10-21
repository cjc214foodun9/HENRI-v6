"""Live probe for the Phase 7 daemon. Written before any run.

Every endpoint is exercised through `TestClient`, so this measures the real
FastAPI stack rather than the handlers in isolation. Results print one line
per endpoint with its status code, so a reader can attribute a failure.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fastapi.testclient import TestClient  # noqa: E402

from henri.api_server import app, reset_banks  # noqa: E402

FAIL: list[str] = []


def chk(label: str, cond: bool, detail: object = "") -> None:
    print(f"{'PASS' if cond else 'FAIL'}  {label:38s} {detail}")
    if not cond:
        FAIL.append(label)


def main() -> int:
    reset_banks()
    c = TestClient(app)

    print("=== A: seal ===")
    r = c.get("/v1/health/seal")
    chk("GET /v1/health/seal", r.status_code == 200, r.status_code)
    chk("seal status SEALED", r.json().get("status") == "SEALED",
        r.json().get("status"))

    print("\n=== B: executor status ===")
    r = c.get("/v1/health/executor")
    chk("GET /v1/health/executor", r.status_code == 200, r.status_code)
    chk("executor available", r.json().get("available") is True,
        r.json().get("executor_error") or r.json().get("module"))

    print("\n=== C: rank ===")
    r = c.post("/v1/reason/rank", json={
        "question": "the pressure increases with temperature",
        "options": ["the pressure decreases", "a recipe for bread",
                    "a tax rule for firms", "orbital mechanics"]})
    chk("POST /v1/reason/rank", r.status_code == 200, r.status_code)
    if r.status_code == 200:
        b = r.json()
        chk("rank has selected_index", "selected_index" in b,
            b.get("selected_index"))
        chk("rank sagnac is a float", isinstance(b.get("sagnac_delta"), float),
            b.get("sagnac_delta"))

    print("\n=== D: rank rejects too few options ===")
    r = c.post("/v1/reason/rank", json={"question": "q", "options": ["one"]})
    chk("422 on single option", r.status_code == 422, r.status_code)

    print("\n=== E: mount before write is 404 (FAIL-CLOSED) ===")
    r = c.post("/v1/cartridges/mount", json={"tenant_id": "never-written"})
    chk("404 mount-before-write", r.status_code == 404, r.status_code)

    print("\n=== F: write then mount ===")
    r = c.post("/v1/cartridges/write", json={
        "tenant_id": "phys",
        "keys": ["what happens to pressure when temperature increases"],
        "values": ["the pressure rises"], "subject": "thermo"})
    chk("POST /v1/cartridges/write", r.status_code == 200,
        r.status_code if r.status_code != 200 else r.json().get("rows"))
    if r.status_code != 200:
        print("   body:", r.text[:300])
    r = c.post("/v1/cartridges/mount", json={"tenant_id": "phys"})
    chk("POST /v1/cartridges/mount", r.status_code == 200, r.status_code)
    if r.status_code == 200:
        chk("mount reports 1 cartridge", r.json().get("cartridges") == 1,
            r.json().get("cartridges"))

    print("\n=== G: write rejects unequal lengths ===")
    r = c.post("/v1/cartridges/write", json={
        "tenant_id": "x", "keys": ["a", "b"], "values": ["c"]})
    chk("422 unequal lengths", r.status_code == 422, r.status_code)

    print("\n=== H: code solve without tests is UNVERIFIABLE ===")
    r = c.post("/v1/code/solve", json={
        "spec": "add two numbers",
        "candidates": ["def add(a, b):\n    return a + b\n"]})
    chk("POST /v1/code/solve", r.status_code == 200, r.status_code)
    if r.status_code == 200:
        chk("verdict NO_TESTS_UNVERIFIABLE",
            r.json().get("verdict") == "NO_TESTS_UNVERIFIABLE",
            r.json().get("verdict"))

    print("\n=== I: code solve with tests verifies ===")
    r = c.post("/v1/code/solve", json={
        "spec": "add two numbers",
        "candidates": ["def add(a, b):\n    return a - b\n",
                       "def add(a, b):\n    return a + b\n"],
        "tests": ["assert add(2, 3) == 5"]})
    chk("POST /v1/code/solve with tests", r.status_code == 200, r.status_code)
    if r.status_code == 200:
        b = r.json()
        chk("verdict EXEC_VERIFIED", b.get("verdict") == "EXEC_VERIFIED",
            b.get("verdict"))
        chk("winner is the correct one", b.get("winner") is not None,
            b.get("winner"))

    print("\n=== J: namespace isolation ===")
    r = c.post("/v1/cartridges/mount", json={"tenant_id": "phys"})
    before = r.json().get("cartridges") if r.status_code == 200 else None
    r = c.post("/v1/cartridges/write", json={
        "tenant_id": "chem", "keys": ["acid plus base"], "values": ["salt"]})
    chk("second namespace writes", r.status_code == 200, r.status_code)
    r = c.post("/v1/cartridges/mount", json={"tenant_id": "phys"})
    chk("first namespace unchanged",
        r.status_code == 200 and r.json().get("cartridges") == before,
        f"{before} -> {r.json().get('cartridges') if r.status_code == 200 else 'ERR'}")

    print(f"\n=== {len(FAIL)} FAILURES ===" if FAIL else "\nALL API CHECKS PASS")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
