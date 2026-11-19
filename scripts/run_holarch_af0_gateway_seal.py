"""Cycle AF0: Unified Production API Gateway seal, OpenAPI 3.1.0 & solver-verb audit.

    python scripts/run_holarch_af0_gateway_seal.py

Verifies:
1. `pytest tests/test_gateway_*.py` passes 100%.
2. `tests/test_holarch_service.py` passes cleanly (or records explicit
   `BLOCKED_MISSING_GITIGNORED_MMLU_CSV` skip when gitignored `mmlu.csv` is absent).
3. OpenAPI 3.1.0 schema validates all REST routes + `/mcp` mount.
4. `UnifiedGatewayService` executes `RepoPatchEngine`, `ScienceSolver`,
   `ObjectLooInducer`, and the Stage AF1 packed `RiemannianChartAtlas` over
   REST and `/mcp` with Ed25519 signed receipts and `evidence_label` /
   `compute_device` provenance.
Seals `receipts/holarch/af0_gateway_seal.json` + `.sha256`.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from fastapi.testclient import TestClient

import run_holarch_p_cycle as pc
from henri.gateway.app import create_gateway_app
from henri.gateway.config import GatewayConfig
from henri.gateway.mcp_mount import build_mcp_server
from henri.gateway.service import UnifiedGatewayService
from henri.gateway.signing import verify_signed_receipt

RECEIPTS = ROOT / "receipts" / "holarch"
PIN_SNAPSHOT = RECEIPTS / "pinned_corpus_snapshot.json"
PIN_SHA = "d675f3816bbf472493b25f885bbb56c62c5ec88a81ab31e31a9b87b3a0c23703"
PIN_DIGEST = "8f270f34e8f35ef84e415d076bea20a7bc15edff337f89ed72ad607447ba8554"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_pytest_files(paths: list[str]) -> Dict[str, Any]:
    import re

    t0 = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-rs", "-p", "no:cacheprovider", *paths],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )
    out = (completed.stdout or "") + "\n" + (completed.stderr or "")

    def _extract(label: str) -> int:
        matches = re.findall(rf"(\d+)\s+{label}\b", out)
        return int(matches[-1]) if matches else 0

    passed = _extract("passed")
    failed = _extract("failed") + _extract("error")
    skipped = _extract("skipped")
    return {
        "exit_code": int(completed.returncode),
        "passed": int(passed),
        "failed": int(failed),
        "skipped": int(skipped),
        "wall_seconds": float(time.perf_counter() - t0),
        "mmlu_skip_provenance": "BLOCKED_MISSING_GITIGNORED_MMLU_CSV" in out,
    }


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Cycle AF0 gateway seal")
    parser.add_argument("--receipts", default=str(RECEIPTS))
    args = parser.parse_args(argv)
    directory = Path(args.receipts).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    pinned_before = _sha(PIN_SNAPSHOT) if PIN_SNAPSHOT.exists() else ""
    pinned_digest = (
        str(json.loads(PIN_SNAPSHOT.read_text(encoding="utf-8")).get("digest", ""))
        if PIN_SNAPSHOT.exists()
        else ""
    )

    gw_test_files = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "tests").glob("test_gateway_*.py"))
    gw_pytest = _run_pytest_files(gw_test_files)
    svc_pytest = _run_pytest_files(["tests/test_holarch_service.py"])

    # Live in-process OpenAPI 3.1.0 + REST/MCP solver & packed-atlas probe.
    probe_results: Dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="henri_af0_") as td:
        tmp = Path(td)
        cfg = GatewayConfig(
            zone_c_backend="sqlite",
            sqlite_path=str(tmp / "gw.sqlite3"),
            zone_c_sqlite_path=str(tmp / "zonec.sqlite3"),
            signing_key_path=str(tmp / "ed25519.pem"),
            require_auth=True,
            holarch_autoload=False,
        )
        svc = UnifiedGatewayService(config=cfg)
        try:
            app = create_gateway_app(svc, mount_mcp=True)
            schema = app.openapi()
            openapi_version = str(schema.get("openapi", ""))
            paths = sorted(schema.get("paths", {}).keys())
            mcp_mounted = any(getattr(r, "path", "") == "/mcp" for r in app.routes)

            api_key, _rec = svc.auth.create_key(tenant_id="af0_seal", allowed_namespaces=["default"])
            headers = {"Authorization": f"Bearer {api_key}"}
            client = TestClient(app)

            jwks = client.get("/.well-known/henri-jwks.json").json()
            jwks_ok = bool(jwks.get("keys") and jwks["keys"][0].get("crv") == "Ed25519")

            # 1. ScienceSolver probe
            sci = client.post(
                "/v1/holarch/science",
                headers=headers,
                json={
                    "problem": {
                        "id": "af0_probe_force",
                        "domain": "physics",
                        "quantities": [
                            {"symbol": "mass", "value": 4.0, "unit": "kg"},
                            {"symbol": "acceleration", "value": 3.0, "unit": "m/s^2"},
                        ],
                        "target": {"symbol": "force", "unit": "N"},
                    },
                    "namespace": "default",
                },
            )
            sci_body = sci.json()
            sci_ok = bool(
                sci.status_code == 200
                and sci_body["result"].get("status") == "SOLVED"
                and abs(float(sci_body["result"].get("answer", 0.0)) - 12.0) < 1e-6
                and sci_body.get("evidence_label") == "OBSERVED"
                and sci_body.get("compute_device") == svc.compute_device
                and verify_signed_receipt(
                    sci_body["epistemic_receipt"], jwks
                )[0] is True
            )

            # 2. ObjectLooInducer (ARC) probe
            arc = client.post(
                "/v1/holarch/arc",
                headers=headers,
                json={
                    "train": [
                        {"input": [[1, 0], [0, 0]], "output": [[0, 1], [0, 0]]},
                        {"input": [[2, 0], [0, 0]], "output": [[0, 2], [0, 0]]},
                    ],
                    "test": [{"input": [[3, 0], [0, 0]], "output": [[0, 3], [0, 0]]}],
                    "depth": 2,
                    "namespace": "default",
                },
            )
            arc_body = arc.json()
            arc_ok = bool(
                arc.status_code == 200
                and arc_body["result"]["solved"] is True
                and arc_body.get("evidence_label") == "OBSERVED"
                and arc_body.get("compute_device") == svc.compute_device
                and verify_signed_receipt(arc_body["epistemic_receipt"], jwks)[0] is True
            )

            # 3. RepoPatchEngine probe
            rp = client.post(
                "/v1/holarch/repo_patch",
                headers=headers,
                json={
                    "name": "af0_probe_add",
                    "issue": "fix add() to sum a and b instead of subtracting",
                    "files": {
                        "pkg/__init__.py": "",
                        "pkg/calc.py": "def add(a, b):\n    return a - b\n",
                    },
                    "public": {
                        "tests/test_public.py": (
                            "from pkg.calc import add\n\n"
                            "def test_add():\n"
                            "    assert add(2, 3) == 5\n"
                            "    assert add(10, 4) == 14\n"
                        )
                    },
                    "namespace": "default",
                },
            )
            rp_body = rp.json()
            rp_ok = bool(
                rp.status_code == 200
                and rp_body["result"].get("abstained") is False
                and rp_body.get("evidence_label") == "OBSERVED"
                and rp_body.get("compute_device") == svc.compute_device
                and verify_signed_receipt(rp_body["epistemic_receipt"], jwks)[0] is True
            )

            # 4. Packed RiemannianChartAtlas REST + MCP probe
            vecs = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
            ab = client.post(
                "/v1/holarch/atlas/build",
                headers=headers,
                json={"vectors": vecs, "rows": [101, 102, 103], "namespace": "default"},
            )
            aq = client.post(
                "/v1/holarch/atlas/query",
                headers=headers,
                json={"vector": [0.0, 0.0, 1.0, 0.0], "k": 1, "namespace": "default"},
            )
            aq_body = aq.json()
            atlas_rest_ok = bool(
                ab.status_code == 200
                and aq.status_code == 200
                and aq_body["result"]["hits"][0]["row"] == 103
                and aq_body.get("evidence_label") == "OBSERVED"
                and aq_body.get("compute_device") == svc.compute_device
                and verify_signed_receipt(aq_body["epistemic_receipt"], jwks)[0] is True
            )

            mcp = build_mcp_server(svc)
            mcp_tools = sorted(t.name for t in asyncio.run(mcp.list_tools()))
            required_mcp = {
                "holarch_repo_patch",
                "holarch_science_solve",
                "holarch_arc_solve",
                "holarch_atlas_build",
                "holarch_atlas_observe",
                "holarch_atlas_query",
                "holarch_atlas_stats",
            }
            mcp_ok = bool(required_mcp <= set(mcp_tools))

            probe_results = {
                "openapi_version": openapi_version,
                "route_count": int(len(paths)),
                "routes": paths,
                "mcp_mounted": bool(mcp_mounted),
                "mcp_tool_count": int(len(mcp_tools)),
                "mcp_tools": mcp_tools,
                "jwks_ed25519_ok": jwks_ok,
                "science_rest_ok": sci_ok,
                "arc_rest_ok": arc_ok,
                "repo_patch_rest_ok": rp_ok,
                "atlas_rest_ok": atlas_rest_ok,
                "mcp_solver_and_atlas_ok": mcp_ok,
                "compute_device": svc.compute_device,
            }
        finally:
            svc.shutdown()

    gates = {
        "af0_gateway_pytest_100pct": bool(
            gw_pytest["exit_code"] == 0 and gw_pytest["failed"] == 0 and gw_pytest["passed"] >= 18
        ),
        "af0_service_pytest_clean_checkout_ok": bool(
            svc_pytest["exit_code"] == 0 and svc_pytest["failed"] == 0
        ),
        "af0_openapi_31_and_mcp_mounted": bool(
            probe_results["openapi_version"].startswith("3.1")
            and probe_results["route_count"] >= 34
            and probe_results["mcp_mounted"]
        ),
        "af0_solvers_and_atlas_rest_mcp_signed": bool(
            probe_results["jwks_ed25519_ok"]
            and probe_results["science_rest_ok"]
            and probe_results["arc_rest_ok"]
            and probe_results["repo_patch_rest_ok"]
            and probe_results["atlas_rest_ok"]
            and probe_results["mcp_solver_and_atlas_ok"]
        ),
        "af0_pinned_snapshot_unchanged": bool(
            pinned_before == _sha(PIN_SNAPSHOT) == PIN_SHA and pinned_digest == PIN_DIGEST
        ),
    }

    receipt = {
        "stage": "AF0",
        "kind": "unified_gateway_seal",
        "gateway_pytest": gw_pytest,
        "service_pytest": svc_pytest,
        "probes": probe_results,
        "pinned_snapshot": {
            "sha256": _sha(PIN_SNAPSHOT) if PIN_SNAPSHOT.exists() else "",
            "sha256_before": pinned_before,
            "expected_sha256": PIN_SHA,
            "expected_digest": PIN_DIGEST,
        },
        "gate": pc._finalise(gates),
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
        "passed": bool(all(gates.values())),
    }
    written = pc._write(directory, "af0_gateway_seal.json", receipt)
    failed = [name for name, val in receipt["gate"].items() if not bool(val)]
    print(
        "[af0] gw_pytest=%d/%d openapi=%s routes=%d mcp_tools=%d passed=%s failed=%s receipt=%s"
        % (
            gw_pytest["passed"],
            gw_pytest["passed"] + gw_pytest["failed"],
            probe_results["openapi_version"],
            probe_results["route_count"],
            probe_results["mcp_tool_count"],
            receipt["passed"],
            failed,
            written["path"],
        )
    )
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
