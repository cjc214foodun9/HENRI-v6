"""Cycle AG unit & integration tests:
  - AG2: Zero-LLM raw-prose dimensional physics parser + ScienceSolver.solve_text + Gateway/MCP raw prompt
  - AG3 Part A: Multi-file (3-5 module) RepoPatchEngine + cross-module RepoSheaf capability/arity trap gating
  - AG3 Part B: Persistent hierarchical RiemannianChartAtlas (save/load mmap, tamper check, UnifiedGatewayService cold mount)
  - AG4: HolarchConfig.production() (D=4096) CPU Parseval norm conservation + save/load checkpoint parity
"""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from henri.gateway.app import create_gateway_app
from henri.gateway.config import GatewayConfig
from henri.gateway.mcp_mount import build_mcp_server
from henri.gateway.service import UnifiedGatewayService
from henri.gateway.signing import verify_signed_receipt
from henri.holarch.config import HolarchConfig
from henri.holarch.model import load_checkpoint, save_checkpoint
from henri.holarch.pipeline import HenriHolarch
from henri.holarch.repo_patch import (
    RepoCase,
    RepoPatchEngine,
    RepoSubprocessSandbox,
    materialise,
)
from henri.holarch.riemannian_atlas import RiemannianChartAtlas
from henri.holarch.science_parser import parse_science_text
from henri.holarch.science_solver import ScienceSolver

ROOT = Path(__file__).resolve().parents[1]


def _make_gateway(tmp_path: Path, persist_dir: Path | None = None):
    cfg = GatewayConfig(
        zone_c_backend="sqlite",
        sqlite_path=str(tmp_path / "gw.sqlite3"),
        zone_c_sqlite_path=str(tmp_path / "zonec.sqlite3"),
        signing_key_path=str(tmp_path / "ed25519.pem"),
        require_auth=True,
        holarch_autoload=False,
        atlas_persist_dir=str(persist_dir) if persist_dir is not None else None,
    )
    svc = UnifiedGatewayService(config=cfg)
    app = create_gateway_app(svc, mount_mcp=True)
    return svc, app


# --------------------------------------------------------------------------- AG2
def test_science_parser_and_solve_text_raw_prose():
    solver = ScienceSolver(ROOT)

    # 1. Photon energy with scientific notation in LaTeX + implicit vacuum constants.
    res_photon = solver.solve_text(
        r"Calculate the energy of a photon of wavelength $5.64 \times 10^{-7}$ m in vacuum."
    )
    assert res_photon["status"] == "SOLVED"
    assert res_photon["input_mode"] == "raw_text"
    assert res_photon["dimensional_ok"] is True
    assert float(res_photon["residual"]) <= 1e-9
    expected_photon = 6.62607015e-34 * 2.99792458e8 / 5.64e-7
    assert math.isclose(float(res_photon["answer"]), expected_photon, rel_tol=1e-3)

    # 2. Compound unit ideal gas pressure (dm^3, g/mol, K, g).
    res_gas = solver.solve_text(
        "A sample of 12.0 g of argon gas (molar mass 39.95 g/mol) occupies 5.0 dm^3 at 300 K. "
        "Calculate the pressure of the gas in Pa."
    )
    assert res_gas["status"] == "SOLVED"
    assert res_gas["dimensional_ok"] is True
    assert float(res_gas["residual"]) <= 1e-9
    expected_p = (12.0 / 39.95) * 8.314462618 * 300.0 / 5.0e-3
    assert math.isclose(float(res_gas["answer"]), expected_p, rel_tol=1e-3)

    # 3. Trap: unsolvable / missing required quantities must abstain fail-closed.
    res_trap = solver.solve_text(
        "Find the kinetic energy in J of a mysterious particle with unknown mass and unknown velocity."
    )
    assert res_trap["status"] == "ABSTAIN"
    assert res_trap.get("answer") is None

    # 4. Parser directly extracts compound units and subatomic particle constants.
    parsed = parse_science_text(
        "Compute the de Broglie wavelength of an electron with kinetic energy 100 eV."
    )
    assert parsed.ok is True
    assert parsed.problem["target"]["symbol"] == "wavelength"
    assert any(m.unit == "eV" for m in parsed.measurements)


def test_gateway_and_mcp_raw_prose_science(tmp_path: Path):
    svc, app = _make_gateway(tmp_path)
    try:
        api_key, _rec = svc.auth.create_key(tenant_id="sci_t", allowed_namespaces=["default"])
        client = TestClient(app)

        # REST POST /v1/holarch/science with raw prompt only (no pre-parsed quantities).
        res = client.post(
            "/v1/holarch/science",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "prompt": "A 2.0 kg block accelerates at 3.5 m/s^2. Calculate the net force in N.",
                "namespace": "default",
            },
        )
        assert res.status_code == 200
        body = res.json()
        assert body["result"]["status"] == "SOLVED"
        assert math.isclose(float(body["result"]["answer"]), 7.0, rel_tol=1e-6)
        rcpt = body.get("epistemic_receipt") or body.get("receipt")
        assert verify_signed_receipt(rcpt, svc.signer.jwks())[0] is True

        # Empty payload (neither prompt nor problem) fails closed with 422.
        res_empty = client.post(
            "/v1/holarch/science",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"namespace": "default"},
        )
        assert res_empty.status_code == 422

        # MCP holarch_science tool executes raw-prose prompt directly.
        mcp = build_mcp_server(svc)
        tools = asyncio.run(mcp.list_tools())
        assert "holarch_science" in {t.name for t in tools}
        mcp_out = asyncio.run(
            mcp._tool_manager.call_tool(
                "holarch_science",
                {
                    "prompt": "A 4.0 kg object moves at 3.0 m/s. Find its kinetic energy in J.",
                    "namespace": "default",
                },
            )
        )
        assert mcp_out["result"]["status"] == "SOLVED"
        assert math.isclose(float(mcp_out["result"]["answer"]), 18.0, rel_tol=1e-6)
        mcp_rcpt = mcp_out.get("epistemic_receipt") or mcp_out.get("receipt")
        assert verify_signed_receipt(mcp_rcpt, svc.signer.jwks())[0] is True
    finally:
        svc.shutdown()


# --------------------------------------------------------------------------- AG3 Part A
def test_multifile_repo_patch_and_cross_module_sheaf_traps(tmp_path: Path):
    base = ROOT / "receipts" / "holarch" / "external" / "repo_patch_multifile"
    manifest = json.loads((base / "MANIFEST.json").read_text(encoding="utf-8"))
    assert len(manifest["cases"]) == 12
    assert len(manifest["traps"]) == 4

    # Verify a 5-module deep defect (mf_08_feature_norm) localizes & passes hidden tests with E_sheaf = 0.0.
    case_raw = json.loads((base / "cases" / "mf_08_feature_norm.json").read_text(encoding="utf-8"))
    case = RepoCase(
        name=case_raw["id"],
        issue=case_raw["issue"],
        files=case_raw["files"],
        public=case_raw["public"],
        hidden=case_raw["hidden"],
    )
    out = RepoPatchEngine(rollouts=48, timeout_s=2.0).solve(case)
    assert out["status"] == "emitted"
    assert float(out["sheaf_after"]["energy"]) == 0.0
    assert out["patch"]["qualname"] == "mlops.kernel:signed_delta"
    assert int(out["llm_calls"]) == 0

    hid_dir = tmp_path / "mf08_hid"
    materialise(case, hid_dir, include_hidden=True, patched=out["patched_files"])
    hrun = RepoSubprocessSandbox(timeout_s=2.0).run_tests(hid_dir, "tests/test_hidden.py")
    assert hrun["passed"] is True

    # Verify cross-module capability & arity traps are refused by RepoSheaf (E_sheaf > 0.0).
    for trap_name in ("trap_mf_01_str_to_numeric_sum.json", "trap_mf_03_cross_module_arity.json"):
        tr = json.loads((base / "traps" / trap_name).read_text(encoding="utf-8"))
        tcase = RepoCase(
            name=tr["id"],
            issue=tr["issue"],
            files=tr["files"],
            public=tr["public"],
            hidden=tr["hidden"],
        )
        tout = RepoPatchEngine(rollouts=48, timeout_s=2.0).solve(tcase)
        assert tout["status"] == "abstained"
        assert not tout.get("patched_files")
        assert float(tout["sheaf_baseline"]["energy"]) > 0.0
        assert tout["reason"] == "sheaf_gate_blocked_all_candidates"


# --------------------------------------------------------------------------- AG3 Part B
def test_persistent_hierarchical_atlas_save_load_tamper_and_gateway_restart(tmp_path: Path):
    rng = np.random.default_rng(20261120)
    dim, domains, n_world, n_online = 64, 24, 2400, 400
    centers = rng.standard_normal((domains, dim)) + 1j * rng.standard_normal((domains, dim))
    centers /= np.linalg.norm(centers, axis=-1, keepdims=True)

    def sample(n: int) -> np.ndarray:
        lbl = rng.integers(0, domains, size=n)
        v = centers[lbl] + 0.30 * (rng.standard_normal((n, dim)) + 1j * rng.standard_normal((n, dim)))
        return np.ascontiguousarray(v / np.linalg.norm(v, axis=-1, keepdims=True), dtype=np.complex128)

    world = sample(n_world)
    online = sample(n_online)

    atlas = RiemannianChartAtlas(rank=8, capacity=128, b_stage=32, min_cluster=6, top_charts=2, top_charts_max=4)
    atlas.build(world, list(range(n_world)), namespace="ns")
    for off, vec in enumerate(online):
        atlas.observe(vec, n_world + off, namespace="ns")
    atlas.finalize()

    save_dir = tmp_path / "atlas_disk"
    meta = atlas.save(save_dir)
    assert meta["total_stored"] >= n_world + n_online

    # Reload via mmap with full SHA-256 verification and zero chart rebuilds.
    loaded = RiemannianChartAtlas.load(save_dir, mmap=True, verify_checksum="full")
    try:
        assert loaded.stats()["rebuilt_charts"] == 0
        assert loaded.stats()["rows"] == n_world + n_online
        for q_idx in rng.choice(n_world, size=32, replace=False):
            r_orig = atlas.query_top1(world[q_idx], namespace="ns")[0]
            r_load = loaded.query_top1(world[q_idx], namespace="ns")[0]
            assert r_load == r_orig
        for off in range(32):
            assert loaded.query_top1(online[off], namespace="ns")[0] == n_world + off
    finally:
        loaded.close()

    # Checksum tamper detection: flipping 1 byte in vectors.bin must raise ValueError.
    vpath = save_dir / "vectors.bin"
    raw = bytearray(vpath.read_bytes())
    raw[0] ^= 0xFF
    vpath.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="SHA-256 verification failed"):
        RiemannianChartAtlas.load(save_dir, mmap=True, verify_checksum="full")

    # Gateway restart persistence: build -> shutdown -> fresh UnifiedGatewayService mounts with rebuilt_charts==0.
    persist_root = tmp_path / "gw_persist"
    svc1, _app1 = _make_gateway(tmp_path / "gw1", persist_dir=persist_root)
    try:
        vecs = rng.standard_normal((48, 16)).tolist()
        svc1.atlas_build(vectors=vecs, rows=list(range(100, 148)), capacity=16, namespace="persist_ns")
        extra = (rng.standard_normal(16) * 2.0).tolist()
        svc1.atlas_observe(vector=extra, row=9999, finalize=True, namespace="persist_ns")
        q1 = svc1.atlas_query(vector=extra, k=1, namespace="persist_ns")
        assert q1["result"]["hits"][0]["row"] == 9999
    finally:
        svc1.shutdown()

    svc2, _app2 = _make_gateway(tmp_path / "gw2", persist_dir=persist_root)
    try:
        health = svc2.healthz()
        pm = health["atlas_persistence"]
        assert "persist_ns" in pm["loaded_namespaces"]
        assert pm["rebuilt_charts_total"] == 0
        q2 = svc2.atlas_query(vector=extra, k=1, namespace="persist_ns")
        assert q2["result"]["hits"][0]["row"] == 9999
    finally:
        svc2.shutdown()


# --------------------------------------------------------------------------- AG4
def test_d4096_production_parseval_and_checkpoint_parity_cpu(tmp_path: Path):
    torch.manual_seed(20261121)
    cfg = HolarchConfig.production(device="cpu", seed=20261121)
    assert cfg.dim == 4096
    model = HenriHolarch(cfg, backend="sqlite", memory_path=str(tmp_path / "z_c.sqlite3"))
    assert model.parameter_count() > 5_000_000

    x = torch.randn(2, 1, 8, 8)
    y = torch.tensor([1, 7], dtype=torch.long)

    # Check Parseval unit-norm conservation on photonic output wave and finite gradients.
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    opt.zero_grad(set_to_none=True)
    out = model("vision", x, training=True, use_memory=False)
    assert float(out.photonic.norm_drift) < 1e-5
    norms = torch.linalg.vector_norm(out.psi_in, dim=-1)
    assert float(torch.max(torch.abs(norms - 1.0)).item()) < 1e-5
    loss = torch.nn.functional.cross_entropy(out.task_logits, y)
    loss.backward()
    grad_norms = [float(p.grad.norm().item()) for p in model.parameters() if p.grad is not None]
    assert grad_norms and all(math.isfinite(g) for g in grad_norms)
    opt.step()

    # Save and reload checkpoint, asserting bit-exact forward parity on CPU.
    model.eval()
    with torch.no_grad():
        ref_out = model("vision", x, training=False, use_memory=False)
        ref_logits = ref_out.task_logits.clone()
        ref_bright = ref_out.photonic.bright.clone()
    ckpt = tmp_path / "holarch_d4096_test.pt"
    info = save_checkpoint(model, ckpt, cfg=cfg, name="ag4_test_d4096")
    assert info["parameters"] == model.parameter_count()

    reloaded, meta = load_checkpoint(ckpt)
    assert meta["config"]["dim"] == 4096
    with torch.no_grad():
        got_out = reloaded("vision", x, training=False, use_memory=False)
        got_logits = got_out.task_logits
        got_bright = got_out.photonic.bright
    assert torch.equal(ref_logits, got_logits)
    assert torch.equal(ref_bright, got_bright)
