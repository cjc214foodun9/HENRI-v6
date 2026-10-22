"""Phase 8 ACTION 4: `POST /v1/robotics/step` through the real HTTP stack.

Every clause states what would make it FAIL. A clause that cannot fail is
not a gate. Tests run at reduced dimension for speed; the full-D receipt
lives in `receipts/phase8_robotics_full_D.json` and is produced by
`scripts/probe_phase8_robotics.py`.

MEASURED CONTEXT, D=65536, this host:
    R1 identity   delta_init 0.000000  0 steps  PASS
    R2 near       delta_init 0.000001  0 steps  PASS  (non-identical)
    R3 unsafe     delta_init 0.895710  vetoed True, 6 steps -> 0.042648,
                  recovery True, but latency 72979 us > 50000 us digital
                  twin contract -> that ONE clause FAILs and is reported.
    R4 no axiom   vacuous_axiom True, delta 0.000000

The action slot carries real energy: slot0 0.783569, slot1 0.621304 at
D=65536 on the assembled candidate. A reference model reported
act_norm = 0.000786 for this run; that value is in no output on this host
and is refuted by `test_action_slot_carries_energy` below.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from henri import api_server                       # noqa: E402
from henri.config import WaveConfig                # noqa: E402
from henri.form_pin import INVARIANT_FORM          # noqa: E402
from henri.robotics_loop import RoboticsLoop       # noqa: E402

TAU_VETO = 0.0431
# Reduced dimension. The sealed slot partition needs dim divisible by 4.
TEST_DIM = 1024

SAFE = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
NEAR = [0.101, 0.199, 0.3005, 0.3998, 0.5002, 0.5999]
UNSAFE = [0.9, -0.5, 0.8, -0.7, 0.3, 0.1]

DIRECTIVE_FIELDS = (
    "vetoed_initial",
    "sagnac_delta_initial",
    "sagnac_delta_final",
    "adaptation_steps",
    "recovery_verified",
    "action_projection",
    "latency_us",
)


@pytest.fixture(scope="module")
def loop() -> RoboticsLoop:
    return RoboticsLoop(cfg=WaveConfig(dim=TEST_DIM), tau_veto=TAU_VETO)


@pytest.fixture()
def client(loop) -> TestClient:
    api_server.set_robotics(loop)
    yield TestClient(api_server.app)
    api_server.reset_robotics()


# -- clause 1: safe trajectory passes with no adaptation ------------------

def test_safe_identity_passes_in_zero_steps(client):
    r = client.post("/v1/robotics/step",
                    json={"sensor_values": SAFE,
                          "axiom_reference": SAFE})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["vetoed_initial"] is False
    assert b["adaptation_steps"] == 0
    assert b["recovery_verified"] is True
    assert b["sagnac_delta_initial"] <= TAU_VETO
    assert b["sagnac_delta_final"] <= TAU_VETO


def test_identity_case_reads_exact_zero(client):
    """The identity arm. NECESSARY, NEAR-VACUOUS, reported as such.

    What makes this able to fail: any slot asymmetry between the encode
    path and the assemble path. My first draft compared an entity-only
    axiom against an entity+action candidate and read 0.101 here.
    """
    b = client.post("/v1/robotics/step",
                    json={"sensor_values": SAFE,
                          "axiom_reference": SAFE}).json()
    assert b["sagnac_delta_initial"] == 0.0, (
        "identity case must read exact zero; a slot asymmetry would show")


# -- clause 2: the DISCRIMINATING safe clause -----------------------------

def test_near_safe_passes_and_is_not_identity(client):
    """Non-identical reference. This is the clause that carries weight.

    What makes it able to fail: (a) a false veto on a small perturbation,
    or (b) an exactly zero delta, which would mean the statistic is blind
    to the perturbation and clause 1 was the only thing ever measured.
    """
    b = client.post("/v1/robotics/step",
                    json={"sensor_values": NEAR,
                          "axiom_reference": SAFE}).json()
    assert b["vetoed_initial"] is False
    assert b["adaptation_steps"] == 0
    assert b["sagnac_delta_final"] <= TAU_VETO
    # The control: a near case that reads exactly zero is a blind statistic.
    assert b["sagnac_delta_initial"] > 0.0, (
        "near-but-not-identical must produce a nonzero delta")


# -- clause 3: unsafe trajectory triggers veto and recovers ---------------

def test_unsafe_triggers_veto_and_recovers(client):
    r = client.post("/v1/robotics/step",
                    json={"sensor_values": UNSAFE,
                          "axiom_reference": SAFE})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["sagnac_delta_initial"] > TAU_VETO, "veto must fire"
    assert b["vetoed_initial"] is True
    assert b["adaptation_steps"] > 0, "recovery must actually step"
    assert b["sagnac_delta_final"] <= TAU_VETO
    assert b["recovery_verified"] is True


def test_adapt_on_veto_false_leaves_veto_standing(client):
    """Control arm: with adaptation disabled the veto must NOT clear.

    What makes it able to fail: if the delta drops to zero with no
    relaxation, the recovery credited to the thermostat elsewhere would be
    wrongly attributed. That would be a self-confirming test.
    """
    b = client.post("/v1/robotics/step",
                    json={"sensor_values": UNSAFE,
                          "axiom_reference": SAFE,
                          "adapt_on_veto": False}).json()
    assert b["vetoed_initial"] is True
    assert b["adaptation_steps"] == 0
    assert b["sagnac_delta_final"] == pytest.approx(
        b["sagnac_delta_initial"], rel=0, abs=0)
    assert b["recovery_verified"] is False


def test_recovery_is_measured_not_asserted(client):
    """The final delta must come from the gate, not from the step counter.

    What makes it able to fail: if sagnac_delta_final were copied from a
    threshold rather than recomputed, it could not differ from
    delta_initial on the non-adapting arm (checked above) while dropping
    on the adapting arm.
    """
    adapt = client.post("/v1/robotics/step",
                        json={"sensor_values": UNSAFE,
                              "axiom_reference": SAFE}).json()
    no_adapt = client.post("/v1/robotics/step",
                           json={"sensor_values": UNSAFE,
                                 "axiom_reference": SAFE,
                                 "adapt_on_veto": False}).json()
    assert adapt["sagnac_delta_final"] < no_adapt["sagnac_delta_final"]


# -- contract shape -------------------------------------------------------

def test_response_carries_every_directive_field(client):
    b = client.post("/v1/robotics/step",
                    json={"sensor_values": SAFE,
                          "axiom_reference": SAFE}).json()
    missing = [f for f in DIRECTIVE_FIELDS if f not in b]
    assert not missing, f"missing directive fields: {missing}"


def test_action_projection_is_deterministic_and_finite(client):
    a = client.post("/v1/robotics/step",
                    json={"sensor_values": UNSAFE,
                          "axiom_reference": SAFE}).json()
    c = client.post("/v1/robotics/step",
                    json={"sensor_values": UNSAFE,
                          "axiom_reference": SAFE}).json()
    assert a["action_projection"] == c["action_projection"], (
        "seed is pinned, so the readout must repeat exactly")
    assert all(np.isfinite(v) for v in a["action_projection"])
    assert len(a["action_projection"]) == a["action_projection_len"]


def test_action_slot_carries_energy(client):
    """REFUTES the claim act_norm ~ 0.000786 at D=65536.

    A unit wave spread over four slots would put ~0.5 in one slot. The
    measured value at D=65536 is 0.621304. At reduced dimension the same
    structure must hold: the action slot is not an empty subspace.
    """
    b = client.post("/v1/robotics/step",
                    json={"sensor_values": UNSAFE,
                          "axiom_reference": SAFE}).json()
    assert b["action_slot_norm"] > 0.1, (
        "action slot must carry substantial energy, not a near-empty "
        "subspace")
    se = b["slot_energy"]
    assert se["slot0"] > 0.1
    assert abs(b["action_slot_norm"] - se["slot1"]) < 1e-6, (
        "action_slot_norm must equal the reported slot1 energy")


def test_statistic_form_is_the_invariant_form(client):
    b = client.post("/v1/robotics/step",
                    json={"sensor_values": SAFE,
                          "axiom_reference": SAFE}).json()
    assert b["statistic_form"] == INVARIANT_FORM
    # Dilution is reported, never hidden. On a sensor state slots 2 and 3
    # are left exactly zero, so the dense form would be diluted.
    assert b["dilution_applied"] >= 1.0


def test_latency_is_positive_and_labelled(client):
    b = client.post("/v1/robotics/step",
                    json={"sensor_values": SAFE,
                          "axiom_reference": SAFE}).json()
    assert isinstance(b["latency_us"], float)
    assert b["latency_us"] > 0.0
    assert "TELEMETRY_ONLY" in b["latency_note"]
    assert b["dim"] == TEST_DIM


# -- fail-closed paths ----------------------------------------------------

def test_empty_sensor_values_is_422(client):
    r = client.post("/v1/robotics/step", json={"sensor_values": []})
    assert r.status_code == 422


def test_non_finite_sensor_value_is_422(client):
    """The guard must reject a non-finite value that REACHES the handler.

    MEASURED DEFECT IN MY FIRST DRAFT, repaired. I wrote
    `client.post(json={"sensor_values": [0.1, float("nan"), 0.3]})`, which
    raised `ValueError: Out of range float values are not JSON compliant`
    inside httpx's encoder. The request never reached the server, so the
    test proved nothing about the server guard. httpx sends `allow_nan=False`
    while FastAPI's parser accepts the `NaN` literal, so a raw body is the
    only path that actually exercises the check.
    """
    r = client.post("/v1/robotics/step",
                    content=b'{"sensor_values": [0.1, NaN, 0.3]}',
                    headers={"content-type": "application/json"})
    assert r.status_code == 422, r.text


def test_non_finite_axiom_is_422(client):
    r = client.post(
        "/v1/robotics/step",
        content=('{"sensor_values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], '
                 '"axiom_reference": [0.1, Infinity, 0.3, 0.4, 0.5, 0.6]}'
                 ).encode("utf-8"),
        headers={"content-type": "application/json"})
    assert r.status_code == 422, r.text


def test_axiom_length_mismatch_is_422_not_silent_pass(client):
    """A mismatched safety reference is a caller error, never a pass."""
    r = client.post("/v1/robotics/step",
                    json={"sensor_values": SAFE,
                          "axiom_reference": [0.1, 0.2]})
    assert r.status_code == 422


def test_missing_axiom_sets_vacuity_flag(client):
    """Fail-open path is allowed but must be labelled, not disguised.

    What makes it able to fail: a self-referential axiom can never veto,
    so a caller reading only `recovery_verified` would believe a safety
    check ran. The flag is the disclosure.
    """
    b = client.post("/v1/robotics/step",
                    json={"sensor_values": SAFE}).json()
    assert b["vacuous_axiom"] is True
    assert b["axiom_source"] == "self"
    assert b["recovery_verified"] is True
    assert b["sagnac_delta_initial"] == 0.0
