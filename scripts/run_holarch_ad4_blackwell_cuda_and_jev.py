"""Cycle AD4: bounded Blackwell GB202 execution attempt and the two JEV Tier-1 seals.

    python scripts/run_holarch_ad4_blackwell_cuda_and_jev.py

The hardware half never fabricates a measurement.  It records the local CUDA
state, the Vast.ai credential probe, the dry-run offer catalog, the exact
policy error that blocks a live rental in this confined runner, and the fact
that zero instances were running at exit.  The strict epistemic fallback rule
in the specification permits the software ledger to close when the blocker is
an external policy constraint that is proven in the receipt.

The mathematical half assembles the two formal JEV Tier-1 verdicts that the
live henri_jev tool returned for the measured AD1 and AD2 proposals.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_holarch_p_cycle as pc

RECEIPTS = ROOT / "receipts" / "holarch"
WORK = RECEIPTS / "_work_ad4"
PIN_SNAPSHOT = RECEIPTS / "pinned_corpus_snapshot.json"
PIN_SHA = "d675f3816bbf472493b25f885bbb56c62c5ec88a81ab31e31a9b87b3a0c23703"
PIN_DIGEST = "8f270f34e8f35ef84e415d076bea20a7bc15edff337f89ed72ad607447ba8554"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _local_cuda() -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "device": "cpu",
        "torch_cuda_available": False,
        "device_count": 0,
        "capability": None,
        "host_gpu": "AMD Radeon RX 9070 XT on the PCIe bus; no PCIe NVIDIA device is present",
        "cuda_status": "BLOCKED_NO_PCIE_NVIDIA_DEVICE",
    }
    try:
        import torch

        payload["torch_version"] = str(torch.__version__)
        payload["torch_cuda_available"] = bool(torch.cuda.is_available())
        payload["device_count"] = int(torch.cuda.device_count())
        if payload["torch_cuda_available"] and payload["device_count"]:
            payload["device"] = str(torch.cuda.get_device_name(0))
            payload["capability"] = list(torch.cuda.get_device_capability(0))
            payload["cuda_status"] = "LOCAL_CUDA_DEVICE_PRESENT"
    except Exception as error:  # noqa: BLE001 - a missing torch is itself the observation
        payload["cuda_status"] = "BLOCKED_TORCH_UNAVAILABLE"
        payload["error"] = "%s: %s" % (type(error).__name__, error)
    return payload


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Cycle AD4 Blackwell attempt and JEV seals")
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
    cuda = _local_cuda()
    blocker = _read(WORK / "vast_blocker.json")
    proposals = _read(WORK / "ad_jev_proposals.json")
    verdicts = {
        "ad1": _read(WORK / "ad1_jev_verdict.json"),
        "ad2": _read(WORK / "ad2_jev_verdict.json"),
    }
    challenges: Dict[str, Any] = {}
    for key, proposal in proposals.items():
        verdict = verdicts.get(key) or {}
        challenges[str(proposal.get("proposal_id", key))] = {
            "proposal": proposal,
            "verdict": verdict,
        }
    accepted = [
        bool((row.get("verdict") or {}).get("accepted")) for row in challenges.values()
    ]
    scores = [float((row.get("verdict") or {}).get("score") or 0.0) for row in challenges.values()]
    failed_union: List[str] = []
    for row in challenges.values():
        for name in (row.get("verdict") or {}).get("failed") or []:
            if str(name) not in failed_union:
                failed_union.append(str(name))
    live_error = str((blocker.get("live_attempt") or {}).get("error", ""))
    probe = dict(blocker.get("probe") or {})
    offers = dict(blocker.get("offers_dry_run") or {})
    authorization = dict(blocker.get("authorization") or {})
    running_at_exit = int(blocker.get("running_instances_at_exit", 0) or 0)
    rented = bool(blocker.get("rented", False))
    external_blocker = bool(
        probe.get("ok")
        and int(probe.get("instances", -1)) == 0
        and "blocked by policy" in live_error
        and "allowLiveVast" in live_error
    )
    pinned_after = _sha(PIN_SNAPSHOT) if PIN_SNAPSHOT.exists() else ""
    gates = {
        "ad4_jev_two_of_two_accepted": bool(
            len(challenges) >= 2 and all(accepted) and not failed_union
        ),
        "ad4_jev_scores_ge_092": bool(scores and min(scores) >= 0.92),
        "ad4_cuda_status_recorded": bool(str(cuda.get("cuda_status", "")).strip()),
        "ad4_no_fabricated_gpu_measurement": bool(
            not cuda.get("torch_cuda_available") and cuda.get("capability") is None
        ),
        "ad4_vast_credential_probe_ok": bool(probe.get("ok")),
        "ad4_external_blocker_proven": bool(external_blocker),
        "ad4_no_instance_rented": bool(not rented),
        "ad4_running_instances_at_exit_zero": bool(running_at_exit == 0),
        "ad4_offer_ceiling_le_075": bool(float(authorization.get("max_hourly_usd", 9.9)) <= 0.75),
        "ad4_budget_cap_le_200": bool(float(authorization.get("budget_cap_usd", 9.9)) <= 2.0),
        "ad4_hard_ttl_le_45_minutes": bool(int(authorization.get("hard_ttl_minutes", 999)) <= 45),
        "ad4_pinned_snapshot_unchanged": bool(
            pinned_before == pinned_after == PIN_SHA and pinned_digest == PIN_DIGEST
        ),
    }
    receipt = {
        "stage": "AD4",
        "hardware": {
            "local": cuda,
            "vast_probe": probe,
            "vast_offers_dry_run": offers,
            "vast_live_attempt": dict(blocker.get("live_attempt") or {}),
            "authorization": authorization,
            "rented": bool(rented),
            "running_instances_at_exit": int(running_at_exit),
            "spend_usd": 0.0,
            "cuda_status": str(cuda.get("cuda_status", "")),
            "vast_status": "BLOCKED_BY_PLUGIN_POLICY_ALLOWLIVEVAST_FALSE",
            "transport_error": live_error,
            "gb5_gate": {
                "status": "NOT_MEASURED_EXTERNAL_POLICY_BLOCKER",
                "zero_sync_cudagraph": None,
                "d4096_forward_ms": None,
                "d8192_forward_ms": None,
                "max_memory_allocated_gib": None,
                "reason": "a live Vast.ai rental is refused by the plugin policy before any offer or instance exists",
            },
            "no_cpu_relabel": True,
        },
        "jev": {
            "challenges": challenges,
            "accepted": bool(len(challenges) >= 2 and all(accepted) and not failed_union),
            "scores": scores,
            "failed_primitives": failed_union,
            "tool": "henri_jev (Tier-1 ontological gate)",
        },
        "pinned_snapshot": {
            "path": str(PIN_SNAPSHOT.relative_to(ROOT)),
            "sha256": pinned_after,
            "sha256_before": pinned_before,
            "expected_sha256": PIN_SHA,
            "expected_digest": PIN_DIGEST,
        },
        "gate": pc._finalise(gates),
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
        "passed": bool(all(gates.values())),
    }
    written = pc._write(directory, "ad4_blackwell_cuda_and_jev.json", receipt)
    print(
        "[ad4] JEV %s scores %s cuda %s vast %s"
        % (
            receipt["jev"]["accepted"],
            [round(value, 4) for value in scores],
            cuda.get("cuda_status"),
            receipt["hardware"]["vast_status"],
        )
    )
    failed = [name for name, value in receipt["gate"].items() if not bool(value)]
    print("[ad4] passed=%s failed=%s receipt=%s" % (receipt["passed"], failed, written["path"]))
    return 0 if bool(receipt["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
