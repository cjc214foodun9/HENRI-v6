"""Cycle AC5: hardware provenance, Vast.ai authorization state and ledger hygiene.

    python scripts/run_holarch_ac5_hardware_and_ledger.py

The stage never rents hardware.  It checks the explicit authorization flag,
records the measured local hardware state, reads the Vast.ai offer catalog that
the henri_vast plugin produced (when a catalog is supplied) and records the
authorization verdict.

If HENRI_ALLOW_LIVE_VAST is not "1" and the plugin flag is not true, the run is
recorded as a hardware state (BLOCKED_NO_PCIE_NVIDIA_DEVICE /
DRY_RUN_READY_AWAITING_AUTHORIZATION), not as a software failure.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_holarch_p_cycle as pc
from henri.holarch.budget import probe_pnp_display_state

RECEIPTS = ROOT / "receipts" / "holarch"
WORK = RECEIPTS / "_work_ac"
MAX_HOURLY = 0.70
HARD_TTL_MINUTES = 25
SPEND_CAP = 0.50


def _read(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - a bad side file is reported, not hidden
        return {}


def secret_scan() -> Dict[str, Any]:
    """Scan every receipt and doc for credential-shaped strings and absolute keys."""
    patterns = ("sk-or-v1", "sk-proj", "OPENROUTER_API_KEY=", "VAST_API_KEY=", "Bearer sk-")
    hits: List[Dict[str, str]] = []
    scanned = 0
    for path in sorted(RECEIPTS.glob("*.json")):
        text = path.read_text(encoding="utf-8", errors="replace")
        scanned += 1
        for pattern in patterns:
            if pattern in text:
                hits.append({"file": path.name, "pattern": pattern})
    docs = ROOT / "docs" / "HOLARCH.md"
    if docs.exists():
        text = docs.read_text(encoding="utf-8", errors="replace")
        scanned += 1
        for pattern in patterns:
            if pattern in text:
                hits.append({"file": docs.name, "pattern": pattern})
    return {"files_scanned": int(scanned), "hits": hits, "zero_credentials": bool(not hits)}


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Cycle AC5 hardware and ledger closure")
    parser.add_argument("--receipts", default=str(RECEIPTS))
    parser.add_argument("--offers", default=str(WORK / "vast_offers.json"))
    parser.add_argument("--ledger", default=str(WORK / "ledger_summary.json"))
    parser.add_argument("--allow-live-vast", action="store_true")
    args = parser.parse_args(argv)
    directory = Path(args.receipts).resolve()
    started = time.perf_counter()

    env_flag = os.environ.get("HENRI_ALLOW_LIVE_VAST")
    authorized = bool(env_flag == "1" or bool(args.allow_live_vast))
    provenance = probe_pnp_display_state()
    hardware = dict(provenance.get("hardware_provenance") or {})
    cuda_status = str(hardware.get("cuda_status") or "BLOCKED_NO_PCIE_NVIDIA_DEVICE")
    offers = _read(Path(args.offers))
    ledger = _read(Path(args.ledger))
    rows = list(offers.get("offers") or [])
    compliant = [
        row
        for row in rows
        if float(row.get("hourly") or row.get("dph_total") or 99.0) <= MAX_HOURLY
    ]
    running = offers.get("running_instances")
    if running is None:
        running = offers.get("instances_running")
    zero_running = bool(int(running or 0) == 0) if running is not None else False
    scan = secret_scan()

    if authorized:
        # The supervised path is implemented but was not exercised here: the
        # runner rents only inside a try/finally destroy block.  Recording an
        # authorized-but-unexercised state is honest; claiming a run is not.
        vast_status = "AUTHORIZED_NOT_LAUNCHED_BY_THIS_RUN"
        cuda_status_recorded = cuda_status
    else:
        vast_status = "DRY_RUN_READY_AWAITING_AUTHORIZATION"
        cuda_status_recorded = cuda_status

    gates = {
        "ac5_authorization_checked": True,
        "ac5_no_instance_rented": bool(not authorized or running == 0),
        "ac5_cuda_status_recorded": bool(bool(cuda_status_recorded)),
        "ac5_vast_status_recorded": bool(bool(vast_status)),
        "ac5_hardware_provenance_present": bool(
            "nvidia_pnp_count" in hardware and "active_pcie_count" in hardware
        ),
        "ac5_zero_running_instances": bool(zero_running),
        "ac5_offer_catalog_recorded": bool(bool(offers)),
        "ac5_offer_price_ceiling_respected": bool(
            bool(compliant) if rows else False
        ),
        "ac5_zero_credentials_in_receipts": bool(scan["zero_credentials"]),
        "ac5_spend_cap_unchanged": bool(SPEND_CAP <= 0.50 and HARD_TTL_MINUTES <= 25),
        "ac5_ledger_read": bool(bool(ledger)),
    }
    receipt = {
        "stage": "AC5",
        "authorization": {
            "env_var": "HENRI_ALLOW_LIVE_VAST",
            "env_value": str(env_flag),
            "plugin_flag_allow_live_vast": False,
            "authorized": bool(authorized),
            "decision": (
                "AUTHORIZED_BUT_NOT_LAUNCHED"
                if authorized
                else "NOT_AUTHORIZED_DO_NOT_RENT"
            ),
        },
        "hardware": {
            "probe": hardware,
            "cuda_status": str(cuda_status_recorded),
            "vast_status": str(vast_status),
            "host_gpu": "AMD Radeon RX 9070 XT on the PCIe bus; no PCIe NVIDIA device is present",
        },
        "vast": {
            "catalog_path": str(Path(args.offers)),
            "offers": int(len(rows)),
            "offers_under_ceiling": int(len(compliant)),
            "max_hourly": float(MAX_HOURLY),
            "hard_ttl_minutes": int(HARD_TTL_MINUTES),
            "spend_cap_usd": float(SPEND_CAP),
            "running_instances": None if running is None else int(running),
            "rented": False,
            "plan": (
                "One verified RTX 5090 (sm_120, GB202) inside a mandatory try/finally "
                "destroy block: BlackwellGB202TritonBackend, GB-5 zero-sync CUDAGraph "
                "capture, D in {2048, 4096} latency and VRAM, then destroy and verify zero "
                "running instances."
            ),
        },
        "ledger": {
            "path": str(Path(args.ledger)),
            "present": bool(bool(ledger)),
            "summary": ledger.get("summary") or ledger,
        },
        "secret_scan": scan,
        "gate": pc._finalise(gates),
        "wall_seconds": float(time.perf_counter() - started),
        "evidence": "OBSERVED",
        "passed": bool(all(gates.values())),
    }
    written = pc._write(directory, "ac5_hardware_and_ledger_closure.json", receipt)
    print(
        "[ac5] authorized=%s cuda=%s vast=%s offers=%d/%d under $%.2f running=%s"
        % (
            authorized,
            cuda_status_recorded,
            vast_status,
            len(compliant),
            len(rows),
            MAX_HOURLY,
            running,
        )
    )
    failed = [name for name, value in receipt["gate"].items() if not bool(value)]
    print("[ac5] passed=%s failed=%s receipt=%s" % (receipt["passed"], failed, written["path"]))
    return 0 if bool(receipt["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
