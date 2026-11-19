"""Gateway IPC client for the `henri-sandbox` sidecar container.

When `sandbox_socket` is configured, all execution jobs are dispatched over
`AF_UNIX` to the isolated `henri-sandbox` container (`network_mode: none`,
read-only rootfs, dropped caps). When not configured (local dev / pytest),
it falls back to `execute_sandbox_job` locally and honestly reports
`mode: "local_child"` in status and receipts.
"""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from henri.gateway.sandbox_worker import (
    execute_sandbox_job,
    recv_frame,
    send_frame,
)
from henri.holarch.sandbox import CapabilityASTGate


class SandboxClient:
    """Two-stage sandbox execution client (sidecar IPC or local child fallback)."""

    def __init__(
        self,
        *,
        socket_path: Optional[str] = None,
        default_timeout_s: float = 10.0,
    ) -> None:
        self.socket_path = str(socket_path) if socket_path else None
        self.default_timeout_s = float(default_timeout_s)

    @property
    def mode(self) -> str:
        return "sidecar_ipc" if self.socket_path else "local_child"

    def _rpc(self, payload: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
        assert self.socket_path is not None
        conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        conn.settimeout(float(timeout_s) + 5.0)
        try:
            conn.connect(self.socket_path)
            send_frame(conn, payload)
            return recv_frame(conn)
        finally:
            conn.close()

    def ping(self) -> Dict[str, Any]:
        if self.socket_path:
            if not Path(self.socket_path).exists():
                return {
                    "ok": False,
                    "mode": "sidecar_ipc",
                    "socket": self.socket_path,
                    "reason": "socket_missing",
                    "evidence_label": "BLOCKED",
                }
            try:
                res = self._rpc({"op": "ping"}, timeout_s=3.0)
                return {
                    "ok": bool(res.get("ok")),
                    "mode": "sidecar_ipc",
                    "socket": self.socket_path,
                    "worker_pid": res.get("pid"),
                    "profiles": res.get("profiles", []),
                    "evidence_label": "OBSERVED",
                }
            except Exception as exc:
                return {
                    "ok": False,
                    "mode": "sidecar_ipc",
                    "socket": self.socket_path,
                    "reason": f"{type(exc).__name__}: {exc}",
                    "evidence_label": "BLOCKED",
                }
        return {
            "ok": True,
            "mode": "local_child",
            "socket": None,
            "note": "local child-process sandbox active (not an OS container jail)",
            "evidence_label": "OBSERVED",
        }

    def execute(
        self,
        source: str,
        *,
        tests: Sequence[str] = (),
        profile: str = "pure_algorithmic",
        timeout_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run Stage-1 AST capability gate in-gateway, then Stage-2 execution."""
        eff_timeout = float(timeout_s or self.default_timeout_s)
        combined = source if not tests else (source + "\n\n" + "\n".join(tests) + "\n")

        # Stage 1 always runs in-gateway before IPC so disallowed code never crosses.
        gate = CapabilityASTGate(profile).check(combined)
        if not gate["ok"]:
            return {
                "ok": False,
                "blocked": True,
                "stage": "ast_gate",
                "mode": self.mode,
                "reason": str(gate["reason"]),
                "blocked_items": list(gate.get("blocked") or []),
                "exit_code": -1,
                "stdout": "",
                "stderr": "",
                "timed_out": False,
                "elapsed_ms": 0.0,
                "evidence_label": "OBSERVED",
            }

        job = {
            "op": "exec",
            "source": str(source),
            "tests": [str(t) for t in tests],
            "profile": str(profile),
            "timeout_s": eff_timeout,
        }
        if self.socket_path:
            res = self._rpc(job, timeout_s=eff_timeout)
            res["mode"] = "sidecar_ipc"
            return res

        res = execute_sandbox_job(job)
        res["mode"] = "local_child"
        return res
