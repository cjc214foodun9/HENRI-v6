"""Hardened sandbox worker daemon for the `henri-sandbox` sidecar container.

Evidence label: OBSERVED for every execution record emitted.

Runs inside a container configured with `network_mode: none`, read-only rootfs,
`cap_drop: ALL`, non-root user, and cgroup memory/CPU/PID ceilings. Listens on
an internal Unix domain socket (`AF_UNIX`) using 4-byte big-endian length-
prefixed UTF-8 JSON frames.

Two-stage fail-closed boundary:
  Stage 1: `CapabilityASTGate(profile)` statically rejects disallowed imports,
           dynamic loaders, dunder escapes, and absolute path literals.
  Stage 2: Executes inside a fresh child Python process with scrubbed env,
           POSIX rlimits (RLIMIT_CPU, RLIMIT_NOFILE, optional RLIMIT_AS),
           wall-clock kill timeout, capped output, and SHA-256 artifact digest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from henri.holarch.sandbox import (
    PROFILE_ALLOWLISTS,
    CapabilityASTGate,
    SubprocessSandboxExecutor,
    extract_child_result,
)

MAX_FRAME_BYTES = 1_048_576  # 1 MiB IPC cap
MAX_OUTPUT_BYTES = 65_536


def _recv_exact(conn: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise EOFError("unexpected EOF on sandbox IPC socket")
        buf.extend(chunk)
    return bytes(buf)


def send_frame(conn: socket.socket, payload: Dict[str, Any]) -> None:
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    if len(raw) > MAX_FRAME_BYTES:
        raise ValueError(f"IPC payload exceeds {MAX_FRAME_BYTES} bytes")
    conn.sendall(struct.pack("!I", len(raw)) + raw)


def recv_frame(conn: socket.socket) -> Dict[str, Any]:
    hdr = _recv_exact(conn, 4)
    (length,) = struct.unpack("!I", hdr)
    if length <= 0 or length > MAX_FRAME_BYTES:
        raise ValueError(f"invalid IPC frame length: {length}")
    body = _recv_exact(conn, length)
    data = json.loads(body.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("IPC frame must decode to JSON object")
    return data


def _apply_posix_rlimits(cpu_seconds: int, mem_mb: int) -> None:
    try:
        import resource

        cpu = max(1, int(cpu_seconds))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
        resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
        if mem_mb > 0:
            mem_bytes = int(mem_mb) * 1024 * 1024
            try:
                resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
            except (ValueError, OSError):
                pass
    except Exception:
        pass


def execute_sandbox_job(job: Dict[str, Any]) -> Dict[str, Any]:
    """Execute one validated sandbox job and return an OBSERVED record."""
    op = str(job.get("op") or "exec")
    if op == "ping":
        return {
            "ok": True,
            "op": "ping",
            "pid": os.getpid(),
            "profiles": sorted(PROFILE_ALLOWLISTS),
            "evidence_label": "OBSERVED",
        }

    source = str(job.get("source") or "")
    tests = [str(t) for t in (job.get("tests") or [])]
    profile = str(job.get("profile") or "pure_algorithmic")
    timeout_s = max(0.5, min(float(job.get("timeout_s", 10.0)), 30.0))
    mem_mb = int(job.get("mem_mb", 0))

    if profile not in PROFILE_ALLOWLISTS:
        return {
            "ok": False,
            "blocked": True,
            "reason": f"unknown sandbox profile: {profile!r}",
            "exit_code": -1,
            "stdout": "",
            "stderr": "",
            "timed_out": False,
            "elapsed_ms": 0.0,
            "artifact_sha256": "",
            "evidence_label": "OBSERVED",
        }

    combined = source if not tests else (source + "\n\n" + "\n".join(tests) + "\n")
    gate = CapabilityASTGate(profile)
    gate_res = gate.check(combined)
    if not gate_res["ok"]:
        return {
            "ok": False,
            "blocked": True,
            "reason": str(gate_res["reason"]),
            "blocked_items": list(gate_res.get("blocked") or []),
            "exit_code": -1,
            "stdout": "",
            "stderr": "",
            "timed_out": False,
            "elapsed_ms": 0.0,
            "artifact_sha256": hashlib.sha256(combined.encode("utf-8")).hexdigest(),
            "evidence_label": "OBSERVED",
        }

    started = time.perf_counter()
    scrub = SubprocessSandboxExecutor(profile=profile, timeout_s=timeout_s).child_env()
    with tempfile.TemporaryDirectory(prefix="henri_sbx_") as tmp:
        script_path = Path(tmp) / "_task.py"
        script_path.write_text(combined, encoding="utf-8")
        artifact_sha = hashlib.sha256(combined.encode("utf-8")).hexdigest()
        preexec = (
            (lambda: _apply_posix_rlimits(int(timeout_s) + 1, mem_mb))
            if os.name == "posix"
            else None
        )
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-S", str(script_path)],
                cwd=tmp,
                env=scrub,
                capture_output=True,
                timeout=timeout_s,
                preexec_fn=preexec,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            stdout = proc.stdout[:MAX_OUTPUT_BYTES].decode("utf-8", errors="replace")
            stderr = proc.stderr[:MAX_OUTPUT_BYTES].decode("utf-8", errors="replace")
            return {
                "ok": proc.returncode == 0,
                "blocked": False,
                "reason": "ok" if proc.returncode == 0 else f"exit_code:{proc.returncode}",
                "exit_code": int(proc.returncode),
                "stdout": stdout,
                "stderr": stderr,
                "result": extract_child_result(stdout),
                "timed_out": False,
                "elapsed_ms": round(elapsed_ms, 3),
                "artifact_sha256": artifact_sha,
                "evidence_label": "OBSERVED",
            }
        except subprocess.TimeoutExpired as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            stdout = (exc.stdout or b"")[:MAX_OUTPUT_BYTES].decode("utf-8", errors="replace")
            stderr = (exc.stderr or b"")[:MAX_OUTPUT_BYTES].decode("utf-8", errors="replace")
            return {
                "ok": False,
                "blocked": False,
                "reason": "timeout",
                "exit_code": -9,
                "stdout": stdout,
                "stderr": stderr,
                "result": None,
                "timed_out": True,
                "elapsed_ms": round(elapsed_ms, 3),
                "artifact_sha256": artifact_sha,
                "evidence_label": "OBSERVED",
            }


class SandboxUnixServer:
    """Threaded AF_UNIX server for `henri-sandbox`."""

    def __init__(self, socket_path: str) -> None:
        self.socket_path = str(socket_path)
        self._sock: Optional[socket.socket] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _handle_client(self, conn: socket.socket) -> None:
        try:
            conn.settimeout(35.0)
            req = recv_frame(conn)
            resp = execute_sandbox_job(req)
            send_frame(conn, resp)
        except Exception as exc:
            try:
                send_frame(
                    conn,
                    {
                        "ok": False,
                        "blocked": True,
                        "reason": f"worker_error: {type(exc).__name__}: {exc}",
                        "exit_code": -1,
                        "stdout": "",
                        "stderr": "",
                        "timed_out": False,
                        "elapsed_ms": 0.0,
                        "artifact_sha256": "",
                        "evidence_label": "OBSERVED",
                    },
                )
            except Exception:
                pass
        finally:
            conn.close()

    def start(self) -> None:
        sock_file = Path(self.socket_path)
        sock_file.parent.mkdir(parents=True, exist_ok=True)
        if sock_file.exists():
            sock_file.unlink()
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(self.socket_path)
        os.chmod(self.socket_path, 0o660)
        srv.listen(32)
        srv.settimeout(0.5)
        self._sock = srv
        self._stop.clear()

        def _loop() -> None:
            while not self._stop.is_set():
                try:
                    conn, _ = srv.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                t = threading.Thread(
                    target=self._handle_client, args=(conn,), daemon=True
                )
                t.start()

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def serve_forever(self) -> None:
        self.start()
        try:
            while not self._stop.is_set():
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self) -> None:
        self._stop.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        sock_file = Path(self.socket_path)
        if sock_file.exists():
            try:
                sock_file.unlink()
            except OSError:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description="HENRI hardened sandbox IPC worker")
    parser.add_argument(
        "--socket",
        default=os.environ.get("HENRI_SANDBOX_SOCKET", "/run/henri-sandbox/exec.sock"),
    )
    args = parser.parse_args()
    server = SandboxUnixServer(args.socket)

    def _sig_handler(_signum, _frame) -> None:
        server.stop()

    signal.signal(signal.SIGTERM, _sig_handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
