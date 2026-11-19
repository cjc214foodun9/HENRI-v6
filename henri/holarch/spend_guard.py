"""Atomic spend guard for the live LLM harness (Cycle AA4).

Two independent guards, both fail-closed:

* a non-blocking process lockfile, so a duplicate runner cannot start a second
  live suite while one is running;
* a per-suite hard call counter, so the live spend can never exceed the
  declared budget even if a retry loop misbehaves.

The lock is an O_CREAT|O_EXCL file holding the pid and the start time.  A stale
lock (a pid that no longer exists on this host) is reclaimed once and recorded;
a live lock makes the second runner exit at zero API calls.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional


class SpendGuardError(RuntimeError):
    """A second runner, or a call above the declared budget."""


def _pid_alive(pid: int) -> bool:
    """True when the pid exists on this host.

    On Windows os.kill(pid, 0) calls TerminateProcess, so it MUST NOT be used
    as a liveness probe: it would kill the process it is asked about.  The
    Windows path opens the process with query-only rights instead and closes
    the handle immediately.
    """
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid)
            )
            if not handle:
                return False
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        except Exception:  # noqa: BLE001 - an unreadable process is not alive
            return False
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    return True


class SpendGuard:
    """One live suite: an exclusive lock plus a hard call budget."""

    def __init__(
        self,
        lock_path: str | Path,
        *,
        budget: int,
        label: str = "live",
    ) -> None:
        self.lock_path = Path(lock_path)
        self.budget = int(budget)
        self.label = str(label)
        self.calls = 0
        self.retries = 0
        self.acquired = False
        self.reclaimed_stale = False
        self.started = time.time()
        self.history: list = []

    # ------------------------------------------------------------- locking
    def acquire(self) -> Dict[str, Any]:
        """Take the lock or raise.  A stale lock is reclaimed once."""
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {"pid": os.getpid(), "started": self.started, "label": self.label}
        )
        for attempt in range(2):
            try:
                handle = os.open(
                    str(self.lock_path),
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                )
            except FileExistsError:
                if attempt == 0 and self._reclaim_stale():
                    self.reclaimed_stale = True
                    continue
                raise SpendGuardError(
                    "a live runner already holds %s: this runner exits at 0 API calls"
                    % self.lock_path
                )
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                stream.write(payload)
            self.acquired = True
            return {"acquired": True, "reclaimed_stale": self.reclaimed_stale}
        raise SpendGuardError("lock acquisition failed")

    def _reclaim_stale(self) -> bool:
        try:
            row = json.loads(self.lock_path.read_text(encoding="utf-8"))
            pid = int(row.get("pid", -1))
        except Exception:  # noqa: BLE001 - an unreadable lock is treated as stale
            pid = -1
        if pid > 0 and _pid_alive(pid):
            return False
        try:
            self.lock_path.unlink()
        except OSError:
            return False
        return True

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            self.lock_path.unlink()
        except OSError:
            pass
        self.acquired = False

    # -------------------------------------------------------------- budget
    def spend(self, *, retry: bool = False) -> int:
        """Reserve one call.  Raises before the budget is exceeded."""
        if not self.acquired:
            raise SpendGuardError("the guard is not held: no live call is allowed")
        if self.calls >= self.budget:
            raise SpendGuardError(
                "hard call budget reached (%d): the live suite stops" % self.budget
            )
        self.calls += 1
        if retry:
            self.retries += 1
        self.history.append({"call": int(self.calls), "retry": bool(retry), "at": time.time()})
        return self.calls

    def report(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "budget": int(self.budget),
            "calls": int(self.calls),
            "retries": int(self.retries),
            "remaining": int(max(0, self.budget - self.calls)),
            "acquired": bool(self.acquired),
            "reclaimed_stale_lock": bool(self.reclaimed_stale),
            "lock": str(self.lock_path),
            "wall_seconds": float(time.time() - self.started),
        }

    def __enter__(self) -> "SpendGuard":
        self.acquire()
        return self

    def __exit__(self, *exc_info) -> None:
        self.release()
