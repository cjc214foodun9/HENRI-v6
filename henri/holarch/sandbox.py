"""Profile-scoped static capability gate and out-of-process task sandbox.

Stage U5 needs a two-stage execution boundary. Stage 1 parses candidate
Python source and checks the import allow-list of one task profile. Stage 2
runs the source in a fresh child Python process with a scrubbed environment,
a wall-clock timeout, capped output and a SHA-256 artifact ledger.

The gate is fail-closed. An unknown module is blocked. A blocked module is
blocked in every profile. A dynamic code-loading builtin is blocked. A dunder
escape chain is blocked. A pathlib call with an absolute literal is blocked.

The child process is not an operating-system jail. The static gate is the
policy boundary. The child process adds process isolation, timeout control,
secret scrubbing and artifact provenance.

Evidence label: OBSERVED for every execution record this module returns.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

# ---------------------------------------------------------------------------
# Task-profile import allow-lists (spec section 4.2, Component 2)
# ---------------------------------------------------------------------------
PURE_ALGORITHMIC_MODULES: frozenset[str] = frozenset(
    {
        "math",
        "cmath",
        "collections",
        "heapq",
        "bisect",
        "itertools",
        "functools",
        "operator",
        "typing",
        "dataclasses",
        "fractions",
        "decimal",
        "re",
        "string",
    }
)
SCIENTIFIC_MODULES: frozenset[str] = PURE_ALGORITHMIC_MODULES | frozenset(
    {"numpy", "scipy", "sympy", "statistics", "json", "csv"}
)
BRIEFCASE_MODULES: frozenset[str] = SCIENTIFIC_MODULES | frozenset({"pandas", "pathlib"})

PROFILE_ALLOWLISTS: Dict[str, frozenset[str]] = {
    "pure_algorithmic": PURE_ALGORITHMIC_MODULES,
    "scientific_scicode": SCIENTIFIC_MODULES,
    "briefcase_workspace": BRIEFCASE_MODULES,
}

BLOCKED_MODULES: frozenset[str] = frozenset(
    {
        "os",
        "sys",
        "subprocess",
        "socket",
        "urllib",
        "http",
        "requests",
        "ctypes",
        "multiprocessing",
        "threading",
        "shutil",
        "importlib",
        "builtins",
        "inspect",
        "gc",
    }
)

BLOCKED_DUNDERS: frozenset[str] = frozenset(
    {
        "__subclasses__",
        "__bases__",
        "__mro__",
        "__globals__",
        "__code__",
        "__builtins__",
        "__class__",
        "__base__",
        "__dict__",
        "__getattribute__",
        "__reduce__",
        "__reduce_ex__",
        "__closure__",
        "__func__",
    }
)

BLOCKED_BUILTINS: frozenset[str] = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "__import__",
        "open",
        "input",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "breakpoint",
        "memoryview",
    }
)

_ABS_PATH_LITERAL = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\|/|~)")
_ARTIFACT_CHUNK = 1 << 20
_RUNNER_NAME = "_henri_task.py"
_RESULT_MARKER = "__HENRI_RESULT__"


def _is_absolute_literal(value: object) -> bool:
    """True when a string literal names an absolute or user-home path."""
    text = str(value).strip()
    if not text:
        return False
    if _ABS_PATH_LITERAL.match(text):
        return True
    return bool(re.match(r"^[A-Za-z]:$", text))


def _module_root(name: str) -> str:
    return str(name).split(".", 1)[0].strip()


class CapabilityASTGate:
    """Static allow-list gate for one task profile."""

    def __init__(self, profile: str) -> None:
        if profile not in PROFILE_ALLOWLISTS:
            raise ValueError(f"unknown sandbox profile: {profile!r}")
        self.profile = str(profile)
        self.allowlist = PROFILE_ALLOWLISTS[self.profile]

    def check(self, source: str) -> Dict[str, object]:
        """Check one source string. Returns ok, reason, imports and blocked."""
        text = str(source)
        try:
            tree = ast.parse(text)
        except SyntaxError as error:
            return {
                "ok": False,
                "reason": f"grammar: {error.msg}",
                "imports": [],
                "blocked": [f"grammar:{error.msg}"],
            }

        imports: List[str] = []
        blocked: List[str] = []

        def block(reason: str) -> None:
            if reason not in blocked:
                blocked.append(reason)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = _module_root(alias.name)
                    imports.append(str(alias.name))
                    if root in BLOCKED_MODULES:
                        block(f"module:{root}")
                    elif root not in self.allowlist:
                        block(f"not-in-profile:{root}")
            elif isinstance(node, ast.ImportFrom):
                if int(getattr(node, "level", 0) or 0) > 0:
                    block("relative-import")
                    continue
                root = _module_root(str(node.module or ""))
                imports.append(root)
                if not root:
                    block("empty-import")
                elif root in BLOCKED_MODULES:
                    block(f"module:{root}")
                elif root not in self.allowlist:
                    block(f"not-in-profile:{root}")

        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                if node.id in BLOCKED_DUNDERS:
                    block(f"dunder:{node.id}")
                if node.id in BLOCKED_BUILTINS:
                    block(f"builtin:{node.id}")
            elif isinstance(node, ast.Attribute):
                if node.attr in BLOCKED_DUNDERS:
                    block(f"dunder-attribute:{node.attr}")
                if node.attr in BLOCKED_BUILTINS:
                    block(f"builtin-attribute:{node.attr}")
            elif isinstance(node, ast.Call):
                callee = node.func
                if isinstance(callee, ast.Name) and callee.id in BLOCKED_BUILTINS:
                    block(f"builtin-call:{callee.id}")
                is_path_call = False
                if isinstance(callee, ast.Name) and callee.id in {
                    "Path",
                    "PurePath",
                    "PosixPath",
                    "WindowsPath",
                }:
                    is_path_call = True
                elif isinstance(callee, ast.Attribute) and callee.attr in {
                    "Path",
                    "PurePath",
                    "PosixPath",
                    "WindowsPath",
                }:
                    is_path_call = True
                if is_path_call and node.args:
                    first = node.args[0]
                    if isinstance(first, ast.Constant) and _is_absolute_literal(first.value):
                        block("absolute-path-literal")

        imports = sorted(dict.fromkeys(imports))
        return {
            "ok": not blocked,
            "reason": "ok" if not blocked else "; ".join(blocked),
            "imports": imports,
            "blocked": blocked,
        }


@dataclass
class ExecutionRecord:
    """One child-process execution, with provenance and timing."""

    ok: bool
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    sha256: str
    artifact_sha256: Dict[str, str] = field(default_factory=dict)
    duration_ms: float = 0.0
    blocked_reason: str = ""

    def as_dict(self) -> Dict[str, object]:
        return {
            "ok": bool(self.ok),
            "exit_code": int(self.exit_code),
            "stdout": str(self.stdout),
            "stderr": str(self.stderr),
            "timed_out": bool(self.timed_out),
            "sha256": str(self.sha256),
            "artifact_sha256": dict(self.artifact_sha256),
            "duration_ms": float(self.duration_ms),
            "blocked_reason": str(self.blocked_reason),
        }


class SubprocessSandboxExecutor:
    """Run one task in a fresh, scrubbed, time-limited child process."""

    def __init__(
        self,
        profile: str,
        timeout_s: float = 10.0,
        output_cap: int = 16384,
        workdir: Optional[str | Path] = None,
    ) -> None:
        if profile not in PROFILE_ALLOWLISTS:
            raise ValueError(f"unknown sandbox profile: {profile!r}")
        self.profile = str(profile)
        self.timeout_s = float(timeout_s)
        self.output_cap = int(output_cap)
        self.workdir = Path(workdir).resolve() if workdir is not None else None
        if self.workdir is not None:
            self.workdir.mkdir(parents=True, exist_ok=True)
        self.ledger: List[ExecutionRecord] = []
        self.last_workdir: Optional[Path] = None

    # ------------------------------------------------------------- env
    @staticmethod
    def _is_sensitive(name: str) -> bool:
        upper = str(name).upper()
        if upper.startswith("HENRI_"):
            return True
        if upper.endswith("_API_KEY"):
            return True
        if upper.startswith("OPENAI") or upper.startswith("OPENROUTER"):
            return True
        if "DSN" in upper:
            return True
        if any(token in upper for token in ("TOKEN", "SECRET", "PASSWORD")):
            return True
        if upper in {"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"}:
            return True
        return False

    def child_env(self) -> Dict[str, str]:
        """Build the child environment explicitly; never inherit secrets."""
        safe_keys = (
            "PATH",
            "SYSTEMROOT",
            "SYSTEMDRIVE",
            "WINDIR",
            "COMSPEC",
            "PATHEXT",
            "TEMP",
            "TMP",
            "TZ",
            "LANG",
            "LC_ALL",
        )
        env: Dict[str, str] = {}
        for key in safe_keys:
            if key in os.environ and not self._is_sensitive(key):
                env[key] = str(os.environ[key])
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        return {key: value for key, value in env.items() if not self._is_sensitive(key)}

    def probe_environment(self, names: Sequence[str]) -> Dict[str, bool]:
        """Measure which named variables survive the scrub inside a child."""
        wanted = [str(name) for name in names]
        if not wanted:
            return {}
        code = (
            "import json, os\n"
            "names = " + repr(wanted) + "\n"
            "print(json.dumps({name: (name in os.environ) for name in names}))\n"
        )
        with tempfile.TemporaryDirectory(prefix="henri_u5_env_") as directory:
            completed = subprocess.run(
                [sys.executable, "-I", "-S", "-c", code],
                env=self.child_env(),
                cwd=directory,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=10.0,
                text=True,
            )
        if completed.returncode != 0:
            raise RuntimeError(f"environment probe failed: {completed.stderr[-400:]}")
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        return {str(key): bool(value) for key, value in json.loads(lines[-1]).items()}

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _safe_join(root: Path, relative: str) -> Path:
        path = Path(str(relative))
        if path.is_absolute() or ".." in path.parts or path.drive:
            raise ValueError(f"refused sandbox file path: {relative!r}")
        return root / path

    @staticmethod
    def _snapshot(root: Path) -> Dict[str, str]:
        snapshot: Dict[str, str] = {}
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path.name == _RUNNER_NAME:
                continue
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(_ARTIFACT_CHUNK), b""):
                    digest.update(chunk)
            snapshot[path.relative_to(root).as_posix()] = digest.hexdigest()
        return snapshot

    @staticmethod
    def _truncate(text: str, cap: int) -> str:
        if cap <= 0:
            return ""
        if len(text) <= cap:
            return text
        marker = "...[truncated]"
        return text[: max(0, cap - len(marker))] + marker

    def _kill_tree(self, process: subprocess.Popen) -> None:
        if os.name == "nt":
            try:
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5.0,
                )
            except Exception:
                pass
        else:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except Exception:
                pass
        try:
            process.kill()
        except Exception:
            pass

    # ------------------------------------------------------------- execute
    def execute(
        self, source: str, files: Optional[Mapping[str, str]] = None
    ) -> ExecutionRecord:
        """Gate, then run one source. The record is appended to the ledger."""
        started = time.perf_counter()
        text = str(source)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        verdict = CapabilityASTGate(self.profile).check(text)
        if not bool(verdict["ok"]):
            record = ExecutionRecord(
                ok=False,
                exit_code=126,
                stdout="",
                stderr=f"BLOCKED: {verdict['reason']}",
                timed_out=False,
                sha256=digest,
                artifact_sha256={},
                duration_ms=float((time.perf_counter() - started) * 1e3),
                blocked_reason=str(verdict["reason"]),
            )
            self.ledger.append(record)
            return record

        root = Path(
            tempfile.mkdtemp(
                prefix=f"u5_{self.profile}_",
                dir=str(self.workdir) if self.workdir is not None else None,
            )
        )
        self.last_workdir = root
        for relative, content in dict(files or {}).items():
            target = self._safe_join(root, str(relative))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(str(content), encoding="utf-8")
        before = self._snapshot(root)

        runner = root / _RUNNER_NAME
        runner.write_text(_runner_source(text), encoding="utf-8")
        command = [sys.executable, "-I"]
        if self.profile == "pure_algorithmic":
            command.append("-S")
        command.append(str(runner))

        stdout = ""
        stderr = ""
        timed_out = False
        exit_code = -1
        process: Optional[subprocess.Popen] = None
        try:
            kwargs: Dict[str, object] = {
                "cwd": str(root),
                "env": self.child_env(),
                "stdin": subprocess.DEVNULL,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
            }
            if os.name == "nt":
                kwargs["creationflags"] = getattr(
                    subprocess, "CREATE_NEW_PROCESS_GROUP", 0
                )
            else:
                kwargs["start_new_session"] = True
            process = subprocess.Popen(command, **kwargs)  # type: ignore[arg-type]
            try:
                out_bytes, err_bytes = process.communicate(timeout=self.timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                self._kill_tree(process)
                try:
                    out_bytes, err_bytes = process.communicate(timeout=5.0)
                except subprocess.TimeoutExpired:
                    out_bytes, err_bytes = b"", b""
            exit_code = int(process.returncode) if process.returncode is not None else -1
            stdout = out_bytes.decode("utf-8", errors="replace")
            stderr = err_bytes.decode("utf-8", errors="replace")
            if timed_out:
                stderr = (stderr + "\nTIMEOUT: child process tree killed").strip()
        except OSError as error:
            stderr = f"spawn-error: {type(error).__name__}: {error}"
            exit_code = -1
        finally:
            if process is not None and process.poll() is None:
                self._kill_tree(process)

        after = self._snapshot(root)
        artifacts = {
            path: value for path, value in after.items() if before.get(path) != value
        }
        duration_ms = float((time.perf_counter() - started) * 1e3)
        record = ExecutionRecord(
            ok=bool(not timed_out and exit_code == 0),
            exit_code=int(exit_code),
            stdout=self._truncate(stdout, self.output_cap),
            stderr=self._truncate(stderr, self.output_cap),
            timed_out=bool(timed_out),
            sha256=digest,
            artifact_sha256=artifacts,
            duration_ms=duration_ms,
            blocked_reason="",
        )
        self.ledger.append(record)
        return record


def _runner_source(source: str) -> str:
    """Build the child runner. The user source stays a literal string."""
    return (
        "import json, sys\n"
        "_SOURCE = " + repr(str(source)) + "\n"
        "_NAMESPACE = {'__name__': '__main__'}\n"
        "try:\n"
        "    exec(compile(_SOURCE, '<henri-u5>', 'exec'), _NAMESPACE, _NAMESPACE)\n"
        "except BaseException as _error:\n"
        "    sys.stderr.write('HENRI_EXEC_ERROR: %s: %s\\n' % "
        "(type(_error).__name__, _error))\n"
        "    raise\n"
        "sys.stdout.write('" + _RESULT_MARKER + "' + "
        "json.dumps(_NAMESPACE.get('result'), default=str, sort_keys=True))\n"
    )


def extract_child_result(stdout: str) -> Optional[object]:
    """Read the runner's result marker from captured stdout."""
    marker_at = str(stdout).rfind(_RESULT_MARKER)
    if marker_at < 0:
        return None
    payload = str(stdout)[marker_at + len(_RESULT_MARKER) :].strip()
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None
