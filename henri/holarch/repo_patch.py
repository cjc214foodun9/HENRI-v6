"""Cycle AD1: standalone multi-file repository patch synthesis.

The stage extends the standalone coding arm from single-function synthesis to
multi-file repository repair at zero external model calls.

The pipeline for one repository is:

1. RepoSubprocessSandbox runs the public test command once in a scrubbed child
   process and returns the exact failing traceback.
2. FaultLocalizer maps the failing traceback frames and the failing test's call
   graph onto the indexed symbol table (CodeGraphIndexer) and isolates a
   candidate function set F_cand of at most six functions over one or two
   impact hops.
3. PatchProposer enumerates single-site structural AST edits over F_cand:
   boundary shifts, operator and sign inversions, guard and empty-sequence
   normalisation, accumulator initialisation, wrong-callee and wrong-attribute
   resolution inside the indexed symbol table, slice repairs and argument-order
   repairs.
4. Before any test process is spawned, the candidate patch is folded into the
   module through CodeGraphIndexer.update_file (delta mode) and the whole
   repository is re-analysed by RepoSheaf.  A candidate whose coboundary energy
   is not exactly zero never reaches the sandbox.
5. A candidate that fails the public suite is encoded as a Zone C
   BoundaryAxiom(polarity="REPULSOR", source="repo_patch_cegis") keyed by
   (target_qualname, patch_ast_signature, failure_kind), so an equivalent
   dead-end edit is deflected instead of executed again.
6. If no candidate passes every public test inside the rollout budget the
   engine abstains and leaves every file untouched.

Every decision is content-addressed: no task identifier, repository name or
file name ever enters a branch.
"""

from __future__ import annotations

import ast
import copy
import difflib
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .repo_sheaf import RepoSheaf, SheafReport
from .sandbox import ExecutionRecord, SubprocessSandboxExecutor
from .symbolic import CodeGraphIndexer
from .zone_c import BoundaryAxiom

MAX_CANDIDATES = 6
MAX_ROLLOUTS = 48
SHEAF_TOLERANCE = 0.0
REPULSOR_SOURCE = "repo_patch_cegis"
AXIOM_DIM = 1024

CMP_SWAP = {
    ast.Lt: ast.LtE,
    ast.LtE: ast.Lt,
    ast.Gt: ast.GtE,
    ast.GtE: ast.Gt,
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
    ast.In: ast.NotIn,
    ast.NotIn: ast.In,
    ast.Is: ast.IsNot,
    ast.IsNot: ast.Is,
}
BIN_SWAP = {
    ast.Add: ast.Sub,
    ast.Sub: ast.Add,
    ast.Mult: ast.Div,
    ast.Div: ast.Mult,
    ast.FloorDiv: ast.Div,
    ast.Mod: ast.FloorDiv,
}
CALL_SWAP = {"min": "max", "max": "min", "sorted": "reversed", "reversed": "sorted"}


@dataclass(frozen=True)
class RepoCase:
    """One author-clean multi-module repair case.

    files is the repository as the agent sees it.  public is the reproduction
    suite that may be executed.  hidden is the unseen maintainer regression
    suite: it exists only in the grading harness and is never visible to the
    engine.
    """

    name: str
    issue: str
    files: Dict[str, str]
    public: Dict[str, str]
    hidden: Dict[str, str]
    test_path: str = "tests/test_public.py"


# --------------------------------------------------------------- materialise
def materialise(
    case: RepoCase,
    root: Path,
    *,
    include_hidden: bool = False,
    patched: Optional[Dict[str, str]] = None,
) -> Path:
    """Write one case to root and return the repository directory."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    files = dict(case.files)
    if patched:
        files.update({path: text for path, text in patched.items()})
    for path, text in files.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(text), encoding="utf-8")
    for path, text in dict(case.public).items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(text), encoding="utf-8")
    if include_hidden:
        for path, text in dict(case.hidden).items():
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(str(text), encoding="utf-8")
    return root


# ------------------------------------------------------------------- sandbox
class RepoSubprocessSandbox:
    """Repository-mode arm of the SubprocessSandbox.

    The environment scrubbing, the timeout policy and the kill-tree rule are
    inherited from SubprocessSandboxExecutor; the difference is that the child
    runs the repository's own test command instead of one embedded source
    string, so a multi-module workspace can be exercised without ever
    inheriting a credential.
    """

    def __init__(
        self,
        profile: str = "briefcase_workspace",
        timeout_s: float = 90.0,
        output_cap: int = 40000,
    ) -> None:
        self.profile = str(profile)
        self.timeout_s = float(timeout_s)
        self.output_cap = int(output_cap)
        self.executor = SubprocessSandboxExecutor(
            profile=self.profile, timeout_s=self.timeout_s, output_cap=self.output_cap
        )
        self.ledger: List[ExecutionRecord] = []

    def run_tests(
        self, root: Path, test_path: str, extra: Sequence[str] = ()
    ) -> Dict[str, Any]:
        """Run one pytest module inside the repository and capture the trace."""
        started = time.perf_counter()
        runner = Path(root) / "_ad1_runner.py"
        runner.write_text(_runner_source(), encoding="utf-8")
        command = [sys.executable, str(runner), str(test_path), *[str(value) for value in extra]]
        env = self.executor.child_env()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        stdout = ""
        stderr = ""
        exit_code = -1
        timed_out = False
        process: Optional[subprocess.Popen] = None
        try:
            kwargs: Dict[str, Any] = {
                "cwd": str(root),
                "env": env,
                "stdin": subprocess.DEVNULL,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
            }
            if os.name == "nt":
                kwargs["creationflags"] = getattr(
                    subprocess, "CREATE_NEW_PROCESS_GROUP", 0
                )
            process = subprocess.Popen(command, **kwargs)  # type: ignore[arg-type]
            try:
                out_bytes, err_bytes = process.communicate(timeout=self.timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                self.executor._kill_tree(process)  # noqa: SLF001 - the sandbox rule
                try:
                    out_bytes, err_bytes = process.communicate(timeout=5.0)
                except subprocess.TimeoutExpired:
                    out_bytes, err_bytes = b"", b""
            exit_code = int(process.returncode) if process.returncode is not None else -1
            stdout = out_bytes.decode("utf-8", errors="replace")
            stderr = err_bytes.decode("utf-8", errors="replace")
        except OSError as error:
            stderr = "spawn-error: %s: %s" % (type(error).__name__, error)
            exit_code = -1
        finally:
            if process is not None and process.poll() is None:
                self.executor._kill_tree(process)  # noqa: SLF001 - the sandbox rule
        parsed = _parse_runner(stdout)
        failures = list(parsed.get("failures") or [])
        text = "\n\n".join(str(row.get("traceback", "")) for row in failures) or (
            stdout + "\n" + stderr
        )
        record = ExecutionRecord(
            ok=bool(not timed_out and exit_code == 0),
            exit_code=int(exit_code),
            stdout=self.executor._truncate(stdout, self.output_cap),  # noqa: SLF001
            stderr=self.executor._truncate(stderr, self.output_cap),  # noqa: SLF001
            timed_out=bool(timed_out),
            sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            artifact_sha256={},
            duration_ms=float((time.perf_counter() - started) * 1e3),
            blocked_reason="",
        )
        self.ledger.append(record)
        return {
            "passed": bool(record.ok and int(parsed.get("failed", 1)) == 0),
            "exit_code": int(exit_code),
            "timed_out": bool(timed_out),
            "stdout": record.stdout,
            "stderr": record.stderr,
            "text": text,
            "failures": failures,
            "total": int(parsed.get("total", 0)),
            "duration_ms": float(record.duration_ms),
            "sha256": record.sha256,
            "profile": self.profile,
            "runner": "in_process_test_module",
        }


# --------------------------------------------------------------- localisation
_FRAME_PYTEST = re.compile(r"^(?P<file>[^\s:][^:\n]*?\.py):(?P<line>\d+): in (?P<func>\S+)", re.M)
_FRAME_STDLIB = re.compile(r'^\s*File "(?P<file>[^"]+)", line (?P<line>\d+), in (?P<func>\S+)', re.M)
_FAILED = re.compile(r"^FAILED\s+([^\s:]+)::(\w+)", re.M)
_ERROR_LINE = re.compile(r"^E\s+([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception|Warning))", re.M)
_ERROR_STDLIB = re.compile(r"^([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception))(?::|$)", re.M)
_ASSERT_LINE = re.compile(r"^E\s+assert\s+(.+?)\s*(==|!=)\s*(.+?)\s*$", re.M)
_ASSERT_STDLIB = re.compile(r"^AssertionError: assert\s+(.+?)\s*==\s*(.+?)\s*$", re.M)


@dataclass
class Localization:
    failing_test: str = ""
    frames: List[Dict[str, Any]] = field(default_factory=list)
    failure_kind: str = "AssertionError"
    assert_detail: str = ""
    numeric_delta: Optional[float] = None
    candidates: List[str] = field(default_factory=list)
    test_calls: List[str] = field(default_factory=list)
    seconds: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "failing_test": self.failing_test,
            "frames": list(self.frames),
            "failure_kind": self.failure_kind,
            "assert_detail": self.assert_detail,
            "numeric_delta": self.numeric_delta,
            "candidates": list(self.candidates),
            "test_calls": list(self.test_calls),
            "seconds": float(self.seconds),
        }


class FaultLocalizer:
    """Traceback plus call-graph localisation over the indexed symbol table."""

    def __init__(self, indexer: CodeGraphIndexer) -> None:
        self.indexer = indexer

    def locate(
        self, repo_root: Path, report: Dict[str, Any], test_path: str
    ) -> Localization:
        started = time.perf_counter()
        out = Localization()
        text = str(report.get("text") or "")
        failed = _FAILED.search(text)
        if failed:
            out.failing_test = str(failed.group(2))
        failures = list(report.get("failures") or [])
        if failures and not out.failing_test:
            out.failing_test = str(failures[0].get("test") or "")
        for pattern in (_FRAME_PYTEST, _FRAME_STDLIB):
            for match in pattern.finditer(text):
                path = str(match.group("file")).replace("\\", "/")
                try:
                    relative = Path(path).resolve().relative_to(Path(repo_root).resolve())
                    path = relative.as_posix()
                except (ValueError, OSError):
                    pass
                out.frames.append(
                    {
                        "file": path,
                        "line": int(match.group("line")),
                        "func": str(match.group("func")),
                    }
                )
        unique: List[Dict[str, Any]] = []
        seen_frames: set = set()
        for frame in out.frames:
            key = (str(frame["file"]), int(frame["line"]), str(frame["func"]))
            if key in seen_frames:
                continue
            seen_frames.add(key)
            unique.append(frame)
        out.frames = unique
        error = _ERROR_LINE.search(text)
        if error:
            out.failure_kind = str(error.group(1))
        else:
            matches = list(_ERROR_STDLIB.finditer(text))
            if matches:
                out.failure_kind = str(matches[-1].group(1)).split(".")[-1]
            elif "assert" in text.lower():
                out.failure_kind = "AssertionError"
        assertion = _ASSERT_LINE.search(text)
        if assertion:
            left, operator, right = assertion.groups()
            out.assert_detail = "%s %s %s" % (left.strip(), operator, right.strip())
            try:
                out.numeric_delta = float(float(left.strip()) - float(right.strip()))
            except (TypeError, ValueError):
                out.numeric_delta = None
        else:
            standard = _ASSERT_STDLIB.search(text)
            if standard:
                left, right = standard.groups()
                out.assert_detail = "%s == %s" % (left.strip(), right.strip())
                try:
                    out.numeric_delta = float(float(left.strip()) - float(right.strip()))
                except (TypeError, ValueError):
                    out.numeric_delta = None
        roots: List[str] = []
        for frame in out.frames:
            if str(frame["file"]).startswith("tests/"):
                continue
            qualname = self._qualname(str(frame["file"]), str(frame["func"]))
            if qualname and qualname not in roots:
                roots.append(qualname)
        test_calls = self._test_calls(repo_root, test_path, out.failing_test)
        out.test_calls = list(test_calls)
        for name in test_calls:
            qualname = self._resolve(name)
            if qualname and qualname not in roots:
                roots.append(qualname)
        expanded: List[str] = list(roots)
        for qualname in list(roots):
            for callee in self._callees(qualname, depth=2):
                if callee not in expanded:
                    expanded.append(callee)
            for caller in self.indexer.impact(qualname, hops=1).get("callers", []):
                if str(caller) not in expanded:
                    expanded.append(str(caller))
        out.candidates = [
            name for name in expanded if name in self.indexer.signatures
        ][:MAX_CANDIDATES]
        out.seconds = float(time.perf_counter() - started)
        return out

    def _qualname(self, relative: str, func: str) -> Optional[str]:
        module = ".".join(Path(relative).with_suffix("").parts)
        for key in ("%s:%s" % (module, func), func):
            if key in self.indexer.signatures:
                return key
        for key in self.indexer.signatures:
            if key.endswith(":" + func):
                if key.split(":")[0].split(".")[-1] == module.split(".")[-1]:
                    return key
        return None

    def _resolve(self, name: str) -> Optional[str]:
        for key in self.indexer.signatures:
            if key.split(":")[-1] == name:
                return key
        return None

    def _callees(self, qualname: str, depth: int = 2) -> List[str]:
        frontier = [qualname]
        seen: List[str] = []
        for _ in range(max(1, int(depth))):
            following: List[str] = []
            for node in frontier:
                for edge in self.indexer.graph.edges:
                    if edge.source == node and edge.target not in seen:
                        seen.append(edge.target)
                        following.append(edge.target)
            frontier = following
        return seen

    @staticmethod
    def _test_calls(repo_root: Path, test_path: str, failing_test: str) -> List[str]:
        path = Path(repo_root) / test_path
        if not path.exists():
            return []
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            return []
        names: List[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if failing_test and node.name != failing_test:
                continue
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    name = _callee_name(child)
                    if name and name not in names:
                        names.append(name)
        return names


def _callee_name(node: ast.Call) -> Optional[str]:
    func = node.func
    if isinstance(func, ast.Name):
        return str(func.id)
    if isinstance(func, ast.Attribute):
        return str(func.attr)
    return None


# ------------------------------------------------------------------- proposals
@dataclass
class Candidate:
    qualname: str
    module: str
    path: str
    kind: str
    signature: str
    source: str
    line: int
    score: float
    note: str = ""

    def summary(self) -> Dict[str, Any]:
        return {
            "qualname": self.qualname,
            "path": self.path,
            "kind": self.kind,
            "line": int(self.line),
            "signature": self.signature,
            "score": float(self.score),
            "note": self.note,
        }


class PatchProposer:
    """Deterministic single-site AST edits over the candidate function set."""

    def __init__(self, indexer: CodeGraphIndexer) -> None:
        self.indexer = indexer
        self.symbols = sorted(
            {key.split(":")[-1] for key in indexer.signatures if ":" in key}
        )

    def propose(
        self,
        files: Dict[str, str],
        localization: Localization,
        budget: int = MAX_ROLLOUTS,
    ) -> List[Candidate]:
        candidates: List[Candidate] = []
        failing_function = localization.candidates[0] if localization.candidates else ""
        failing_line = int(localization.frames[-1]["line"]) if localization.frames else 0
        evidence = _evidence_weights(localization)
        for index, qualname in enumerate(localization.candidates):
            module, _, function = qualname.rpartition(":")
            path = module.replace(".", "/") + ".py"
            if module.startswith("tests") or path.startswith("tests/"):
                continue
            source = files.get(path)
            if source is None:
                continue
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            node = _find_function(tree, function)
            if node is None:
                continue
            func_path = _path_to(tree, node)
            if func_path is None:
                continue
            candidates.extend(
                self._function_candidates(
                    files,
                    source,
                    tree,
                    qualname,
                    module,
                    path,
                    node,
                    func_path,
                    index,
                    qualname == failing_function,
                    failing_line,
                    evidence,
                )
            )
        candidates.sort(
            key=lambda row: (-row.score, row.qualname, row.kind, row.signature)
        )
        return candidates[: int(budget)]

    def _function_candidates(
        self,
        files: Dict[str, str],
        source: str,
        tree: ast.Module,
        qualname: str,
        module: str,
        path: str,
        node: ast.AST,
        func_path: Tuple[int, ...],
        index: int,
        is_failing: bool,
        failing_line: int,
        evidence: Dict[str, float],
    ) -> List[Candidate]:
        out: List[Candidate] = []
        params = [
            arg.arg
            for arg in getattr(getattr(node, "args", None), "args", []) or []
            if arg.arg not in {"self", "cls"}
        ]
        neutral = _neutral_return(node)
        base = 40.0 - 8.0 * float(index) + (25.0 if is_failing else 0.0)
        if _is_delegation(node):
            base -= 15.0

        def add(kind: str, mutated: Optional[str], line: int, signature: str, note: str) -> None:
            if mutated is None or mutated == source:
                return
            proximity = 0.0
            if is_failing and failing_line and abs(int(line) - int(failing_line)) <= 3:
                proximity = 25.0
            out.append(
                Candidate(
                    qualname=qualname,
                    module=module,
                    path=path,
                    kind=kind,
                    signature=signature,
                    source=mutated,
                    line=int(line),
                    score=float(base + evidence.get(kind, 0.0) + proximity),
                    note=note,
                )
            )

        def signature(kind: str, site: str, arg: Any) -> str:
            payload = "%s|%s|%s|%s" % (qualname, kind, site, arg)
            return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

        # 1. Missing None / empty-sequence guards, inserted after the docstring.
        for param in (params[:2] if evidence.get("guard_none", 0.0) > 0.0 else []):
            for guard_kind, test_source in (
                ("guard_none", "%s is None" % param),
                ("guard_empty", "not %s" % param),
            ):
                mutated = _apply_guard(tree, source, func_path, test_source, neutral)
                add(
                    guard_kind,
                    mutated,
                    int(getattr(node, "lineno", 0)),
                    signature(guard_kind, "guard", test_source),
                    "insert guard %s" % test_source,
                )

        # 2. Structural sites.
        sites = _collect_sites(node, tree, files, path)
        for ordinal, site in enumerate(sites):
            kind = str(site["kind"])
            site_path = tuple(site["path"])
            line = int(site["line"])
            site_key = "%s@%d" % (kind, ordinal)
            if kind == "compare":
                mutated = _apply_site(tree, source, func_path, site_path, _act_compare, None)
                add("compare", mutated, line, signature("compare", site_key, 0),
                    "compare operator swap")
            elif kind == "binop":
                mutated = _apply_site(tree, source, func_path, site_path, _act_binop, None)
                add("binop", mutated, line, signature("binop", site_key, 0),
                    "binary operator swap")
            elif kind == "int":
                for delta in (1, -1):
                    mutated = _apply_site(tree, source, func_path, site_path, _act_int, delta)
                    add("int", mutated, line, signature("int", site_key, delta),
                        "integer constant %+d" % delta)
            elif kind == "index":
                for delta in (1, -1):
                    mutated = _apply_site(tree, source, func_path, site_path, _act_index, delta)
                    add("index", mutated, line, signature("index", site_key, delta),
                        "subscript index %+d" % delta)
            elif kind == "attr":
                for alternative in site["alternatives"]:
                    mutated = _apply_site(tree, source, func_path, site_path, _act_attr, alternative)
                    add("attr", mutated, line,
                        signature("attr", site_key, alternative),
                        "attribute self.%s" % alternative)
            elif kind == "call":
                for alternative in site["alternatives"]:
                    mutated = _apply_site(tree, source, func_path, site_path, _act_call, alternative)
                    add("call", mutated, line,
                        signature("call", site_key, alternative),
                        "callee %s" % alternative)
                if int(site["arity"]) >= 2:
                    mutated = _apply_site(tree, source, func_path, site_path, _act_argswap, None)
                    add("call", mutated, line, signature("call", site_key, "swap"),
                        "argument order swap")
            elif kind == "boolop":
                mutated = _apply_site(tree, source, func_path, site_path, _act_boolop, None)
                add("boolop", mutated, line, signature("boolop", site_key, 0),
                    "boolean operator swap")
            elif kind == "slice":
                if site["lower"] is not None:
                    mutated = _apply_site(tree, source, func_path, site_path, _act_slice_lower, None)
                    add("slice", mutated, line,
                        signature("slice", site_key, "lower"),
                        "slice lower bound removed")
                if site["upper"] is not None:
                    mutated = _apply_site(tree, source, func_path, site_path, _act_slice_upper, None)
                    add("slice", mutated, line,
                        signature("slice", site_key, "upper"),
                        "slice upper bound removed")
            elif kind == "str":
                for alternative in site["alternatives"]:
                    mutated = _apply_site(tree, source, func_path, site_path, _act_str, alternative)
                    add("str", mutated, line,
                        signature("str", site_key, alternative),
                        "string literal %r" % alternative)
        return out


def _evidence_weights(localization: Localization) -> Dict[str, float]:
    kind = str(localization.failure_kind)
    weights: Dict[str, float] = {}
    if kind in {"TypeError", "AttributeError"}:
        weights = {"guard_none": 60.0, "guard_empty": 55.0, "attr": 50.0, "call": 40.0}
    elif kind == "IndexError":
        weights = {"guard_none": 55.0, "guard_empty": 50.0, "index": 50.0}
    elif kind == "KeyError":
        weights = {"guard_none": 50.0, "guard_empty": 45.0, "attr": 25.0}
    elif kind == "ZeroDivisionError":
        weights = {"guard_none": 60.0, "guard_empty": 60.0}
    elif kind == "AssertionError":
        weights = {
            "compare": 40.0,
            "int": 35.0,
            "binop": 30.0,
            "index": 25.0,
            "attr": 20.0,
            "call": 20.0,
            "guard_none": 0.0,
            "guard_empty": 0.0,
            "slice": 20.0,
            "str": 30.0,
            "boolop": 15.0,
        }
        delta = localization.numeric_delta
        if delta is not None and abs(abs(delta) - 1.0) < 1e-9:
            weights["int"] = 75.0
    return weights


def _is_delegation(node: ast.AST) -> bool:
    """A function whose whole body is one return of a call is a pass-through."""
    body = list(getattr(node, "body", []) or [])
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        if isinstance(body[0].value.value, str):
            body = body[1:]
    if len(body) != 1 or not isinstance(body[0], ast.Return):
        return False
    return isinstance(body[0].value, ast.Call)


def _neutral_return(node: ast.AST) -> str:
    annotation = ""
    returns = getattr(node, "returns", None)
    if returns is not None:
        try:
            annotation = ast.unparse(returns)
        except Exception:  # noqa: BLE001 - unparse of an exotic node
            annotation = ""
    base = annotation.split("[")[0].strip()
    return {
        "int": "0",
        "float": "0.0",
        "bool": "False",
        "str": '""',
        "list": "[]",
        "List": "[]",
        "dict": "{}",
        "Dict": "{}",
        "set": "set()",
        "tuple": "()",
        "Tuple": "()",
    }.get(base, "None")


def _find_function(tree: ast.AST, name: str) -> Optional[ast.AST]:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def _path_to(root: ast.AST, target: ast.AST) -> Optional[Tuple[int, ...]]:
    """Child-index path from root to target, or None when target is absent."""
    path: List[int] = []

    def walk(node: ast.AST) -> bool:
        if node is target:
            return True
        for position, child in enumerate(ast.iter_child_nodes(node)):
            path.append(position)
            if walk(child):
                return True
            path.pop()
        return False

    return tuple(path) if walk(root) else None


def _navigate(root: ast.AST, path: Sequence[int]) -> ast.AST:
    node = root
    for position in path:
        children = list(ast.iter_child_nodes(node))
        node = children[int(position)]
    return node


def _site_kind(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Compare):
        if len(node.ops) == 1 and type(node.ops[0]) in CMP_SWAP:
            return "compare"
        return None
    if isinstance(node, ast.BinOp) and type(node.op) in BIN_SWAP:
        return "binop"
    if isinstance(node, ast.Subscript):
        if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, int):
            if not isinstance(node.slice.value, bool):
                return "index"
        return None
    if isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name):
            return "attr"
        return None
    if isinstance(node, ast.Call):
        return "call"
    if isinstance(node, ast.BoolOp):
        return "boolop"
    if isinstance(node, ast.Slice):
        return "slice"
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return None
        if isinstance(node.value, int):
            return "int"
        if isinstance(node.value, str):
            return "str"
    return None


def _collect_sites(
    func: ast.AST, tree: ast.Module, files: Dict[str, str], path: str
) -> List[Dict[str, Any]]:
    parents: Dict[int, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[id(child)] = parent
    attrs: Dict[str, set] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            attrs.setdefault(node.value.id, set()).add(node.attr)
    strings = sorted(
        {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and _is_docstring(node, parents) is False
        }
    )
    sites: List[Dict[str, Any]] = []

    def walk(node: ast.AST, relative: Tuple[int, ...]) -> None:
        kind = _site_kind(node)
        if kind is not None:
            record: Optional[Dict[str, Any]] = None
            if kind == "attr":
                base = node.value.id if isinstance(node.value, ast.Name) else ""
                parent = parents.get(id(node))
                if isinstance(parent, ast.Call) and parent.func is node:
                    record = None
                else:
                    alternatives = sorted(
                        value for value in attrs.get(base, set()) if value != node.attr
                    )[:3]
                    if alternatives:
                        record = {"kind": kind, "path": relative, "line": node.lineno,
                                  "alternatives": alternatives}
            elif kind == "call":
                if isinstance(node.func, ast.Name):
                    alternatives = _call_alternatives(
                        node, files, path, enclosing=_enclosing_function(node, parents)
                    )
                    if alternatives:
                        record = {"kind": kind, "path": relative, "line": node.lineno,
                                  "alternatives": alternatives, "arity": len(node.args)}
            elif kind == "str":
                if _is_docstring(node, parents):
                    record = None
                else:
                    alternatives = [value for value in strings if value != node.value][:3]
                    if alternatives:
                        record = {"kind": kind, "path": relative, "line": node.lineno,
                                  "alternatives": alternatives}
            elif kind == "slice":
                lower = None
                upper = None
                if isinstance(node.lower, ast.Constant) and node.lower.value == 1:
                    lower = 1
                if isinstance(node.upper, ast.Constant) and node.upper.value == -1:
                    upper = -1
                if (
                    isinstance(node.upper, ast.UnaryOp)
                    and isinstance(node.upper.op, ast.USub)
                    and isinstance(node.upper.operand, ast.Constant)
                    and node.upper.operand.value == 1
                ):
                    upper = -1
                if lower is not None or upper is not None:
                    record = {"kind": kind, "path": relative, "line": node.lineno,
                              "lower": lower, "upper": upper}
            else:
                record = {"kind": kind, "path": relative, "line": node.lineno}
            if record is not None:
                sites.append(record)
        for position, child in enumerate(ast.iter_child_nodes(node)):
            walk(child, relative + (position,))

    walk(func, ())
    return sites


def _is_docstring(node: ast.Constant, parents: Dict[int, ast.AST]) -> bool:
    parent = parents.get(id(node))
    if parent is None:
        return False
    if not isinstance(parent, ast.Expr) or parent.value is not node:
        return False
    grand = parents.get(id(parent))
    return isinstance(grand, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))


def _call_alternatives(
    node: ast.Call, files: Dict[str, str], path: str, enclosing: str = ""
) -> List[str]:
    """Repo symbols with the same arity that are visible at this call site."""
    name = node.func.id if isinstance(node.func, ast.Name) else ""
    arity = len(node.args)
    out: List[str] = []
    if name in CALL_SWAP:
        out.append(CALL_SWAP[name])
    visible = _visible_symbols(files, path)
    definitions: Dict[str, int] = {}
    for other_path, text in files.items():
        if not other_path.endswith(".py") or other_path.startswith("tests/"):
            continue
        try:
            other = ast.parse(text)
        except SyntaxError:
            continue
        for candidate in ast.walk(other):
            if isinstance(candidate, (ast.FunctionDef, ast.AsyncFunctionDef)):
                params = [
                    arg for arg in candidate.args.args if arg.arg not in {"self", "cls"}
                ]
                definitions.setdefault(candidate.name, len(params))
    for symbol in sorted(visible):
        if symbol == name or symbol == enclosing or symbol in out:
            continue
        if definitions.get(symbol) == arity:
            out.append(symbol)
    return out[:3]


def _visible_symbols(files: Dict[str, str], path: str) -> set:
    """Module-level definitions plus imported names visible in one module."""
    text = files.get(path, "")
    visible: set = set()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return visible
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            visible.add(node.name)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                visible.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                visible.add((alias.asname or alias.name).split(".")[0])
    return visible


def _enclosing_function(node: ast.AST, parents: Dict[int, ast.AST]) -> str:
    current = parents.get(id(node))
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return str(current.name)
        current = parents.get(id(current))
    return ""


def _apply_site(
    tree: ast.Module,
    source: str,
    func_path: Sequence[int],
    site_path: Sequence[int],
    action,
    argument: Any,
) -> Optional[str]:
    clone = copy.deepcopy(tree)
    try:
        function = _navigate(clone, func_path)
        node = _navigate(function, site_path)
        action(node, argument)
    except (IndexError, TypeError, ValueError):
        return None
    ast.fix_missing_locations(clone)
    return _splice_function(source, tree, func_path, clone)


def _apply_guard(
    tree: ast.Module,
    source: str,
    func_path: Sequence[int],
    test_source: str,
    neutral: str,
) -> Optional[str]:
    clone = copy.deepcopy(tree)
    try:
        function = _navigate(clone, func_path)
        statement = ast.parse(
            "if %s:\n    return %s" % (test_source, neutral)
        ).body[0]
        index = 0
        if function.body:
            first = function.body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                index = 1
        function.body.insert(index, statement)
    except (AttributeError, IndexError, SyntaxError):
        return None
    ast.fix_missing_locations(clone)
    return _splice_function(source, tree, func_path, clone)


def _splice_function(
    source: str,
    original_tree: ast.Module,
    func_path: Sequence[int],
    mutated_tree: ast.Module,
) -> Optional[str]:
    """Replace exactly one function's source segment with its mutated text.

    Only the edited function is reformatted, so the emitted patch is confined
    to the repaired function and cannot touch an unrelated module line.
    """
    try:
        original = _navigate(original_tree, func_path)
        mutated = _navigate(mutated_tree, func_path)
    except (IndexError, TypeError):
        return None
    decorators = list(getattr(original, "decorator_list", []) or [])
    start = min([int(getattr(node, "lineno", 1)) for node in decorators] or [int(original.lineno)])
    end = int(getattr(original, "end_lineno", original.lineno))
    lines = source.splitlines(keepends=True)
    if start < 1 or end > len(lines):
        return None
    prefix = lines[start - 1][: len(lines[start - 1]) - len(lines[start - 1].lstrip())]
    try:
        text = ast.unparse(mutated)
    except Exception:  # noqa: BLE001 - an unparsable mutation is skipped
        return None
    if prefix:
        text = "\n".join(prefix + line for line in text.splitlines())
    head = "".join(lines[: start - 1])
    tail = "".join(lines[end:])
    candidate = head + text + "\n" + tail
    try:
        ast.parse(candidate)
    except SyntaxError:
        return None
    return candidate


def _act_compare(node: ast.AST, argument: Any) -> None:
    node.ops[0] = CMP_SWAP[type(node.ops[0])]()


def _act_binop(node: ast.AST, argument: Any) -> None:
    node.op = BIN_SWAP[type(node.op)]()


def _act_int(node: ast.AST, argument: Any) -> None:
    node.value = int(node.value) + int(argument)


def _act_index(node: ast.AST, argument: Any) -> None:
    node.slice.value = int(node.slice.value) + int(argument)


def _act_attr(node: ast.AST, argument: Any) -> None:
    node.attr = str(argument)


def _act_call(node: ast.AST, argument: Any) -> None:
    node.func = ast.Name(id=str(argument), ctx=ast.Load())


def _act_argswap(node: ast.AST, argument: Any) -> None:
    node.args = list(reversed(node.args))


def _act_boolop(node: ast.AST, argument: Any) -> None:
    node.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()


def _act_slice_lower(node: ast.AST, argument: Any) -> None:
    node.lower = None


def _act_slice_upper(node: ast.AST, argument: Any) -> None:
    node.upper = None


def _act_str(node: ast.AST, argument: Any) -> None:
    node.value = str(argument)


# ----------------------------------------------------------- conflict analysis
def detect_public_conflict(public: Dict[str, str]) -> Dict[str, Any]:
    """Find two public assertions that demand different values of one call."""
    seen: Dict[str, Dict[str, Any]] = {}
    for path, text in sorted(public.items()):
        try:
            tree = ast.parse(str(text))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assert):
                continue
            for comparison in ast.walk(node.test):
                if not isinstance(comparison, ast.Compare):
                    continue
                if len(comparison.ops) != 1 or not isinstance(comparison.ops[0], ast.Eq):
                    continue
                left, right = comparison.left, comparison.comparators[0]
                for call, constant in ((left, right), (right, left)):
                    if not isinstance(call, ast.Call):
                        continue
                    if not isinstance(constant, (ast.Constant, ast.List, ast.Tuple, ast.Dict)):
                        continue
                    try:
                        key = ast.dump(call, annotate_fields=False)
                        value = ast.dump(constant, annotate_fields=False)
                    except Exception:  # noqa: BLE001 - an exotic node is skipped
                        continue
                    record = seen.setdefault(
                        key, {"expressions": [], "values": [], "lines": []}
                    )
                    if value not in record["values"]:
                        record["values"].append(value)
                        record["expressions"].append(ast.unparse(call))
                    record["lines"].append(
                        {"path": str(path), "line": int(getattr(node, "lineno", 0))}
                    )
    conflicts = [
        {
            "expression": record["expressions"][0],
            "values": record["values"],
            "assertions": record["lines"],
        }
        for record in seen.values()
        if len(record["values"]) >= 2
    ]
    return {
        "detected": bool(conflicts),
        "kind": "conflicting_assertions" if conflicts else "",
        "conflicts": conflicts,
    }


# -------------------------------------------------------------------- repulsor
@dataclass
class RepulsorRecord:
    qualname: str
    signature: str
    failure_kind: str
    axiom_id: str
    name: str
    engram_id: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "qualname": self.qualname,
            "patch_ast_signature": self.signature,
            "failure_kind": self.failure_kind,
            "axiom_id": self.axiom_id,
            "name": self.name,
            "engram_id": int(self.engram_id),
            "polarity": "REPULSOR",
            "source": REPULSOR_SOURCE,
        }


def _repulsor_wave(key: str, dim: int) -> np.ndarray:
    seed = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16) % (2**32)
    generator = np.random.default_rng(seed)
    wave = generator.standard_normal(int(dim)) + 1j * generator.standard_normal(int(dim))
    norm = float(np.linalg.norm(wave))
    return (wave / max(norm, 1e-12)).astype(np.complex64)


# ---------------------------------------------------------------------- engine
class RepoPatchEngine:
    """Localise, propose, gate, execute and prune until one patch is verified."""

    def __init__(
        self,
        rollouts: int = MAX_ROLLOUTS,
        timeout_s: float = 90.0,
        memory: Any = None,
        namespace: str = "repo_patch",
        workdir: Optional[str | Path] = None,
    ) -> None:
        self.rollouts = max(1, int(rollouts))
        self.timeout_s = float(timeout_s)
        self.memory = memory
        self.namespace = str(namespace)
        self.workdir = Path(workdir) if workdir is not None else None
        if self.workdir is not None:
            self.workdir.mkdir(parents=True, exist_ok=True)
        self.sandbox = RepoSubprocessSandbox(timeout_s=self.timeout_s)
        self.sheaf = RepoSheaf()
        self.repulsors: List[RepulsorRecord] = []
        self._registry: Dict[Tuple[str, str], str] = {}

    def solve(self, case: RepoCase, workspace: Optional[Path] = None) -> Dict[str, Any]:
        started = time.perf_counter()
        root = (
            Path(workspace)
            if workspace is not None
            else Path(
                tempfile.mkdtemp(
                    prefix="ad1_", dir=str(self.workdir) if self.workdir else None
                )
            )
        )
        materialise(case, root)
        package = {
            path: text
            for path, text in case.files.items()
            if path.endswith(".py") and not path.startswith("tests/")
        }
        baseline = self.sheaf.analyze(package)
        conflict = detect_public_conflict(case.public)
        result: Dict[str, Any] = {
            "case": case.name,
            "issue": case.issue,
            "files_seen": sorted(case.files),
            "test_path": case.test_path,
            "sheaf_baseline": baseline.as_dict(),
            "public_conflict": conflict,
            "llm_calls": 0,
            "repulsors": [],
            "rollouts": [],
            "deflections": 0,
            "seconds": 0.0,
        }
        if conflict["detected"]:
            return self._abstain(
                result,
                "contradictory_public_specification",
                provenance={"kind": "conflicting_assertions", "detail": conflict["conflicts"]},
                started=started,
            )
        indexer = CodeGraphIndexer()
        indexer.index(root, package="workspace")
        first = self.sandbox.run_tests(root, case.test_path)
        result["initial_tests"] = {
            "passed": bool(first["passed"]),
            "exit_code": int(first["exit_code"]),
            "duration_ms": float(first["duration_ms"]),
            "failure_kind": "",
        }
        if first["passed"]:
            return self._abstain(
                result,
                "issue_conflict_public_tests_pass",
                provenance={
                    "kind": "stale_issue",
                    "detail": "the published reproduction suite already passes; no patch can be verified",
                },
                started=started,
            )
        localization = FaultLocalizer(indexer).locate(root, first, case.test_path)
        result["initial_tests"]["failure_kind"] = localization.failure_kind
        result["localization"] = localization.as_dict()
        proposer = PatchProposer(indexer)
        candidates = proposer.propose(package, localization, self.rollouts)
        result["candidates_proposed"] = int(len(candidates))
        result["candidate_kinds"] = sorted({row.kind for row in candidates})
        if not candidates:
            return self._abstain(
                result,
                "no_structural_candidate_in_candidate_set",
                provenance={"kind": "localisation_empty", "detail": localization.as_dict()},
                started=started,
            )
        blocked_by_sheaf = 0
        emitted: Optional[Candidate] = None
        for rollout, candidate in enumerate(candidates, start=1):
            key = (candidate.qualname, candidate.signature)
            if key in self._registry:
                result["deflections"] = int(result["deflections"]) + 1
                result["rollouts"].append(
                    {
                        "rollout": rollout,
                        "qualname": candidate.qualname,
                        "kind": candidate.kind,
                        "signature": candidate.signature,
                        "deflected": True,
                        "failure_kind": self._registry[key],
                        "duration_ms": 0.0,
                    }
                )
                continue
            patched = dict(package)
            patched[candidate.path] = candidate.source
            delta_report = indexer.update_file(
                root, candidate.path, source_text=candidate.source
            )
            delta_ms = float(delta_report.seconds * 1e3)
            sheaf = self.sheaf.analyze(patched)
            row: Dict[str, Any] = {
                "rollout": rollout,
                "qualname": candidate.qualname,
                "kind": candidate.kind,
                "signature": candidate.signature,
                "line": int(candidate.line),
                "note": candidate.note,
                "sheaf_energy": float(sheaf.energy),
                "delta_ms": float(delta_ms),
                "deflected": False,
            }
            if float(sheaf.energy) != float(SHEAF_TOLERANCE):
                blocked_by_sheaf += 1
                indexer.update_file(
                    root, candidate.path, source_text=package[candidate.path]
                )
                row["public_ok"] = False
                row["blocked"] = "sheaf_gate"
                row["violations"] = [violation.as_dict() for violation in sheaf.violations]
                result["rollouts"].append(row)
                continue
            target = root / candidate.path
            original_text = target.read_text(encoding="utf-8")
            target.write_text(candidate.source, encoding="utf-8")
            run = self.sandbox.run_tests(root, case.test_path)
            row["public_ok"] = bool(run["passed"])
            row["exit_code"] = int(run["exit_code"])
            row["duration_ms"] = float(run["duration_ms"])
            row["failure_kind"] = "" if run["passed"] else _failure_kind(run["text"])
            result["rollouts"].append(row)
            if run["passed"]:
                emitted = candidate
                result["emitted_line"] = int(candidate.line)
                break
            target.write_text(original_text, encoding="utf-8")
            indexer.update_file(root, candidate.path, source_text=original_text)
            self._register_repulsor(
                candidate.qualname, candidate.signature, str(row["failure_kind"])
            )
        result["blocked_by_sheaf"] = int(blocked_by_sheaf)
        result["repulsors"] = [row.as_dict() for row in self.repulsors]
        if emitted is None:
            reason = (
                "sheaf_gate_blocked_all_candidates"
                if blocked_by_sheaf and blocked_by_sheaf == len(result["rollouts"])
                else "no_candidate_passed_public_tests"
            )
            provenance: Dict[str, Any] = {
                "kind": "pre_execution_sheaf_gate" if "sheaf" in reason else "rollout_exhausted",
                "baseline_energy": float(baseline.energy),
                "blocked_by_sheaf": int(blocked_by_sheaf),
                "rollouts": int(len(result["rollouts"])),
            }
            return self._abstain(result, reason, provenance=provenance, started=started)
        diff = "".join(
            difflib.unified_diff(
                str(package[emitted.path]).splitlines(keepends=True),
                str(emitted.source).splitlines(keepends=True),
                fromfile="a/" + emitted.path,
                tofile="b/" + emitted.path,
            )
        )
        result["status"] = "emitted"
        result["reason"] = "verified_public_suite"
        result["patch"] = {
            "path": emitted.path,
            "qualname": emitted.qualname,
            "kind": emitted.kind,
            "line": int(emitted.line),
            "note": emitted.note,
            "diff": diff,
            "ast_signature": emitted.signature,
            "source_sha256": hashlib.sha256(emitted.source.encode("utf-8")).hexdigest(),
        }
        result["patched_files"] = {emitted.path: emitted.source}
        result["sheaf_after"] = self.sheaf.analyze(
            {**package, emitted.path: emitted.source}
        ).as_dict()
        result["seconds"] = float(time.perf_counter() - started)
        return result

    def _abstain(
        self,
        result: Dict[str, Any],
        reason: str,
        *,
        provenance: Dict[str, Any],
        started: float,
    ) -> Dict[str, Any]:
        result["status"] = "abstained"
        result["reason"] = str(reason)
        result["provenance"] = provenance
        result["patched_files"] = {}
        result["patch"] = None
        result["seconds"] = float(time.perf_counter() - started)
        return result

    def _register_repulsor(
        self, qualname: str, signature: str, failure_kind: str
    ) -> RepulsorRecord:
        key = "%s|%s|%s" % (qualname, signature, failure_kind)
        name = "repo_patch_repulsor_%s" % hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
        dim = int(getattr(self.memory, "dim", AXIOM_DIM) or AXIOM_DIM)
        wave = _repulsor_wave(key, dim)
        engram_id = -1
        axiom_id = "%s:REPULSOR:%s" % (self.namespace, name)
        if self.memory is not None:
            try:
                axiom = self.memory.register_axiom(
                    wave,
                    name=name,
                    polarity="REPULSOR",
                    source=REPULSOR_SOURCE,
                    slot_mask=(True, True, True, True),
                    namespace=self.namespace,
                )
                engram_id = int(getattr(axiom, "engram_id", -1))
                axiom_id = str(getattr(axiom, "axiom_id", axiom_id))
            except Exception:  # noqa: BLE001 - a memory refusal never hides the patch result
                engram_id = -1
        record = RepulsorRecord(
            qualname=str(qualname),
            signature=str(signature),
            failure_kind=str(failure_kind),
            axiom_id=axiom_id,
            name=name,
            engram_id=int(engram_id),
        )
        self.repulsors.append(record)
        self._registry[(str(qualname), str(signature))] = str(failure_kind)
        return record


def _failure_kind(text: str) -> str:
    match = _ERROR_LINE.search(str(text))
    if match:
        return str(match.group(1))
    if "assert" in str(text).lower():
        return "AssertionError"
    return "unknown"


# ------------------------------------------------------------------ test runner
_RUNNER = '''import importlib.util, json, sys, traceback
path = sys.argv[1]
spec = importlib.util.spec_from_file_location("ad1_tests", path)
module = importlib.util.module_from_spec(spec)
sys.modules["ad1_tests"] = module
failures = []
total = 0
try:
    spec.loader.exec_module(module)
except BaseException:
    failures.append({"test": "<import>", "traceback": traceback.format_exc()})
else:
    for name in sorted(dir(module)):
        if not name.startswith("test_"):
            continue
        item = getattr(module, name)
        if not callable(item):
            continue
        total += 1
        try:
            item()
        except BaseException:
            failures.append({"test": name, "traceback": traceback.format_exc()})
print("AD1_RESULT " + json.dumps({"total": total, "failed": len(failures), "failures": failures}))
sys.exit(1 if failures else 0)
'''


def _runner_source() -> str:
    return _RUNNER


def _parse_runner(stdout: str) -> Dict[str, Any]:
    marker = "AD1_RESULT "
    at = str(stdout).rfind(marker)
    if at < 0:
        return {}
    payload = str(stdout)[at + len(marker) :].strip().splitlines()
    if not payload:
        return {}
    try:
        return json.loads(payload[0])
    except json.JSONDecodeError:
        return {}
