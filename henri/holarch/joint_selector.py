"""Joint epistemic selection of one adaptation or a RepoSheaf-admissible DAG.

Cycle AC2.  The AB2 harness pruned a type-incompatible plan BEFORE it was
ranked, which closed one class of miss, but the warm path still emitted the
FIRST verified candidate in retrieval order.  On ``z2_composed_03`` that first
candidate was a single adaptation of ``is_palindrome``: it passes the two
public examples (both answers are ``False``) and contradicts the hidden test
(``racecar`` has four distinct characters, and four is a perfect square).  A
two-stage DAG ``count_distinct_characters -> is_perfect_square`` answers the
same prompt and is in the PlanStore, but a first-pass exit never looked at it.

This module supplies the decision rule that compares the two:

    Cov_IDF(h, I)          the IDF-weighted share of the prompt intent I that
                           the program h explains, read from h's identifier
                           tokens, its AST identifiers and literals, and the
                           names of the plans it chains.
    DeltaResidue_IDF(h,I)  the residue reduction of h against the strongest
                           single-plan alternative (Coverage(h) - Coverage(ref)).
    |h|_AST                the AST node count of the emitted module.
    E_sheaf(h)             the count of RepoSheaf domain/codomain violations on
                           the plan-DAG edges plus the return edge of h.
    J(h)                   Cov_IDF + eta*DeltaResidue_IDF - lambda_MDL*|h|_AST

Normalisation (the one place a number is chosen): ``lambda_MDL`` is
``MDL_NODE_COST / n_clauses``, with ``MDL_NODE_COST = 0.005`` per AST node and
``n_clauses`` the number of discourse clauses the prompt was split into.  An
extra stage of ``d`` nodes then costs ``0.005*d/n_clauses`` while it earns
``(1 + eta)*DeltaResidue_IDF``, so it is rejected unless the residue actually
falls.  For the reference extra stage measured in this cycle (d ~ 38 nodes,
two clauses, residue gain 0.121) the charge is 0.095 and the crossing sits at
``DeltaResidue_IDF = 0.047``.  The cycle specification suggested a 0.25
crossing; the receipt records that 0.25 is NOT met and why: on this corpus the
strongest compositional candidate covers 0.136 of the intent's IDF mass, so a
0.25 crossing would reject every composition, including the ones whose hidden
tests pass.  The measured crossing is reported instead of the suggested one.
``synthetic_ab()`` in the AC2 runner measures the crossing rather than
asserting it.

The IDF table is built from the eligible PROMPT set only.  Hidden tests never
enter it, so the statistic cannot be fitted to the scoring set.
"""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

MDL_NODE_COST = 0.005
"""lambda_MDL numerator: cost of one AST node, before the clause normalisation."""
ETA_COMP = 1.0
"""Weight of the residue reduction term."""
COVERAGE_FLOOR = 0.80
"""A single plan that covers at least this share of the intent needs no DAG."""
TAU_RESIDUE = 0.25
"""The residue reduction the cycle specification asked an extra stage to earn."""
MAX_PLANS = 8
"""How many retrieved plans may contribute a single-stage adaptation."""
MAX_STAGES = 3
"""Longest composed DAG the two-stage builder is iterated into."""
CANDIDATE_CAP = 32
"""Hard cap on the candidate set one task may be judged on."""

_LEXEME = re.compile(r"[A-Za-z][A-Za-z0-9]*")
_CAMEL = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9]+")
_TRAILING_DIGITS = re.compile(r"[0-9]+$")

_CLAUSE_CUE = re.compile(
    r"(?:\s*(?:;|->|\u2192)\s*|\s+(?:and\s+then|then|after\s+that|afterwards|"
    r"followed\s+by|subsequently|next|finally)\s+)",
    re.IGNORECASE,
)
_CLAUSE_PREFIX = re.compile(
    r"(?i)^(?:first(?:ly)?|start\s+by|to\s+begin|begin\s+by)\s*[,:]?\s*"
)
_CONNECTIVES = (
    "followed by",
    "and then",
    "after that",
    "subsequently",
    "afterwards",
    "finally",
    "then",
    "next",
    "after",
    "->",
    "\u2192",
    ";",
)
_ANAPHOR_WORDS = frozenset({"apply", "result"})
"""The 'apply ... to the result' anaphor, as a word set, not a literal test."""
_SINGLE_KIND = "single"
_COMPOSE_KIND = "compose"
_IMPERATIVE_VERBS = frozenset(
    {
        "write", "return", "find", "compute", "calculate", "count", "check",
        "determine", "remove", "sort", "reverse", "convert", "apply", "add",
        "print", "create", "generate", "sum", "multiply", "capitalize",
        "split", "join", "filter", "map", "replace", "insert", "delete",
        "merge", "flatten", "rotate", "shift", "normalize", "normalise",
        "rescale", "encode", "decode", "pad", "truncate", "dedupe", "group",
        "search", "locate", "extract", "read", "parse", "build", "construct",
    }
)


# --------------------------------------------------------------- tokenising
def word_tokens(text: str) -> List[str]:
    """Whole-word tokens of a text, splitting snake_case and camelCase.

    A substring test would bind a sketch to a letter rather than a word, so a
    compound identifier is read as the sequence of words it names.
    """
    out: List[str] = []
    for part in _LEXEME.findall(str(text)):
        for piece in _CAMEL.findall(part):
            token = _TRAILING_DIGITS.sub("", piece.lower())
            if token:
                out.append(token)
    return out


def program_features(source: str) -> Tuple[Set[str], int]:
    """(token set, AST node count) of one emitted program.

    The token set is the identifier surface the program presents to a reader:
    function names, argument names, every Name read or written, attribute
    names, imported symbols and string literals.  The node count is the size
    the minimum-description-length term charges for.
    """
    text = str(source)
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return set(word_tokens(text)), 0
    tokens: Set[str] = set()
    nodes = 0
    for node in ast.walk(tree):
        nodes += 1
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            tokens.update(word_tokens(node.name))
        elif isinstance(node, ast.Name):
            tokens.update(word_tokens(node.id))
        elif isinstance(node, ast.Attribute):
            tokens.update(word_tokens(node.attr))
        elif isinstance(node, ast.arg):
            tokens.update(word_tokens(node.arg))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            tokens.update(word_tokens(node.value))
        elif isinstance(node, ast.ImportFrom):
            tokens.update(word_tokens(node.module or ""))
        elif isinstance(node, ast.alias):
            tokens.update(word_tokens(node.name))
    return tokens, int(nodes)


# --------------------------------------------------------------------- IDF
@dataclass
class PromptIDF:
    """Document frequencies of the eligible prompt set, and nothing else."""

    documents: int = 0
    document_frequency: Dict[str, int] = field(default_factory=dict)
    source: str = "eligible_prompt_intents"

    @classmethod
    def from_intents(
        cls, intents: Sequence[str], source: str = "eligible_prompt_intents"
    ) -> "PromptIDF":
        frequency: Dict[str, int] = {}
        documents = 0
        for text in intents:
            tokens = set(word_tokens(text))
            if not tokens:
                continue
            documents += 1
            for token in tokens:
                frequency[token] = frequency.get(token, 0) + 1
        return cls(
            documents=int(documents),
            document_frequency=dict(sorted(frequency.items())),
            source=str(source),
        )

    def idf(self, token: str) -> float:
        """Smoothed inverse document frequency; an unseen token is maximally rare."""
        frequency = int(self.document_frequency.get(str(token), 0))
        return float(math.log((self.documents + 1.0) / (frequency + 1.0)) + 1.0)

    def weights(self, tokens: Sequence[str]) -> Dict[str, float]:
        return {str(token): self.idf(str(token)) for token in tokens}

    def summary(self) -> Dict[str, Any]:
        return {
            "source": str(self.source),
            "documents": int(self.documents),
            "vocabulary": int(len(self.document_frequency)),
            "idf_max": float(max([self.idf(t) for t in self.document_frequency] or [1.0])),
            "idf_min": float(min([self.idf(t) for t in self.document_frequency] or [1.0])),
        }


def coverage_idf(
    tokens_h: Set[str], intent_tokens: Sequence[str], idf: PromptIDF
) -> float:
    """IDF-weighted share of the intent that the program's token set explains."""
    total = 0.0
    hit = 0.0
    for token in intent_tokens:
        weight = idf.idf(token)
        total += weight
        if token in tokens_h:
            hit += weight
    return float(hit / total) if total > 0.0 else 0.0


# --------------------------------------------------------------- discourse
def discourse_report(text: str) -> Dict[str, Any]:
    """Deterministic reading of a prompt's compositional structure.

    Only generic linguistic facts are used: the ordered connectives of a
    compound instruction, how many clauses they delimit, how many clauses open
    with an imperative verb, and the "apply ... to the result" anaphor.  No
    task name, no task id and no benchmark phrase is consulted.
    """
    plain = re.sub(r"\s+", " ", str(text)).strip()
    lowered = " " + plain.lower() + " "
    markers = [cue for cue in _CONNECTIVES if cue in lowered]
    parts = [part.strip() for part in _CLAUSE_CUE.split(plain)]
    clauses: List[str] = []
    for index, part in enumerate(parts):
        if index == 0:
            part = _CLAUSE_PREFIX.sub("", part).strip()
        part = part.strip(" \t,;.")
        if part:
            clauses.append(part)
    sentences = [part.strip() for part in re.split(r"[.;]\s+", plain) if part.strip()]
    imperative = [
        sentence
        for sentence in sentences
        if word_tokens(sentence)[:1]
        and word_tokens(sentence)[0] in _IMPERATIVE_VERBS
    ]
    anaphor = bool(_ANAPHOR_WORDS <= set(word_tokens(plain)))
    compositional = bool(
        (markers and len(clauses) >= 2) or len(imperative) >= 3 or anaphor
    )
    return {
        "markers": list(markers),
        "clauses": clauses,
        "clause_count": int(max(1, len(clauses))),
        "imperative_clauses": [str(row)[:80] for row in imperative],
        "imperative_count": int(len(imperative)),
        "apply_to_result": bool(anaphor),
        "compositional": bool(compositional),
    }


# --------------------------------------------------------- source builders
class _Rename(ast.NodeTransformer):
    def __init__(self, mapping: Dict[str, str]) -> None:
        self.mapping = dict(mapping)

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in self.mapping:
            node.id = self.mapping[node.id]
        return node


def adapt_source(source: str, name: str, params: Sequence[str]) -> str:
    """Rebind a verified plan to a new signature (name-based role binding).

    A byte-faithful descendant of the AB2 rebound used by the transfer arm: the
    parameter ORDER may change, a rename is applied through the body, and the
    declared annotations travel with the definition.
    """
    try:
        tree = ast.parse(str(source))
    except SyntaxError:
        return ""
    definition = next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef)), None
    )
    if definition is None:
        return ""
    old_params = [
        str(arg.arg)
        for arg in definition.args.args
        if str(arg.arg) not in {"self", "cls"}
    ]
    new_params = [str(param) for param in params]
    if len(old_params) != len(new_params):
        return ""
    old_function = str(definition.name)
    if set(old_params) == set(new_params):
        # A reordering of the same names is a name-preserving rebind.
        for index, arg in enumerate(definition.args.args):
            if str(arg.arg) not in {"self", "cls"}:
                arg.arg = new_params[index]
    else:
        mapping = {
            old: new for old, new in zip(old_params, new_params) if old != new
        }
        if old_function and old_function != name:
            mapping[old_function] = str(name)
        for index, arg in enumerate(definition.args.args):
            if str(arg.arg) not in {"self", "cls"}:
                arg.arg = new_params[index]
        if mapping:
            transformer = _Rename(mapping)
            definition.body = [
                transformer.visit(statement) for statement in definition.body
            ]
    if old_function != name:
        mapping = {old_function: str(name)}
        transformer = _Rename(mapping)
        definition.body = [transformer.visit(statement) for statement in definition.body]
    definition.name = str(name)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"


def compose_sources(first: str, second: str, name: str, param: str) -> str:
    """Chain two verified behaviours into one wrapper module."""
    stage_a = adapt_source(first, "_z2_stage_1", ["value"])
    stage_b = adapt_source(second, "_z2_stage_2", ["value"])
    if not stage_a or not stage_b:
        return ""
    wrapper = "def %s(%s):\n    return _z2_stage_2(_z2_stage_1(%s))\n" % (
        str(name),
        str(param),
        str(param),
    )
    return stage_a.strip() + "\n\n\n" + stage_b.strip() + "\n\n\n" + wrapper


def compose_chain_sources(
    sources: Sequence[str], name: str, param: str
) -> str:
    """Chain N >= 2 verified behaviours into one wrapper module.

    A two-stage chain is byte-identical to ``compose_sources``, so the shape a
    transfer task's reference implementation has is the shape the selector
    judges.
    """
    stages = [str(source) for source in sources]
    if len(stages) < 2:
        return ""
    if len(stages) == 2:
        return compose_sources(stages[0], stages[1], name, param)
    adapted: List[str] = []
    for index, source in enumerate(stages):
        stage = adapt_source(source, "_z2_stage_%d" % (index + 1), ["value"])
        if not stage:
            return ""
        adapted.append(stage.strip())
    call = "%s" % str(param)
    for index in range(len(stages), 0, -1):
        call = "_z2_stage_%d(%s)" % (index, call)
    wrapper = "def %s(%s):\n    return %s\n" % (str(name), str(param), call)
    return "\n\n\n".join(adapted) + "\n\n\n" + wrapper


# --------------------------------------------------------- sheaf energies
def return_edge_energy(source: str, declared: str) -> int:
    """1 when the module returns a capability outside its declared annotation.

    This is the RepoSheaf pullback on the return edge, evaluated with the
    module's OWN function table so a wrapper that returns a call to a local
    stage is read instead of being skipped as un-inferable.
    """
    if not str(declared or "").strip():
        return 0
    try:
        from henri.holarch.repo_sheaf import _annotation_vector, _default_provision

        tree = ast.parse(str(source))
        functions = {}
        definition = None
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions[str(node.name)] = node
        definition = next(
            (node for node in tree.body if isinstance(node, ast.FunctionDef)), None
        )
        if definition is None:
            return 1
        declared_vector = _annotation_vector(definition.returns)
        if declared_vector is None:
            return 0
        env: Dict[str, np.ndarray] = {}
        for arg in definition.args.args:
            vector = _annotation_vector(arg.annotation)
            if vector is not None:
                env[arg.arg] = vector
        table: Dict[str, Any] = {}
        for node in functions.values():
            vector = _annotation_vector(node.returns)
            if vector is not None:
                table[str(node.name)] = _ReturnOnly(vector)
        for node in ast.walk(definition):
            if not isinstance(node, ast.Return) or node.value is None:
                continue
            provision = _default_provision(node.value, env, table)
            if float(np.abs(np.asarray(provision)).sum()) <= 0.0:
                continue
            if bool(np.any(np.asarray(provision) > np.asarray(declared_vector))):
                return 1
    except Exception:  # noqa: BLE001 - an unreadable body is not refused
        return 0
    return 0


@dataclass
class _ReturnOnly:
    """The minimum shape ``_default_provision`` reads from a callee profile."""

    return_type: Optional[str] = None


def chain_sheaf_energy(plans: Sequence[Any], gate: Any) -> Tuple[int, List[Dict[str, Any]]]:
    """Count RepoSheaf domain/codomain violations on the plan-DAG edges."""
    edges: List[Dict[str, Any]] = []
    violations = 0
    for left, right in zip(plans, list(plans)[1:]):
        verdict = gate.check(left, right) if gate is not None else {"ok": True}
        ok = bool(verdict.get("ok", True))
        if not ok:
            violations += 1
        edges.append(
            {
                "from": str(left.name),
                "to": str(right.name),
                "ok": bool(ok),
            }
        )
    return int(violations), edges


# ------------------------------------------------------------- retrieval
def rank_plans(store: Any, intent: str) -> List[Any]:
    """Rank stored playbooks by how much of their intent the request contains.

    A byte-faithful descendant of the AB2 ranking: containment of the plan's
    IDF-weighted intent tokens in the full request, plus the argument-type verb
    bias, with wave coherence breaking the ties.
    """
    key = store.key(str(intent))
    weights = store.token_weight
    intent_tokens = {int(token) for token in store.tokenizer.encode(str(intent))}
    scored = []
    for plan in store.plans.values():
        plan_tokens = {
            int(token) for token in store.tokenizer.encode(plan.intent)[:96]
        }
        if not plan_tokens:
            continue
        total = 0.0
        hit = 0.0
        for token in plan_tokens:
            weight = float(weights[token]) if 0 <= token < weights.size else 1.0
            total += weight
            if token in intent_tokens:
                hit += weight
        containment = float(hit / total) if total > 0.0 else 0.0
        coherence = float(np.real(np.vdot(key, store.key(plan.intent))))
        bias = 0.0
        try:
            bias = float(store._type_verb_bias(str(intent), plan))
        except Exception:  # noqa: BLE001 - the bias is optional
            bias = 0.0
        scored.append((min(1.0, containment + bias), coherence + bias, plan))
    scored.sort(key=lambda row: (-row[0], -row[1], row[2].name))
    return [plan for _containment, _coherence, plan in scored]


def _args_of(plan: Any) -> List[Any]:
    return list((getattr(plan, "slots", {}) or {}).get("args", []) or [])


def candidate_tokens(candidate: Dict[str, Any]) -> Set[str]:
    """The identifier surface of one candidate, plan names included.

    A composed module renames its stages to ``_z2_stage_1`` / ``_z2_stage_2``,
    so the behaviour a stage contributes is named by the PLAN it came from, not
    by the private symbol it was bound to.  The plan names are therefore part
    of the program's identifier surface for coverage purposes.
    """
    tokens, _nodes = program_features(str(candidate["source"]))
    for name in candidate.get("names") or []:
        tokens.update(word_tokens(str(name)))
    return tokens


def warm_candidates(
    store: Any,
    *,
    name: str,
    params: Sequence[str],
    intent: str,
    coverage_floor: float = COVERAGE_FLOOR,
    compositional: bool = False,
    cap: int = CANDIDATE_CAP,
    max_plans: int = MAX_PLANS,
    max_stages: int = MAX_STAGES,
    idf: Optional[PromptIDF] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Every candidate the warm path must judge before it may choose one.

    Order is the AB2 policy order (single-plan adaptations first, in retrieval
    order, then composed DAGs), so a first-pass policy over this list is
    exactly the AB2 baseline and the joint selector is a strict re-ranking of
    it.  Composition is attempted when the strongest single adaptation covers
    less than ``coverage_floor`` of the intent OR the prompt carries discourse
    structure, which is the Cycle AC2 no-premature-exit rule.
    """
    plans = list(getattr(store, "plans", {}) or {})
    if not plans:
        return [], {"engaged": False, "reason": "empty_plan_store"}
    gate = getattr(store, "type_gate", None)
    ranked = rank_plans(store, intent)
    declared_singles: List[Dict[str, Any]] = []
    intent_tokens = set(word_tokens(intent))
    for plan in ranked[: int(max_plans)]:
        source = adapt_source(plan.source, name, params)
        if not source:
            continue
        declared_singles.append(
            {
                "label": "adapt:" + str(plan.name),
                "kind": "single",
                "names": [str(plan.name)],
                "source": source,
                "plans": [plan],
                "sheaf_edges": [],
                "sheaf_violations": 0,
            }
        )
    best = 0.0
    for candidate in declared_singles:
        tokens = candidate_tokens(candidate)
        if idf is None:
            continue
        value = coverage_idf(tokens, sorted(intent_tokens), idf)
        candidate["single_coverage"] = float(value)
        best = max(best, value)
    single_coverage = float(best)
    needs_dag = bool(
        single_coverage < float(coverage_floor) or bool(compositional)
    )
    info: Dict[str, Any] = {
        "engaged": True,
        "ranked": [str(plan.name) for plan in ranked[: int(max_plans)]],
        "max_single_coverage": float(single_coverage),
        "coverage_floor": float(coverage_floor),
        "compositional": bool(compositional),
        "needs_dag": bool(needs_dag),
        "dag_chains": [],
        "pairs_offered": 0,
        "pairs_masked": 0,
        "sheaf_checked": 0,
        "cap": int(cap),
    }
    candidates: List[Dict[str, Any]] = list(declared_singles)
    if not needs_dag or len(params) != 1:
        info["reason"] = (
            "single_plan_sufficient" if not needs_dag else "multi_argument_prompt"
        )
        return candidates[: int(cap)], info

    composed: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, ...]] = set()
    if int(max_stages) >= 2:
        dag = store.retrieve_dag(intent, max_plans=int(max_stages), arity=len(params))
        if len(dag) >= 2:
            chain = [plan for plan, _score in dag]
            violations, edges = chain_sheaf_energy(chain, gate)
            key = tuple(str(plan.name) for plan in chain)
            source = compose_chain_sources(
                [plan.source for plan in chain], name, str(params[0])
            )
            if source and key not in seen:
                seen.add(key)
                info["sheaf_checked"] = int(info["sheaf_checked"]) + len(edges)
                if violations:
                    info["pairs_masked"] = int(info["pairs_masked"]) + 1
                else:
                    composed.append(
                        {
                            "label": "dag:" + "+".join(key),
                            "kind": "compose",
                            "names": list(key),
                            "source": source,
                            "plans": chain,
                            "sheaf_edges": edges,
                            "sheaf_violations": int(violations),
                        }
                    )
                    info["dag_chains"].append(list(key))
    unary = [plan for plan in ranked[: int(max_plans)] if len(_args_of(plan)) == 1]
    ordered: List[Tuple[int, int, int, Any, Any]] = []
    for first_index, first in enumerate(unary):
        for second_index, second in enumerate(unary):
            if first.name == second.name:
                continue
            ordered.append(
                (first_index + second_index, first_index, second_index, first, second)
            )
    ordered.sort(key=lambda item: (item[0], item[1], item[2]))
    for _cost, _first_index, _second_index, first, second in ordered:
        info["pairs_offered"] = int(info["pairs_offered"]) + 1
        violations, edges = chain_sheaf_energy([first, second], gate)
        info["sheaf_checked"] = int(info["sheaf_checked"]) + len(edges)
        if violations:
            info["pairs_masked"] = int(info["pairs_masked"]) + 1
            continue
        key = (str(first.name), str(second.name))
        if key in seen:
            continue
        source = compose_sources(first.source, second.source, name, str(params[0]))
        if not source:
            continue
        seen.add(key)
        composed.append(
            {
                "label": "compose:" + "+".join(key),
                "kind": "compose",
                "names": list(key),
                "source": source,
                "plans": [first, second],
                "sheaf_edges": edges,
                "sheaf_violations": 0,
            }
        )
    info["reason"] = "composed_dag_admissible"
    return (candidates + composed)[: int(cap)], info


# --------------------------------------------------------------- scoring
def score_table(
    intent: str,
    clause_count: int,
    candidates: Sequence[Dict[str, Any]],
    idf: PromptIDF,
    *,
    eta_comp: float = ETA_COMP,
    mdl_node_cost: float = MDL_NODE_COST,
) -> Dict[str, Any]:
    """Score every candidate and return the auditable table plus the winner.

    ``J(h) = Cov_IDF + eta*DeltaResidue_IDF - lambda_MDL*|h|_AST`` with
    ``lambda_MDL = mdl_node_cost / max(1, clause_count)`` and the residue
    measured against the strongest single-plan candidate that clears the sheaf
    gate.  The winner is the admissible (verified, E_sheaf == 0) argmax.
    """
    intent_tokens = sorted(set(word_tokens(intent)))
    clauses = max(1, int(clause_count))
    lam = float(mdl_node_cost) / float(clauses)
    rows: List[Dict[str, Any]] = []
    for candidate in candidates:
        tokens, nodes = program_features(str(candidate["source"]))
        for plan_name in candidate.get("names") or []:
            tokens.update(word_tokens(str(plan_name)))
        candidate["_tokens"] = tokens
        candidate["_nodes"] = int(nodes)
        rows.append(
            {
                "label": str(candidate["label"]),
                "kind": str(candidate["kind"]),
                "names": list(candidate.get("names") or []),
                "sheaf_energy": int(candidate.get("sheaf_violations", 0))
                + int(candidate.get("return_edge", 0)),
                "ast_nodes": int(nodes),
                "cov_idf": float(coverage_idf(tokens, intent_tokens, idf)),
                "verified": bool(candidate.get("verified", False)),
                "detail": str(candidate.get("detail", ""))[:160],
                "sheaf_edges": list(candidate.get("sheaf_edges") or []),
            }
        )
    singles = [
        row
        for row in rows
        if row["kind"] == _SINGLE_KIND
        and int(row["sheaf_energy"]) == 0
        and bool(row["verified"])
    ]
    reference = None
    if singles:
        # The residue is measured against the strongest single-plan candidate
        # that PASSES the public examples: the strongest answer one stage could
        # actually deliver.  A single that cannot be emitted is not an
        # alternative, so it cannot set the bar for the second stage.
        reference = min(
            singles,
            key=lambda row: (-float(row["cov_idf"]), int(row["ast_nodes"]), row["label"]),
        )
    elif rows:
        reference = min(
            [row for row in rows if row["kind"] == _SINGLE_KIND] or rows,
            key=lambda row: (-float(row["cov_idf"]), int(row["ast_nodes"]), row["label"]),
        )
    reference_coverage = float(reference["cov_idf"]) if reference is not None else 0.0
    for row in rows:
        row["residue"] = float(1.0 - float(row["cov_idf"]))
        row["delta_residue"] = float(row["cov_idf"]) - reference_coverage
        row["mdl_penalty"] = float(lam) * float(row["ast_nodes"])
        row["j"] = float(
            float(row["cov_idf"])
            + float(eta_comp) * float(row["delta_residue"])
            - float(row["mdl_penalty"])
        )
        row["admissible"] = bool(
            bool(row["verified"]) and int(row["sheaf_energy"]) == 0
        )
    admissible = [row for row in rows if row["admissible"]]
    admissible.sort(key=lambda row: (-float(row["j"]), int(row["ast_nodes"]), row["label"]))
    first_pass = next((row for row in rows if row["admissible"]), None)
    chosen = admissible[0] if admissible else None
    by_label = {row["label"]: row for row in rows}
    return {
        "rows": rows,
        "by_label": by_label,
        "chosen": chosen,
        "first_pass": first_pass,
        "reference": reference,
        "reference_coverage": float(reference_coverage),
        "lambda_mdl": float(lam),
        "eta_comp": float(eta_comp),
        "clause_count": int(clauses),
        "crossover_delta_residue": (
            float(lam) * float(chosen["ast_nodes"]) - float(lam) * float(reference["ast_nodes"])
        )
        / (1.0 + float(eta_comp))
        if (chosen is not None and reference is not None)
        else 0.0,
        "intent_tokens": int(len(intent_tokens)),
        "idf_documents": int(idf.documents),
    }
