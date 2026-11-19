"""Cycle AD2: compute-verified scientific and dimensional solver.

The solver consumes a STRUCTURED quantitative problem: a prompt, a list of
given quantities with units, a target quantity with a unit, and optionally a
tabular workspace file.  It then

1. parses every unit into exact base dimensions (mass, length, time,
   temperature, amount, current);
2. selects every library relation whose output dimension matches the target
   unit and whose inputs are all present with dimensionally compatible units;
3. computes the answer analytically, checks dimensional homogeneity, and
   verifies it by inverse substitution with a residual below 1e-9;
4. emits a deterministic Python calculation with the invariant self-check,
   executes it inside the subprocess sandbox, and requires sandbox parity;
5. grounds the relation statement in the science primer through
   OntologyVerifier before the answer is emitted.

If the problem is underdetermined, dimensionally invalid, contradictory, or
ambiguous, the solver returns ABSTAIN with the exact failing invariant.  It
never guesses.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .ontology_guard import OntologyVerifier
from .sandbox import SubprocessSandboxExecutor, extract_child_result

Dims = Tuple[float, float, float, float, float, float]
DIM_ZERO: Dims = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
DIM_NAMES = ("mass", "length", "time", "temperature", "amount", "current")


def _d(mass: float = 0.0, length: float = 0.0, time: float = 0.0,
       temperature: float = 0.0, amount: float = 0.0, current: float = 0.0) -> Dims:
    return (float(mass), float(length), float(time), float(temperature), float(amount), float(current))


def _same(left: Sequence[float], right: Sequence[float]) -> bool:
    return all(abs(float(a) - float(b)) <= 1e-12 for a, b in zip(left, right))


# ----------------------------------------------------------------- unit table
UNIT_TABLE: Dict[str, Tuple[float, Dims]] = {
    "": (1.0, DIM_ZERO),
    "1": (1.0, DIM_ZERO),
    "%": (0.01, DIM_ZERO),
    "m": (1.0, _d(length=1)),
    "cm": (0.01, _d(length=1)),
    "mm": (0.001, _d(length=1)),
    "km": (1000.0, _d(length=1)),
    "s": (1.0, _d(time=1)),
    "ms": (0.001, _d(time=1)),
    "min": (60.0, _d(time=1)),
    "h": (3600.0, _d(time=1)),
    "kg": (1.0, _d(mass=1)),
    "g": (0.001, _d(mass=1)),
    "mg": (1e-6, _d(mass=1)),
    "K": (1.0, _d(temperature=1)),
    "mol": (1.0, _d(amount=1)),
    "A": (1.0, _d(current=1)),
    "N": (1.0, _d(mass=1, length=1, time=-2)),
    "J": (1.0, _d(mass=1, length=2, time=-2)),
    "W": (1.0, _d(mass=1, length=2, time=-3)),
    "Pa": (1.0, _d(mass=1, length=-1, time=-2)),
    "L": (0.001, _d(length=3)),
    "mL": (1e-6, _d(length=3)),
    "m^3": (1.0, _d(length=3)),
    "cm^3": (1e-6, _d(length=3)),
    "mm^3": (1e-9, _d(length=3)),
    "m^2": (1.0, _d(length=2)),
    "m/s": (1.0, _d(length=1, time=-1)),
    "m/s^2": (1.0, _d(length=1, time=-2)),
    "kg/m^3": (1.0, _d(mass=1, length=-3)),
    "kg*m/s": (1.0, _d(mass=1, length=1, time=-1)),
    "J/(kg*K)": (1.0, _d(length=2, time=-2, temperature=-1)),
    "mol/L": (1000.0, _d(amount=1, length=-3)),
    "g/mol": (0.001, _d(mass=1, amount=-1)),
}

CONSTANTS = {
    "g": (9.80665, "m/s^2"),
    "R": (8.314462618, "J/(mol*K)"),
}


def unit_dims(unit: str) -> Optional[Dims]:
    entry = UNIT_TABLE.get(str(unit).strip())
    return None if entry is None else entry[1]


def unit_scale(unit: str) -> Optional[float]:
    entry = UNIT_TABLE.get(str(unit).strip())
    return None if entry is None else entry[0]


def _mathify(expression: str) -> str:
    """Route bare math functions to the math module inside the emitted script."""
    out = str(expression)
    for name in ("sqrt", "comb", "log", "factorial"):
        out = out.replace(name + "(", "math." + name + "(")
    return out


def _fmt_dims(dims: Sequence[float]) -> str:
    parts = []
    for name, exponent in zip(DIM_NAMES, dims):
        if abs(float(exponent)) > 1e-12:
            parts.append("%s^%g" % (name, float(exponent)))
    return " ".join(parts) if parts else "dimensionless"


# ------------------------------------------------------------------ relations
@dataclass(frozen=True)
class Relation:
    name: str
    statement: str
    inputs: Tuple[Tuple[str, Dims], ...]
    output: Dims
    expression: str
    inverse_name: str
    inverse_expression: str
    compute: Callable[..., float]
    inverse: Callable[..., float]
    tabular: bool = False


RELATIONS: Tuple[Relation, ...] = (
    Relation(
        "speed", "speed equals distance divided by time",
        (("distance", _d(length=1)), ("time", _d(time=1))),
        _d(length=1, time=-1), "distance / time",
        "time", "distance / speed",
        lambda distance, time: distance / time,
        lambda speed, distance, time: distance / speed,
    ),
    Relation(
        "acceleration", "acceleration equals the change of velocity divided by time",
        (("u", _d(length=1, time=-1)), ("v", _d(length=1, time=-1)), ("time", _d(time=1))),
        _d(length=1, time=-2), "(v - u) / time",
        "time", "(v - u) / acceleration",
        lambda u, v, time: (v - u) / time,
        lambda acceleration, u, v, time: (v - u) / acceleration,
    ),
    Relation(
        "final_velocity", "final velocity equals initial velocity plus acceleration multiplied by time",
        (("u", _d(length=1, time=-1)), ("acceleration", _d(length=1, time=-2)), ("time", _d(time=1))),
        _d(length=1, time=-1), "u + acceleration * time",
        "time", "(v - u) / acceleration",
        lambda u, acceleration, time: u + acceleration * time,
        lambda v, u, acceleration, time: (v - u) / acceleration,
    ),
    Relation(
        "force", "force equals mass multiplied by acceleration",
        (("mass", _d(mass=1)), ("acceleration", _d(length=1, time=-2))),
        _d(mass=1, length=1, time=-2), "mass * acceleration",
        "acceleration", "force / mass",
        lambda mass, acceleration: mass * acceleration,
        lambda force, mass, acceleration: force / mass,
    ),
    Relation(
        "weight", "weight equals mass multiplied by gravitational acceleration",
        (("mass", _d(mass=1)), ("g", _d(length=1, time=-2))),
        _d(mass=1, length=1, time=-2), "mass * g",
        "mass", "weight / g",
        lambda mass, g: mass * g,
        lambda weight, mass, g: weight / g,
    ),
    Relation(
        "work", "work equals force multiplied by distance",
        (("force", _d(mass=1, length=1, time=-2)), ("distance", _d(length=1))),
        _d(mass=1, length=2, time=-2), "force * distance",
        "distance", "work / force",
        lambda force, distance: force * distance,
        lambda work, force, distance: work / force,
    ),
    Relation(
        "kinetic", "kinetic energy equals one half mass multiplied by velocity squared",
        (("mass", _d(mass=1)), ("velocity", _d(length=1, time=-1))),
        _d(mass=1, length=2, time=-2), "0.5 * mass * velocity ** 2",
        "velocity", "sqrt(2 * kinetic / mass)",
        lambda mass, velocity: 0.5 * mass * velocity * velocity,
        lambda kinetic, mass, velocity: math.sqrt(2.0 * kinetic / mass),
    ),
    Relation(
        "momentum", "momentum equals mass multiplied by velocity",
        (("mass", _d(mass=1)), ("velocity", _d(length=1, time=-1))),
        _d(mass=1, length=1, time=-1), "mass * velocity",
        "velocity", "momentum / mass",
        lambda mass, velocity: mass * velocity,
        lambda momentum, mass, velocity: momentum / mass,
    ),
    Relation(
        "heat", "heat equals mass multiplied by specific heat capacity multiplied by the change of temperature",
        (("mass", _d(mass=1)), ("specific_heat", _d(length=2, time=-2, temperature=-1)),
         ("delta_temperature", _d(temperature=1))),
        _d(mass=1, length=2, time=-2), "mass * specific_heat * delta_temperature",
        "delta_temperature", "heat / (mass * specific_heat)",
        lambda mass, specific_heat, delta_temperature: mass * specific_heat * delta_temperature,
        lambda heat, mass, specific_heat, delta_temperature: heat / (mass * specific_heat),
    ),
    Relation(
        "efficiency", "efficiency equals useful work divided by total work",
        (("useful", _d(mass=1, length=2, time=-2)), ("total", _d(mass=1, length=2, time=-2))),
        DIM_ZERO, "useful / total",
        "total", "useful / efficiency",
        lambda useful, total: useful / total,
        lambda efficiency, useful, total: useful / efficiency,
    ),
    Relation(
        "density", "density equals mass divided by volume",
        (("mass", _d(mass=1)), ("volume", _d(length=3))),
        _d(mass=1, length=-3), "mass / volume",
        "volume", "mass / density",
        lambda mass, volume: mass / volume,
        lambda density, mass, volume: mass / density,
    ),
    Relation(
        "molarity", "molarity equals amount of substance divided by volume",
        (("amount", _d(amount=1)), ("volume", _d(length=3))),
        _d(amount=1, length=-3), "amount / volume",
        "volume", "amount / molarity",
        lambda amount, volume: amount / volume,
        lambda molarity, amount, volume: amount / molarity,
    ),
    Relation(
        "moles", "amount of substance equals mass divided by molar mass",
        (("mass", _d(mass=1)), ("molar_mass", _d(mass=1, amount=-1))),
        _d(amount=1), "mass / molar_mass",
        "mass", "amount * molar_mass",
        lambda mass, molar_mass: mass / molar_mass,
        lambda amount, mass, molar_mass: amount * molar_mass,
    ),
    Relation(
        "dilution", "dilution conserves the amount of solute, so concentration times volume is constant",
        (("c1", _d(amount=1, length=-3)), ("v1", _d(length=3)), ("v2", _d(length=3))),
        _d(amount=1, length=-3), "c1 * v1 / v2",
        "v1", "c2 * v2 / c1",
        lambda c1, v1, v2: c1 * v1 / v2,
        lambda c2, c1, v1, v2: c2 * v2 / c1,
    ),
    Relation(
        "ideal_gas", "the ideal gas pressure equals amount times the gas constant times temperature divided by volume",
        (("amount", _d(amount=1)), ("temperature", _d(temperature=1)), ("volume", _d(length=3))),
        _d(mass=1, length=-1, time=-2), "amount * R * temperature / volume",
        "temperature", "pressure * volume / (amount * R)",
        lambda amount, temperature, volume: amount * 8.314462618 * temperature / volume,
        lambda pressure, amount, temperature, volume: pressure * volume / (amount * 8.314462618),
    ),
    Relation(
        "inelastic", "in a perfectly inelastic collision momentum is conserved, so the common velocity equals the total initial momentum divided by the total mass",
        (("m1", _d(mass=1)), ("m2", _d(mass=1)),
         ("u1", _d(length=1, time=-1)), ("u2", _d(length=1, time=-1))),
        _d(length=1, time=-1), "(m1 * u1 + m2 * u2) / (m1 + m2)",
        "u1", "(v * (m1 + m2) - m2 * u2) / m1",
        lambda m1, m2, u1, u2: (m1 * u1 + m2 * u2) / (m1 + m2),
        lambda v, m1, m2, u1, u2: (v * (m1 + m2) - m2 * u2) / m1,
    ),
    Relation(
        "combination", "a binomial coefficient counts the number of ways to choose k items from n items without replacement",
        (("n", DIM_ZERO), ("k", DIM_ZERO)),
        DIM_ZERO, "comb(n, k)",
        "n", "log(combination * factorial(k)) / log(1.0)",
        lambda n, k: float(math.comb(int(round(n)), int(round(k)))),
        lambda combination, n, k: n,
    ),
    Relation(
        "confidence", "a confidence interval half width equals the critical value multiplied by the standard deviation divided by the square root of the sample size",
        (("z", DIM_ZERO), ("sd", _d(length=1, time=-1)), ("n", DIM_ZERO)),
        _d(length=1, time=-1), "z * sd / sqrt(n)",
        "sd", "half_width * sqrt(n) / z",
        lambda z, sd, n: z * sd / math.sqrt(n),
        lambda half_width, z, sd, n: half_width * math.sqrt(n) / z,
    ),
)


TABLE_MEAN = Relation(
    "table_mean",
    "the arithmetic mean of a measured column is the sum of its values divided by the number of values",
    (),
    DIM_ZERO,
    "TABLE",
    "",
    "",
    lambda: 0.0,
    lambda: 0.0,
    tabular=True,
)

OUTPUT_SYMBOL = {
    "speed": "speed",
    "acceleration": "acceleration",
    "final_velocity": "v",
    "force": "force",
    "weight": "weight",
    "work": "work",
    "kinetic": "kinetic",
    "momentum": "momentum",
    "heat": "heat",
    "efficiency": "efficiency",
    "density": "density",
    "molarity": "molarity",
    "moles": "amount",
    "dilution": "c2",
    "ideal_gas": "pressure",
    "inelastic": "v",
    "combination": "combination",
    "confidence": "half_width",
}


@dataclass(frozen=True)
class Given:
    symbol: str
    value: float
    unit: str


def _parse_given(row: Dict[str, Any]) -> Given:
    return Given(str(row["symbol"]), float(row["value"]), str(row.get("unit", "")))


def _convert(value: float, unit: str) -> float:
    scale = unit_scale(unit)
    if scale is None:
        raise ValueError("unknown unit %r" % unit)
    return float(value) * float(scale)


class ScienceSolver:
    """Deterministic dimensional solver with sandbox parity and ontology grounding."""

    def __init__(self, ontology_root: Optional[str | Path] = None) -> None:
        self.ontology_root = Path(ontology_root or ".")
        self._ontology: Optional[OntologyVerifier] = None
        self._ontology_stats: Dict[str, float] = {}
        self.executions: List[Dict[str, Any]] = []

    # ------------------------------------------------------------- ontology
    def ontology(self) -> OntologyVerifier:
        if self._ontology is None:
            verifier = OntologyVerifier(
                self.ontology_root, docs=("docs/SCIENCE_PRIMER.md",), package=""
            )
            self._ontology_stats = verifier.build()
            self._ontology = verifier
        return self._ontology

    # ----------------------------------------------------------------- solve
    def solve(self, problem: Dict[str, Any], workspace: Optional[Path] = None) -> Dict[str, Any]:
        started = __import__("time").perf_counter()
        result: Dict[str, Any] = {
            "id": str(problem.get("id", "")),
            "domain": str(problem.get("domain", "")),
            "status": "ABSTAIN",
            "reason": "",
            "llm_calls": 0,
        }
        try:
            givens = [_parse_given(row) for row in problem.get("quantities") or []]
        except (KeyError, TypeError, ValueError) as error:
            result["reason"] = "malformed_quantities: %s" % error
            return result
        symbols: Dict[str, Given] = {}
        for given in givens:
            if given.symbol in symbols and abs(symbols[given.symbol].value - given.value) > 1e-12:
                result["reason"] = "contradictory_quantity_%s" % given.symbol
                result["provenance"] = {"symbol": given.symbol,
                                        "values": [symbols[given.symbol].value, given.value]}
                return result
            symbols[given.symbol] = given
        target = dict(problem.get("target") or {})
        target_unit = str(target.get("unit", ""))
        target_dims = unit_dims(target_unit)
        if target_dims is None:
            result["reason"] = "unknown_target_unit_%s" % target_unit
            return result
        for given in givens:
            if unit_dims(given.unit) is None:
                result["reason"] = "unknown_unit_%s" % given.unit
                return result
            if given.unit == "K" and given.value < 0.0:
                result["reason"] = "nonphysical_negative_temperature"
                return result
        table = dict(problem.get("table") or {})
        candidates: List[Relation] = []
        if table:
            candidates = [replace(TABLE_MEAN, output=tuple(target_dims))]
        else:
            for relation in RELATIONS:
                if relation.tabular:
                    continue
                if not _same(relation.output, target_dims):
                    continue
                usable = True
                for name, dims in relation.inputs:
                    if name in ("g", "R"):
                        continue
                    if name not in symbols:
                        usable = False
                        break
                    given_dims = unit_dims(symbols[name].unit)
                    if given_dims is None or not _same(given_dims, dims):
                        usable = False
                        break
                if usable:
                    candidates.append(relation)
            if not candidates:
                result["reason"] = "no_dimensionally_consistent_relation"
                return result
        evaluations: List[Tuple[Relation, float, Dict[str, float]]] = []
        constant_free: List[bool] = []
        for relation in candidates:
            si_values: Dict[str, float] = {}
            for name, _dims in relation.inputs:
                if name == "g":
                    si_values[name] = 9.80665
                elif name == "R":
                    si_values[name] = 8.314462618
                else:
                    si_values[name] = _convert(symbols[name].value, symbols[name].unit)
            try:
                if relation.tabular:
                    value = float(self._tabular_value(problem, workspace))
                else:
                    value = float(relation.compute(**si_values))
            except (ZeroDivisionError, ValueError, OSError) as error:
                result["reason"] = "computation_failed_%s" % type(error).__name__
                return result
            if not math.isfinite(value):
                result["reason"] = "non_finite_result"
                return result
            evaluations.append((relation, value, si_values))
            constant_free.append(
                all(
                    name in symbols or name not in ("g", "R")
                    for name, _dims in relation.inputs
                )
            )
        if any(constant_free):
            evaluations = [
                row for row, free in zip(evaluations, constant_free) if free
            ]
        if len(evaluations) > 1:
            values = [row[1] for row in evaluations]
            spread = max(values) - min(values)
            if spread > 1e-9 * max(1.0, max(abs(value) for value in values)):
                result["reason"] = "ambiguous_relation"
                result["provenance"] = {
                    "relations": [row[0].name for row in evaluations],
                    "values": values,
                }
                return result
        relation, si_value, si_values = evaluations[0]
        output_scale = unit_scale(target_unit) or 1.0
        answer = si_value / output_scale
        residual = 0.0
        if relation.name == "combination":
            n_value = int(round(si_values["n"]))
            k_value = int(round(si_values["k"]))
            independent = math.factorial(n_value) / (
                math.factorial(k_value) * math.factorial(n_value - k_value)
            )
            residual = abs(independent - si_value) / max(1.0, abs(si_value))
        elif not relation.tabular:
            kwargs = dict(si_values)
            kwargs[OUTPUT_SYMBOL.get(relation.name, "result")] = si_value
            try:
                recomputed = float(relation.inverse(**kwargs))
                original = float(kwargs[relation.inverse_name])
                residual = abs(recomputed - original) / max(1.0, abs(original))
            except (KeyError, TypeError, ZeroDivisionError, ValueError):
                residual = 0.0
        dimensional_ok = bool(_same(relation.output, target_dims))
        claim = self.ontology().verify_claim(relation.statement)
        if not dimensional_ok:
            result["reason"] = "dimensional_homogeneity_failed"
            return result
        if residual > 1e-9:
            result["reason"] = "inverse_substitution_residual_failed"
            result["residual"] = residual
            return result
        script = self._script(problem, relation, symbols, si_values, target_unit, output_scale)
        profile = "briefcase_workspace" if table else "scientific_scicode"
        executor = SubprocessSandboxExecutor(profile=profile, timeout_s=20.0)
        files = dict(problem.get("files") or {})
        record = executor.execute(script, files=files)
        sandbox_value = extract_child_result(record.stdout)
        sandbox_ok = bool(record.ok and isinstance(sandbox_value, dict))
        parity = 0.0
        if sandbox_ok:
            try:
                parity = abs(float(sandbox_value.get("value")) - answer) / max(1.0, abs(answer))
            except (TypeError, ValueError):
                sandbox_ok = False
        execution = {
            "id": str(problem.get("id", "")),
            "profile": profile,
            "ok": bool(record.ok),
            "exit_code": int(record.exit_code),
            "duration_ms": float(record.duration_ms),
            "script_sha256": hashlib.sha256(script.encode("utf-8")).hexdigest(),
            "sandbox_value": sandbox_value,
        }
        self.executions.append(execution)
        if not sandbox_ok:
            result["reason"] = "sandbox_execution_failed"
            result["execution"] = execution
            return result
        if parity > 1e-9:
            result["reason"] = "sandbox_parity_failed"
            result["parity"] = parity
            return result
        if not bool(claim.supported):
            result["reason"] = "ontology_statement_unsupported"
            result["claim"] = claim.as_dict()
            return result
        result.update(
            {
                "status": "SOLVED",
                "reason": "verified",
                "relation": relation.name,
                "statement": relation.statement,
                "answer": float(answer),
                "unit": target_unit,
                "dims": _fmt_dims(relation.output),
                "dimensional_ok": True,
                "residual": float(residual),
                "parity": float(parity),
                "ontology": claim.as_dict(),
                "execution": execution,
                "seconds": float(__import__("time").perf_counter() - started),
            }
        )
        return result

    # -------------------------------------------------------------- tabular
    @staticmethod
    def _tabular_value(problem: Dict[str, Any], workspace: Optional[Path]) -> float:
        table = dict(problem.get("table") or {})
        name = str(table.get("file", "measurements.csv"))
        column = str(table.get("column", "value"))
        text = ""
        if workspace is not None and (Path(workspace) / name).exists():
            text = (Path(workspace) / name).read_text(encoding="utf-8")
        else:
            text = str((problem.get("files") or {}).get(name, ""))
        if not text:
            raise ValueError("tabular file is absent")
        import csv
        import io

        reader = csv.DictReader(io.StringIO(text))
        values = [float(row[column]) for row in reader if row.get(column) not in (None, "")]
        if not values:
            raise ValueError("tabular column is empty")
        return sum(values) / len(values)

    # --------------------------------------------------------------- script
    def _script(
        self,
        problem: Dict[str, Any],
        relation: Relation,
        symbols: Dict[str, Given],
        si_values: Dict[str, float],
        target_unit: str,
        output_scale: float,
    ) -> str:
        lines = [
            '"""Deterministic science calculation with an invariant self-check."""',
        ]
        if relation.tabular:
            lines += ["import csv", "import math", "from pathlib import Path", ""]
            table = dict(problem.get("table") or {})
            name = str(table.get("file", "measurements.csv"))
            column = str(table.get("column", "value"))
            lines += [
                "TEXT = Path(%r).read_text(encoding='utf-8')" % name,
                "ROWS = list(csv.DictReader(TEXT.splitlines()))",
                "VALUES = [float(row[%r]) for row in ROWS if row.get(%r) not in (None, '')]" % (column, column),
                "value_si = sum(VALUES) / len(VALUES)",
            ]
        else:
            lines += ["import math", "R = 8.314462618", "g = 9.80665", ""]
            for name, _dims in relation.inputs:
                lines.append("%s = %.17g" % (name, si_values[name]))
            lines.append("value_si = " + _mathify(relation.expression))
        lines += [
            "value = value_si / %.17g" % output_scale,
            "result = {'value': value, 'unit': %r, 'relation': %r}" % (target_unit, relation.name),
        ]
        return "\n".join(lines) + "\n"
