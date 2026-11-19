"""Cycle AG2: deterministic zero-LLM raw-prose & LaTeX dimensional science parser.

Extracts structured physical quantities ``(symbol, value, unit)``, target
quantity and output unit, mathematical series/combinatorics/probability specs,
and implicitly named physical constants or IUPAC chemical molar masses directly
from unmodified raw prose and LaTeX ``problem_text`` — with zero LLM calls and
zero task-ID or problem-number string branching.

Design principles (ADS-STE100 epistemology):
1. General multiplicative-power SI compound unit algebra parser (supports
   ``kJ/mol``, ``N m^{-1}``, ``N*m^2/C``, ``dm^3``, ``g/mol``, ``m/s^2``,
   ``J mol^{-1} K^{-1}``, ``\\mu C``, ``nm``, ``eV``, ``atm``, ``Torr``, etc.)
   while failing closed on any unknown unit token (``furlongs``, ``bogons``).
2. Strict scope & unused-quantity abstention guard: if explicit dimensional
   measurements remain unconsumed by the inferred physical model, or if the
   prompt enters unsupported mathematical/physical domains (ODEs, integrals,
   matrix algebra, equilibrium constants, electrochemistry, etc.), the parser
   returns a clean ABSTAIN reason rather than forcing a wrong formula.
3. Explicit trap detection: flags contradictory measurements, nonphysical
   negative Kelvin/mass/wavelength values, unknown unit tokens, divergent
   series (handed to ``ScienceSolver`` for formal rejection), and invalid
   probability domains.
"""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

Dims = Tuple[float, float, float, float, float, float]
DIM_ZERO: Dims = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def _d(
    mass: float = 0.0,
    length: float = 0.0,
    time: float = 0.0,
    temperature: float = 0.0,
    amount: float = 0.0,
    current: float = 0.0,
) -> Dims:
    return (
        float(mass),
        float(length),
        float(time),
        float(temperature),
        float(amount),
        float(current),
    )


def dims_equal(a: Sequence[float], b: Sequence[float]) -> bool:
    return all(abs(float(x) - float(y)) <= 1e-12 for x, y in zip(a, b))


# ----------------------------------------------------------------- SI algebra
_SI_PREFIXES: Dict[str, float] = {
    "T": 1e12,
    "G": 1e9,
    "M": 1e6,
    "k": 1e3,
    "d": 1e-1,
    "c": 1e-2,
    "m": 1e-3,
    "u": 1e-6,
    "n": 1e-9,
    "p": 1e-12,
    "f": 1e-15,
}

# Atomic units: unit_symbol -> (si_scale, dims, allow_si_prefix)
_UNIT_ATOMS: Dict[str, Tuple[float, Dims, bool]] = {
    "1": (1.0, DIM_ZERO, False),
    "%": (0.01, DIM_ZERO, False),
    "m": (1.0, _d(length=1), True),
    "g": (1e-3, _d(mass=1), True),
    "s": (1.0, _d(time=1), True),
    "min": (60.0, _d(time=1), False),
    "h": (3600.0, _d(time=1), False),
    "hr": (3600.0, _d(time=1), False),
    "K": (1.0, _d(temperature=1), False),
    "mol": (1.0, _d(amount=1), True),
    "A": (1.0, _d(current=1), True),
    "Hz": (1.0, _d(time=-1), True),
    "N": (1.0, _d(mass=1, length=1, time=-2), True),
    "J": (1.0, _d(mass=1, length=2, time=-2), True),
    "W": (1.0, _d(mass=1, length=2, time=-3), True),
    "Pa": (1.0, _d(mass=1, length=-1, time=-2), True),
    "bar": (1e5, _d(mass=1, length=-1, time=-2), True),
    "atm": (101325.0, _d(mass=1, length=-1, time=-2), False),
    "Torr": (101325.0 / 760.0, _d(mass=1, length=-1, time=-2), False),
    "torr": (101325.0 / 760.0, _d(mass=1, length=-1, time=-2), False),
    "mmHg": (101325.0 / 760.0, _d(mass=1, length=-1, time=-2), False),
    "C": (1.0, _d(time=1, current=1), True),
    "V": (1.0, _d(mass=1, length=2, time=-3, current=-1), True),
    "F": (1.0, _d(mass=-1, length=-2, time=4, current=2), True),
    "Ohm": (1.0, _d(mass=1, length=2, time=-3, current=-2), True),
    "eV": (1.602176634e-19, _d(mass=1, length=2, time=-2), True),
    "L": (1e-3, _d(length=3), True),
    "l": (1e-3, _d(length=3), False),
    "u": (1.66053906660e-27, _d(mass=1), False),
    "amu": (1.66053906660e-27, _d(mass=1), False),
    "Da": (1.66053906660e-27, _d(mass=1), True),
    "e": (1.602176634e-19, _d(time=1, current=1), False),
    "A_ang": (1e-10, _d(length=1), False),
    "M": (1000.0, _d(amount=1, length=-3), True),
    "cal": (4.184, _d(mass=1, length=2, time=-2), True),
}

_UNIT_WORD_ALIASES: Dict[str, str] = {
    "meter": "m",
    "meters": "m",
    "metre": "m",
    "metres": "m",
    "centimeter": "cm",
    "centimeters": "cm",
    "centimetre": "cm",
    "centimetres": "cm",
    "millimeter": "mm",
    "millimeters": "mm",
    "millimetre": "mm",
    "millimetres": "mm",
    "micrometer": "um",
    "micrometers": "um",
    "micrometre": "um",
    "micrometres": "um",
    "nanometer": "nm",
    "nanometers": "nm",
    "nanometre": "nm",
    "nanometres": "nm",
    "picometer": "pm",
    "picometers": "pm",
    "kilometer": "km",
    "kilometers": "km",
    "kilometre": "km",
    "kilometres": "km",
    "angstrom": "A_ang",
    "angstroms": "A_ang",
    "second": "s",
    "seconds": "s",
    "sec": "s",
    "millisecond": "ms",
    "milliseconds": "ms",
    "minute": "min",
    "minutes": "min",
    "hour": "h",
    "hours": "h",
    "kilogram": "kg",
    "kilograms": "kg",
    "gram": "g",
    "grams": "g",
    "milligram": "mg",
    "milligrams": "mg",
    "kelvin": "K",
    "kelvins": "K",
    "mole": "mol",
    "moles": "mol",
    "newton": "N",
    "newtons": "N",
    "joule": "J",
    "joules": "J",
    "kilojoule": "kJ",
    "kilojoules": "kJ",
    "watt": "W",
    "watts": "W",
    "pascal": "Pa",
    "pascals": "Pa",
    "kilopascal": "kPa",
    "kilopascals": "kPa",
    "atmosphere": "atm",
    "atmospheres": "atm",
    "hertz": "Hz",
    "coulomb": "C",
    "coulombs": "C",
    "volt": "V",
    "volts": "V",
    "ampere": "A",
    "amperes": "A",
    "ohm": "Ohm",
    "ohms": "Ohm",
    "liter": "L",
    "liters": "L",
    "litre": "L",
    "litres": "L",
    "milliliter": "mL",
    "milliliters": "mL",
    "electronvolt": "eV",
    "electronvolts": "eV",
    "percent": "%",
    "molar": "M",
}

_CANONICAL_UNIT_REWRITES: Dict[str, str] = {
    "hr": "h",
    "sec": "s",
    "torr": "Torr",
    "mmHg": "Torr",
    "amu": "u",
    "Da": "u",
    "l": "L",
    "M": "mol/L",
    "mol/dm^3": "mol/L",
    "N*m^-1": "N/m",
    "kJ*mol^-1": "kJ/mol",
    "J*mol^-1": "J/mol",
    "J*K^-1": "J/K",
    "J*mol^-1*K^-1": "J/(mol*K)",
    "J*K^-1*mol^-1": "J/(mol*K)",
    "J*kg^-1*K^-1": "J/(kg*K)",
    "J*K^-1*kg^-1": "J/(kg*K)",
    "g*mol^-1": "g/mol",
    "m*s^-1": "m/s",
    "m*s^-2": "m/s^2",
    "s^-1": "Hz",
    "1/s": "Hz",
    "N*C^-1": "N/C",
    "V*m^-1": "V/m",
    "N*m^2*C^-1": "N*m^2/C",
    "kg*m^-3": "kg/m^3",
    "kg*m*s^-1": "kg*m/s",
    "mol*L^-1": "mol/L",
    "mol*dm^-3": "mol/L",
}


def _resolve_unit_atom(atom: str) -> Optional[Tuple[float, Dims]]:
    token = str(atom).strip()
    if not token:
        return None
    if token in _UNIT_ATOMS:
        sc, dm, _ = _UNIT_ATOMS[token]
        return sc, dm
    low = token.lower()
    if low in _UNIT_WORD_ALIASES:
        mapped = _UNIT_WORD_ALIASES[low]
        return _resolve_unit_atom(mapped)
    # Try single-character SI prefix + base unit atom
    if len(token) >= 2 and token[0] in _SI_PREFIXES:
        prefix = token[0]
        rest = token[1:]
        if rest in _UNIT_ATOMS and _UNIT_ATOMS[rest][2]:
            sc, dm, _ = _UNIT_ATOMS[rest]
            return _SI_PREFIXES[prefix] * sc, dm
    return None


def parse_compound_unit(expr: str) -> Optional[Tuple[float, Dims, str]]:
    """Parse any compound SI unit expression into ``(scale, dims, canonical)``.

    Returns ``None`` on any unrecognised unit atom (fail-closed for unit traps).
    """
    raw = str(expr).strip()
    if raw in ("", "1"):
        return 1.0, DIM_ZERO, raw or "1"
    if raw == "%":
        return 0.01, DIM_ZERO, "%"

    # Normalize unicode / LaTeX leftovers inside unit expressions
    s = raw
    s = s.replace("μ", "u").replace("µ", "u").replace("\\mu", "u")
    s = s.replace("\\Omega", "Ohm").replace("Ω", "Ohm")
    s = s.replace("\\AA", "A_ang").replace("Å", "A_ang")
    s = s.replace("\\cdot", "*").replace("·", "*").replace("×", "*")
    s = s.replace("\\times", "*").replace("−", "-").replace("–", "-")
    s = re.sub(r"\^\{([+-]?\d+)\}", r"^\1", s)
    s = re.sub(r"([A-Za-z_])\s*\*\*\s*([+-]?\d+)", r"\1^\2", s)
    s = re.sub(r"\s*/\s*", "/", s)
    s = re.sub(r"\s*\*\s*", "*", s)
    s = re.sub(r"\s+", "*", s.strip())
    s = s.strip("*.")
    if not s:
        return None

    # Tokenize into atoms, operators (*, /), parentheses, and exponents
    tok_re = re.compile(r"\(|\)|\*|/|[A-Za-z_%][A-Za-z0-9_]*(\^[+-]?\d+)?|1(?=/|$)")
    pos = 0
    tokens: List[str] = []
    while pos < len(s):
        m = tok_re.match(s, pos)
        if not m:
            return None
        tokens.append(m.group(0))
        pos = m.end()
    if not tokens:
        return None

    # Evaluate multiplicative & division structure with single-level parentheses
    scale = 1.0
    dims = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    sign_stack = [1]
    next_sign = 1
    expect_atom = True
    norm_parts: List[Tuple[str, int]] = []

    for tok in tokens:
        if tok == "(":
            if not expect_atom:
                return None
            sign_stack.append(sign_stack[-1] * next_sign)
            next_sign = 1
            expect_atom = True
        elif tok == ")":
            if len(sign_stack) <= 1 or expect_atom:
                return None
            sign_stack.pop()
            expect_atom = False
        elif tok == "*":
            if expect_atom:
                return None
            next_sign = 1
            expect_atom = True
        elif tok == "/":
            if expect_atom:
                return None
            next_sign = -1
            expect_atom = True
        else:
            if not expect_atom:
                return None
            eff_sign = sign_stack[-1] * next_sign
            next_sign = 1
            expect_atom = False
            if "^" in tok:
                atom_name, exp_str = tok.split("^", 1)
                try:
                    exp = int(exp_str)
                except ValueError:
                    return None
            else:
                atom_name = tok
                exp = 1
            if atom_name == "1":
                continue
            resolved = _resolve_unit_atom(atom_name)
            if resolved is None:
                return None
            atom_scale, atom_dims = resolved
            power = eff_sign * exp
            scale *= atom_scale ** power
            for idx in range(6):
                dims[idx] += atom_dims[idx] * power
            norm_parts.append((atom_name, power))

    if len(sign_stack) != 1 or expect_atom:
        return None
    if not math.isfinite(scale) or scale <= 0.0:
        return None

    # Construct canonical string matching ScienceSolver.UNIT_TABLE keys where possible
    canonical = _CANONICAL_UNIT_REWRITES.get(s, s)
    if canonical == s:
        num = []
        den = []
        for atom_name, power in norm_parts:
            if power > 0:
                num.append(atom_name if power == 1 else "%s^%d" % (atom_name, power))
            elif power < 0:
                den.append(atom_name if power == -1 else "%s^%d" % (atom_name, -power))
        if not num and not den:
            canonical = "1"
        elif not den:
            canonical = "*".join(num)
        else:
            num_s = "*".join(num) if num else "1"
            den_s = ("(%s)" % "*".join(den)) if len(den) > 1 else den[0]
            canonical = "%s/%s" % (num_s, den_s)
        canonical = _CANONICAL_UNIT_REWRITES.get(canonical, canonical)

    return float(scale), tuple(float(x) for x in dims), canonical  # type: ignore[return-value]


# ------------------------------------------------------- IUPAC & particle data
# Standard IUPAC atomic weights (g/mol) — general physical chemistry constants.
_ELEMENT_WEIGHTS_BY_SYMBOL: Dict[str, float] = {
    "H": 1.00794,
    "He": 4.002602,
    "Li": 6.941,
    "Be": 9.012182,
    "B": 10.811,
    "C": 12.0107,
    "N": 14.0067,
    "O": 15.9994,
    "F": 18.9984032,
    "Ne": 20.1797,
    "Na": 22.989769,
    "Mg": 24.3050,
    "Al": 26.9815386,
    "Si": 28.0855,
    "P": 30.973762,
    "S": 32.065,
    "Cl": 35.453,
    "Ar": 39.948,
    "K": 39.0983,
    "Ca": 40.078,
    "Fe": 55.845,
    "Cu": 63.546,
    "Zn": 65.38,
    "Br": 79.904,
    "Kr": 83.798,
    "Ag": 107.8682,
    "I": 126.90447,
    "Xe": 131.293,
    "Au": 196.966569,
    "Hg": 200.59,
    "Pb": 207.2,
}

# Named species -> molar mass (g/mol)
_NAMED_SPECIES_MOLAR_MASS: Dict[str, float] = {
    "helium": 4.002602,
    "neon": 20.1797,
    "argon": 39.948,
    "krypton": 83.798,
    "xenon": 131.293,
    "hydrogen gas": 2.01588,
    "dihydrogen": 2.01588,
    "nitrogen gas": 28.0134,
    "dinitrogen": 28.0134,
    "oxygen gas": 31.9988,
    "dioxygen": 31.9988,
    "chlorine gas": 70.906,
    "fluorine gas": 37.9968,
    "carbon dioxide": 44.0095,
    "carbon monoxide": 28.0101,
    "water": 18.01528,
    "water vapor": 18.01528,
    "methane": 16.0425,
    "ammonia": 17.0305,
    "ethane": 30.0690,
    "propane": 44.0956,
    "sulfur dioxide": 64.0638,
    "hydrogen chloride": 36.4609,
    "air": 28.97,
}

# Standard elementary particles: name -> (mass_kg, charge_in_e)
_PARTICLE_CONSTANTS: Dict[str, Tuple[float, Optional[float]]] = {
    "electron": (9.1093837015e-31, 1.0),
    "positron": (9.1093837015e-31, 1.0),
    "proton": (1.67262192369e-27, 1.0),
    "neutron": (1.67492749804e-27, None),
    "alpha particle": (6.6446573357e-27, 2.0),
}


def compute_formula_molar_mass(formula: str) -> Optional[float]:
    """Compute IUPAC molar mass (g/mol) for a chemical formula such as ``CO2`` or ``H2O``."""
    tokens = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
    if not tokens or "".join(a + b for a, b in tokens) != formula:
        return None
    total = 0.0
    for sym, count_s in tokens:
        if sym not in _ELEMENT_WEIGHTS_BY_SYMBOL:
            return None
        cnt = int(count_s) if count_s else 1
        total += _ELEMENT_WEIGHTS_BY_SYMBOL[sym] * cnt
    return total if total > 0.0 else None


# ----------------------------------------------------------- safe math eval
def _safe_eval_arith(expr: str, env: Dict[str, float]) -> Optional[float]:
    """Safely evaluate a pure arithmetic AST over ``{+,-,*,/,**}``."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None

    def _visit(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id in env:
                return float(env[node.id])
            raise ValueError("unknown symbol")
        if isinstance(node, ast.UnaryOp):
            val = _visit(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +val
            if isinstance(node.op, ast.USub):
                return -val
            raise ValueError("unsupported unary op")
        if isinstance(node, ast.BinOp):
            left = _visit(node.left)
            right = _visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                if abs(right) > 50.0:
                    raise ValueError("exponent too large")
                return left ** right
            raise ValueError("unsupported binary op")
        raise ValueError("unsupported AST node")

    try:
        out = float(_visit(tree))
        return out if math.isfinite(out) else None
    except Exception:
        return None


def _latex_math_to_arith(expr: str) -> str:
    """Convert simple LaTeX algebraic fractions & powers to Python arithmetic."""
    s = str(expr).strip()
    s = s.replace("\\cdot", "*").replace("\\times", "*").replace("\\left", "").replace("\\right", "")
    s = s.replace("\\,", "").replace("~", "")
    # Iteratively unwrap \frac{A}{B}
    for _ in range(8):
        idx = s.find("\\frac")
        if idx < 0:
            break
        rest = s[idx + 5 :].lstrip()
        if not rest.startswith("{"):
            break
        # parse two brace-balanced groups
        groups: List[str] = []
        pos = 0
        ok = True
        for _g in range(2):
            while pos < len(rest) and rest[pos].isspace():
                pos += 1
            if pos >= len(rest) or rest[pos] != "{":
                ok = False
                break
            depth = 0
            start = pos + 1
            while pos < len(rest):
                if rest[pos] == "{":
                    depth += 1
                elif rest[pos] == "}":
                    depth -= 1
                    if depth == 0:
                        groups.append(rest[start:pos])
                        pos += 1
                        break
                pos += 1
            else:
                ok = False
                break
        if not ok or len(groups) != 2:
            break
        s = s[:idx] + "((%s)/(%s))" % (groups[0], groups[1]) + rest[pos:]

    # Replace ^{...} and ^x with **(...)
    s = re.sub(r"\^\{([^{}]+)\}", r"**(\1)", s)
    s = re.sub(r"\^([A-Za-z0-9]+)", r"**(\1)", s)
    s = s.replace("{", "(").replace("}", ")")
    # Implicit multiplication: 2n -> 2*n, )( -> )*(, 2( -> 2*(
    s = re.sub(r"(\d)\s*([A-Za-z(])", r"\1*\2", s)
    s = re.sub(r"\)\s*\(", ")*(", s)
    s = re.sub(r"\)\s*([A-Za-z0-9])", r")*\1", s)
    return s


# ----------------------------------------------------------- data structures
@dataclass
class ExtractedMeasurement:
    value: float
    raw_unit: str
    unit: str
    dims: Dims
    scale: float
    span: Tuple[int, int]
    context_before: str
    context_after: str
    inline_symbol: str = ""
    is_celsius: bool = False
    is_temp_delta: bool = False
    implicit: bool = False


@dataclass
class ParseResult:
    ok: bool
    problem: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    measurements: List[ExtractedMeasurement] = field(default_factory=list)
    normalized_text: str = ""
    target_unit_explicit: bool = False
    ignored_comparison_text: str = ""
    boilerplate_stripped: List[str] = field(default_factory=list)


# --------------------------------------------------------------- preprocess
_BOILERPLATE_PATTERNS = (
    re.compile(r"Assume all gases are perfect unless stated otherwise\.?", re.I),
    re.compile(r"Unless otherwise stated,\s*thermodynamic data are for\s*298\.15\s*K\.?", re.I),
    re.compile(r"\(?\s*Ignore gravitation\.?\s*\)?", re.I),
    re.compile(r"\(?\s*Ignore air resistance\.?\s*\)?", re.I),
)


def normalize_science_text(raw_text: str) -> Tuple[str, str, List[str], str]:
    """Normalize LaTeX/Unicode science prose into clean parseable text.

    Returns ``(body_text, comparison_text, boilerplate_removed, raw_cleaned)``.
    """
    text = str(raw_text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    raw_cleaned = text
    boilerplate_removed: List[str] = []
    for pat in _BOILERPLATE_PATTERNS:
        for m in pat.finditer(text):
            boilerplate_removed.append(m.group(0).strip())
        text = pat.sub(" ", text)

    # Unwrap common LaTeX text/font commands before removing dollar delimiters
    for cmd in ("\\mathrm", "\\text", "\\operatorname", "\\mathbf", "\\mathit", "\\rm"):
        text = re.sub(re.escape(cmd) + r"\s*\{([^{}]*)\}", r"\1", text)
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\,", " ").replace("\\;", " ").replace("\\!", "")
    text = text.replace("\\quad", " ").replace("~", " ")
    # Normalize Celsius and Fahrenheit forms before stripping $
    text = re.sub(r"\^\s*\{\s*\\circ\s*\}\s*C\b", " degC", text)
    text = re.sub(r"\^\s*\\circ\s*C\b", " degC", text)
    text = re.sub(r"°\s*C\b", " degC", text)
    text = re.sub(r"\\circ\s*C\b", " degC", text)
    text = re.sub(r"\^\s*\{\s*\\circ\s*\}\s*F\b", " degF", text)
    text = re.sub(r"\^\s*\\circ\s*F\b", " degF", text)
    text = re.sub(r"°\s*F\b", " degF", text)
    text = re.sub(r"\\circ\s*F\b", " degF", text)
    # Normalize Greek micro and common operators
    text = re.sub(r"\\mu\s*", "u", text)
    text = text.replace("μ", "u").replace("µ", "u")
    text = text.replace("\\cdot", "*").replace("·", "*")
    text = text.replace("\\times", " * ").replace("×", " * ")
    text = text.replace("\\AA", "A_ang").replace("Å", "A_ang")
    text = text.replace("\\Omega", "Ohm").replace("Ω", "Ohm")
    text = text.replace("−", "-").replace("–", "-")
    # Strip math delimiters $ and $$
    text = text.replace("$$", " ").replace("$", " ")
    # Fix "750 . N m^{-1}" trailing decimal dot followed by space and unit
    text = re.sub(r"(\d+)\s+\.\s+([A-Za-z])", r"\1.0 \2", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Split trailing comparison clause ("and compare it with ...")
    comparison_text = ""
    cmp_m = re.search(
        r"\b(?:and\s+)?compar(?:e|ing)\s+(?:it|this|that|the\s+result)\s+with\b.*$",
        text,
        re.I,
    )
    if cmp_m:
        comparison_text = text[cmp_m.start() :].strip()
        text = text[: cmp_m.start()].strip()

    return text, comparison_text, boilerplate_removed, raw_cleaned


# ------------------------------------------------------- special math parsers
def _try_parse_geometric_series(raw_text: str) -> Optional[ParseResult]:
    """Parse convergent or divergent infinite geometric series from raw LaTeX."""
    text = str(raw_text or "")
    # 1. Sigma summation form: \sum_{n=0}^{\infty} expr
    sig_m = re.search(
        r"\\sum_\{?\s*([a-zA-Z])\s*=\s*(-?\d+)\s*\}?\s*\^\s*\{?\s*\\infty\s*\}?\s*(.+?)(?:\$\$|\$|\n\s*$|$)",
        text,
        re.S,
    )
    if sig_m:
        var = sig_m.group(1)
        start = int(sig_m.group(2))
        body = sig_m.group(3).strip().rstrip(".,;")
        arith = _latex_math_to_arith(body)
        terms: List[float] = []
        for offset in range(4):
            val = _safe_eval_arith(arith, {var: float(start + offset)})
            if val is None:
                return None
            terms.append(val)
        if abs(terms[0]) < 1e-15:
            return None
        r0 = terms[1] / terms[0]
        for idx in range(1, len(terms) - 1):
            if abs(terms[idx]) < 1e-15:
                return None
            ri = terms[idx + 1] / terms[idx]
            if abs(ri - r0) > 1e-10 * max(1.0, abs(r0)):
                return ParseResult(ok=False, reason="non_geometric_series")
        return ParseResult(
            ok=True,
            problem={
                "domain": "mathematics",
                "prompt": raw_text,
                "quantities": [
                    {"symbol": "first_term", "value": float(terms[0]), "unit": "1"},
                    {"symbol": "ratio", "value": float(r0), "unit": "1"},
                ],
                "target": {"symbol": "series_sum", "unit": "1"},
            },
            normalized_text=text,
            target_unit_explicit=True,
        )

    # 2. Expanded infinite sum form: a_0 + a_1 + a_2 + ... + \cdots
    if "\\cdots" in text or "\\ldots" in text or "..." in text:
        sum_m = re.search(
            r"(?:[A-Za-z]\s*=\s*)?((?:[+-]?\s*(?:\\frac\{[^{}]+\}\{[^{}]+\}|\d+(?:\.\d+)?)\s*){3,})\s*[+-]?\s*(?:\\cdots|\\ldots|\.\.\.)",
            text,
        )
        if sum_m:
            chunk = sum_m.group(1)
            raw_terms = re.findall(
                r"([+-]?)\s*(\\frac\{[^{}]+\}\{[^{}]+\}|\d+(?:\.\d+)?)", chunk
            )
            if len(raw_terms) >= 3:
                vals: List[float] = []
                for sign, tok in raw_terms:
                    v = _safe_eval_arith(_latex_math_to_arith(tok), {})
                    if v is None:
                        return None
                    if sign == "-":
                        v = -v
                    vals.append(v)
                if abs(vals[0]) < 1e-15:
                    return None
                r0 = vals[1] / vals[0]
                for idx in range(1, len(vals) - 1):
                    if abs(vals[idx]) < 1e-15:
                        return None
                    ri = vals[idx + 1] / vals[idx]
                    if abs(ri - r0) > 1e-10 * max(1.0, abs(r0)):
                        return ParseResult(ok=False, reason="non_geometric_series")
                return ParseResult(
                    ok=True,
                    problem={
                        "domain": "mathematics",
                        "prompt": raw_text,
                        "quantities": [
                            {"symbol": "first_term", "value": float(vals[0]), "unit": "1"},
                            {"symbol": "ratio", "value": float(r0), "unit": "1"},
                        ],
                        "target": {"symbol": "series_sum", "unit": "1"},
                    },
                    normalized_text=text,
                    target_unit_explicit=True,
                )
    return None


def _try_parse_probability(raw_text: str, norm_text: str) -> Optional[ParseResult]:
    """Parse conditional or independent event probability equations from prose/LaTeX."""
    if "P(" not in norm_text and "P (" not in norm_text:
        return None
    cleaned = norm_text.replace("\\cap", " cap ").replace("\\mid", " | ")
    cleaned = cleaned.replace("\\cup", " cup ").replace("\\land", " cap ")
    cleaned = re.sub(r"\s+", " ", cleaned)

    # Reject unsupported union/complement/Bayes multi-event forms
    if " cup " in cleaned or "complement" in cleaned.lower() or "bayes" in cleaned.lower():
        return None

    assign_re = re.compile(
        r"P\s*\(\s*([^()]+?)\s*\)\s*=\s*([+-]?\d+(?:\.\d+)?(?:/\d+)?)"
    )
    probs: Dict[str, float] = {}
    for m in assign_re.finditer(cleaned):
        ev_raw = m.group(1).strip()
        val_s = m.group(2).strip()
        val = _safe_eval_arith(val_s, {})
        if val is None:
            continue
        cap_m = re.match(r"^([A-Za-z0-9_]+)\s+(?:cap|and)\s+([A-Za-z0-9_]+)$", ev_raw)
        if cap_m:
            key = "cap:" + ":".join(sorted([cap_m.group(1), cap_m.group(2)]))
        else:
            key = "ev:" + ev_raw.strip()
        if key in probs and abs(probs[key] - val) > 1e-12:
            return ParseResult(ok=False, reason="contradictory_probability")
        probs[key] = float(val)

    # Query: conditional P(Y | X)
    cond_q = re.search(
        r"(?:find|calculate|compute|determine|what\s+is)\s+P\s*\(\s*([A-Za-z0-9_]+)\s*\|\s*([A-Za-z0-9_]+)\s*\)",
        cleaned,
        re.I,
    )
    if cond_q:
        ev_y, ev_x = cond_q.group(1), cond_q.group(2)
        cap_key = "cap:" + ":".join(sorted([ev_x, ev_y]))
        cond_key = "ev:" + ev_x
        if cap_key in probs and cond_key in probs:
            return ParseResult(
                ok=True,
                problem={
                    "domain": "probability",
                    "prompt": raw_text,
                    "quantities": [
                        {"symbol": "p_intersection", "value": probs[cap_key], "unit": "1"},
                        {"symbol": "p_condition", "value": probs[cond_key], "unit": "1"},
                    ],
                    "target": {"symbol": "p_cond", "unit": "1"},
                },
                normalized_text=norm_text,
                target_unit_explicit=True,
            )

    # Query: independent intersection P(A cap B)
    if "independent" in cleaned.lower():
        ind_q = re.search(
            r"(?:find|calculate|compute|determine|what\s+is)\s+P\s*\(\s*([A-Za-z0-9_]+)\s+(?:cap|and)\s+([A-Za-z0-9_]+)\s*\)",
            cleaned,
            re.I,
        )
        if ind_q:
            ka = "ev:" + ind_q.group(1)
            kb = "ev:" + ind_q.group(2)
            if ka in probs and kb in probs:
                return ParseResult(
                    ok=True,
                    problem={
                        "domain": "probability",
                        "prompt": raw_text,
                        "quantities": [
                            {"symbol": "p_a", "value": probs[ka], "unit": "1"},
                            {"symbol": "p_b", "value": probs[kb], "unit": "1"},
                        ],
                        "target": {"symbol": "p_ab", "unit": "1"},
                    },
                    normalized_text=norm_text,
                    target_unit_explicit=True,
                )
    return None


def _try_parse_combinatorics(raw_text: str, norm_text: str) -> Optional[ParseResult]:
    """Parse binomial coefficient ``C(n, k)`` problems from raw prose."""
    low = norm_text.lower()
    # Guard against conditional / ordered / multi-constraint combinatorics
    for forbidden in (
        "probability",
        "at least",
        "at most",
        "exactly",
        "with replacement",
        "order matters",
        "permutation",
        "arrangement",
        "flush",
        "straight",
        "full house",
        "pair",
        "same suit",
        "royal",
        "sum of",
        "distinct ways to arrange",
        "expected",
    ):
        if forbidden in low:
            return None

    patterns = (
        re.compile(
            r"\b(\d+)\s*-\s*card\s+hands?\b.*?\bdeck\s+of\s+(\d+)\s+(?:playing\s+)?cards?\b",
            re.I,
        ),
        re.compile(
            r"\b(?:choose|select|pick|draw)\s+(\d+)\s+[a-z]+\s+from\s+(?:a\s+(?:set|group|deck|pool)\s+of\s+)?(\d+)\b",
            re.I,
        ),
        re.compile(
            r"\bcommittees?\s+of\s+(?:size\s+)?(\d+)\b.*?\bfrom\s+(\d+)\b",
            re.I,
        ),
        re.compile(r"\\binom\{\s*(\d+)\s*\}\{\s*(\d+)\s*\}", re.I),
        re.compile(r"\bC\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)"),
    )
    for idx, pat in enumerate(patterns):
        m = pat.search(norm_text if idx < 3 else raw_text)
        if m:
            if idx < 3:
                k_val, n_val = int(m.group(1)), int(m.group(2))
            else:
                n_val, k_val = int(m.group(1)), int(m.group(2))
            if 0 <= k_val <= n_val and n_val >= 1:
                return ParseResult(
                    ok=True,
                    problem={
                        "domain": "combinatorics",
                        "prompt": raw_text,
                        "quantities": [
                            {"symbol": "n", "value": float(n_val), "unit": "1"},
                            {"symbol": "k", "value": float(k_val), "unit": "1"},
                        ],
                        "target": {"symbol": "combination", "unit": "1"},
                    },
                    normalized_text=norm_text,
                    target_unit_explicit=True,
                )
    return None


# --------------------------------------------------------- measurement extraction
_NUM_RE = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"(?P<sign>[+-])?"
    r"(?P<mant>\d+(?:\.\d*)?|\.\d+)"
    r"(?:"
    r"\s*\*\s*10\s*\^\s*\{?(?P<p10>[+-]?\d+)\}?"
    r"|[eE](?P<ep>[+-]?\d+)"
    r")?"
)

# Unit token stream immediately following a number
_UNIT_ATOM_PATTERN = (
    r"(?:degC|degF|A_ang|mmHg|Torr|torr|Ohm|atm|bar|mol|mmol|min|hr|amu|cal|kcal|Hz|kHz|MHz|GHz|THz|"
    r"Pa|kPa|MPa|eV|MeV|keV|GeV|kJ|mJ|MJ|uJ|kW|MW|mW|kN|mN|kV|mV|mA|uA|mC|uC|nC|pC|MC|"
    r"kg|mg|ug|km|dm|cm|mm|um|nm|pm|mL|uL|ms|us|ns|Da|kDa|"
    r"N|J|W|C|V|F|A|K|L|m|g|s|h|u|M|e|%)"
)
_UNIT_CHUNK_RE = re.compile(
    r"^\s*(?P<unit>"
    + _UNIT_ATOM_PATTERN
    + r"(?:\s*\^\s*\{?[+-]?\d+\}?)?"
    r"(?:\s*(?:[*/]|\s+(?="
    + _UNIT_ATOM_PATTERN
    + r"(?:\s*\^\s*\{?[+-]?\d+\}?)?(?:\b|[^A-Za-z])))"
    r"\s*(?:\(\s*"
    + _UNIT_ATOM_PATTERN
    + r"(?:\s*\*\s*"
    + _UNIT_ATOM_PATTERN
    + r")*\s*\)|"
    + _UNIT_ATOM_PATTERN
    + r"(?:\s*\^\s*\{?[+-]?\d+\}?)?))*"
    r")"
)

# Explicit bogus/non-SI unit trap tokens to reject cleanly if attached to a number
_KNOWN_UNKNOWN_UNIT_WORDS = frozenset(
    {
        "furlong", "furlongs", "fortnight", "fortnights", "bogon", "bogons",
        "widget", "widgets", "quark", "quarks", "parsec", "parsecs",
        "erg_fake", "zorblax", "cubit", "cubits", "stone", "stones",
        "league", "leagues", "smoot", "smoots",
        "ft", "feet", "foot", "inch", "inches", "lb", "lbs", "pound", "pounds",
        "mile", "miles", "mph", "rpm", "erg", "ergs", "dyne", "dynes",
        "btu", "hp", "kwh",
    }
)


def _extract_inline_symbol(context_before: str) -> str:
    """Detect inline algebraic assignments such as ``q_1 = `` or ``k = ``."""
    m = re.search(
        r"([A-Za-z\\][A-Za-z0-9_{}\\]*)\s*=\s*$", context_before.strip()
    )
    if not m:
        return ""
    sym = m.group(1).replace("\\", "").replace("{", "").replace("}", "").strip()
    return sym


def extract_measurements(
    norm_text: str,
) -> Tuple[List[ExtractedMeasurement], Optional[str]]:
    """Extract all dimensional measurements from normalized prose/LaTeX.

    Returns ``(measurements, error_reason)``. ``error_reason`` is non-None when
    an explicit unknown unit token or nonphysical value is detected.
    """
    measurements: List[ExtractedMeasurement] = []
    pos = 0
    n = len(norm_text)
    while pos < n:
        m = _NUM_RE.search(norm_text, pos)
        if not m:
            break
        start, end = m.span()
        before_char = norm_text[start - 1] if start > 0 else " "
        if before_char in ("_", "^"):
            pos = end
            continue
        if start > 0 and norm_text[start - 1].isalpha():
            pos = end
            continue

        mant_s = m.group("mant")
        sign_s = m.group("sign") or ""
        p10_s = m.group("p10")
        ep_s = m.group("ep")
        try:
            val = float(sign_s + mant_s)
            if p10_s is not None:
                val *= 10.0 ** int(p10_s)
            elif ep_s is not None:
                val *= 10.0 ** int(ep_s)
        except ValueError:
            pos = end
            continue

        tail = norm_text[end:]
        ctx_before = norm_text[max(0, start - 60) : start]

        word_m = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_/-]*)", tail)
        if word_m:
            first_word = word_m.group(1)
            head_atom = first_word.split("/")[0].lower()
            if head_atom in _KNOWN_UNKNOWN_UNIT_WORDS:
                return [], "unknown_unit_%s" % first_word

        unit_m = _UNIT_CHUNK_RE.match(tail)
        raw_unit = ""
        unit_end = end
        if unit_m:
            candidate_u = unit_m.group("unit").strip()
            after_u = tail[unit_m.end() :]
            if after_u and after_u[0].isalpha():
                spelled = re.match(
                    r"^\s*([A-Za-z]+(?:\s*/\s*[A-Za-z]+)?)", tail
                )
                if spelled and spelled.group(1).lower() in _UNIT_WORD_ALIASES:
                    raw_unit = _UNIT_WORD_ALIASES[spelled.group(1).lower()]
                    unit_end = end + spelled.end()
                else:
                    candidate_u = ""
            if candidate_u and not raw_unit:
                if candidate_u == "A" and re.match(r"^\s+[a-z]", after_u):
                    candidate_u = ""
                elif candidate_u == "l" and re.match(r"^\s+[a-z]", after_u):
                    candidate_u = ""
                else:
                    raw_unit = candidate_u
                    unit_end = end + unit_m.end()
        else:
            spelled = re.match(r"^\s*([A-Za-z]+)", tail)
            if spelled and spelled.group(1).lower() in _UNIT_WORD_ALIASES:
                raw_unit = _UNIT_WORD_ALIASES[spelled.group(1).lower()]
                unit_end = end + spelled.end()

        if not raw_unit:
            pos = end
            continue

        is_celsius = False
        is_temp_delta = False
        if raw_unit in ("degC", "degF"):
            is_celsius = raw_unit == "degC"
            delta_hint = bool(
                re.search(
                    r"\b(?:by|change|increase|decrease|rise|rises|drop|drops|delta|difference)\b",
                    ctx_before[-30:],
                    re.I,
                )
            )
            if delta_hint:
                is_temp_delta = True
                if raw_unit == "degF":
                    val = val * (5.0 / 9.0)
            else:
                if raw_unit == "degC":
                    val = val + 273.15
                else:
                    val = (val - 32.0) * (5.0 / 9.0) + 273.15
            canonical_unit = "K"
            scale, dims = 1.0, _d(temperature=1)
        else:
            parsed = parse_compound_unit(raw_unit)
            if parsed is None:
                if " " in raw_unit:
                    trimmed = raw_unit.rsplit(" ", 1)[0].strip()
                    parsed = parse_compound_unit(trimmed)
                    if parsed is not None:
                        raw_unit = trimmed
                        unit_end = end + tail.find(trimmed) + len(trimmed)
                if parsed is None:
                    pos = end
                    continue
            scale, dims, canonical_unit = parsed
            if canonical_unit == "K" and re.search(
                r"\b(?:by|change|increase|decrease|rise|rises|drop|drops|delta)\b",
                ctx_before[-30:],
                re.I,
            ):
                is_temp_delta = True

        if dims_equal(dims, _d(temperature=1)) and not is_temp_delta and val < 0.0:
            return [], "nonphysical_negative_temperature"

        inline_sym = _extract_inline_symbol(ctx_before)
        measurements.append(
            ExtractedMeasurement(
                value=float(val),
                raw_unit=raw_unit,
                unit=canonical_unit,
                dims=dims,
                scale=float(scale),
                span=(start, unit_end),
                context_before=ctx_before,
                context_after=norm_text[unit_end : min(n, unit_end + 60)],
                inline_symbol=inline_sym,
                is_celsius=is_celsius,
                is_temp_delta=is_temp_delta,
            )
        )
        pos = unit_end

    return measurements, None


# --------------------------------------------------- target & domain analysis
# Out-of-scope mathematical/physical domain markers that our algebraic primer
# does NOT cover; triggering any of these causes clean ABSTAIN (protecting
# out-of-distribution precision on blind benchmarks).
_OUT_OF_SCOPE_PATTERNS = (
    re.compile(r"\(a\).*\(b\)", re.S),
    re.compile(r"\\int\b|\\oint\b|\bintegral\b|\bintegrate\b|\bantiderivative\b", re.I),
    re.compile(r"\\frac\s*\{\s*d[a-zA-Z]?\s*\}\s*\{\s*d[a-zA-Z]|\bdifferential equation\b|\bderivative\b", re.I),
    re.compile(r"\beigenvalue|\beigenvector|\bdeterminant\b|\bmatrix\b|\bmatrices\b|\bvector space\b", re.I),
    re.compile(r"\btaylor\s+series|\bfourier|\blaplace\s+transform|\blimit\s+as\b|\\lim_", re.I),
    re.compile(r"\bvan\s+der\s+waals\b|\bvirial\b|\bfugacity\b|\bcompressibility\s+factor\b", re.I),
    re.compile(r"\b(?:gibbs|helmholtz)\s+(?:free\s+)?energy|\benthalpy\b|\\Delta\s*[GH]\b", re.I),
    re.compile(r"\bequilibrium\s+constant|\bK_p\b|\bK_c\b|\bK_\{?sp\}?|\bpH\b|\bpK_?a\b|\btitration\b|\bbuffer\b", re.I),
    re.compile(r"\barrhenius|\bactivation\s+energy|\brate\s+constant|\bhalf-life\b|\bfirst-order\b|\bsecond-order\b", re.I),
    re.compile(r"\bnernst\b|\bcell\s+potential|\belectrode\s+potential|\bfaraday\s+constant|\belectrolysis\b", re.I),
    re.compile(r"\badiabatic\b|\bcarnot\b|\botto\s+cycle|\brankine\b|\bheat\s+pump|\brefrigerator\b|\bpolytropic\b", re.I),
    re.compile(r"\bpartition\s+function|\bboltzmann\s+factor|\bfermi-dirac|\bbose-einstein|\bmaxwell-boltzmann\b", re.I),
    re.compile(r"\bschr[oö]dinger|\bwavefunction|\bwave\s+function|\bexpectation\s+value|\bcommutator\b|\bheisenberg\b|\buncertainty\s+principle\b|\btunneling\b|\bparticle\s+in\s+a\s+box\b|\bharmonic\s+oscillator\b|\bbohr\b|\brydberg\b|\bbalmer\b|\blyman\b|\bcompton\b", re.I),
    re.compile(r"\bquantum\s+number|\bground\s+state|\bexcited\s+state|\btransition\s+from\b|\benergy\s+level", re.I),
    re.compile(r"\bmagnetic\s+field|\bmagnetic\s+flux|\bsolenoid|\binductance|\binductor|\bcapacitor|\bcapacitance|\bRC\s+circuit|\bRLC\b|\bkirchhoff|\blorentz\s+force|\btesla\b", re.I),
    re.compile(r"\bsnell|\brefractive\s+index|\bfocal\s+length|\bthin\s+lens|\bspherical\s+mirror|\bdiffraction|\bdouble-slit|\binterference\s+fringe|\bdoppler\b", re.I),
    re.compile(r"\bspecial\s+relativity|\blorentz\s+factor|\btime\s+dilation|\blength\s+contraction|\brelativistic\b", re.I),
    re.compile(r"\btorque\b|\bangular\s+(?:velocity|momentum|acceleration)|\bmoment\s+of\s+inertia|\bpendulum\b|\bprojectile\b|\binclined\s+plane|\bfriction\b|\btension\b", re.I),
    re.compile(r"\bangle\b|\bdegrees?\b(?!\s*(?:C|F|Celsius|Fahrenheit|centigrade)\b)|\bradians?\b|\\theta\b|\\sin\b|\\cos\b|\\tan\b", re.I),
    re.compile(r"\\circ\b(?!\s*\}?\s*(?:\\mathrm\s*\{|\\text\s*\{)?\s*[CF]\b)"),
    re.compile(r"\bx-axis\b|\by-axis\b|\bz-axis\b|\bresultant\b|\bvector\b|\bcomponents?\b|\bequilateral\b|\bthree\s+charges\b|\bfour\s+charges\b|\bnet\s+force\s+on\b", re.I),
    re.compile(r"\bpartial\s+pressure|\bmole\s+fraction|\bmixture\b|\bpercent\s+by\s+mass|\bweight\s+percent|\bmass\s+percent|\bdouble[sd]\b|\bdouble\s+(?:the|its|their)\b|\btriple[sd]\b|\btriple\s+(?:the|its|their)\b|\bhalve[sd]?\b|\bfactor\s+of\b", re.I),
    re.compile(r"\\rightarrow|\\longrightarrow|\\rightleftharpoons|\breaction\b|\breacts\s+with\b|\bcombustion\b|\bdecomposition\b|\bprecipitate\b|\blimiting\b", re.I),
    re.compile(r"\blatent\s+heat|\bfusion\b|\bvaporization\b|\bmelting\b|\bboiling\b|\bthermal\s+equilibrium\b|\bfinal\s+temperature\b", re.I),
    re.compile(r"\bbuoyancy|\barchimedes|\bbernoulli|\bviscosity|\bpoiseuille|\breynolds\s+number", re.I),
    re.compile(r"\bnormal\s+distribution|\bpoisson\b|\bbinomial\s+distribution|\bgaussian\s+distribution|\bchi-square|\bregression\b|\bmarkov\b|\bexpected\s+value|\bvariance\b", re.I),
    re.compile(r"\bcolligative|\bosmotic\s+pressure|\bboiling\s+point\s+elevation|\bfreezing\s+point\s+depression|\braoult\b|\bhenry's\s+law\b", re.I),
    re.compile(r"\bisothermal\s+reversible|\breversible\s+expansion|\bexpansion\s+work|\bheat\s+capacity\s+at\s+constant", re.I),
    re.compile(r"\breal\s+gas|\bnon-?ideal\s+gas|\bdegF\b|\bfahrenheit\b", re.I),
    re.compile(r"\b(?:linear|surface|volume)\s+charge\s+density|\buniformly\s+charged\s+(?:rod|ring|disk|sheet|wire|cylinder|plane|line)|\bparallel\s+plates?|\bdielectric\b", re.I),
    re.compile(r"\bcentripetal|\bangular\s+frequency|\borbit(?:al|ing)?\b|\bkepler|\buniversal\s+gravitation|\bescape\s+(?:speed|velocity)|\bblack\s+hole", re.I),
)


def _extract_question_clause(norm_text: str) -> str:
    """Return the sentence/clause that specifies what quantity to compute."""
    sentences = [
        s.strip()
        for s in re.split(r"(?<=[?!])\s+|(?<=[A-Za-z0-9)])\.\s+(?=[A-Z])", norm_text)
        if s.strip()
    ]
    if not sentences:
        return norm_text
    cue_re = re.compile(
        r"\b(calculate|find|determine|compute|evaluate|what\s+(?:is|are|was|were|must|would|pressure|temperature|wavelength|frequency|force|work|energy|distance|speed|velocity|acceleration|mass|volume|flux|moment|heat|entropy|power|density|molarity|concentration|efficiency)|how\s+(?:many|much|fast|far|long|high)|report)\b|\?",
        re.I,
    )
    matches = [s for s in sentences if cue_re.search(s)]
    if matches:
        return " ".join(matches[-2:]) if len(matches) > 1 else matches[-1]
    return sentences[-1]


def _extract_requested_unit(
    question: str, comparison_text: str
) -> Tuple[Optional[str], bool]:
    """Extract an explicitly requested output unit such as ``in m/s`` or ``in percent``."""
    req_re = re.compile(
        r"\b(?:in\s+units\s+of|expressed\s+in|report(?:ed)?\s+in|give(?:n)?\s+in|in)\s+"
        r"([A-Za-z%][A-Za-z0-9_*/^() -]{0,24}?)(?=[.,;?!)]|\s+(?:and|or|for|at|if|when|of|the|to|by|with|on|from)\b|$)",
        re.I,
    )
    for m in req_re.finditer(question):
        cand = m.group(1).strip()
        low = cand.lower()
        if any(w in low.split() for w in ("a", "an", "the", "this", "that", "which", "both")):
            continue
        if low.split()[0] in _KNOWN_UNKNOWN_UNIT_WORDS:
            return cand.split()[0], True
        if low in _UNIT_WORD_ALIASES:
            return _UNIT_WORD_ALIASES[low], True
        parsed = parse_compound_unit(cand)
        if parsed is not None:
            return parsed[2], True

    if comparison_text:
        cmp_ms, _err = extract_measurements(comparison_text)
        if cmp_ms:
            return cmp_ms[0].unit, True

    return None, False


# Head-initial question cue anchor
_QUESTION_CUE_RE = re.compile(
    r"\b(?:calculate|find|determine|compute|evaluate|report|what\s+must\s+be|what\s+would|what\s+is|what\s+are|what|how\s+many|how\s+much|how\s+fast|how\s+far)\b",
    re.I,
)


def _identify_target_spec(
    question: str,
    full_norm: str,
    measurements: List[ExtractedMeasurement],
    comparison_text: str,
) -> Tuple[Optional[str], Optional[str], bool]:
    """Identify ``(target_symbol, target_unit, explicit_unit)`` by head-initial syntax."""
    q_low = question.lower()
    full_low = full_norm.lower()
    explicit_unit, has_explicit_unit = _extract_requested_unit(question, comparison_text)

    # Locate the last imperative/interrogative cue in the question clause so that
    # the head noun phrase immediately following it takes precedence over nouns
    # mentioned later in conditioning subclauses ("for the electrostatic force...").
    cue_matches = list(_QUESTION_CUE_RE.finditer(q_low))
    anchor = cue_matches[-1].start() if cue_matches else 0
    tail_low = q_low[anchor:]

    # Guard: questions asking for ratios, fractions, percentages, deltas, or
    # unsupported discrete counts ("how many photons/atoms/molecules/times...")
    # must not be mis-mapped onto an absolute physical quantity.
    if re.search(
        r"\b(?:ratio\s+of|fraction\s+of|percentage\s+of|percent\s+of|how\s+many\s+times|"
        r"difference\s+(?:in|between)|increase\s+in|decrease\s+in|"
        r"change\s+in\s+(?:pressure|volume|temperature|energy|kinetic|potential|speed|velocity|force|momentum|mass|length|wavelength|frequency))\b",
        tail_low,
    ):
        return None, None, False
    if re.search(r"\bhow\s+many\b", tail_low) and not re.search(
        r"\bhow\s+many\s+(?:moles?|electrons?)\b", tail_low
    ):
        return None, None, False

    has_voltage = any(dims_equal(m.dims, _d(mass=1, length=2, time=-3, current=-1)) for m in measurements)
    has_k = any(dims_equal(m.dims, _d(mass=1, time=-2)) for m in measurements)
    has_force = any(dims_equal(m.dims, _d(mass=1, length=1, time=-2)) for m in measurements)
    has_mass = any(dims_equal(m.dims, _d(mass=1)) for m in measurements)
    has_ohm = any(dims_equal(m.dims, _d(mass=1, length=2, time=-3, current=-2)) for m in measurements)

    n_press = sum(1 for m in measurements if dims_equal(m.dims, _d(mass=1, length=-1, time=-2)))
    n_temp = sum(1 for m in measurements if dims_equal(m.dims, _d(temperature=1)))
    n_vol = sum(1 for m in measurements if dims_equal(m.dims, _d(length=3)))
    n_conc = sum(1 for m in measurements if dims_equal(m.dims, _d(amount=1, length=-3)))

    def _work_kind() -> Tuple[str, str]:
        if has_k or "hooke" in full_low or "spring" in full_low or "force constant" in full_low:
            return "hooke_work", "J"
        if has_force:
            return "work", "J"
        if has_mass or "climb" in full_low or "lift" in full_low or "earth" in full_low or "gravity" in full_low:
            return "potential_work", "J"
        return "work", "J"

    def _pressure_kind() -> Tuple[str, str]:
        if n_press == 1 and (n_temp == 2 or n_vol == 2):
            p_unit = next(m.unit for m in measurements if dims_equal(m.dims, _d(mass=1, length=-1, time=-2)))
            return "p2", p_unit
        return "pressure", "Pa"

    def _volume_kind() -> Tuple[str, str]:
        if n_vol == 1 and n_temp == 2:
            v_unit = next(m.unit for m in measurements if dims_equal(m.dims, _d(length=3)))
            return "v2", v_unit
        return "volume", "m^3"

    def _conc_kind() -> Tuple[str, str]:
        if n_conc == 1 and n_vol == 2:
            c_unit = next(m.unit for m in measurements if dims_equal(m.dims, _d(amount=1, length=-3)))
            return "c2", c_unit
        return "molarity", "mol/L"

    # Candidate table: (regex, (target_symbol, default_unit))
    # Ranked strictly by (earliest position after question cue, -matched_span_len).
    candidates: List[Tuple[int, int, str, str]] = []
    rules: List[Tuple[str, Tuple[str, str]]] = [
        (r"\bwork\s+function\b", ("energy", "J")),
        (r"\belectric\s+dipole\s+moment\b|\bdipole\s+moment\b", ("dipole_moment", "C*m")),
        (r"\b(?:net\s+)?electric\s+flux\b|\bnet\s+flux\b", ("flux", "N*m^2/C")),
        (r"\breduced\s+mass\b", ("mu", "u")),
        (r"\bde\s+broglie\s+wavelength\b", ("wavelength", "m")),
        (r"\b(?:maximum|peak)\s+wavelength\b", ("wavelength", "m")),
        (r"\bwavelength\b", ("wavelength", "m")),
        (r"\bfrequency\b", ("frequency", "Hz")),
        (r"\bmolar\s+(?:photon\s+)?energy\b|\benergy\s+of\s+(?:one|1|a)\s+mol(?:e)?\b", ("molar_energy", "J/mol")),
        (r"\bkinetic\s+energy\b", ("energy" if has_voltage else "kinetic", "J")),
        (r"\bpotential\s+energy\b", ("potential_work", "J")),
        (r"\bphoton\s+energy\b|\benergy\s+of\s+a\s+photon\b|\benergy\b", ("energy", "J")),
        (r"\bwork\b", _work_kind()),
        (r"\bacceleration\b", ("acceleration", "m/s^2")),
        (r"\b(?:electrostatic|electric|coulomb|net)\s+force\b|\bforce\b", ("force", "N")),
        (r"\bweight\b", ("weight", "N")),
        (r"\belectric\s+field\b", ("electric_field", "N/C")),
        (r"\bdistance\b|\bseparation\b|\bhow\s+far\b", ("distance", "m")),
        (r"\bpressure\b", _pressure_kind()),
        (r"\btemperature\b", ("temperature", "K")),
        (r"\bvolume\b", _volume_kind()),
        (r"\b(?:rms|root[- ]mean[- ]square)\s+speed\b", ("rms_speed", "m/s")),
        (r"\b(?:common|final)\s+velocity\b", ("v", "m/s")),
        (r"\bspeed\b|\bvelocity\b|\bhow\s+fast\b", ("speed", "m/s")),
        (r"\bmomentum\b", ("momentum", "kg*m/s")),
        (r"\bpower\b", ("power", "W")),
        (r"\bentropy\b", ("entropy", "J/K")),
        (r"\bheat\b", ("heat", "J")),
        (r"\befficiency\b", ("efficiency", "1")),
        (r"\bdensity\b", ("density", "kg/m^3")),
        (r"\bmolarity\b|\bconcentration\b", _conc_kind()),
        (r"\bnumber\s+of\s+moles\b|\bhow\s+many\s+moles\b|\bamount\s+of\s+substance\b", ("amount", "mol")),
        (r"\bnumber\s+of\s+electrons\b|\bhow\s+many\s+electrons\b", ("count", "1")),
        (r"\bvoltage\b|\bpotential\s+difference\b", ("voltage", "V")),
        (r"\bhalf\s+width\b|\bmargin\s+of\s+error\b", ("half_width", "1")),
    ]
    if has_ohm:
        rules.append((r"\bcurrent\b", ("current", "A")))
    if explicit_unit is not None:
        rules.append((r"\bresult\b", ("target", explicit_unit)))

    for pattern, (sym, dunit) in rules:
        m = re.search(pattern, tail_low)
        if m:
            candidates.append((m.start(), -(m.end() - m.start()), sym, dunit))

    if not candidates:
        return None, None, False

    candidates.sort(key=lambda row: (row[0], row[1]))
    _, _, target_symbol, default_unit = candidates[0]

    if explicit_unit is not None:
        parsed_ex = parse_compound_unit(explicit_unit)
        if parsed_ex is None:
            return target_symbol, explicit_unit, True
        if default_unit is not None and target_symbol not in ("target", "half_width"):
            parsed_def = parse_compound_unit(default_unit)
            if parsed_def is not None and dims_equal(parsed_ex[1], parsed_def[1]):
                return target_symbol, parsed_ex[2], True
            if re.search(r"\b(?:report|express|give)\b.*\bin\b", q_low):
                return target_symbol, parsed_ex[2], True
        else:
            return target_symbol, parsed_ex[2], True

    return target_symbol, default_unit, has_explicit_unit


# --------------------------------------------------------- symbol assignment
def _ground_chemical_molar_mass(full_norm: str, raw_text: str) -> Optional[float]:
    """Identify a single unambiguously named gas/chemical species and return its molar mass (g/mol)."""
    low = full_norm.lower()
    found: List[Tuple[str, float]] = []
    for species in sorted(_NAMED_SPECIES_MOLAR_MASS, key=len, reverse=True):
        if re.search(r"\b" + re.escape(species) + r"\b", low):
            if species == "water" and "triple point" in low:
                continue
            found.append((species, _NAMED_SPECIES_MOLAR_MASS[species]))
    if len(found) == 1:
        return found[0][1]
    if len(found) > 1:
        return None

    for m in re.finditer(r"\\mathrm\{([A-Z][a-z]?)\}(?:_\{?(\d+)\}?)?", raw_text):
        sym = m.group(1)
        cnt = m.group(2) or "1"
        if sym in _ELEMENT_WEIGHTS_BY_SYMBOL:
            return _ELEMENT_WEIGHTS_BY_SYMBOL[sym] * float(int(cnt))
    return None


def _ground_particle_constants(full_norm: str) -> Optional[Tuple[float, Optional[float]]]:
    """Return ``(mass_kg, charge_in_e)`` if a single standard particle is referenced."""
    low = full_norm.lower()
    hits: List[str] = []
    for pname in ("alpha particle", "electron", "positron", "proton", "neutron"):
        if re.search(r"\b" + re.escape(pname) + r"s?\b", low):
            hits.append(pname)
    # Remove duplicate hit when both "electron" and "alpha" absent; abstain if >1 particle species
    unique = list(dict.fromkeys(hits))
    if len(unique) == 1:
        return _PARTICLE_CONSTANTS[unique[0]]
    return None


def _bind_quantities(
    target_symbol: str,
    target_unit: str,
    measurements: List[ExtractedMeasurement],
    norm_text: str,
    raw_text: str,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Bind extracted dimensional measurements + implicit constants to solver symbols.

    Returns ``(quantities, error_reason)``. Enforces the Unused-Quantity Guard so
    multi-step / out-of-scope problems abstain instead of emitting wrong answers.
    """
    low = norm_text.lower()
    quantities: List[Dict[str, Any]] = []
    consumed: set[int] = set()

    def _add(sym: str, val: float, unit: str, idx: Optional[int] = None) -> None:
        quantities.append({"symbol": sym, "value": float(val), "unit": str(unit)})
        if idx is not None:
            consumed.add(idx)

    # Check for explicit contradictory same-dimension duplicate assignments (e.g. "mass is 1 kg and also 2 kg")
    if re.search(r"\band\s+also\b|\breported\s+as\b.*\band\b", low):
        by_dim: Dict[Dims, List[ExtractedMeasurement]] = {}
        for meas in measurements:
            by_dim.setdefault(meas.dims, []).append(meas)
        for dm, group in by_dim.items():
            if len(group) >= 2 and dims_equal(dm, _d(mass=1)) and target_symbol not in ("mu", "v"):
                for item in group:
                    quantities.append({"symbol": "mass", "value": item.value, "unit": item.unit})
                return quantities, None

    # Special case: reduced mass of homonuclear or heteronuclear molecule
    if target_symbol == "mu":
        mass_idx = [i for i, m in enumerate(measurements) if dims_equal(m.dims, _d(mass=1))]
        if len(mass_idx) == 2:
            i1, i2 = mass_idx
            _add("m1", measurements[i1].value, measurements[i1].unit, i1)
            _add("m2", measurements[i2].value, measurements[i2].unit, i2)
        elif len(mass_idx) == 1 and re.search(r"\b(both|identical|equal|two\s+same)\b", low):
            i1 = mass_idx[0]
            _add("m1", measurements[i1].value, measurements[i1].unit, i1)
            _add("m2", measurements[i1].value, measurements[i1].unit, i1)
        else:
            # Dimensionless atomic mass wording: "both ... atoms have an atomic mass of 14.00"
            am_m = re.search(
                r"\b(?:both|each)\b.*?\batomic\s+mass(?:es)?\s+(?:of|=)\s*(\d+(?:\.\d+)?)",
                low,
            )
            if am_m:
                v = float(am_m.group(1))
                _add("m1", v, "u")
                _add("m2", v, "u")
            else:
                two_am = re.findall(r"\batomic\s+mass(?:es)?\s+(?:of|=)\s*(\d+(?:\.\d+)?)", low)
                if len(two_am) == 2:
                    _add("m1", float(two_am[0]), "u")
                    _add("m2", float(two_am[1]), "u")
        if len(consumed) != len(measurements):
            return [], "unconsumed_physical_quantities"
        return quantities, None

    # Index measurements by base dimension
    by_dims: Dict[Dims, List[int]] = {}
    for idx, meas in enumerate(measurements):
        by_dims.setdefault(meas.dims, []).append(idx)

    D_M = _d(mass=1)
    D_L = _d(length=1)
    D_T = _d(time=1)
    D_TEMP = _d(temperature=1)
    D_MOL = _d(amount=1)
    D_I = _d(current=1)
    D_Q = _d(time=1, current=1)
    D_VEL = _d(length=1, time=-1)
    D_ACC = _d(length=1, time=-2)
    D_FORCE = _d(mass=1, length=1, time=-2)
    D_K = _d(mass=1, time=-2)
    D_E = _d(mass=1, length=2, time=-2)
    D_P = _d(mass=1, length=-1, time=-2)
    D_V = _d(length=3)
    D_VOLT = _d(mass=1, length=2, time=-3, current=-1)
    D_EFIELD = _d(mass=1, length=1, time=-3, current=-1)
    D_FREQ = _d(time=-1)
    D_MM = _d(mass=1, amount=-1)
    D_CONC = _d(amount=1, length=-3)
    D_CP = _d(length=2, time=-2, temperature=-1)
    D_OHM = _d(mass=1, length=2, time=-3, current=-2)

    # Check if particle constants are mentioned
    particle = _ground_particle_constants(norm_text)

    # Helper: grab exactly N indices of a dimension
    def _idxs(dm: Dims) -> List[int]:
        return list(by_dims.get(dm, []))

    has_em_wave_ctx = bool(
        re.search(
            r"\b(?:wavelength|photon|light|laser|radiation|electromagnetic|ultraviolet|uv|infrared|x-?ray|gamma|threshold|photoelectric|work\s+function|radio|microwave|vacuum)\b",
            low,
        )
    )
    has_gravity_height_ctx = bool(
        re.search(
            r"\b(?:height|climb|lift|raise|elevat|above|high|stair|cliff|tower|earth|gravity|gravitational|altitude|vertical)\b",
            low,
        )
    )
    has_nonconst_motion_ctx = bool(
        re.search(
            r"\b(?:accelerat|gravity|gravitational|free\s+fall|dropped|thrown|spring|oscillat|harmonic|circular|period)\b",
            low,
        )
    )
    has_continuous_charge_ctx = bool(
        re.search(
            r"\b(?:rod|ring|disk|sheet|plane|wire|cylinder|dipole|line\s+of\s+charge)\b",
            low,
        )
    )

    # --- Dispatch by physically inferred target quantity (zero task-ID branching!) ---
    if target_symbol == "potential_work":
        if len(_idxs(D_M)) == 1 and len(_idxs(D_L)) == 1 and has_gravity_height_ctx:
            im, il = _idxs(D_M)[0], _idxs(D_L)[0]
            _add("mass", measurements[im].value, measurements[im].unit, im)
            _add("height", measurements[il].value, measurements[il].unit, il)

    elif target_symbol == "hooke_work":
        if len(_idxs(D_K)) == 1 and len(_idxs(D_L)) == 1:
            ik, il = _idxs(D_K)[0], _idxs(D_L)[0]
            _add("spring_constant", measurements[ik].value, measurements[ik].unit, ik)
            _add("displacement", measurements[il].value, measurements[il].unit, il)

    elif target_symbol == "work":
        if len(_idxs(D_FORCE)) == 1 and len(_idxs(D_L)) == 1:
            iforce, il = _idxs(D_FORCE)[0], _idxs(D_L)[0]
            _add("force", measurements[iforce].value, measurements[iforce].unit, iforce)
            _add("distance", measurements[il].value, measurements[il].unit, il)

    elif target_symbol == "kinetic":
        if len(_idxs(D_M)) == 1 and len(_idxs(D_VEL)) == 1:
            im, iv = _idxs(D_M)[0], _idxs(D_VEL)[0]
            _add("mass", measurements[im].value, measurements[im].unit, im)
            _add("velocity", measurements[iv].value, measurements[iv].unit, iv)
        elif not _idxs(D_M) and particle is not None and len(_idxs(D_VEL)) == 1:
            iv = _idxs(D_VEL)[0]
            _add("mass", particle[0], "kg")
            _add("velocity", measurements[iv].value, measurements[iv].unit, iv)

    elif target_symbol == "momentum":
        if len(_idxs(D_M)) == 1 and len(_idxs(D_VEL)) == 1:
            im, iv = _idxs(D_M)[0], _idxs(D_VEL)[0]
            _add("mass", measurements[im].value, measurements[im].unit, im)
            _add("velocity", measurements[iv].value, measurements[iv].unit, iv)
        elif not _idxs(D_M) and particle is not None and len(_idxs(D_VEL)) == 1:
            iv = _idxs(D_VEL)[0]
            _add("mass", particle[0], "kg")
            _add("velocity", measurements[iv].value, measurements[iv].unit, iv)

    elif target_symbol == "energy":
        if len(_idxs(D_VOLT)) == 1:
            iv = _idxs(D_VOLT)[0]
            _add("voltage", measurements[iv].value, measurements[iv].unit, iv)
            if len(_idxs(D_Q)) == 1:
                iq = _idxs(D_Q)[0]
                _add("charge", measurements[iq].value, measurements[iq].unit, iq)
            elif particle is not None and particle[1] is not None:
                _add("charge", particle[1], "e")
        elif len(_idxs(D_L)) == 1 and has_em_wave_ctx:
            il = _idxs(D_L)[0]
            _add("wavelength", measurements[il].value, measurements[il].unit, il)
        elif len(_idxs(D_FREQ)) == 1 and has_em_wave_ctx:
            ifq = _idxs(D_FREQ)[0]
            _add("frequency", measurements[ifq].value, measurements[ifq].unit, ifq)

    elif target_symbol == "molar_energy":
        if len(_idxs(D_L)) == 1 and has_em_wave_ctx:
            il = _idxs(D_L)[0]
            _add("wavelength", measurements[il].value, measurements[il].unit, il)

    elif target_symbol == "wavelength":
        if len(_idxs(D_E)) == 1 and ("de broglie" in low or particle is not None or _idxs(D_M)):
            ie = _idxs(D_E)[0]
            _add("kinetic_energy", measurements[ie].value, measurements[ie].unit, ie)
            if len(_idxs(D_M)) == 1:
                im = _idxs(D_M)[0]
                _add("mass", measurements[im].value, measurements[im].unit, im)
            elif particle is not None:
                _add("mass", particle[0], "kg")
        elif len(_idxs(D_VEL)) == 1 and ("de broglie" in low or particle is not None or _idxs(D_M)):
            iv = _idxs(D_VEL)[0]
            _add("velocity", measurements[iv].value, measurements[iv].unit, iv)
            if len(_idxs(D_M)) == 1:
                im = _idxs(D_M)[0]
                _add("mass", measurements[im].value, measurements[im].unit, im)
            elif particle is not None:
                _add("mass", particle[0], "kg")
        elif len(_idxs(D_TEMP)) == 1 and ("blackbody" in low or "wien" in low or "maximum" in low or "peak" in low):
            it = _idxs(D_TEMP)[0]
            _add("temperature", measurements[it].value, measurements[it].unit, it)
        elif len(_idxs(D_FREQ)) == 1 and has_em_wave_ctx:
            ifq = _idxs(D_FREQ)[0]
            _add("frequency", measurements[ifq].value, measurements[ifq].unit, ifq)
        elif len(_idxs(D_E)) == 1 and ("photon" in low or "light" in low or "laser" in low or "radiation" in low):
            ie = _idxs(D_E)[0]
            _add("energy", measurements[ie].value, measurements[ie].unit, ie)

    elif target_symbol == "frequency":
        if len(_idxs(D_L)) == 1 and has_em_wave_ctx:
            il = _idxs(D_L)[0]
            _add("wavelength", measurements[il].value, measurements[il].unit, il)

    elif target_symbol == "temperature":
        if len(_idxs(D_L)) == 1 and ("blackbody" in low or "wien" in low or "lambda" in low or "max" in low):
            il = _idxs(D_L)[0]
            _add("wavelength", measurements[il].value, measurements[il].unit, il)
        elif len(_idxs(D_P)) == 1 and len(_idxs(D_V)) == 1 and len(_idxs(D_MOL)) == 1:
            ip, iv, in_ = _idxs(D_P)[0], _idxs(D_V)[0], _idxs(D_MOL)[0]
            _add("pressure", measurements[ip].value, measurements[ip].unit, ip)
            _add("volume", measurements[iv].value, measurements[iv].unit, iv)
            _add("amount", measurements[in_].value, measurements[in_].unit, in_)

    elif target_symbol == "p2":
        if len(_idxs(D_P)) == 1 and len(_idxs(D_TEMP)) == 2 and re.search(
            r"\b(?:constant[- ]volume|rigid|fixed\s+volume|thermometer|sealed|closed\s+vessel|same\s+volume|isochoric)\b",
            low,
        ):
            ip = _idxs(D_P)[0]
            it1, it2 = _idxs(D_TEMP)
            _add("p1", measurements[ip].value, measurements[ip].unit, ip)
            _add("t1", measurements[it1].value, measurements[it1].unit, it1)
            _add("t2", measurements[it2].value, measurements[it2].unit, it2)
        elif len(_idxs(D_P)) == 1 and len(_idxs(D_V)) == 2 and re.search(
            r"\b(?:constant[- ]temperature|isothermal|boyle|same\s+temperature|fixed\s+temperature)\b",
            low,
        ):
            ip = _idxs(D_P)[0]
            iv1, iv2 = _idxs(D_V)
            _add("p1", measurements[ip].value, measurements[ip].unit, ip)
            _add("v1", measurements[iv1].value, measurements[iv1].unit, iv1)
            _add("v2", measurements[iv2].value, measurements[iv2].unit, iv2)

    elif target_symbol == "pressure":
        if len(_idxs(D_V)) == 1 and len(_idxs(D_TEMP)) == 1:
            iv, it = _idxs(D_V)[0], _idxs(D_TEMP)[0]
            _add("volume", measurements[iv].value, measurements[iv].unit, iv)
            _add("temperature", measurements[it].value, measurements[it].unit, it)
            if len(_idxs(D_MOL)) == 1:
                in_ = _idxs(D_MOL)[0]
                _add("amount", measurements[in_].value, measurements[in_].unit, in_)
            elif re.search(r"\b(?:one|1)\s+mol(?:e)?\b", low) and not _idxs(D_M):
                _add("amount", 1.0, "mol")
            elif len(_idxs(D_M)) == 1:
                im = _idxs(D_M)[0]
                _add("mass", measurements[im].value, measurements[im].unit, im)
                if len(_idxs(D_MM)) == 1:
                    imm = _idxs(D_MM)[0]
                    _add("molar_mass", measurements[imm].value, measurements[imm].unit, imm)
                else:
                    mm = _ground_chemical_molar_mass(norm_text, raw_text)
                    if mm is not None:
                        _add("molar_mass", mm, "g/mol")

    elif target_symbol == "v2":
        if len(_idxs(D_V)) == 1 and len(_idxs(D_TEMP)) == 2 and re.search(
            r"\b(?:constant[- ]pressure|isobaric|charles|same\s+pressure|fixed\s+pressure)\b",
            low,
        ):
            iv = _idxs(D_V)[0]
            it1, it2 = _idxs(D_TEMP)
            _add("v1", measurements[iv].value, measurements[iv].unit, iv)
            _add("t1", measurements[it1].value, measurements[it1].unit, it1)
            _add("t2", measurements[it2].value, measurements[it2].unit, it2)

    elif target_symbol == "volume":
        if len(_idxs(D_P)) == 1 and len(_idxs(D_TEMP)) == 1:
            ip, it = _idxs(D_P)[0], _idxs(D_TEMP)[0]
            _add("pressure", measurements[ip].value, measurements[ip].unit, ip)
            _add("temperature", measurements[it].value, measurements[it].unit, it)
            if len(_idxs(D_MOL)) == 1:
                in_ = _idxs(D_MOL)[0]
                _add("amount", measurements[in_].value, measurements[in_].unit, in_)
            elif len(_idxs(D_M)) == 1:
                im = _idxs(D_M)[0]
                _add("mass", measurements[im].value, measurements[im].unit, im)
                if len(_idxs(D_MM)) == 1:
                    imm = _idxs(D_MM)[0]
                    _add("molar_mass", measurements[imm].value, measurements[imm].unit, imm)
                else:
                    mm = _ground_chemical_molar_mass(norm_text, raw_text)
                    if mm is not None:
                        _add("molar_mass", mm, "g/mol")

    elif target_symbol == "force":
        if len(_idxs(D_Q)) == 2 and len(_idxs(D_L)) == 1 and not has_continuous_charge_ctx:
            iq1, iq2 = _idxs(D_Q)
            il = _idxs(D_L)[0]
            _add("q1", measurements[iq1].value, measurements[iq1].unit, iq1)
            _add("q2", measurements[iq2].value, measurements[iq2].unit, iq2)
            _add("distance", measurements[il].value, measurements[il].unit, il)
        elif len(_idxs(D_M)) == 1 and len(_idxs(D_ACC)) == 1:
            im, ia = _idxs(D_M)[0], _idxs(D_ACC)[0]
            _add("mass", measurements[im].value, measurements[im].unit, im)
            _add("acceleration", measurements[ia].value, measurements[ia].unit, ia)
        elif len(_idxs(D_EFIELD)) == 1:
            ief = _idxs(D_EFIELD)[0]
            _add("electric_field", measurements[ief].value, measurements[ief].unit, ief)
            if len(_idxs(D_Q)) == 1:
                iq = _idxs(D_Q)[0]
                _add("charge", measurements[iq].value, measurements[iq].unit, iq)
            elif particle is not None and particle[1] is not None:
                _add("charge", particle[1], "e")

    elif target_symbol == "weight":
        if len(_idxs(D_M)) == 1:
            im = _idxs(D_M)[0]
            _add("mass", measurements[im].value, measurements[im].unit, im)

    elif target_symbol == "electric_field":
        if len(_idxs(D_Q)) == 1 and len(_idxs(D_L)) == 1 and not has_continuous_charge_ctx:
            iq, il = _idxs(D_Q)[0], _idxs(D_L)[0]
            _add("charge", measurements[iq].value, measurements[iq].unit, iq)
            _add("distance", measurements[il].value, measurements[il].unit, il)

    elif target_symbol == "distance":
        if len(_idxs(D_Q)) == 2 and len(_idxs(D_FORCE)) == 1 and not has_continuous_charge_ctx:
            iq1, iq2 = _idxs(D_Q)
            iforce = _idxs(D_FORCE)[0]
            _add("q1", measurements[iq1].value, measurements[iq1].unit, iq1)
            _add("q2", measurements[iq2].value, measurements[iq2].unit, iq2)
            _add("force", measurements[iforce].value, measurements[iforce].unit, iforce)
        elif len(_idxs(D_VEL)) == 1 and len(_idxs(D_T)) == 1 and not has_nonconst_motion_ctx:
            iv, it = _idxs(D_VEL)[0], _idxs(D_T)[0]
            _add("speed", measurements[iv].value, measurements[iv].unit, iv)
            _add("time", measurements[it].value, measurements[it].unit, it)

    elif target_symbol == "acceleration":
        if len(_idxs(D_EFIELD)) == 1:
            ief = _idxs(D_EFIELD)[0]
            _add("electric_field", measurements[ief].value, measurements[ief].unit, ief)
            if len(_idxs(D_Q)) == 1:
                iq = _idxs(D_Q)[0]
                _add("charge", measurements[iq].value, measurements[iq].unit, iq)
            elif particle is not None and particle[1] is not None:
                _add("charge", particle[1], "e")
            if len(_idxs(D_M)) == 1:
                im = _idxs(D_M)[0]
                _add("mass", measurements[im].value, measurements[im].unit, im)
            elif particle is not None:
                _add("mass", particle[0], "kg")
        elif len(_idxs(D_VEL)) == 2 and len(_idxs(D_T)) == 1:
            iu, iv = _idxs(D_VEL)
            it = _idxs(D_T)[0]
            _add("u", measurements[iu].value, measurements[iu].unit, iu)
            _add("v", measurements[iv].value, measurements[iv].unit, iv)
            _add("time", measurements[it].value, measurements[it].unit, it)

    elif target_symbol == "dipole_moment":
        if len(_idxs(D_Q)) == 1 and len(_idxs(D_L)) == 1:
            iq, il = _idxs(D_Q)[0], _idxs(D_L)[0]
            _add("charge", measurements[iq].value, measurements[iq].unit, iq)
            _add("separation", measurements[il].value, measurements[il].unit, il)
            # Physically, intrinsic electric dipole moment p = |q|*d is independent
            # of an external electric field strength E stated in the prompt.
            for ief in _idxs(D_EFIELD):
                consumed.add(ief)

    elif target_symbol == "flux":
        if len(_idxs(D_Q)) == 1:
            iq = _idxs(D_Q)[0]
            _add("charge", measurements[iq].value, measurements[iq].unit, iq)
            # By Gauss's law Phi = q / eps0, closed Gaussian surface geometry
            # (cube edge / sphere radius) is physically independent of net flux.
            if "gauss" in low or "closed surface" in low or "cube" in low or "sphere" in low:
                for il in _idxs(D_L):
                    consumed.add(il)

    elif target_symbol == "speed":
        if len(_idxs(D_L)) == 1 and len(_idxs(D_T)) == 1 and not has_nonconst_motion_ctx:
            il, it = _idxs(D_L)[0], _idxs(D_T)[0]
            _add("distance", measurements[il].value, measurements[il].unit, il)
            _add("time", measurements[it].value, measurements[it].unit, it)

    elif target_symbol == "rms_speed":
        if len(_idxs(D_TEMP)) == 1:
            it = _idxs(D_TEMP)[0]
            _add("temperature", measurements[it].value, measurements[it].unit, it)
            if len(_idxs(D_MM)) == 1:
                imm = _idxs(D_MM)[0]
                _add("molar_mass", measurements[imm].value, measurements[imm].unit, imm)
            else:
                mm = _ground_chemical_molar_mass(norm_text, raw_text)
                if mm is not None:
                    _add("molar_mass", mm, "g/mol")

    elif target_symbol == "v":
        if len(_idxs(D_M)) == 2 and len(_idxs(D_VEL)) == 2:
            im1, im2 = _idxs(D_M)
            iv1, iv2 = _idxs(D_VEL)
            _add("m1", measurements[im1].value, measurements[im1].unit, im1)
            _add("m2", measurements[im2].value, measurements[im2].unit, im2)
            _add("u1", measurements[iv1].value, measurements[iv1].unit, iv1)
            _add("u2", measurements[iv2].value, measurements[iv2].unit, iv2)
        elif len(_idxs(D_VEL)) == 1 and len(_idxs(D_ACC)) == 1 and len(_idxs(D_T)) == 1:
            iu, ia, it = _idxs(D_VEL)[0], _idxs(D_ACC)[0], _idxs(D_T)[0]
            _add("u", measurements[iu].value, measurements[iu].unit, iu)
            _add("acceleration", measurements[ia].value, measurements[ia].unit, ia)
            _add("time", measurements[it].value, measurements[it].unit, it)

    elif target_symbol == "power":
        if len(_idxs(D_E)) == 1 and len(_idxs(D_T)) == 1:
            ie, it = _idxs(D_E)[0], _idxs(D_T)[0]
            _add("work", measurements[ie].value, measurements[ie].unit, ie)
            _add("time", measurements[it].value, measurements[it].unit, it)
        elif len(_idxs(D_VOLT)) == 1 and len(_idxs(D_I)) == 1:
            iv, ii = _idxs(D_VOLT)[0], _idxs(D_I)[0]
            _add("voltage", measurements[iv].value, measurements[iv].unit, iv)
            _add("current", measurements[ii].value, measurements[ii].unit, ii)

    elif target_symbol == "heat":
        if len(_idxs(D_M)) == 1 and len(_idxs(D_CP)) == 1 and len(_idxs(D_TEMP)) == 1:
            im, ic, it = _idxs(D_M)[0], _idxs(D_CP)[0], _idxs(D_TEMP)[0]
            _add("mass", measurements[im].value, measurements[im].unit, im)
            _add("specific_heat", measurements[ic].value, measurements[ic].unit, ic)
            _add("delta_temperature", measurements[it].value, measurements[it].unit, it)

    elif target_symbol == "entropy":
        if len(_idxs(D_E)) == 1 and len(_idxs(D_TEMP)) == 1:
            ie, it = _idxs(D_E)[0], _idxs(D_TEMP)[0]
            _add("heat", measurements[ie].value, measurements[ie].unit, ie)
            _add("temperature", measurements[it].value, measurements[it].unit, it)

    elif target_symbol == "efficiency":
        if len(_idxs(D_E)) == 2:
            i1, i2 = _idxs(D_E)
            _add("useful", measurements[i1].value, measurements[i1].unit, i1)
            _add("total", measurements[i2].value, measurements[i2].unit, i2)

    elif target_symbol == "density":
        if len(_idxs(D_M)) == 1 and len(_idxs(D_V)) == 1:
            im, iv = _idxs(D_M)[0], _idxs(D_V)[0]
            _add("mass", measurements[im].value, measurements[im].unit, im)
            _add("volume", measurements[iv].value, measurements[iv].unit, iv)

    elif target_symbol == "molarity":
        if len(_idxs(D_MOL)) == 1 and len(_idxs(D_V)) == 1:
            in_, iv = _idxs(D_MOL)[0], _idxs(D_V)[0]
            _add("amount", measurements[in_].value, measurements[in_].unit, in_)
            _add("volume", measurements[iv].value, measurements[iv].unit, iv)

    elif target_symbol == "c2":
        if len(_idxs(D_CONC)) == 1 and len(_idxs(D_V)) == 2:
            ic = _idxs(D_CONC)[0]
            iv1, iv2 = _idxs(D_V)
            _add("c1", measurements[ic].value, measurements[ic].unit, ic)
            _add("v1", measurements[iv1].value, measurements[iv1].unit, iv1)
            _add("v2", measurements[iv2].value, measurements[iv2].unit, iv2)

    elif target_symbol == "amount":
        if len(_idxs(D_M)) == 1:
            im = _idxs(D_M)[0]
            _add("mass", measurements[im].value, measurements[im].unit, im)
            if len(_idxs(D_MM)) == 1:
                imm = _idxs(D_MM)[0]
                _add("molar_mass", measurements[imm].value, measurements[imm].unit, imm)
            else:
                mm = _ground_chemical_molar_mass(norm_text, raw_text)
                if mm is not None:
                    _add("molar_mass", mm, "g/mol")
        elif len(_idxs(D_P)) == 1 and len(_idxs(D_V)) == 1 and len(_idxs(D_TEMP)) == 1:
            ip, iv, it = _idxs(D_P)[0], _idxs(D_V)[0], _idxs(D_TEMP)[0]
            _add("pressure", measurements[ip].value, measurements[ip].unit, ip)
            _add("volume", measurements[iv].value, measurements[iv].unit, iv)
            _add("temperature", measurements[it].value, measurements[it].unit, it)

    elif target_symbol == "count":
        if len(_idxs(D_Q)) == 1:
            iq = _idxs(D_Q)[0]
            _add("charge", measurements[iq].value, measurements[iq].unit, iq)

    elif target_symbol == "current":
        if len(_idxs(D_VOLT)) == 1 and len(_idxs(D_OHM)) == 1:
            iv, ir = _idxs(D_VOLT)[0], _idxs(D_OHM)[0]
            _add("voltage", measurements[iv].value, measurements[iv].unit, iv)
            _add("resistance", measurements[ir].value, measurements[ir].unit, ir)

    elif target_symbol == "voltage":
        if len(_idxs(D_I)) == 1 and len(_idxs(D_OHM)) == 1:
            ii, ir = _idxs(D_I)[0], _idxs(D_OHM)[0]
            _add("current", measurements[ii].value, measurements[ii].unit, ii)
            _add("resistance", measurements[ir].value, measurements[ir].unit, ir)

    elif target_symbol == "target":
        # Preserve raw measurements for dimensional trap checking (e.g. distance+time asked in J)
        if len(_idxs(D_L)) == 1 and len(_idxs(D_T)) == 1:
            il, it = _idxs(D_L)[0], _idxs(D_T)[0]
            _add("distance", measurements[il].value, measurements[il].unit, il)
            _add("time", measurements[it].value, measurements[it].unit, it)

    # Unused-Quantity Guard: if explicit dimensional measurements remain unconsumed,
    # the problem requires a richer physical model -> abstain cleanly!
    unconsumed = [i for i in range(len(measurements)) if i not in consumed]
    if unconsumed:
        return [], "unconsumed_physical_quantities"

    if not quantities:
        return [], "underdetermined_raw_prose"

    return quantities, None


# --------------------------------------------------------------- public entry
def parse_science_text(
    problem_text: str,
    *,
    target_unit: Optional[str] = None,
    problem_id: str = "",
    domain: str = "",
) -> ParseResult:
    """Parse raw physics/chemistry/math prose + LaTeX into a ``ScienceSolver`` dict.

    Returns a ``ParseResult`` whose ``problem`` dictionary can be passed directly
    to ``ScienceSolver.solve()`` when ``ok`` is True; otherwise ``ok`` is False
    with an explicit fail-closed ``reason``.
    """
    raw = str(problem_text or "").strip()
    if not raw:
        return ParseResult(ok=False, reason="empty_problem_text")

    # 1. Special exact symbolic math structures (geometric series, probability, combinatorics)
    geom = _try_parse_geometric_series(raw)
    if geom is not None:
        if geom.ok:
            geom.problem["id"] = problem_id
            if domain:
                geom.problem["domain"] = domain
        return geom

    norm_text, comparison_text, boilerplate, _raw_clean = normalize_science_text(raw)

    prob = _try_parse_probability(raw, norm_text)
    if prob is not None:
        if prob.ok:
            prob.problem["id"] = problem_id
            if domain:
                prob.problem["domain"] = domain
        return prob

    comb = _try_parse_combinatorics(raw, norm_text)
    if comb is not None:
        if comb.ok:
            comb.problem["id"] = problem_id
            if domain:
                comb.problem["domain"] = domain
        return comb

    # 2. Out-of-scope mathematical/physical domain guard (protects blind precision)
    for pat in _OUT_OF_SCOPE_PATTERNS:
        if pat.search(raw) or pat.search(norm_text):
            return ParseResult(
                ok=False,
                reason="out_of_verified_primer_scope",
                normalized_text=norm_text,
                boilerplate_stripped=boilerplate,
            )

    # 3. Extract explicit dimensional measurements
    measurements, meas_err = extract_measurements(norm_text)
    if meas_err is not None:
        return ParseResult(
            ok=False,
            reason=meas_err,
            normalized_text=norm_text,
            boilerplate_stripped=boilerplate,
        )

    # 4. Identify target quantity and output unit from the question clause
    question = _extract_question_clause(norm_text)
    t_sym, t_unit, explicit_u = _identify_target_spec(
        question, norm_text, measurements, comparison_text
    )
    if target_unit is not None and str(target_unit).strip():
        t_unit = str(target_unit).strip()
        explicit_u = True
    if not t_sym or t_unit is None:
        return ParseResult(
            ok=False,
            reason="target_quantity_unidentified",
            measurements=measurements,
            normalized_text=norm_text,
            boilerplate_stripped=boilerplate,
        )

    # 5. Bind measurements & grounded constants under the Unused-Quantity Guard
    quantities, bind_err = _bind_quantities(
        t_sym, t_unit, measurements, norm_text, raw
    )
    if bind_err is not None:
        return ParseResult(
            ok=False,
            reason=bind_err,
            measurements=measurements,
            normalized_text=norm_text,
            ignored_comparison_text=comparison_text,
            boilerplate_stripped=boilerplate,
        )

    problem_dict: Dict[str, Any] = {
        "id": str(problem_id),
        "domain": str(domain or "raw_prose_science"),
        "prompt": raw,
        "quantities": quantities,
        "target": {"symbol": t_sym, "unit": t_unit},
    }
    return ParseResult(
        ok=True,
        problem=problem_dict,
        measurements=measurements,
        normalized_text=norm_text,
        target_unit_explicit=bool(explicit_u),
        ignored_comparison_text=comparison_text,
        boilerplate_stripped=boilerplate,
    )
