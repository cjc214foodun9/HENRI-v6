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
from .science_parser import parse_compound_unit, parse_science_text

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
    "um": (1e-6, _d(length=1)),
    "nm": (1e-9, _d(length=1)),
    "pm": (1e-12, _d(length=1)),
    "A_ang": (1e-10, _d(length=1)),
    "km": (1000.0, _d(length=1)),
    "s": (1.0, _d(time=1)),
    "ms": (0.001, _d(time=1)),
    "min": (60.0, _d(time=1)),
    "h": (3600.0, _d(time=1)),
    "Hz": (1.0, _d(time=-1)),
    "1/s": (1.0, _d(time=-1)),
    "GHz": (1e9, _d(time=-1)),
    "kg": (1.0, _d(mass=1)),
    "g": (0.001, _d(mass=1)),
    "mg": (1e-6, _d(mass=1)),
    "u": (1.66053906660e-27, _d(mass=1)),
    "K": (1.0, _d(temperature=1)),
    "mol": (1.0, _d(amount=1)),
    "A": (1.0, _d(current=1)),
    "C": (1.0, _d(time=1, current=1)),
    "mC": (1e-3, _d(time=1, current=1)),
    "uC": (1e-6, _d(time=1, current=1)),
    "nC": (1e-9, _d(time=1, current=1)),
    "pC": (1e-12, _d(time=1, current=1)),
    "MC": (1e6, _d(time=1, current=1)),
    "e": (1.602176634e-19, _d(time=1, current=1)),
    "N": (1.0, _d(mass=1, length=1, time=-2)),
    "N/m": (1.0, _d(mass=1, time=-2)),
    "J": (1.0, _d(mass=1, length=2, time=-2)),
    "mJ": (1e-3, _d(mass=1, length=2, time=-2)),
    "kJ": (1e3, _d(mass=1, length=2, time=-2)),
    "eV": (1.602176634e-19, _d(mass=1, length=2, time=-2)),
    "MeV": (1.602176634e-13, _d(mass=1, length=2, time=-2)),
    "J/mol": (1.0, _d(mass=1, length=2, time=-2, amount=-1)),
    "kJ/mol": (1e3, _d(mass=1, length=2, time=-2, amount=-1)),
    "J/K": (1.0, _d(mass=1, length=2, time=-2, temperature=-1)),
    "J/(mol*K)": (1.0, _d(mass=1, length=2, time=-2, temperature=-1, amount=-1)),
    "J*s": (1.0, _d(mass=1, length=2, time=-1)),
    "W": (1.0, _d(mass=1, length=2, time=-3)),
    "Pa": (1.0, _d(mass=1, length=-1, time=-2)),
    "kPa": (1e3, _d(mass=1, length=-1, time=-2)),
    "bar": (1e5, _d(mass=1, length=-1, time=-2)),
    "atm": (101325.0, _d(mass=1, length=-1, time=-2)),
    "Torr": (101325.0 / 760.0, _d(mass=1, length=-1, time=-2)),
    "V": (1.0, _d(mass=1, length=2, time=-3, current=-1)),
    "N/C": (1.0, _d(mass=1, length=1, time=-3, current=-1)),
    "V/m": (1.0, _d(mass=1, length=1, time=-3, current=-1)),
    "C*m": (1.0, _d(length=1, time=1, current=1)),
    "N*m^2/C": (1.0, _d(mass=1, length=3, time=-3, current=-1)),
    "N*m^2/C^2": (1.0, _d(mass=1, length=3, time=-4, current=-2)),
    "F/m": (1.0, _d(mass=-1, length=-3, time=4, current=2)),
    "m*K": (1.0, _d(length=1, temperature=1)),
    "1/mol": (1.0, _d(amount=-1)),
    "L": (0.001, _d(length=3)),
    "dm^3": (0.001, _d(length=3)),
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
    "Ohm": (1.0, _d(mass=1, length=2, time=-3, current=-2)),
}

CONSTANTS: Dict[str, Tuple[float, str]] = {
    "g": (9.80665, "m/s^2"),
    "R": (8.314462618, "J/(mol*K)"),
    "c": (299792458.0, "m/s"),
    "h": (6.62607015e-34, "J*s"),
    "k_B": (1.380649e-23, "J/K"),
    "N_A": (6.02214076e23, "1/mol"),
    "e_charge": (1.602176634e-19, "C"),
    "eps0": (8.8541878128e-12, "F/m"),
    "k_e": (8.9875517923e9, "N*m^2/C^2"),
    "b_wien": (2.897771955e-3, "m*K"),
}


def unit_dims(unit: str) -> Optional[Dims]:
    key = str(unit).strip()
    entry = UNIT_TABLE.get(key)
    if entry is not None:
        return entry[1]
    parsed = parse_compound_unit(key)
    return None if parsed is None else parsed[1]


def unit_scale(unit: str) -> Optional[float]:
    key = str(unit).strip()
    entry = UNIT_TABLE.get(key)
    if entry is not None:
        return entry[0]
    parsed = parse_compound_unit(key)
    return None if parsed is None else parsed[0]


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


def _checked_geom(first_term: float, ratio: float) -> float:
    if abs(float(ratio)) >= 1.0:
        raise ValueError("geometric series diverges for |ratio| >= 1")
    return float(first_term) / (1.0 - float(ratio))


def _checked_cond_prob(p_intersection: float, p_condition: float) -> float:
    if p_intersection < 0.0 or p_condition <= 0.0 or p_condition > 1.0 or p_intersection > p_condition:
        raise ValueError("invalid probability domain")
    return float(p_intersection) / float(p_condition)


def _checked_indep_prob(p_a: float, p_b: float) -> float:
    if p_a < 0.0 or p_a > 1.0 or p_b < 0.0 or p_b > 1.0:
        raise ValueError("invalid probability domain")
    return float(p_a) * float(p_b)


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
        "potential_work", "gravitational potential work equals mass multiplied by gravitational acceleration multiplied by height",
        (("mass", _d(mass=1)), ("g", _d(length=1, time=-2)), ("height", _d(length=1))),
        _d(mass=1, length=2, time=-2), "mass * g * height",
        "height", "potential_work / (mass * g)",
        lambda mass, g, height: mass * g * height,
        lambda potential_work, mass, g, height: potential_work / (mass * g),
    ),
    Relation(
        "hooke_work", "elastic spring work equals one half spring constant multiplied by displacement squared",
        (("spring_constant", _d(mass=1, time=-2)), ("displacement", _d(length=1))),
        _d(mass=1, length=2, time=-2), "0.5 * spring_constant * displacement ** 2",
        "displacement", "sqrt(2 * hooke_work / spring_constant)",
        lambda spring_constant, displacement: 0.5 * spring_constant * displacement * displacement,
        lambda hooke_work, spring_constant, displacement: math.sqrt(2.0 * hooke_work / spring_constant),
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
        "isothermal_entropy", "isothermal entropy change equals reversible heat divided by temperature",
        (("heat", _d(mass=1, length=2, time=-2)), ("temperature", _d(temperature=1))),
        _d(mass=1, length=2, time=-2, temperature=-1), "heat / temperature",
        "temperature", "heat / entropy",
        lambda heat, temperature: heat / temperature,
        lambda entropy, heat, temperature: heat / entropy,
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
        "ideal_gas_mass", "the ideal gas pressure from sample mass equals mass divided by molar mass times the gas constant times temperature divided by volume",
        (("mass", _d(mass=1)), ("molar_mass", _d(mass=1, amount=-1)),
         ("temperature", _d(temperature=1)), ("volume", _d(length=3))),
        _d(mass=1, length=-1, time=-2), "(mass / molar_mass) * R * temperature / volume",
        "temperature", "pressure * volume * molar_mass / (mass * R)",
        lambda mass, molar_mass, temperature, volume: (mass / molar_mass) * 8.314462618 * temperature / volume,
        lambda pressure, mass, molar_mass, temperature, volume: pressure * volume * molar_mass / (mass * 8.314462618),
    ),
    Relation(
        "gay_lussac_pressure", "at constant volume ideal gas pressure is proportional to absolute temperature",
        (("p1", _d(mass=1, length=-1, time=-2)), ("t1", _d(temperature=1)), ("t2", _d(temperature=1))),
        _d(mass=1, length=-1, time=-2), "p1 * t2 / t1",
        "t2", "p2 * t1 / p1",
        lambda p1, t1, t2: p1 * t2 / t1,
        lambda p2, p1, t1, t2: p2 * t1 / p1,
    ),
    Relation(
        "wave_frequency", "electromagnetic wave frequency equals the speed of light divided by wavelength",
        (("c", _d(length=1, time=-1)), ("wavelength", _d(length=1))),
        _d(time=-1), "c / wavelength",
        "wavelength", "c / frequency",
        lambda c, wavelength: c / wavelength,
        lambda frequency, c, wavelength: c / frequency,
    ),
    Relation(
        "photon_energy", "photon energy equals Planck constant multiplied by the speed of light divided by wavelength",
        (("h", _d(mass=1, length=2, time=-1)), ("c", _d(length=1, time=-1)), ("wavelength", _d(length=1))),
        _d(mass=1, length=2, time=-2), "h * c / wavelength",
        "wavelength", "h * c / energy",
        lambda h, c, wavelength: h * c / wavelength,
        lambda energy, h, c, wavelength: h * c / energy,
    ),
    Relation(
        "molar_photon_energy", "molar photon energy equals Avogadro constant times Planck constant times the speed of light divided by wavelength",
        (("N_A", _d(amount=-1)), ("h", _d(mass=1, length=2, time=-1)),
         ("c", _d(length=1, time=-1)), ("wavelength", _d(length=1))),
        _d(mass=1, length=2, time=-2, amount=-1), "N_A * h * c / wavelength",
        "wavelength", "N_A * h * c / molar_energy",
        lambda N_A, h, c, wavelength: N_A * h * c / wavelength,
        lambda molar_energy, N_A, h, c, wavelength: N_A * h * c / molar_energy,
    ),
    Relation(
        "wien_temperature", "blackbody temperature equals the Wien displacement constant divided by peak wavelength",
        (("b_wien", _d(length=1, temperature=1)), ("wavelength", _d(length=1))),
        _d(temperature=1), "b_wien / wavelength",
        "wavelength", "b_wien / temperature",
        lambda b_wien, wavelength: b_wien / wavelength,
        lambda temperature, b_wien, wavelength: b_wien / temperature,
    ),
    Relation(
        "wien_wavelength", "peak blackbody wavelength equals the Wien displacement constant divided by temperature",
        (("b_wien", _d(length=1, temperature=1)), ("temperature", _d(temperature=1))),
        _d(length=1), "b_wien / temperature",
        "temperature", "b_wien / wavelength",
        lambda b_wien, temperature: b_wien / temperature,
        lambda wavelength, b_wien, temperature: b_wien / wavelength,
    ),
    Relation(
        "de_broglie_velocity", "de Broglie wavelength equals Planck constant divided by mass multiplied by velocity",
        (("h", _d(mass=1, length=2, time=-1)), ("mass", _d(mass=1)), ("velocity", _d(length=1, time=-1))),
        _d(length=1), "h / (mass * velocity)",
        "velocity", "h / (mass * wavelength)",
        lambda h, mass, velocity: h / (mass * velocity),
        lambda wavelength, h, mass, velocity: h / (mass * wavelength),
    ),
    Relation(
        "de_broglie_energy", "de Broglie wavelength from kinetic energy equals Planck constant divided by the square root of two times mass times kinetic energy",
        (("h", _d(mass=1, length=2, time=-1)), ("mass", _d(mass=1)), ("kinetic_energy", _d(mass=1, length=2, time=-2))),
        _d(length=1), "h / sqrt(2.0 * mass * kinetic_energy)",
        "kinetic_energy", "(h / wavelength) ** 2 / (2.0 * mass)",
        lambda h, mass, kinetic_energy: h / math.sqrt(2.0 * mass * kinetic_energy),
        lambda wavelength, h, mass, kinetic_energy: (h / wavelength) ** 2 / (2.0 * mass),
    ),
    Relation(
        "electrostatic_energy", "electrostatic work equals electric charge multiplied by potential difference",
        (("charge", _d(time=1, current=1)), ("voltage", _d(mass=1, length=2, time=-3, current=-1))),
        _d(mass=1, length=2, time=-2), "abs(charge) * voltage",
        "voltage", "energy / abs(charge)",
        lambda charge, voltage: abs(charge) * voltage,
        lambda energy, charge, voltage: energy / abs(charge),
    ),
    Relation(
        "coulomb_force", "electrostatic force equals Coulomb constant times the product of charge magnitudes divided by distance squared",
        (("k_e", _d(mass=1, length=3, time=-4, current=-2)),
         ("q1", _d(time=1, current=1)), ("q2", _d(time=1, current=1)), ("distance", _d(length=1))),
        _d(mass=1, length=1, time=-2), "k_e * abs(q1 * q2) / (distance ** 2)",
        "distance", "sqrt(k_e * abs(q1 * q2) / force)",
        lambda k_e, q1, q2, distance: k_e * abs(q1 * q2) / (distance * distance),
        lambda force, k_e, q1, q2, distance: math.sqrt(k_e * abs(q1 * q2) / force),
    ),
    Relation(
        "coulomb_distance", "separation between point charges equals the square root of Coulomb constant times the product of charge magnitudes divided by force",
        (("k_e", _d(mass=1, length=3, time=-4, current=-2)),
         ("q1", _d(time=1, current=1)), ("q2", _d(time=1, current=1)), ("force", _d(mass=1, length=1, time=-2))),
        _d(length=1), "sqrt(k_e * abs(q1 * q2) / force)",
        "force", "k_e * abs(q1 * q2) / (distance ** 2)",
        lambda k_e, q1, q2, force: math.sqrt(k_e * abs(q1 * q2) / force),
        lambda distance, k_e, q1, q2, force: k_e * abs(q1 * q2) / (distance * distance),
    ),
    Relation(
        "electric_acceleration", "acceleration in a uniform electric field equals charge magnitude multiplied by electric field divided by mass",
        (("charge", _d(time=1, current=1)), ("electric_field", _d(mass=1, length=1, time=-3, current=-1)), ("mass", _d(mass=1))),
        _d(length=1, time=-2), "abs(charge) * electric_field / mass",
        "electric_field", "acceleration * mass / abs(charge)",
        lambda charge, electric_field, mass: abs(charge) * electric_field / mass,
        lambda acceleration, charge, electric_field, mass: acceleration * mass / abs(charge),
    ),
    Relation(
        "dipole_moment", "electric dipole moment equals charge magnitude multiplied by separation",
        (("charge", _d(time=1, current=1)), ("separation", _d(length=1))),
        _d(length=1, time=1, current=1), "abs(charge) * separation",
        "separation", "dipole_moment / abs(charge)",
        lambda charge, separation: abs(charge) * separation,
        lambda dipole_moment, charge, separation: dipole_moment / abs(charge),
    ),
    Relation(
        "gauss_flux", "net electric flux through a closed surface equals enclosed charge divided by vacuum permittivity",
        (("charge", _d(time=1, current=1)), ("eps0", _d(mass=-1, length=-3, time=4, current=2))),
        _d(mass=1, length=3, time=-3, current=-1), "charge / eps0",
        "charge", "flux * eps0",
        lambda charge, eps0: charge / eps0,
        lambda flux, charge, eps0: flux * eps0,
    ),
    Relation(
        "electron_count", "elementary charge count equals total charge magnitude divided by the elementary charge",
        (("charge", _d(time=1, current=1)), ("e_charge", _d(time=1, current=1))),
        DIM_ZERO, "abs(charge) / e_charge",
        "charge", "count * e_charge",
        lambda charge, e_charge: abs(charge) / e_charge,
        lambda count, charge, e_charge: count * e_charge,
    ),
    Relation(
        "reduced_mass", "reduced mass equals the product of two masses divided by their sum",
        (("m1", _d(mass=1)), ("m2", _d(mass=1))),
        _d(mass=1), "(m1 * m2) / (m1 + m2)",
        "m1", "(mu * m2) / (m2 - mu)",
        lambda m1, m2: (m1 * m2) / (m1 + m2),
        lambda mu, m1, m2: (mu * m2) / (m2 - mu),
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
        "combination", "factorial(int(round(n))) / (factorial(int(round(k))) * factorial(int(round(n)) - int(round(k))))",
        lambda n, k: float(math.comb(int(round(n)), int(round(k)))),
        lambda combination, n, k: float(
            math.factorial(int(round(n)))
            / (math.factorial(int(round(k))) * math.factorial(int(round(n)) - int(round(k))))
        ),
    ),
    Relation(
        "geometric_series", "the sum of a convergent infinite geometric series equals the first term divided by one minus the common ratio",
        (("first_term", DIM_ZERO), ("ratio", DIM_ZERO)),
        DIM_ZERO, "first_term / (1.0 - ratio)",
        "first_term", "series_sum * (1.0 - ratio)",
        _checked_geom,
        lambda series_sum, first_term, ratio: series_sum * (1.0 - ratio),
    ),
    Relation(
        "conditional_prob", "conditional probability equals joint probability divided by the conditioning event probability",
        (("p_intersection", DIM_ZERO), ("p_condition", DIM_ZERO)),
        DIM_ZERO, "p_intersection / p_condition",
        "p_intersection", "p_cond * p_condition",
        _checked_cond_prob,
        lambda p_cond, p_intersection, p_condition: p_cond * p_condition,
    ),
    Relation(
        "independent_intersection", "for independent events the joint probability equals the product of the individual probabilities",
        (("p_a", DIM_ZERO), ("p_b", DIM_ZERO)),
        DIM_ZERO, "p_a * p_b",
        "p_a", "p_ab / p_b",
        _checked_indep_prob,
        lambda p_ab, p_a, p_b: p_ab / p_b,
    ),
    Relation(
        "confidence", "a confidence interval half width equals the critical value multiplied by the standard deviation divided by the square root of the sample size",
        (("z", DIM_ZERO), ("sd", _d(length=1, time=-1)), ("n", DIM_ZERO)),
        _d(length=1, time=-1), "z * sd / sqrt(n)",
        "sd", "half_width * sqrt(n) / z",
        lambda z, sd, n: z * sd / math.sqrt(n),
        lambda half_width, z, sd, n: half_width * math.sqrt(n) / z,
    ),
    Relation(
        "power", "power equals work divided by time",
        (("work", _d(mass=1, length=2, time=-2)), ("time", _d(time=1))),
        _d(mass=1, length=2, time=-3), "work / time",
        "time", "work / power",
        lambda work, time: work / time,
        lambda power, work, time: work / power,
    ),
    Relation(
        "distance_from_speed", "distance equals speed multiplied by time",
        (("speed", _d(length=1, time=-1)), ("time", _d(time=1))),
        _d(length=1), "speed * time",
        "time", "distance / speed",
        lambda speed, time: speed * time,
        lambda distance, speed, time: distance / speed,
    ),
    Relation(
        "boyle_pressure", "at constant temperature ideal gas pressure times volume is constant",
        (("p1", _d(mass=1, length=-1, time=-2)), ("v1", _d(length=3)), ("v2", _d(length=3))),
        _d(mass=1, length=-1, time=-2), "p1 * v1 / v2",
        "v2", "p1 * v1 / p2",
        lambda p1, v1, v2: p1 * v1 / v2,
        lambda p2, p1, v1, v2: p1 * v1 / p2,
    ),
    Relation(
        "charles_volume", "at constant pressure ideal gas volume is proportional to absolute temperature",
        (("v1", _d(length=3)), ("t1", _d(temperature=1)), ("t2", _d(temperature=1))),
        _d(length=3), "v1 * t2 / t1",
        "t2", "v2 * t1 / v1",
        lambda v1, t1, t2: v1 * t2 / t1,
        lambda v2, v1, t1, t2: v2 * t1 / v1,
    ),
    Relation(
        "ideal_gas_volume", "the ideal gas volume equals amount times the gas constant times temperature divided by pressure",
        (("amount", _d(amount=1)), ("temperature", _d(temperature=1)), ("pressure", _d(mass=1, length=-1, time=-2))),
        _d(length=3), "amount * R * temperature / pressure",
        "temperature", "pressure * volume / (amount * R)",
        lambda amount, temperature, pressure: amount * 8.314462618 * temperature / pressure,
        lambda volume, amount, temperature, pressure: pressure * volume / (amount * 8.314462618),
    ),
    Relation(
        "ideal_gas_volume_mass", "the ideal gas volume from sample mass equals mass divided by molar mass times the gas constant times temperature divided by pressure",
        (("mass", _d(mass=1)), ("molar_mass", _d(mass=1, amount=-1)),
         ("temperature", _d(temperature=1)), ("pressure", _d(mass=1, length=-1, time=-2))),
        _d(length=3), "(mass / molar_mass) * R * temperature / pressure",
        "temperature", "pressure * volume * molar_mass / (mass * R)",
        lambda mass, molar_mass, temperature, pressure: (mass / molar_mass) * 8.314462618 * temperature / pressure,
        lambda volume, mass, molar_mass, temperature, pressure: pressure * volume * molar_mass / (mass * 8.314462618),
    ),
    Relation(
        "ideal_gas_temperature", "the ideal gas temperature equals pressure times volume divided by amount times the gas constant",
        (("pressure", _d(mass=1, length=-1, time=-2)), ("volume", _d(length=3)), ("amount", _d(amount=1))),
        _d(temperature=1), "pressure * volume / (amount * R)",
        "pressure", "amount * R * temperature / volume",
        lambda pressure, volume, amount: pressure * volume / (amount * 8.314462618),
        lambda temperature, pressure, volume, amount: amount * 8.314462618 * temperature / volume,
    ),
    Relation(
        "ideal_gas_amount", "the ideal gas amount equals pressure times volume divided by the gas constant times temperature",
        (("pressure", _d(mass=1, length=-1, time=-2)), ("volume", _d(length=3)), ("temperature", _d(temperature=1))),
        _d(amount=1), "pressure * volume / (R * temperature)",
        "pressure", "amount * R * temperature / volume",
        lambda pressure, volume, temperature: pressure * volume / (8.314462618 * temperature),
        lambda amount, pressure, volume, temperature: amount * 8.314462618 * temperature / volume,
    ),
    Relation(
        "rms_speed", "root mean square gas speed equals the square root of three times the gas constant times temperature divided by molar mass",
        (("R", _d(mass=1, length=2, time=-2, temperature=-1, amount=-1)),
         ("temperature", _d(temperature=1)), ("molar_mass", _d(mass=1, amount=-1))),
        _d(length=1, time=-1), "sqrt(3.0 * R * temperature / molar_mass)",
        "temperature", "(rms_speed ** 2) * molar_mass / (3.0 * R)",
        lambda R, temperature, molar_mass: math.sqrt(3.0 * R * temperature / molar_mass),
        lambda rms_speed, R, temperature, molar_mass: (rms_speed * rms_speed) * molar_mass / (3.0 * R),
    ),
    Relation(
        "wave_wavelength", "electromagnetic wavelength equals the speed of light divided by frequency",
        (("c", _d(length=1, time=-1)), ("frequency", _d(time=-1))),
        _d(length=1), "c / frequency",
        "frequency", "c / wavelength",
        lambda c, frequency: c / frequency,
        lambda wavelength, c, frequency: c / wavelength,
    ),
    Relation(
        "photon_energy_frequency", "photon energy equals Planck constant multiplied by frequency",
        (("h", _d(mass=1, length=2, time=-1)), ("frequency", _d(time=-1))),
        _d(mass=1, length=2, time=-2), "h * frequency",
        "frequency", "energy / h",
        lambda h, frequency: h * frequency,
        lambda energy, h, frequency: energy / h,
    ),
    Relation(
        "photon_wavelength_from_energy", "photon wavelength equals Planck constant multiplied by the speed of light divided by energy",
        (("h", _d(mass=1, length=2, time=-1)), ("c", _d(length=1, time=-1)), ("energy", _d(mass=1, length=2, time=-2))),
        _d(length=1), "h * c / energy",
        "energy", "h * c / wavelength",
        lambda h, c, energy: h * c / energy,
        lambda wavelength, h, c, energy: h * c / wavelength,
    ),
    Relation(
        "coulomb_electric_field", "point charge electric field equals Coulomb constant times charge magnitude divided by distance squared",
        (("k_e", _d(mass=1, length=3, time=-4, current=-2)),
         ("charge", _d(time=1, current=1)), ("distance", _d(length=1))),
        _d(mass=1, length=1, time=-3, current=-1), "k_e * abs(charge) / (distance ** 2)",
        "distance", "sqrt(k_e * abs(charge) / electric_field)",
        lambda k_e, charge, distance: k_e * abs(charge) / (distance * distance),
        lambda electric_field, k_e, charge, distance: math.sqrt(k_e * abs(charge) / electric_field),
    ),
    Relation(
        "electric_force_field", "electric force in a uniform electric field equals charge magnitude multiplied by electric field",
        (("charge", _d(time=1, current=1)), ("electric_field", _d(mass=1, length=1, time=-3, current=-1))),
        _d(mass=1, length=1, time=-2), "abs(charge) * electric_field",
        "electric_field", "force / abs(charge)",
        lambda charge, electric_field: abs(charge) * electric_field,
        lambda force, charge, electric_field: force / abs(charge),
    ),
    Relation(
        "ohm_current", "electric current equals voltage divided by resistance",
        (("voltage", _d(mass=1, length=2, time=-3, current=-1)), ("resistance", _d(mass=1, length=2, time=-3, current=-2))),
        _d(current=1), "voltage / resistance",
        "voltage", "current * resistance",
        lambda voltage, resistance: voltage / resistance,
        lambda current, voltage, resistance: current * resistance,
    ),
    Relation(
        "ohm_voltage", "electric potential difference equals current multiplied by resistance",
        (("current", _d(current=1)), ("resistance", _d(mass=1, length=2, time=-3, current=-2))),
        _d(mass=1, length=2, time=-3, current=-1), "current * resistance",
        "current", "voltage / resistance",
        lambda current, resistance: current * resistance,
        lambda voltage, current, resistance: voltage / resistance,
    ),
    Relation(
        "electric_power", "electrical power equals voltage multiplied by current",
        (("voltage", _d(mass=1, length=2, time=-3, current=-1)), ("current", _d(current=1))),
        _d(mass=1, length=2, time=-3), "voltage * current",
        "current", "power / voltage",
        lambda voltage, current: voltage * current,
        lambda power, voltage, current: power / voltage,
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
    "potential_work": "potential_work",
    "hooke_work": "hooke_work",
    "kinetic": "kinetic",
    "momentum": "momentum",
    "heat": "heat",
    "isothermal_entropy": "entropy",
    "efficiency": "efficiency",
    "density": "density",
    "molarity": "molarity",
    "moles": "amount",
    "dilution": "c2",
    "ideal_gas": "pressure",
    "ideal_gas_mass": "pressure",
    "gay_lussac_pressure": "p2",
    "wave_frequency": "frequency",
    "photon_energy": "energy",
    "molar_photon_energy": "molar_energy",
    "wien_temperature": "temperature",
    "wien_wavelength": "wavelength",
    "de_broglie_velocity": "wavelength",
    "de_broglie_energy": "wavelength",
    "electrostatic_energy": "energy",
    "coulomb_force": "force",
    "coulomb_distance": "distance",
    "electric_acceleration": "acceleration",
    "dipole_moment": "dipole_moment",
    "gauss_flux": "flux",
    "electron_count": "count",
    "reduced_mass": "mu",
    "inelastic": "v",
    "combination": "combination",
    "geometric_series": "series_sum",
    "conditional_prob": "p_cond",
    "independent_intersection": "p_ab",
    "confidence": "half_width",
    "power": "power",
    "distance_from_speed": "distance",
    "boyle_pressure": "p2",
    "charles_volume": "v2",
    "ideal_gas_volume": "volume",
    "ideal_gas_volume_mass": "volume",
    "ideal_gas_temperature": "temperature",
    "ideal_gas_amount": "amount",
    "rms_speed": "rms_speed",
    "wave_wavelength": "wavelength",
    "photon_energy_frequency": "energy",
    "photon_wavelength_from_energy": "wavelength",
    "coulomb_electric_field": "electric_field",
    "electric_force_field": "force",
    "ohm_current": "current",
    "ohm_voltage": "voltage",
    "electric_power": "power",
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

    # ---------------------------------------------------------- raw prose API
    def solve_text(
        self,
        problem_text: str,
        *,
        target_unit: Optional[str] = None,
        problem_id: str = "",
        domain: str = "",
        workspace: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Parse unmodified raw prose/LaTeX and verify with ``solve()`` (0 LLM calls)."""
        started = __import__("time").perf_counter()
        parsed = parse_science_text(
            problem_text,
            target_unit=target_unit,
            problem_id=problem_id,
            domain=domain,
        )
        if not parsed.ok:
            return {
                "id": str(problem_id),
                "domain": str(domain or "raw_prose_science"),
                "status": "ABSTAIN",
                "reason": str(parsed.reason or "raw_prose_parse_abstained"),
                "llm_calls": 0,
                "input_mode": "raw_text",
                "parsed": {
                    "ok": False,
                    "reason": str(parsed.reason),
                    "measurements_found": len(parsed.measurements),
                    "target_unit_explicit": bool(parsed.target_unit_explicit),
                },
                "seconds": float(__import__("time").perf_counter() - started),
            }
        res = self.solve(parsed.problem, workspace=workspace)
        res["input_mode"] = "raw_text"
        res["parsed"] = {
            "ok": True,
            "quantities": list(parsed.problem.get("quantities") or []),
            "target": dict(parsed.problem.get("target") or {}),
            "target_unit_explicit": bool(parsed.target_unit_explicit),
            "measurements_found": len(parsed.measurements),
            "boilerplate_stripped": list(parsed.boilerplate_stripped),
        }
        return res

    # ----------------------------------------------------------------- solve
    def solve(self, problem: Dict[str, Any], workspace: Optional[Path] = None) -> Dict[str, Any]:
        if (
            "quantities" not in problem
            and "target" not in problem
            and "table" not in problem
            and (problem.get("prompt") or problem.get("problem_text"))
        ):
            raw_text = str(problem.get("problem_text") or problem.get("prompt") or "")
            return self.solve_text(
                raw_text,
                target_unit=problem.get("target_unit"),
                problem_id=str(problem.get("id", "")),
                domain=str(problem.get("domain", "")),
                workspace=workspace,
            )
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
            if given.symbol in ("mass", "m1", "m2", "molar_mass", "wavelength") and given.value < 0.0:
                result["reason"] = "nonphysical_negative_%s" % given.symbol
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
                    if name in CONSTANTS and name not in symbols:
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
                if name in symbols:
                    si_values[name] = _convert(symbols[name].value, symbols[name].unit)
                elif name in CONSTANTS:
                    si_values[name] = float(CONSTANTS[name][0])
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
                    name in symbols or name not in CONSTANTS
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
        if not relation.tabular:
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
            lines += ["import math"]
            for cname, (cval, _cunit) in CONSTANTS.items():
                lines.append("%s = %.17g" % (cname, cval))
            lines.append("")
            for name, _dims in relation.inputs:
                lines.append("%s = %.17g" % (name, si_values[name]))
            lines.append("value_si = " + _mathify(relation.expression))
        lines += [
            "value = value_si / %.17g" % output_scale,
            "result = {'value': value, 'unit': %r, 'relation': %r}" % (target_unit, relation.name),
        ]
        return "\n".join(lines) + "\n"
