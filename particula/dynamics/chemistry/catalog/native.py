"""Compile selected closed gas channels to Particula's native schema."""

import math
import re
from collections import Counter

from particula.dynamics.chemistry.mechanism import ReactionMechanism, _units
from particula.util.constants import AVOGADRO_NUMBER, GAS_CONSTANT

# Reviewed gas identities. Never interpret a lumped source name as a formula.
FORMULAS = {
    name: name
    for name in (
        "O",
        "O1D",
        "O2",
        "O3",
        "H",
        "H2",
        "OH",
        "HO2",
        "H2O",
        "H2O2",
        "N",
        "N2",
        "NO",
        "NO2",
        "NO3",
        "N2O",
        "N2O5",
        "HONO",
        "HNO3",
        "CO",
        "CO2",
        "CH4",
        "SO",
        "SO2",
        "SO3",
        "H2SO4",
        "H2S",
        "CH3O2",
        "CH3OOH",
        "CH2O",
        "HCHO",
        "C2H6",
        "C2H4",
    )
}
FORMULAS.update(O1D="O", HO="OH", O3P="O", O1D_CB4="O")
ALIASES = {"HO": "OH", "O3P": "O", "O1D_CB4": "O1D", "HCHO": "CH2O"}


def _elements(name):
    counts: Counter[str] = Counter()
    for atom, n in re.findall(r"([A-Z][a-z]?)(\d*)", FORMULAS[name]):
        counts[atom] += int(n or "1")
    return dict(counts)


def _normalize(side, declared):
    terms: Counter[str] = Counter()
    for species, n in side.items():
        if species not in FORMULAS and species not in declared:
            raise ValueError(f"unmapped molecular identity {species}")
        if not float(n).is_integer() or n <= 0:
            raise ValueError("requires fractional or diagnostic stoichiometry")
        terms[ALIASES.get(species, species)] += int(n)
    return dict(sorted(terms.items()))


def _reservoirs(record, lhs, rhs, close_oxygen):
    fixed = set(record["metadata"].get("fixed_species", []))
    titration = lhs == {"NO": 1, "O3": 1} and rhs in (
        {"NO2": 1},
        {"NO2": 1, "O2": 1},
    )
    if titration and ("O2" not in rhs or "O2" in fixed):
        if not close_oxygen:
            raise ValueError("requires explicit closed-box O2 (--close-oxygen)")
        rhs["O2"] = 1
        fixed.discard("O2")
    if fixed.intersection(set(record["reactants"]) | set(record["products"])):
        raise ValueError("requires a prescribed reservoir")
    annotations = re.findall(r"\{([^{}]*)\}", record["equation"])
    if annotations and not (
        titration
        and [a.strip().replace(" ", "") for a in annotations] == ["+O2"]
    ):
        raise ValueError(
            "requires review of implicit species in KPP annotation"
        )


def compile_channel(record, *, close_oxygen=False):
    """Return a native reaction and reviewed atomic compositions."""
    if record["reason"]:
        raise ValueError(record["reason"])
    if record["kinetics"] is None:
        raise ValueError("requires a source rate implementation")
    declared = record["metadata"].get("elements", {})
    lhs = _normalize(record["reactants"], declared)
    rhs = _normalize(record["products"], declared)
    _reservoirs(record, lhs, rhs, close_oxygen)
    order = sum(lhs.values())
    if (
        not lhs
        or not rhs
        or order not in (1, 2)
        or order != record["metadata"].get("kinetic_order", order)
    ):
        raise ValueError(
            "requires a different reaction order or an open budget"
        )
    delta: Counter[str] = Counter()
    species = {}
    for side, sign in ((lhs, -1), (rhs, 1)):
        for name, n in side.items():
            elements = _elements(name) if name in FORMULAS else declared[name]
            species["gas:" + name] = {"elements": elements}
            for elem, count in elements.items():
                delta[elem] += sign * n * count
    if any(delta.values()):
        raise ValueError("source equation requires an external material budget")
    a, c, b, tref = record["kinetics"]
    if not all(map(math.isfinite, (a, c, b, tref))) or a <= 0 or tref <= 0:
        raise ValueError("invalid or inactive kinetic coefficient")
    rate = dict(
        type="arrhenius",
        A=a * (AVOGADRO_NUMBER * 1e-6) ** (order - 1),
        b=b,
        Ea_J_per_mol=-GAS_CONSTANT * c,
        T_ref_K=tref,
        units=_units(order),
    )
    return dict(
        id=record["id"],
        reactants={"gas:" + k: v for k, v in lhs.items()},
        products={"gas:" + k: v for k, v in rhs.items()},
        rate=rate,
    ), species


def selected_channel(record, *, close_oxygen=False):
    """Read a released native reaction or convert a locally imported record."""
    if "mechanism" not in record:
        return compile_channel(record, close_oxygen=close_oxygen)
    if record["mechanism"] is None:
        raise ValueError(record["native_issue"])
    if record["requires_closed_oxygen"] and not close_oxygen:
        raise ValueError("requires explicit closed-box O2 (--close-oxygen)")
    mechanism = record["mechanism"]
    (reaction,) = mechanism["reactions"]
    return reaction, mechanism["species"]


def native_mechanism(records, *, close_oxygen=False):
    """Build one validated closed mechanism from a pinned selection."""
    reactions = []
    species: dict[str, dict] = {}
    seen: dict[tuple, str] = {}
    for record in records:
        try:
            reaction, catalog = selected_channel(
                record, close_oxygen=close_oxygen
            )
        except ValueError as error:
            raise ValueError(f"{record['id']}: {error}") from error
        key = (
            tuple(sorted(reaction["reactants"].items())),
            tuple(sorted(reaction["products"].items())),
        )
        owner = record["snapshot"]
        if key in seen and seen[key] != owner:
            raise ValueError(
                f"{record['id']}: overlapping channel from two snapshots"
            )
        seen[key] = owner
        reactions.append(reaction)
        for name, properties in catalog.items():
            if name in species and properties != species[name]:
                raise ValueError(f"conflicting molecular identities for {name}")
            species[name] = properties
    return ReactionMechanism.from_dict(
        dict(
            version=1, basis="mol/m^3-air", species=species, reactions=reactions
        )
    )
