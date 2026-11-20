"""Versioned native mechanism records for irreversible gas mass action."""

import json
from dataclasses import dataclass
from math import isclose, isfinite
from pathlib import Path

import yaml  # type: ignore[import-untyped]


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key {key!r}")
        result[key] = value
    return result


class _MechanismLoader(yaml.SafeLoader):
    """Preserve duplicate-key diagnostics for scientific input."""


def _yaml_mapping(loader, node):
    return _unique(loader.construct_pairs(node, deep=True))


_MechanismLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _yaml_mapping
)


def _read_mapping(path):
    """Read JSON or YAML without silently discarding duplicate keys."""
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        if path.suffix.lower() == ".json":
            return json.load(stream, object_pairs_hook=_unique)
        if path.suffix.lower() in (".yaml", ".yml"):
            loader = _MechanismLoader(stream)
            try:
                return loader.get_single_data()
            finally:
                loader.dispose()
    raise ValueError("mechanism file must use .json, .yaml or .yml")


def _fields(value, required, optional=()):
    if not isinstance(value, dict):
        raise ValueError("expected a mapping")
    missing = set(required).difference(value)
    extra = set(value).difference(required, optional)
    if missing or extra:
        raise ValueError(f"missing fields {missing}, unexpected fields {extra}")


def _number(value, name):
    if type(value) not in (int, float) or not isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _counts(values, names, label, *, minimum):
    if not isinstance(values, dict):
        raise ValueError(f"{label} must be a mapping")
    result = []
    for key, count in values.items():
        if key not in names:
            raise ValueError(f"{label} references unknown species {key!r}")
        if type(count) is not int or count < minimum:
            raise ValueError(f"{label} requires integer counts >= {minimum}")
        result.append((key, count))
    return tuple(sorted(result))


def _units(order):
    if order == 0:
        return "mol/m^3/s"
    if order == 1:
        return "1/s"
    if order == 2:
        return "m^3/mol/s"
    return f"m^{3 * (order - 1)}/mol^{order - 1}/s"


@dataclass(frozen=True)
class Reaction:
    """Validated immutable reaction with SI molar coefficients."""

    name: str
    reactants: tuple
    products: tuple
    orders: tuple
    law: str
    coefficient: float
    exponent: float
    energy: float
    reference_temperature: float

    @classmethod
    def parse(cls, record, names):
        """Check a reaction against the declared species and SI units."""
        _fields(record, ("id", "reactants", "products", "rate"), ("orders",))
        name = record["id"]
        if not isinstance(name, str) or not name.strip():
            raise ValueError("reaction id must be a nonempty string")
        reactants = _counts(record["reactants"], names, "reactants", minimum=1)
        products = _counts(record["products"], names, "products", minimum=1)
        if not reactants or not products:
            raise ValueError("closed gas reactions need reactants and products")
        orders = dict(reactants)
        orders.update(
            _counts(record.get("orders", {}), names, "orders", minimum=0)
        )
        rate = record["rate"]
        law = rate.get("type") if isinstance(rate, dict) else None
        exponent, energy, reference = 0.0, 0.0, 1.0
        if law == "constant":
            _fields(rate, ("type", "k", "units"))
            coefficient = _number(rate["k"], "k")
        elif law == "reactive_uptake":
            _fields(rate, ("type", "gamma"))
            coefficient = _number(rate["gamma"], "gamma")
            if not 0 < coefficient <= 1:
                raise ValueError("gamma must be in (0, 1]")
        elif law == "arrhenius":
            _fields(
                rate, ("type", "A", "b", "Ea_J_per_mol", "T_ref_K", "units")
            )
            coefficient = _number(rate["A"], "A")
            exponent = _number(rate["b"], "b")
            energy = _number(rate["Ea_J_per_mol"], "Ea_J_per_mol")
            reference = _number(rate["T_ref_K"], "T_ref_K")
        else:
            raise ValueError(f"unsupported rate type {law!r}")
        if coefficient < 0 or reference <= 0:
            raise ValueError(
                "rate coefficient must be nonnegative and T_ref_K positive"
            )
        expected = _units(sum(orders.values()))
        if law != "reactive_uptake" and rate["units"] != expected:
            raise ValueError(f"rate units must be {expected!r}")
        return cls(
            name,
            reactants,
            products,
            tuple(sorted(orders.items())),
            law,
            coefficient,
            exponent,
            energy,
            reference,
        )

    def signed_coefficients(self):
        """Return affected species and net molecular counts per reaction."""
        lhs, rhs = dict(self.reactants), dict(self.products)
        return {
            name: rhs.get(name, 0) - lhs.get(name, 0)
            for name in sorted(lhs.keys() | rhs.keys())
            if rhs.get(name, 0) != lhs.get(name, 0)
        }

    def validate_masses(self, signed, masses):
        """Check closed mass balance when every net participant has a mass."""
        if signed.keys() <= masses.keys():
            consumed = sum(-v * masses[n] for n, v in signed.items() if v < 0)
            produced = sum(v * masses[n] for n, v in signed.items() if v > 0)
            if not isclose(consumed, produced, rel_tol=1e-12):
                raise ValueError(
                    f"reaction {self.name}: molar masses do not balance"
                )

    def to_dict(self):
        """Serialize reaction orders and rate parameters."""
        rate = {
            "type": self.law,
            "units": _units(sum(v for _, v in self.orders)),
        }
        if self.law == "reactive_uptake":
            rate = {"type": self.law, "gamma": self.coefficient}
        elif self.law == "constant":
            rate["k"] = self.coefficient
        else:
            rate.update(
                A=self.coefficient,
                b=self.exponent,
                Ea_J_per_mol=self.energy,
                T_ref_K=self.reference_temperature,
            )
        return {
            "id": self.name,
            "reactants": dict(self.reactants),
            "products": dict(self.products),
            "orders": dict(self.orders),
            "rate": rate,
        }


def _check_phases(reaction):
    """Restrict the kinetic collision law to one gas and particle products."""
    if reaction.law == "reactive_uptake":
        if (
            len(reaction.reactants) != 1
            or not reaction.reactants[0][0].startswith("gas:")
            or reaction.reactants[0][1] != 1
            or reaction.orders != reaction.reactants
            or not all(
                name.startswith("particle:") for name, _ in reaction.products
            )
        ):
            raise ValueError(
                "reactive_uptake consumes one gas and forms particle products"
            )
    else:
        phases = {
            name.split(":", 1)[0]
            for name, _ in (
                *reaction.reactants,
                *reaction.products,
                *reaction.orders,
            )
        }
        if len(phases) != 1:
            raise ValueError(
                "mass action requires a single concentration phase"
            )


def _catalog(catalog, phases):
    if not isinstance(catalog, dict) or not catalog:
        raise ValueError("species must be a nonempty mapping")
    species = []
    for name, meta in catalog.items():
        if (
            not isinstance(name, str)
            or name.partition(":")[0] not in phases
            or not name.partition(":")[2].strip()
        ):
            raise ValueError(f"invalid {'/'.join(phases)} species {name!r}")
        _fields(meta, (), ("elements", "molar_mass_kg_per_mol"))
        elements = meta.get("elements", {})
        if not isinstance(elements, dict) or any(
            not isinstance(key, str)
            or not key.isalpha()
            or type(v) is not int
            or v <= 0
            for key, v in elements.items()
        ):
            raise ValueError(f"invalid elemental composition for {name}")
        species.append((name, tuple(sorted(elements.items()))))
    return species


def _declared_masses(catalog):
    declared_masses = []
    for name, meta in catalog.items():
        if name.startswith("particle:") and "molar_mass_kg_per_mol" not in meta:
            raise ValueError(f"{name} requires molar_mass_kg_per_mol")
        if "molar_mass_kg_per_mol" in meta:
            mass = _number(meta["molar_mass_kg_per_mol"], f"{name} molar mass")
            if mass <= 0:
                raise ValueError(f"{name} molar mass must be positive")
            declared_masses.append((name, mass))
    return tuple(sorted(declared_masses))


@dataclass(frozen=True, init=False)
class ReactionMechanism:
    """Closed reaction mechanism, independent of storage order and solver.

    Construct with a version-1 Python mapping or load its JSON/YAML equivalent.
    Version 2 admits explicit particle materials and reactive gas uptake.
    """

    species: tuple
    reactions: tuple[Reaction, ...]
    declared_masses: tuple
    version: int

    def __init__(self, record: dict):
        """Validate the versioned fields and copy the reaction definitions."""
        _fields(record, ("version", "basis", "species", "reactions"))
        version = record["version"]
        if type(version) is not int or version not in (1, 2):
            raise ValueError("unsupported native mechanism version")
        basis = "mol/m^3-air" if version == 1 else "mol/m^3-phase"
        if record["basis"] != basis:
            raise ValueError(f"version {version} basis must be {basis}")
        object.__setattr__(self, "version", version)
        phases = ("gas",) if version == 1 else ("gas", "particle")
        species = _catalog(record["species"], phases)
        object.__setattr__(
            self, "declared_masses", _declared_masses(record["species"])
        )
        catalog = dict(species)
        known_masses = dict(self.declared_masses)
        if not isinstance(record["reactions"], list) or not record["reactions"]:
            raise ValueError("reactions must be a nonempty list")
        reactions = []
        seen = set()
        for idx, item in enumerate(record["reactions"]):
            try:
                reaction = Reaction.parse(item, catalog)
                reaction.validate_masses(
                    reaction.signed_coefficients(), known_masses
                )
                if version == 2:
                    _check_phases(reaction)
                elif reaction.law == "reactive_uptake":
                    raise ValueError("reactive_uptake requires version 2")
                if reaction.name in seen:
                    raise ValueError(f"duplicate reaction id {reaction.name!r}")
            except ValueError as exc:
                label = item.get("id", idx) if isinstance(item, dict) else idx
                raise ValueError(f"reaction {label!r}: {exc}") from exc
            seen.add(reaction.name)
            reactions.append(reaction)
        object.__setattr__(self, "species", tuple(sorted(species)))
        object.__setattr__(self, "reactions", tuple(reactions))

    @classmethod
    def from_dict(cls, record):
        """Validate and detach native Python mechanism data."""
        return cls(record)

    def to_dict(self):
        """Return a detached, normalized representation for interchange."""
        catalog = {
            name: {"elements": dict(elements)} if elements else {}
            for name, elements in self.species
        }
        for name, mass in self.declared_masses:
            catalog[name]["molar_mass_kg_per_mol"] = mass
        return {
            "version": self.version,
            "basis": "mol/m^3-air" if self.version == 1 else "mol/m^3-phase",
            "species": catalog,
            "reactions": [r.to_dict() for r in self.reactions],
        }

    @classmethod
    def load(cls, path):
        """Read a native .json, .yaml or .yml file with unique mapping keys."""
        return cls(_read_mapping(path))

    def save(self, path):
        """Write a normalized native mechanism file."""
        path = Path(path)
        if path.suffix.lower() == ".json":
            value = json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n"
        elif path.suffix.lower() in (".yaml", ".yml"):
            value = yaml.safe_dump(self.to_dict(), sort_keys=False)
        else:
            raise ValueError("mechanism file must use .json, .yaml or .yml")
        path.write_text(value, encoding="utf-8")

    def bind(self, gas, overrides=None):
        """Resolve gas identities using exact names and explicit overrides."""
        from particula.dynamics.chemistry.binding import BoundMechanism

        if self.version != 1:
            raise ValueError("use bind_resolved for a version 2 mechanism")
        return BoundMechanism.from_gas(self, gas, overrides)

    def bind_resolved(self, aerosol, particle_columns, gas_overrides=None):
        """Bind explicit material columns and gas names in resolved state."""
        from particula.dynamics.chemistry.resolved_binding import (
            ResolvedBinding,
        )

        if self.version != 2:
            raise ValueError("resolved chemistry requires version 2")
        return ResolvedBinding.create(
            self, aerosol, particle_columns, gas_overrides
        )
