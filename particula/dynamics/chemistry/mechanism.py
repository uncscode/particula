"""Versioned native mechanism records for irreversible gas mass action."""

import json
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

import yaml


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
        if rate["units"] != expected:
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

    def to_dict(self):
        """Serialize reaction orders and rate parameters."""
        rate = {
            "type": self.law,
            "units": _units(sum(v for _, v in self.orders)),
        }
        if self.law == "constant":
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


def _catalog(catalog):
    if not isinstance(catalog, dict) or not catalog:
        raise ValueError("species must be a nonempty mapping")
    species = []
    for name, meta in catalog.items():
        if (
            not isinstance(name, str)
            or not name.startswith("gas:")
            or not name[4:].strip()
        ):
            raise ValueError(f"invalid gas species {name!r}")
        _fields(meta, (), ("elements",))
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


@dataclass(frozen=True, init=False)
class ReactionMechanism:
    """Closed gas mechanism, independent of storage order and solver.

    Construct with a version-1 Python mapping or load its JSON/YAML equivalent.
    Species IDs use the gas: prefix. Concentrations are mol/m³ of air.
    """

    species: tuple
    reactions: tuple[Reaction, ...]

    def __init__(self, record: dict):
        """Validate the v1 fields and copy the reaction definitions."""
        _fields(record, ("version", "basis", "species", "reactions"))
        if type(record["version"]) is not int or record["version"] != 1:
            raise ValueError("unsupported native mechanism version")
        if record["basis"] != "mol/m^3-air":
            raise ValueError("basis must be mol/m^3-air")
        species = _catalog(record["species"])
        catalog = dict(species)
        if not isinstance(record["reactions"], list) or not record["reactions"]:
            raise ValueError("reactions must be a nonempty list")
        reactions = []
        seen = set()
        for idx, item in enumerate(record["reactions"]):
            try:
                reaction = Reaction.parse(item, catalog)
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
        return {
            "version": 1,
            "basis": "mol/m^3-air",
            "species": {
                name: {"elements": dict(elements)} if elements else {}
                for name, elements in self.species
            },
            "reactions": [r.to_dict() for r in self.reactions],
        }

    @classmethod
    def load(cls, path):
        """Read a native .json, .yaml or .yml file with unique mapping keys."""
        path = Path(path)
        with path.open(encoding="utf-8") as stream:
            if path.suffix.lower() == ".json":
                return cls(json.load(stream, object_pairs_hook=_unique))
            if path.suffix.lower() in (".yaml", ".yml"):
                loader = _MechanismLoader(stream)
                try:
                    return cls(loader.get_single_data())
                finally:
                    loader.dispose()
        raise ValueError("mechanism file must use .json, .yaml or .yml")

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

        return BoundMechanism.from_gas(self, gas, overrides)
