"""Import the closed single-gas-phase subset of NCAR 1.3.0 mechanisms."""

import argparse
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile

from particula.dynamics.chemistry.mechanism import (
    ReactionMechanism,
    _fields,
    _number,
    _read_mapping,
    _units,
)
from particula.util.constants import AVOGADRO_NUMBER, GAS_CONSTANT


@contextmanager
def _at(location):
    try:
        yield
    except ValueError as exc:
        raise ValueError(f"{location}: {exc}") from exc


def _entries(records, location, label="name") -> Iterator[dict]:
    if not isinstance(records, list) or not records:
        raise ValueError(f"{location} must be a nonempty list")
    seen = set()
    for index, item in enumerate(records):
        if not isinstance(item, dict):
            raise ValueError(f"{location}[{index}] must be a mapping")
        name = item.get(label)
        if not isinstance(name, str) or not name.strip() or name in seen:
            raise ValueError(
                f"{location}[{index}]: invalid or duplicate {label} {name!r}"
            )
        seen.add(name)
        yield item


def _catalog(records, phases):
    weights_key = "molecular weight [kg mol-1]"
    catalog = {}
    for item in _entries(records, "species"):
        name = item["name"]
        with _at(f"species {name!r}"):
            _fields(item, ("name",), (weights_key,))
            metadata = {}
            if weights_key in item:
                mass = _number(item[weights_key], weights_key)
                if mass <= 0:
                    raise ValueError(f"{weights_key} must be positive")
                metadata["molar_mass_kg_per_mol"] = mass
            catalog[f"gas:{name}"] = metadata
    phases = list(_entries(phases, "phases"))
    if len(phases) != 1:
        raise ValueError("phases must contain a single gas phase")
    phase = phases[0]
    _fields(phase, ("name", "species"))
    members = set()
    for item in _entries(phase["species"], "phase species"):
        with _at(f"phase species {item['name']!r}"):
            _fields(item, ("name",))
        members.add("gas:" + item["name"])
    if members != catalog.keys():
        raise ValueError("phase species must match the species catalog")
    return catalog, phase["name"]


def _side(records, side, catalog):
    terms = {}
    for item in _entries(records, side, "species name"):
        name = "gas:" + item["species name"]
        with _at(f"{side} {name}"):
            _fields(item, ("species name",), ("coefficient",))
            if name not in catalog:
                raise ValueError("unknown species")
            count = _number(item.get("coefficient", 1), "coefficient")
            if (
                not count.is_integer()
                or count < 1
                or (side == "reactants" and count != 1)
            ):
                raise ValueError("unsupported stoichiometric coefficient")
        terms[name] = int(count)
    return terms


def _reaction(item, catalog, phase):
    _fields(
        item,
        ("type", "name", "gas phase", "reactants", "products"),
        ("A", "B", "C", "D", "E", "Ea"),
    )
    if item["type"] != "ARRHENIUS" or item["gas phase"] != phase:
        raise ValueError("expected ARRHENIUS in the declared gas phase")
    if _number(item.get("E", 0), "E") != 0:
        raise ValueError("nonzero E requires a pressure-dependent rate")
    if "C" in item and "Ea" in item:
        raise ValueError("C and Ea are mutually exclusive")
    lhs = _side(item["reactants"], "reactants", catalog)
    rhs = _side(item["products"], "products", catalog)
    if len(lhs) not in (1, 2):
        raise ValueError("only first- and second-order reactants are supported")
    energy = (
        AVOGADRO_NUMBER * _number(item["Ea"], "Ea")
        if "Ea" in item
        else -GAS_CONSTANT * _number(item.get("C", 0), "C")
    )
    return {
        "id": item["name"],
        "reactants": lhs,
        "products": rhs,
        "rate": {
            "type": "arrhenius",
            "A": _number(item.get("A", 1), "A"),
            "b": _number(item.get("B", 0), "B"),
            "Ea_J_per_mol": energy,
            "T_ref_K": _number(item.get("D", 300), "D"),
            "units": _units(len(lhs)),
        },
    }


def import_ncar(path) -> ReactionMechanism:
    """Read an NCAR 1.3.0 JSON or YAML file into a native gas mechanism.

    Gas concentrations are mol/m³ of air. Optional source molecular weights
    (kg/mol) are retained for validation when binding to simulation species.
    """
    data = _read_mapping(path)
    _fields(data, ("version", "name", "species", "phases", "reactions"))
    if not isinstance(data["name"], str) or not data["name"].strip():
        raise ValueError("mechanism name must be a nonempty string")
    if data["version"] != "1.3.0":
        raise ValueError("expected NCAR version 1.3.0")
    catalog, phase = _catalog(data["species"], data["phases"])
    reactions = []
    for item in _entries(data["reactions"], "reactions"):
        with _at(f"reaction {item['name']!r}"):
            reactions.append(_reaction(item, catalog, phase))
    return ReactionMechanism.from_dict(
        {
            "version": 1,
            "basis": "mol/m^3-air",
            "species": catalog,
            "reactions": reactions,
        }
    )


def convert_ncar(source, destination):
    """Validate and atomically write native JSON or YAML at destination."""
    mechanism = import_ncar(source)
    target = Path(destination)
    if Path(source).resolve() == target.resolve():
        raise ValueError("source and destination must be distinct")
    with NamedTemporaryFile(
        dir=target.parent, suffix=target.suffix, delete=False
    ) as temp:
        temporary = Path(temp.name)
    try:
        mechanism.save(temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return mechanism


def main():
    """Convert a supported NCAR file to inspectable native JSON or YAML."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    try:
        convert_ncar(args.source, args.destination)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
