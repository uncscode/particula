"""Native, element-balanced reaction discovery for specified chamber hypotheses."""

from pathlib import Path
import json
import numpy as np
from particula.dynamics.chemistry.discovery import (
    balanced_steps,
    compile_network,
    enumerate_networks,
)

HERE = Path(__file__).resolve().parent
NAMES = (
    "NaCl",
    "readyNaCl",
    "NaHSO3",
    "NaHSO4",
    "Na2SO4",
    "H2SO3",
    "Oeq",
    "H2SO4",
    "HCl",
)
ATOMS = ("Na", "Cl", "S", "H", "O")
FORMULA = np.array(
    [
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 0, 1, 1, 3],
        [1, 0, 1, 1, 4],
        [2, 0, 1, 0, 4],
        [0, 0, 1, 2, 3],
        [0, 0, 0, 0, 1],
        [0, 0, 1, 2, 4],
        [0, 1, 0, 1, 0],
    ]
)
# Oeq is an oxygen atom balance equivalent; no atomic O concentration is assumed.
MASSES = FORMULA @ np.array([22.98977, 35.453, 32.065, 1.00794, 15.9994]) / 1000
SPECIES = {
    f"particle:{name}": {
        "elements": {k: int(v) for k, v in zip(ATOMS, row) if v},
        "molar_mass_kg_per_mol": float(mass),
    }
    for name, row, mass in zip(NAMES, FORMULA, MASSES)
}


def step(lhs, rhs):
    return tuple(rhs.count(n) - lhs.count(n) for n in NAMES)


CHANNELS = {
    "direct": step(["NaCl", "H2SO4"], ["NaHSO4", "HCl"]),
    "capture": step(["NaCl", "H2SO3"], ["NaHSO3", "HCl"]),
    "oxidize": step(["NaHSO3", "Oeq"], ["NaHSO4"]),
    "expose": step(["NaCl"], ["readyNaCl"]),
    "react_exposed": step(["readyNaCl", "H2SO4"], ["NaHSO4", "HCl"]),
    "neutralize": step(["NaHSO4", "NaCl"], ["Na2SO4", "HCl"]),
}
CANDIDATES = {
    "direct": ["direct"],
    "chloride_activity": ["direct"],
    "access_delay": ["expose", "react_exposed"],
    "siv_lag": ["capture", "oxidize"],
    "siv_parallel": ["direct", "capture", "oxidize"],
    "siv_sink": ["direct", "capture"],
    "neutral": ["direct", "neutralize"],
    "lag_neutral": ["capture", "oxidize", "neutralize"],
    "parallel_neutral": ["direct", "capture", "oxidize", "neutralize"],
    "activity_neutral": ["direct", "neutralize"],
    "retention": ["direct", "expose"],
    "retention_neutral": ["direct", "expose", "neutralize"],
    "retention_activity": ["direct", "expose"],
}


def catalog():
    generated = set(balanced_steps(["particle"] * len(NAMES), FORMULA.tolist()))
    assert set(CHANNELS.values()) <= generated
    search = enumerate_networks(
        list(CHANNELS.values()),
        initial=[n in ["NaCl", "H2SO3", "Oeq", "H2SO4"] for n in NAMES],
        products=[3],
        max_steps=4,
        max_trials=100,
        max_candidates=100,
    )
    assert search.limit is None
    selected = {
        tuple(sorted(CHANNELS[k] for k in names)) for names in CANDIDATES.values()
    }
    assert selected <= set(search.networks)
    banks = {}
    reverse = {row: k for k, row in CHANNELS.items()}
    for number, network in enumerate(search.networks):
        native = compile_network(network, SPECIES, [1.0] * len(network))
        assert not np.any(np.asarray(network) @ FORMULA)
        names = [reverse[row] for row in network]
        doc = native.to_dict()
        for label, rx in zip(names, doc["reactions"]):
            rx["id"] = label
        banks["+".join(sorted(names))] = doc
    return dict(
        species=NAMES,
        generated_balanced_steps=len(generated),
        screened_channels=CHANNELS,
        examined=search.examined,
        reachable=len(banks),
        search_limit=search.limit,
        candidates=CANDIDATES,
        mechanisms=banks,
    )


def sodium_stoichiometry():
    bank = catalog()["mechanisms"]
    from particula.dynamics.chemistry import ReactionMechanism

    rows = {}
    for doc in bank.values():
        for r in ReactionMechanism.from_dict(doc).reactions:
            co = r.signed_coefficients()
            rows[r.name] = [co.get("particle:" + n, 0) for n in NAMES[:5]]
    return np.array([rows[n] for n in CHANNELS], dtype=float)


if __name__ == "__main__":
    (HERE / "native-networks.json").write_text(json.dumps(catalog(), indent=2) + "\n")
    print(sodium_stoichiometry())
