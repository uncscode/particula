"""Enumerate reachable, element-balanced ways of adding one condensed sulfate."""

import json
from pathlib import Path
import numpy as np
from particula.dynamics.chemistry.discovery import (
    balanced_steps,
    enumerate_networks,
)

HERE = Path(__file__).resolve().parent
SPECIES = ("SO2", "H2O", "O3", "H2O2", "H2SO3", "H2SO4", "O2")
ATOMS = [
    (1, 0, 2),
    (0, 2, 1),
    (0, 0, 3),
    (0, 2, 2),
    (1, 2, 3),
    (1, 2, 4),
    (0, 0, 2),
]


def equation(row):
    def side(sign):
        return " + ".join(
            (str(abs(n)) if abs(n) > 1 else "") + name
            for name, n in zip(SPECIES, row)
            if sign * n > 0
        )

    return side(-1) + " → " + side(1)


def enumerate_bank():
    raw = balanced_steps(["particle"] * len(SPECIES), ATOMS)
    # Admit one sulfur carrier in and one sulfur carrier out at each step.
    # Sulfur enters as dissolved SO2 and retained(VI) is terminal.
    bank = [
        r
        for r in raw
        if sum(max(-v, 0) * a[0] for v, a in zip(r, ATOMS)) == 1
        and r[0] <= 0
        and r[5] >= 0
        and r[6] >= 0
        and r[2] <= 0
        and r[3] <= 0
    ]
    result = {}
    for name, target in {
        "ozone": (-1, -1, -1, 0, 0, 1, 1),
        "peroxide": (-1, 0, 0, -1, 0, 1, 0),
    }.items():
        search = enumerate_networks(
            bank,
            [v < 0 or i == 1 for i, v in enumerate(target)],
            [5],
            max_steps=3,
            max_trials=10000,
            max_candidates=3000,
            net=target,
        )
        assert search.networks and search.limit is None
        for network in search.networks:
            assert np.all(np.array(network) @ np.asarray(ATOMS) == 0)
        result[name] = dict(
            net=equation(target),
            examined=search.examined,
            networks=[
                [equation(row) for row in network]
                for network in search.networks
            ],
        )
    return dict(
        species=SPECIES,
        atom_order=["S", "H", "O"],
        atoms=ATOMS,
        balanced_steps=len(raw),
        eligible=[equation(x) for x in bank],
        families=result,
    )


if __name__ == "__main__":
    (HERE / "networks.json").write_text(
        json.dumps(enumerate_bank(), indent=2) + "\n"
    )
