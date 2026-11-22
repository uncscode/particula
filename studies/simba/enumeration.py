"""Find the published finalists given compositional priors."""

import json
import sys
from pathlib import Path

from particula.dynamics.chemistry.discovery import (
    balanced_steps,
    enumerate_networks,
)
from published_rankings import TABLES

# Each list is ordered by A, B, ... . The hidden-species budgets are supplied
# hypotheses; candidate equations are generated without using table rows.
# Hypothetical: conserved A equivalents. Aldol: elemental C/H/O. Fructose:
# molecule as a retained HMF skeleton and three removable water equivalents.
A, B, C, D = (8, 8, 1), (7, 6, 1), (15, 12, 1), (0, 2, 1)
ADDUCT = (15, 14, 2)
BUDGETS = {
    "hypoth": [
        [(1,), (2,), (2,)],
        [(1,), (2,), (2,), (2,)],
        [(1,), (2,), (2,), (1,), (1,)],
        [(1,), (2,), (2,), (2,), (1,), (1,)],
    ],
    "aldol_condensation": [
        [A, B, C, D],
        [A, B, C, D, A],
        [A, B, C, D, A, ADDUCT],
        [A, B, C, D, A, C, D],
    ],
    "fruc_HMF2": [
        [(1, 3), (0, 1), (1, 0), (1, 2), (1, 1)],
        [(1, 3), (0, 1), (1, 0), (1, 3), (1, 2), (1, 1)],
        [(1, 3), (0, 1), (1, 0), (1, 2), (1, 1), (0, 1), (1, 1)],
    ],
}


def enumerate_case(case, iteration):
    """Search up to each reported complexity and check table membership."""
    atoms = BUDGETS[case][iteration - 1]
    reactions, _ = TABLES[case][iteration - 1]
    sides = [w.split(">") for w in reactions.split()]
    columns = "ABCDEFG"[: len(atoms)]
    target_network = tuple(
        sorted(
            tuple(b.count(c) - a.count(c) for c in columns) for a, b in sides
        )
    )
    net = {
        "hypoth": (-4, 1, 1),
        "aldol_condensation": (-1, -1, 1, 1),
        "fruc_HMF2": (-1, 3, 1),
    }[case]
    net = net + (0,) * (len(atoms) - len(net))
    bank = [
        step
        for step in balanced_steps(["gas"] * len(atoms), atoms)
        if all(
            not (v > 0 and n < 0 or v < 0 and n > 0)
            for v, n in zip(step, net, strict=True)
        )
    ]
    result = enumerate_networks(
        bank,
        [n < 0 for n in net],
        [i for i, n in enumerate(net) if n > 0],
        max_steps=len(sides),
        max_trials=2_000_000,
        max_candidates=500_000,
        net=net,
    )
    found = target_network in result.networks
    if not found:
        raise AssertionError(
            (case, iteration, bank, result.examined, result.limit)
        )
    return {
        "case": case,
        "iteration": iteration,
        "species_budgets": atoms,
        "allowed_steps": len(bank),
        "networks_returned": len(result.networks),
        "examined": result.examined,
        "stopped_at": result.limit,
        "table_finalist_found": found,
    }


if __name__ == "__main__":
    rows = [
        enumerate_case(case, i)
        for case, table in TABLES.items()
        for i in range(1, len(table) + 1)
    ]
    Path(sys.argv[1]).write_text(json.dumps(rows, indent=2) + "\n")
