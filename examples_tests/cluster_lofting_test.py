"""Reconstruct the published group budgets from individual particle outcomes."""

import json
from pathlib import Path

import numpy as np


def test_resolved_species_and_final_destinations():
    """Each ID contributes one physical particle and its own component masses."""
    path = (
        Path(__file__).resolve().parents[1]
        / "website/results/cluster-lofting/data.json"
    )
    runs = json.loads(path.read_text())
    ids = runs["row"]["release_catalog"]["id"]
    assert len(set(ids)) == len(ids) == 1024

    for case in ("stack", "row", "block"):
        run = runs[case]
        catalog = run["release_catalog"]
        assert catalog["id"] == ids
        sizes = np.asarray(catalog["radius_index"])
        recipes = np.asarray(catalog["recipe_index"])
        np.testing.assert_array_equal(
            np.bincount(4 * sizes + recipes), np.full(16, 64)
        )
        radii = 1e-6 * np.asarray(run["radii_um"])[sizes]
        fractions = np.asarray(run["volume_fractions"])[recipes]
        component_mass = (
            (4 * np.pi / 3 * radii**3)[:, None]
            * fractions
            * run["material_densities_kg_m3"]
        )
        final = run["final"]
        destinations = np.array(
            [
                "outflow" if exited else run["groups"][cell]
                for cell, exited in zip(
                    final["cell"], final["exited"], strict=True
                )
            ]
        )
        for group, budget in zip(
            run["budget_groups"], run["budget_history"][-1], strict=True
        ):
            selected = destinations == group
            assert selected.sum() == budget[0]
            np.testing.assert_allclose(
                component_mass[selected].sum(axis=0),
                budget[1:],
                rtol=1e-13,
                atol=0,
            )

    row = runs["row"]
    changes = np.asarray(row["final_displacement_by_radius_recipe_m"])
    height = row["displacement_fields"].index("height_world_m")
    largest = row["radii_um"].index(10)
    assert changes[largest, row["recipes"].index("Sucrose"), height] > 0
    assert changes[largest, row["recipes"].index("Sodium chloride"), height] < 0
