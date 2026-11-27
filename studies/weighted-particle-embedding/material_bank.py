"""Reproducible synthetic material mixtures with a fixed water lane."""

import numpy as np
from vapor_properties import Materials, materials


def mixture(count, chemistry):
    """Specify neutral materials across a range of kinetic properties."""
    if not 2 <= count <= 64:
        raise ValueError(
            "Use two through sixty-four simultaneous volatile species"
        )
    water = materials(1)
    rows = []
    for j in range(1, count):
        rng = np.random.default_rng([int(chemistry), j, 3361])
        rows.append(
            (
                rng.uniform(0.07, 0.21),
                rng.uniform(900.0, 1550.0),
                np.exp(rng.uniform(np.log(0.2), np.log(45.0))),
                np.exp(rng.uniform(np.log(3e-6), np.log(1.2e-5))),
                np.exp(rng.uniform(np.log(0.025), np.log(0.6))),
            )
        )
    mol, rho, pressure, diffusion, alpha = np.asarray(rows).T
    return Materials(
        ("water", *(f"organic_{j}" for j in range(1, count))),
        np.r_[water.molar[0], mol, water.molar[1:]],
        np.r_[water.density[0], rho, water.density[1:]],
        np.r_[water.pressure, pressure],
        np.r_[water.diffusivity, diffusion],
        np.r_[water.accommodation, alpha],
    )
