"""Scientific regression checks for migrated native condensation examples."""

import runpy
from pathlib import Path
from unittest.mock import patch

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def test_radius_bin_example_evaporation_and_species_inventory():
    """The undersaturated example evaporates without losing material."""
    with patch.object(plt, "show"):
        values = runpy.run_path(
            str(
                ROOT
                / "docs/examples/processes/condensation/condensation_1_bin.py"
            )
        )
    plt.close("all")
    aerosol = values["aerosol"]
    particles, gas = aerosol.particles, aerosol.gas
    masses = particles.masses.copy()
    gas_before = gas.concentration.copy()
    inventory = (
        np.sum(masses * particles.concentration[..., None], axis=1) + gas_before
    )
    assert values["rate"].shape == (1, 500, 1)
    assert np.all(values["rate"] < 0)
    values["condensation_process"].execute(aerosol, time_step=1e-9, sub_steps=4)
    assert aerosol.particles is particles and aerosol.gas is gas
    assert np.any(particles.masses < masses)
    assert np.all(gas.concentration >= gas_before)
    after = (
        np.sum(particles.masses * particles.concentration[..., None], axis=1)
        + gas.concentration
    )
    np.testing.assert_allclose(after, inventory, rtol=1e-12, atol=1e-30)
    assert np.all(particles.masses >= 0)
