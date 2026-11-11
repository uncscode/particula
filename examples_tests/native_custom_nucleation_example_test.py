"""Volume-aware accounting for the illustrative native custom source."""

import runpy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

EXAMPLE = (
    Path(__file__).resolve().parents[1]
    / "docs/Examples/Nucleation/Notebooks/Custom_Nucleation_Single_Species.py"
)


def test_custom_source_preserves_extensive_mass_and_capacity(monkeypatch):
    """Two unit-count nuclei debit their mass divided by physical volume."""
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    example = runpy.run_path(str(EXAMPLE))
    aerosol = example["build_aerosol"]()
    volume = aerosol.particles.volume[0]
    nucleus_mass = example["mass_nucleated_particle"]
    gas_before = aerosol.gas.concentration.copy()
    counts_before = aerosol.particles.concentration.sum()
    particle_before = np.sum(
        aerosol.particles.masses * aerosol.particles.concentration[..., None]
    )
    count = example["inject_custom_nuclei"](
        aerosol, 2.5 * nucleus_mass / volume, 1.0, nucleus_mass
    )
    assert count == 2
    assert aerosol.particles.concentration.sum() == counts_before + 2
    np.testing.assert_allclose(
        aerosol.gas.concentration,
        gas_before - 2 * nucleus_mass / volume,
        rtol=1e-12,
        atol=1e-30,
    )
    particle_after = np.sum(
        aerosol.particles.masses * aerosol.particles.concentration[..., None]
    )
    np.testing.assert_allclose(
        particle_after + aerosol.gas.concentration[0, 0] * volume,
        particle_before + gas_before[0, 0] * volume,
        rtol=1e-12,
        atol=1e-30,
    )
    # Capacity rejection preserves both gas and particle records.
    aerosol.particles.masses[:] = nucleus_mass
    aerosol.particles.concentration[:] = 1.0
    gas_before = aerosol.gas.concentration.copy()
    masses_before = aerosol.particles.masses.copy()
    with pytest.raises(ValueError, match="free slot capacity"):
        example["inject_custom_nuclei"](
            aerosol, 2.5 * nucleus_mass / volume, 1.0, nucleus_mass
        )
    np.testing.assert_array_equal(aerosol.gas.concentration, gas_before)
    np.testing.assert_array_equal(aerosol.particles.masses, masses_before)
    plt.close("all")
