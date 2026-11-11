"""Native coagulation examples preserve units and physical trends."""

import runpy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

EXAMPLES = (
    Path(__file__).resolve().parents[1] / "docs/Examples/Dynamics/Coagulation"
)


@pytest.mark.parametrize("index,kind", [(1, "pmf"), (2, "pdf")])
def test_binned_coagulation_number_loss_and_accounting(
    monkeypatch, index, kind
):
    """Independent radius integrals distinguish PDF density from bin counts."""
    monkeypatch.setattr(plt, "show", lambda: None)
    result = runpy.run_path(
        str(EXAMPLES / f"Coagulation_{index}_{kind.upper()}_Pattern.py")
    )
    particles = result[f"{kind}_aerosol"].particles
    radius = result["radius_bins"]
    before = result[f"{kind}_before"]
    after = result[f"{kind}_after_step_2"]
    particle_mass = 4.0 / 3.0 * np.pi * radius**3 * 1000.0
    if kind == "pdf":
        initial_number = np.trapezoid(before, radius)
        final_number = np.trapezoid(after, radius)
        final_mass = np.trapezoid(after * particle_mass, radius)
    else:
        initial_number = np.sum(before)
        final_number = np.sum(after)
        final_mass = np.sum(after * particle_mass)
    np.testing.assert_allclose(initial_number, 1e10, rtol=1e-12)
    assert 0 < final_number < initial_number
    assert np.all(np.isfinite(after))
    assert np.all(after >= 0)
    np.testing.assert_allclose(
        particles.species_mass_density[0, 0],
        final_mass,
        rtol=1e-12,
        atol=1e-30,
    )
    np.testing.assert_allclose(
        particles.masses[0, :, 0],
        particle_mass,
        rtol=1e-12,
        atol=1e-30,
    )
    plt.close("all")
