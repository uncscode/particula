"""Independent checks of the research comparison's numerical oracles."""

import numpy as np
import numpy.testing as npt
from scipy.integrate import quad
from scripts.assess_native_condensation import (
    DENSITY,
    DIFFUSION,
    GAS_CONSTANT,
    MASS,
    MOLAR_MASS,
    NUMBER,
    PRESSURE,
    RADIUS,
    TEMPERATURE,
    VAPOR,
    VISCOSITY,
    CondensationIsothermal,
    fixed_run,
    reference,
)
from scripts.assess_process_splitting import assess


def test_growth_reference_against_separated_quadrature():
    """An inverse radius integral independently fixes the terminal radius."""
    mean_free_path = (2 * VISCOSITY / PRESSURE) / np.sqrt(
        8 * MOLAR_MASS / (np.pi * GAS_CONSTANT * TEMPERATURE)
    )

    def inverse(x):
        knudsen = mean_free_path / (RADIUS * x)
        correction = (
            0.75 * (1 + knudsen) / (knudsen**2 + 1.283 * knudsen + 0.75)
        )
        vapor = VAPOR - NUMBER * MASS * (x**3 - 1)
        return DENSITY * RADIUS**2 * x / (DIFFUSION * correction * vapor)

    duration, error = quad(inverse, 1, 1.2, epsabs=1e-12, epsrel=1e-12)
    assert error < 1e-10
    radii, uncertainty = reference(duration, 0.0, np.array([0, duration]))
    npt.assert_allclose(radii, [1, 1.2], rtol=0, atol=1e-10)
    assert uncertainty < 1e-9


def test_native_growth_refinement_and_budget():
    """Native explicit steps converge while preserving closed material."""
    coarse = fixed_run(CondensationIsothermal, 1.4, 0.0, 16)
    fine = fixed_run(CondensationIsothermal, 1.4, 0.0, 32)
    assert fine["radius_error"] < 0.6 * coarse["radius_error"]
    assert fine["relative_budget_error"] < 2e-14


def test_noncommuting_split_orders():
    """Exact component maps expose first versus second order composition."""
    finest = {row["method"]: row for row in assess() if row["steps"] == 128}
    for method in ("sequential", "reversed"):
        assert 0.98 < finest[method]["observed_order"] < 1.02
    assert 1.98 < finest["strang"]["observed_order"] < 2.02
    assert finest["strang"]["maximum_trajectory_error"] < 2e-5


def test_patankar_linear_zero_species_and_conservation():
    """Zero donors use linear limits while transfer retains total inventory."""
    from scripts.assess_conservative_transfer import assess, patankar_step

    matrix = np.array(
        [[-1000.0, 1.0, 0.0], [1000.0, -1.0, 0.0], [0.0, 0.0, 0.0]]
    )
    result = patankar_step(matrix, np.array([1.0, 0.0, 0.0]), 100.0)
    assert np.all(result >= 0)
    npt.assert_allclose(result.sum(), 1, rtol=2e-14)
    assert result[2] == 0
    smooth = [row for row in assess() if row["case"] == "smooth"]
    assert 1.9 < smooth[-1]["observed_order"] < 2.1


def test_local_error_reference_from_noninitial_state():
    """An independent inverse integral checks local physical mass/gas units."""
    from scripts.assess_native_adaptive import VISCOSITY, local_reference

    old_radius, new_radius = 1.1, 1.2
    old_gas = VAPOR - NUMBER * MASS * (old_radius**3 - 1)
    mean_free_path = (2 * VISCOSITY / PRESSURE) / np.sqrt(
        8 * MOLAR_MASS / (np.pi * GAS_CONSTANT * TEMPERATURE)
    )

    def inverse(x):
        kn = mean_free_path / (RADIUS * x)
        correction = 0.75 * (1 + kn) / (kn**2 + 1.283 * kn + 0.75)
        gas = VAPOR - NUMBER * MASS * (x**3 - 1)
        return DENSITY * RADIUS**2 * x / (DIFFUSION * correction * gas)

    duration, _ = quad(
        inverse, old_radius, new_radius, epsabs=1e-12, epsrel=1e-12
    )
    values, uncertainty = local_reference(old_radius, old_gas, duration, 0.0)
    expected = np.array(
        [MASS * new_radius**3, VAPOR - NUMBER * MASS * (new_radius**3 - 1)]
    )
    npt.assert_allclose(values, expected, rtol=1e-10)
    assert np.max(uncertainty / expected) < 1e-10
