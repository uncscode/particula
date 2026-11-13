"""Numerical checks for the native foundational tutorials."""

import runpy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def test_aerosol_tutorial_population_and_held_state():
    """Retain the prescribed lognormal number, mass and saturation state."""
    values = runpy.run_path(
        str(ROOT / "docs/examples/fundamentals/native_aerosol.py")
    )
    aerosol = values["aerosol"]
    particles = values["particles"]
    assert aerosol.particles is particles
    assert aerosol.gas is values["glycerol_gas"]
    assert aerosol.environment is values["environment"]
    assert particles.distribution_type == "discrete"
    np.testing.assert_allclose(particles.concentration.sum(), 1e10, rtol=1e-12)
    # Independent third moment of the prescribed lognormal radius population.
    expected_mass = (
        1e10
        * 4
        * np.pi
        / 3
        * 1260
        * (100e-9) ** 3
        * np.exp(4.5 * np.log(1.5) ** 2)
    )
    actual_mass = np.sum(particles.masses[:, :, 0] * particles.concentration)
    # The published 250-point grid approximates the continuous third moment.
    np.testing.assert_allclose(actual_mass, expected_mass, rtol=0.01)
    np.testing.assert_array_equal(particles.volume, [1.0])
    np.testing.assert_allclose(
        aerosol.gas.concentration, [[values["sat_concentration"] * 0.5]]
    )
    assert aerosol.gas.name == ["Glycerol"]
    np.testing.assert_array_equal(aerosol.environment.temperature, [298.15])


def test_atmosphere_tutorial_prescribed_history():
    """Retain full gas order and the independent water altitude diagnostic."""
    import matplotlib

    matplotlib.use("Agg")
    from unittest.mock import patch

    import matplotlib.pyplot as plt

    with patch.object(plt, "show"):
        values = runpy.run_path(
            str(ROOT / "docs/examples/fundamentals/gas/environment_data.py")
        )
    plt.close("all")
    gas = values["gas_mixture"]
    assert gas.name == ["water", "butanol", "styrene"]
    np.testing.assert_array_equal(gas.concentration, [[1e-3, 2e-6, 1e-9]])
    temperature = values["temperature_at_altitudes"]
    np.testing.assert_allclose(temperature[[0, -1]], [288.15, 223.15])
    # Buck's water/ice equations, independently evaluated in Celsius.
    celsius = temperature - 273.15
    pressure = np.where(
        celsius >= 0,
        611.21
        * np.exp((18.678 - celsius / 234.5) * celsius / (257.14 + celsius)),
        611.15
        * np.exp((23.036 - celsius / 333.7) * celsius / (279.82 + celsius)),
    )
    expected = 1e-3 * 8.31446261815324 * temperature / (0.01801528 * pressure)
    np.testing.assert_allclose(values["saturation_ratio"], expected, rtol=1e-12)


def test_gas_species_tutorial_units_and_pressure_ratios():
    """Check ordered species, SI conversion and ideal-gas saturation ratios."""
    import matplotlib

    matplotlib.use("Agg")
    from unittest.mock import patch

    import matplotlib.pyplot as plt

    with patch.object(plt, "show"):
        values = runpy.run_path(
            str(ROOT / "docs/examples/fundamentals/gas/gas_data.py")
        )
    plt.close("all")
    water = values["water_species"]
    organic = values["organic_species"]
    assert organic.name == ["butanol", "styrene"]
    np.testing.assert_allclose(water.concentration, [[1e-7]], rtol=1e-14)
    np.testing.assert_allclose(water.molar_mass, [0.01801528], rtol=1e-14)
    temperature = values["temperature_range"]
    # Independent evaluation of the retained tutorial coefficient convention.
    coefficients = [(7.838, 1558.19, 196.881), (6.924, 1420, 226)]
    pressures = np.array(
        [
            133.32238741499998 * 10 ** (a - b / (temperature - c))
            for a, b, c in coefficients
        ]
    )
    np.testing.assert_allclose(
        values["organic_pure_vapor_pressure"], pressures, rtol=1e-12
    )
    expected = (
        organic.concentration[0, :, None]
        * 8.31446261815324
        * temperature
        / organic.molar_mass[:, None]
        / pressures
    )
    np.testing.assert_allclose(
        values["organic_saturation_ratio"], expected, rtol=1e-12
    )


def test_particle_construction_units_and_material_inventory():
    """Preserve spherical masses, all material lanes and modal number density."""
    import matplotlib

    matplotlib.use("Agg")
    from unittest.mock import patch

    import matplotlib.pyplot as plt

    with patch.object(plt, "show"):
        values = runpy.run_path(
            str(ROOT / "docs/examples/fundamentals/particles/particle_data.py")
        )
    plt.close("all")
    particles = values["particle_data"]
    expected_mass = (
        4 * np.pi / 3 * 2500 * (np.array([100, 200, 300]) * 1e-9) ** 3
    )
    np.testing.assert_allclose(
        particles.total_mass[0], expected_mass, rtol=1e-14
    )
    np.testing.assert_allclose(particles.concentration[0], [1e8, 1e9, 1e10])
    expected = np.sum(expected_mass * [1e8, 1e9, 1e10])
    np.testing.assert_allclose(
        particles.species_mass_density, [[expected]], rtol=1e-14
    )
    speciated = values["speciated_data"]
    expected_lanes = np.sum(
        values["mass_distribution_pg"]
        * 1e-15
        * values["concentration_cm3"][:, None]
        * 1e6,
        axis=0,
    )
    np.testing.assert_allclose(
        speciated.species_mass_density[0], expected_lanes, rtol=1e-14
    )
    np.testing.assert_allclose(
        values["lognormal_data"].number_density, [2e10], rtol=1e-12
    )


def test_distribution_tutorial_material_and_volume_accounting():
    """Verify illustrative inventories and resolved count/volume arithmetic."""
    values = runpy.run_path(
        str(
            ROOT
            / "docs/examples/fundamentals/particles/distribution_tutorial.py"
        )
    )
    np.testing.assert_allclose(
        values["mass_bins"].species_mass_inventory, [[25.0]]
    )
    np.testing.assert_allclose(
        values["speciated_bins"].species_mass_inventory, [[25.0, 25.0]]
    )
    np.testing.assert_allclose(values["resolved"].number_density, [32.0])
    np.testing.assert_allclose(
        values["resolved"].species_mass_inventory, [[50.0, 50.0]]
    )
    np.testing.assert_allclose(
        values["resolved"].species_mass_density, [[200.0, 200.0]]
    )
    np.testing.assert_allclose(
        values["radius_bins"].radii[0],
        np.linspace(0, 0.1, 5),
        atol=1e-17,
    )


def test_diameter_to_native_radius_pdf_preserves_population():
    """Coordinate conversion must include the diameter/radius Jacobian."""
    import matplotlib

    matplotlib.use("Agg")
    from unittest.mock import patch

    import matplotlib.pyplot as plt

    with patch.object(plt, "show"):
        values = runpy.run_path(
            str(
                ROOT
                / "docs/examples/fundamentals/particles/aerosol_distributions.py"
            )
        )
    plt.close("all")
    data = values["native_pdf"]
    diameter_integral = np.trapezoid(
        values["multi_mode_distribution"], values["x_values"]
    )
    np.testing.assert_allclose(
        data.number_density, [diameter_integral], rtol=1e-14
    )
    np.testing.assert_allclose(data.number_density, [2000], rtol=1e-4)
    np.testing.assert_allclose(
        data.radius_grid, values["x_values"] * 0.5e-6, rtol=1e-14
    )
    expected = np.trapezoid(
        (4 * np.pi / 3 * 1000 * data.radius_grid**3)
        * values["multi_mode_distribution"]
        * 2e6,
        data.radius_grid,
    )
    np.testing.assert_allclose(
        data.species_mass_density, [[expected]], rtol=1e-14
    )
