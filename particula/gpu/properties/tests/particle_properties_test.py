"""Tests for Warp particle property functions.

These tests validate parity between Warp @wp.func implementations and the
NumPy reference functions.
"""

from typing import Any

import numpy as np
import numpy.testing as npt
import pytest

wp = pytest.importorskip("warp")

from particula.gpu.properties.particle_properties import (  # noqa: E402
    aerodynamic_mobility_wp,
    cunningham_slip_correction_wp,
    friction_factor_wp,
    kelvin_radius_wp,
    kelvin_term_wp,
    knudsen_number_wp,
    mean_thermal_speed_wp,
    partial_pressure_delta_wp,
    vapor_transition_correction_wp,
)
from particula.particles.properties.aerodynamic_mobility_module import (  # noqa: E402
    get_aerodynamic_mobility,
)
from particula.particles.properties.friction_factor_module import (  # noqa: E402
    get_friction_factor,
)
from particula.particles.properties.kelvin_effect_module import (  # noqa: E402
    MAX_KELVIN_RATIO,
    get_kelvin_radius,
    get_kelvin_term,
)
from particula.particles.properties.knudsen_number_module import (  # noqa: E402
    get_knudsen_number,
)
from particula.particles.properties.mean_thermal_speed_module import (  # noqa: E402
    get_mean_thermal_speed,
)
from particula.particles.properties.partial_pressure_module import (  # noqa: E402
    get_partial_pressure_delta,
)
from particula.particles.properties.slip_correction_module import (  # noqa: E402
    get_cunningham_slip_correction,
)
from particula.particles.properties.vapor_correction_module import (  # noqa: E402
    get_vapor_transition_correction,
)
from particula.util.constants import (  # noqa: E402
    BOLTZMANN_CONSTANT,
    GAS_CONSTANT,
)


@wp.kernel
def _knudsen_number_kernel(
    mean_free_paths: Any,
    particle_radii: Any,
    result: Any,
) -> None:
    """Compute Knudsen number for each sample.

    Args:
        mean_free_paths: Mean free paths for each gas sample [m].
        particle_radii: Particle radii for each sample [m].
        result: Output array for Knudsen numbers (dimensionless).
    """
    tid = wp.tid()
    result[tid] = knudsen_number_wp(mean_free_paths[tid], particle_radii[tid])


@wp.kernel
def _slip_correction_kernel(
    knudsen_numbers: Any,
    result: Any,
) -> None:
    """Compute slip correction factor for each Knudsen number.

    Args:
        knudsen_numbers: Knudsen numbers for each sample (dimensionless).
        result: Output array for slip correction factors (dimensionless).
    """
    tid = wp.tid()
    result[tid] = cunningham_slip_correction_wp(knudsen_numbers[tid])


@wp.kernel
def _aerodynamic_mobility_kernel(
    particle_radii: Any,
    slip_corrections: Any,
    dynamic_viscosities: Any,
    result: Any,
) -> None:
    """Compute aerodynamic mobility for each sample.

    Args:
        particle_radii: Particle radii for each sample [m].
        slip_corrections: Slip correction factors (dimensionless).
        dynamic_viscosities: Dynamic viscosities for each sample [Pa·s].
        result: Output array for aerodynamic mobility [m²/s].
    """
    tid = wp.tid()
    result[tid] = aerodynamic_mobility_wp(
        particle_radii[tid],
        slip_corrections[tid],
        dynamic_viscosities[tid],
    )


@wp.kernel
def _mean_thermal_speed_kernel(
    particle_masses: Any,
    temperatures: Any,
    boltzmann_constant: Any,
    result: Any,
) -> None:
    """Compute mean thermal speed for each sample.

    Args:
        particle_masses: Particle masses for each sample [kg].
        temperatures: Gas temperatures for each sample [K].
        boltzmann_constant: Boltzmann constant [J/K].
        result: Output array for mean thermal speeds [m/s].
    """
    tid = wp.tid()
    result[tid] = mean_thermal_speed_wp(
        particle_masses[tid],
        temperatures[tid],
        boltzmann_constant,
    )


@wp.kernel
def _friction_factor_kernel(
    particle_radii: Any,
    dynamic_viscosities: Any,
    slip_corrections: Any,
    result: Any,
) -> None:
    """Compute friction factor for each particle sample.

    Args:
        particle_radii: Particle radii for each sample [m].
        dynamic_viscosities: Dynamic viscosities for each sample [Pa·s].
        slip_corrections: Slip correction factors (dimensionless).
        result: Output array for friction factors [N·s/m].
    """
    tid = wp.tid()
    result[tid] = friction_factor_wp(
        particle_radii[tid],
        dynamic_viscosities[tid],
        slip_corrections[tid],
    )


@wp.kernel
def _vapor_transition_correction_kernel(
    knudsen_numbers: Any,
    mass_accommodations: Any,
    result: Any,
) -> None:
    """Compute vapor transition correction factors for each sample.

    Args:
        knudsen_numbers: Knudsen numbers (dimensionless).
        mass_accommodations: Mass accommodation coefficients (dimensionless).
        result: Output array for correction factors (dimensionless).
    """
    tid = wp.tid()
    result[tid] = vapor_transition_correction_wp(
        knudsen_numbers[tid],
        mass_accommodations[tid],
    )


@wp.kernel
def _kelvin_radius_kernel(
    surface_tensions: Any,
    densities: Any,
    molar_masses: Any,
    temperatures: Any,
    gas_constant: Any,
    result: Any,
) -> None:
    """Compute Kelvin radii for each sample.

    Args:
        surface_tensions: Surface tensions for each sample [N/m].
        densities: Densities for each sample [kg/m³].
        molar_masses: Molar masses for each sample [kg/mol].
        temperatures: Temperatures for each sample [K].
        gas_constant: Gas constant [J/(mol·K)].
        result: Output array for Kelvin radii [m].
    """
    tid = wp.tid()
    result[tid] = kelvin_radius_wp(
        surface_tensions[tid],
        densities[tid],
        molar_masses[tid],
        temperatures[tid],
        gas_constant,
    )


@wp.kernel
def _kelvin_term_kernel(
    particle_radii: Any,
    kelvin_radii: Any,
    result: Any,
) -> None:
    """Compute Kelvin terms for each sample.

    Args:
        particle_radii: Particle radii for each sample [m].
        kelvin_radii: Kelvin radii for each sample [m].
        result: Output array for Kelvin terms (dimensionless).
    """
    tid = wp.tid()
    result[tid] = kelvin_term_wp(particle_radii[tid], kelvin_radii[tid])


@wp.kernel
def _partial_pressure_delta_kernel(
    partial_pressures_gas: Any,
    partial_pressures_particle: Any,
    kelvin_terms: Any,
    result: Any,
) -> None:
    """Compute partial pressure deltas for each sample.

    Args:
        partial_pressures_gas: Gas partial pressures [Pa].
        partial_pressures_particle: Particle partial pressures [Pa].
        kelvin_terms: Kelvin terms (dimensionless).
        result: Output array for partial pressure deltas [Pa].
    """
    tid = wp.tid()
    result[tid] = partial_pressure_delta_wp(
        partial_pressures_gas[tid],
        partial_pressures_particle[tid],
        kelvin_terms[tid],
    )


@pytest.fixture(params=["cpu"] + (["cuda"] if wp.is_cuda_available() else []))
def device(request) -> str:
    """Provide available Warp devices for testing."""
    return request.param


def test_knudsen_number_matches_numpy(device: str) -> None:
    """Ensure knudsen_number_wp matches NumPy reference values."""
    mean_free_paths = np.array([6.5e-8, 1.0e-7, 2.0e-7], dtype=np.float64)
    particle_radii = np.array([1.0e-7, 2.0e-7, 5.0e-8], dtype=np.float64)
    expected = np.array(
        [
            get_knudsen_number(mean_free_path, radius)
            for mean_free_path, radius in zip(
                mean_free_paths,
                particle_radii,
                strict=True,
            )
        ],
        dtype=np.float64,
    )

    mean_free_paths_wp = wp.array(
        mean_free_paths, dtype=wp.float64, device=device
    )
    particle_radii_wp = wp.array(
        particle_radii, dtype=wp.float64, device=device
    )
    result_wp = wp.zeros(len(mean_free_paths), dtype=wp.float64, device=device)

    wp.launch(
        _knudsen_number_kernel,
        dim=len(mean_free_paths),
        inputs=[mean_free_paths_wp, particle_radii_wp],
        outputs=[result_wp],
        device=device,
    )
    wp.synchronize()

    npt.assert_allclose(
        result_wp.numpy(),
        expected,
        rtol=1e-10,
        atol=1e-7,
    )


def test_slip_correction_matches_numpy(device: str) -> None:
    """Ensure cunningham_slip_correction_wp matches NumPy reference values."""
    knudsen_numbers = np.array([0.01, 0.1, 1.0, 10.0], dtype=np.float64)
    expected = np.array(
        [get_cunningham_slip_correction(value) for value in knudsen_numbers],
        dtype=np.float64,
    )

    knudsen_numbers_wp = wp.array(
        knudsen_numbers, dtype=wp.float64, device=device
    )
    result_wp = wp.zeros(len(knudsen_numbers), dtype=wp.float64, device=device)

    wp.launch(
        _slip_correction_kernel,
        dim=len(knudsen_numbers),
        inputs=[knudsen_numbers_wp],
        outputs=[result_wp],
        device=device,
    )
    wp.synchronize()

    npt.assert_allclose(
        result_wp.numpy(),
        expected,
        rtol=1e-10,
        atol=1e-6,
    )


def test_aerodynamic_mobility_matches_numpy(device: str) -> None:
    """Ensure aerodynamic_mobility_wp matches NumPy reference values."""
    particle_radii = np.array([5e-8, 1e-7, 2e-7], dtype=np.float64)
    slip_corrections = np.array([1.2, 1.1, 1.05], dtype=np.float64)
    dynamic_viscosities = np.array([1.8e-5, 1.85e-5, 1.9e-5], dtype=np.float64)
    expected = np.array(
        [
            get_aerodynamic_mobility(radius, slip, viscosity)
            for radius, slip, viscosity in zip(
                particle_radii,
                slip_corrections,
                dynamic_viscosities,
                strict=True,
            )
        ],
        dtype=np.float64,
    )

    particle_radii_wp = wp.array(
        particle_radii, dtype=wp.float64, device=device
    )
    slip_corrections_wp = wp.array(
        slip_corrections, dtype=wp.float64, device=device
    )
    dynamic_viscosities_wp = wp.array(
        dynamic_viscosities, dtype=wp.float64, device=device
    )
    result_wp = wp.zeros(len(particle_radii), dtype=wp.float64, device=device)

    wp.launch(
        _aerodynamic_mobility_kernel,
        dim=len(particle_radii),
        inputs=[particle_radii_wp, slip_corrections_wp, dynamic_viscosities_wp],
        outputs=[result_wp],
        device=device,
    )
    wp.synchronize()

    npt.assert_allclose(
        result_wp.numpy(),
        expected,
        rtol=1e-10,
        atol=2e3,
    )


def test_mean_thermal_speed_matches_numpy(device: str) -> None:
    """Ensure mean_thermal_speed_wp matches NumPy reference values."""
    particle_masses = np.array([1e-18, 1e-17, 5e-17], dtype=np.float64)
    temperatures = np.array([250.0, 298.15, 350.0], dtype=np.float64)
    expected = np.array(
        [
            get_mean_thermal_speed(mass, temp)
            for mass, temp in zip(particle_masses, temperatures, strict=True)
        ],
        dtype=np.float64,
    )

    particle_masses_wp = wp.array(
        particle_masses, dtype=wp.float64, device=device
    )
    temperatures_wp = wp.array(temperatures, dtype=wp.float64, device=device)
    result_wp = wp.zeros(len(particle_masses), dtype=wp.float64, device=device)

    wp.launch(
        _mean_thermal_speed_kernel,
        dim=len(particle_masses),
        inputs=[
            particle_masses_wp,
            temperatures_wp,
            wp.float64(BOLTZMANN_CONSTANT),
        ],
        outputs=[result_wp],
        device=device,
    )
    wp.synchronize()

    npt.assert_allclose(
        result_wp.numpy(),
        expected,
        rtol=1e-10,
        atol=1e-6,
    )


def test_friction_factor_matches_numpy(device: str) -> None:
    """Ensure friction_factor_wp matches NumPy reference values."""
    particle_radii = np.array([1.0e-7, 2.0e-7, 4.0e-7], dtype=np.float64)
    dynamic_viscosities = np.array([1.8e-5, 1.9e-5, 2.0e-5], dtype=np.float64)
    slip_corrections = np.array([1.1, 1.05, 1.01], dtype=np.float64)
    expected = np.array(
        [
            get_friction_factor(radius, viscosity, slip)
            for radius, viscosity, slip in zip(
                particle_radii,
                dynamic_viscosities,
                slip_corrections,
                strict=True,
            )
        ],
        dtype=np.float64,
    )

    particle_radii_wp = wp.array(
        particle_radii, dtype=wp.float64, device=device
    )
    dynamic_viscosities_wp = wp.array(
        dynamic_viscosities, dtype=wp.float64, device=device
    )
    slip_corrections_wp = wp.array(
        slip_corrections, dtype=wp.float64, device=device
    )
    result_wp = wp.zeros(len(particle_radii), dtype=wp.float64, device=device)

    wp.launch(
        _friction_factor_kernel,
        dim=len(particle_radii),
        inputs=[
            particle_radii_wp,
            dynamic_viscosities_wp,
            slip_corrections_wp,
        ],
        outputs=[result_wp],
        device=device,
    )
    wp.synchronize()

    npt.assert_allclose(
        result_wp.numpy(),
        expected,
        rtol=1e-10,
        atol=1e-7,
    )


def test_vapor_transition_correction_matches_numpy(device: str) -> None:
    """Ensure vapor_transition_correction_wp matches NumPy values."""
    knudsen_numbers = np.array([0.01, 1.0, 100.0], dtype=np.float64)
    mass_accommodations = np.array([0.5, 1.0], dtype=np.float64)
    knudsen_grid, accommodation_grid = np.meshgrid(
        knudsen_numbers,
        mass_accommodations,
    )
    knudsen_flat = knudsen_grid.ravel()
    accommodation_flat = accommodation_grid.ravel()
    expected = np.array(
        [
            get_vapor_transition_correction(kn, alpha)
            for kn, alpha in zip(knudsen_flat, accommodation_flat, strict=True)
        ],
        dtype=np.float64,
    )

    knudsen_wp = wp.array(knudsen_flat, dtype=wp.float64, device=device)
    accommodations_wp = wp.array(
        accommodation_flat,
        dtype=wp.float64,
        device=device,
    )
    result_wp = wp.zeros(len(knudsen_flat), dtype=wp.float64, device=device)

    wp.launch(
        _vapor_transition_correction_kernel,
        dim=len(knudsen_flat),
        inputs=[knudsen_wp, accommodations_wp],
        outputs=[result_wp],
        device=device,
    )
    wp.synchronize()

    npt.assert_allclose(
        result_wp.numpy(),
        expected,
        rtol=1e-10,
        atol=1e-8,
    )


def test_kelvin_radius_matches_numpy(device: str) -> None:
    """Ensure kelvin_radius_wp matches NumPy reference values."""
    surface_tensions = np.array([0.072, 0.072, 0.072], dtype=np.float64)
    densities = np.array([1000.0, 1000.0, 1000.0], dtype=np.float64)
    molar_masses = np.array([0.018, 0.018, 0.018], dtype=np.float64)
    temperatures = np.array([250.0, 298.15, 320.0], dtype=np.float64)
    expected = np.array(
        [
            get_kelvin_radius(surface, density, molar_mass, temp)
            for surface, density, molar_mass, temp in zip(
                surface_tensions,
                densities,
                molar_masses,
                temperatures,
                strict=True,
            )
        ],
        dtype=np.float64,
    )

    surface_wp = wp.array(surface_tensions, dtype=wp.float64, device=device)
    density_wp = wp.array(densities, dtype=wp.float64, device=device)
    molar_mass_wp = wp.array(molar_masses, dtype=wp.float64, device=device)
    temperature_wp = wp.array(temperatures, dtype=wp.float64, device=device)
    result_wp = wp.zeros(len(surface_tensions), dtype=wp.float64, device=device)

    wp.launch(
        _kelvin_radius_kernel,
        dim=len(surface_tensions),
        inputs=[
            surface_wp,
            density_wp,
            molar_mass_wp,
            temperature_wp,
            wp.float64(GAS_CONSTANT),
        ],
        outputs=[result_wp],
        device=device,
    )
    wp.synchronize()

    npt.assert_allclose(
        result_wp.numpy(),
        expected,
        rtol=1e-10,
        atol=1e-12,
    )


def test_kelvin_term_matches_numpy(device: str) -> None:
    """Ensure kelvin_term_wp matches NumPy reference values."""
    particle_radii = np.array([5.0e-8, 1.0e-10], dtype=np.float64)
    kelvin_radii = np.array([1.0e-10, 1.0e-7], dtype=np.float64)
    expected = np.array(
        [
            get_kelvin_term(radius, kelvin_radius)
            for radius, kelvin_radius in zip(
                particle_radii,
                kelvin_radii,
                strict=True,
            )
        ],
        dtype=np.float64,
    )

    particle_radii_wp = wp.array(
        particle_radii, dtype=wp.float64, device=device
    )
    kelvin_radii_wp = wp.array(kelvin_radii, dtype=wp.float64, device=device)
    result_wp = wp.zeros(len(particle_radii), dtype=wp.float64, device=device)

    wp.launch(
        _kelvin_term_kernel,
        dim=len(particle_radii),
        inputs=[particle_radii_wp, kelvin_radii_wp],
        outputs=[result_wp],
        device=device,
    )
    wp.synchronize()

    npt.assert_allclose(
        result_wp.numpy(),
        expected,
        rtol=1e-10,
        atol=1e-12,
    )


def test_kelvin_term_zero_radius_clamps_matches_numpy(device: str) -> None:
    """Ensure kelvin_term_wp clamps zero-radius values safely."""
    particle_radii = np.array([0.0, 1.0e-10], dtype=np.float64)
    kelvin_radii = np.array([1.0e-9, 1.0e-9], dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = np.array(
            [
                get_kelvin_term(radius, kelvin_radius)
                for radius, kelvin_radius in zip(
                    particle_radii,
                    kelvin_radii,
                    strict=True,
                )
            ],
            dtype=np.float64,
        )
    expected = np.minimum(expected, np.exp(MAX_KELVIN_RATIO))

    particle_radii_wp = wp.array(
        particle_radii, dtype=wp.float64, device=device
    )
    kelvin_radii_wp = wp.array(kelvin_radii, dtype=wp.float64, device=device)
    result_wp = wp.zeros(len(particle_radii), dtype=wp.float64, device=device)

    wp.launch(
        _kelvin_term_kernel,
        dim=len(particle_radii),
        inputs=[particle_radii_wp, kelvin_radii_wp],
        outputs=[result_wp],
        device=device,
    )
    wp.synchronize()

    npt.assert_allclose(
        result_wp.numpy(),
        expected,
        rtol=1e-10,
        atol=1e-12,
    )


def test_partial_pressure_delta_matches_numpy(device: str) -> None:
    """Ensure partial_pressure_delta_wp matches NumPy reference values."""
    partial_pressures_gas = np.array([100.0, 80.0, 50.0], dtype=np.float64)
    partial_pressures_particle = np.array([60.0, 90.0, 50.0], dtype=np.float64)
    kelvin_terms = np.array([1.1, 1.2, 1.0], dtype=np.float64)
    expected = np.array(
        [
            get_partial_pressure_delta(gas, particle, kelvin)
            for gas, particle, kelvin in zip(
                partial_pressures_gas,
                partial_pressures_particle,
                kelvin_terms,
                strict=True,
            )
        ],
        dtype=np.float64,
    )

    gas_wp = wp.array(partial_pressures_gas, dtype=wp.float64, device=device)
    particle_wp = wp.array(
        partial_pressures_particle, dtype=wp.float64, device=device
    )
    kelvin_wp = wp.array(kelvin_terms, dtype=wp.float64, device=device)
    result_wp = wp.zeros(
        len(partial_pressures_gas),
        dtype=wp.float64,
        device=device,
    )

    wp.launch(
        _partial_pressure_delta_kernel,
        dim=len(partial_pressures_gas),
        inputs=[gas_wp, particle_wp, kelvin_wp],
        outputs=[result_wp],
        device=device,
    )
    wp.synchronize()

    npt.assert_allclose(
        result_wp.numpy(),
        expected,
        rtol=1e-10,
        atol=1e-12,
    )
