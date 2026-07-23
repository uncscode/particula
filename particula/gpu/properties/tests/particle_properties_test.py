"""Tests for Warp particle property functions.

These tests validate parity between Warp @wp.func implementations and the
NumPy reference functions.
"""

from typing import Any

import numpy as np
import numpy.testing as npt
import pytest

pytestmark = pytest.mark.warp

wp = pytest.importorskip("warp")

from particula.gpu.properties.particle_properties import (  # noqa: E402
    aerodynamic_mobility_wp,
    cunningham_slip_correction_wp,
    debye_1_wp,
    diffusion_coefficient_wp,
    effective_density_wp,
    friction_factor_wp,
    kelvin_radius_wp,
    kelvin_term_wp,
    knudsen_number_wp,
    mean_thermal_speed_wp,
    partial_pressure_delta_wp,
    particle_radius_from_volume_wp,
    settling_velocity_stokes_from_transport_wp,
    vapor_transition_correction_wp,
    x_coth_x_wp,
)
from particula.gpu.tests.cuda_availability import warp_devices  # noqa: E402
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
    STANDARD_GRAVITY,
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


@wp.kernel
def _debye_kernel(values: Any, result: Any) -> None:
    """Evaluate the Debye geometry factor for each dimensionless input."""
    tid = wp.tid()
    result[tid] = debye_1_wp(values[tid])


@wp.kernel
def _x_coth_x_kernel(values: Any, result: Any) -> None:
    """Evaluate the rectangular wall-loss geometry factor for each input."""
    tid = wp.tid()
    result[tid] = x_coth_x_wp(values[tid])


@wp.kernel
def _radius_kernel(values: Any, result: Any) -> None:
    """Evaluate radii from particle volumes."""
    tid = wp.tid()
    result[tid] = particle_radius_from_volume_wp(values[tid])


@wp.kernel
def _diffusion_kernel(temperatures: Any, mobilities: Any, result: Any) -> None:
    """Evaluate Stokes-Einstein diffusion coefficients."""
    tid = wp.tid()
    result[tid] = diffusion_coefficient_wp(
        temperatures[tid], mobilities[tid], wp.float64(BOLTZMANN_CONSTANT)
    )


@wp.kernel
def _effective_density_kernel(masses: Any, volumes: Any, result: Any) -> None:
    """Evaluate effective mixture densities from scalar total lanes."""
    tid = wp.tid()
    result[tid] = effective_density_wp(masses[tid], volumes[tid])


@wp.kernel
def _settling_from_transport_kernel(
    radii: Any,
    densities: Any,
    viscosities: Any,
    mean_free_paths: Any,
    result: Any,
) -> None:
    """Evaluate Stokes settling velocity from supplied transport lanes."""
    tid = wp.tid()
    result[tid] = settling_velocity_stokes_from_transport_wp(
        radii[tid],
        densities[tid],
        viscosities[tid],
        mean_free_paths[tid],
    )


@pytest.fixture(params=warp_devices(wp))
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


def _evaluate_scalar_kernel(
    kernel: Any, values: np.ndarray, device: str
) -> np.ndarray:
    """Evaluate a scalar Warp property kernel and synchronize before readback."""
    values_wp = wp.array(values, dtype=wp.float64, device=device)
    result_wp = wp.zeros(len(values), dtype=wp.float64, device=device)
    wp.launch(
        kernel,
        dim=len(values),
        inputs=[values_wp],
        outputs=[result_wp],
        device=device,
    )
    wp.synchronize()
    return result_wp.numpy()


def test_debye_1_matches_independent_quadrature(device: str) -> None:
    """Ensure Debye geometry factor is accurate across branch boundaries."""
    from scipy.integrate import quad

    values = np.array(
        [
            0.0,
            1.0e-12,
            np.nextafter(1.0, 0.0),
            1.0,
            np.nextafter(1.0, np.inf),
            5.0,
            np.nextafter(20.0, 0.0),
            20.0,
            np.nextafter(20.0, np.inf),
        ],
        dtype=np.float64,
    )

    def integrand(value: float) -> float:
        return 1.0 if value == 0.0 else value / np.expm1(value)

    expected = np.array(
        [
            1.0 if value == 0.0 else quad(integrand, 0.0, value)[0] / value
            for value in values
        ],
        dtype=np.float64,
    )
    result = _evaluate_scalar_kernel(_debye_kernel, values, device)
    assert np.all(np.isfinite(result))
    npt.assert_allclose(result, expected, rtol=1e-9, atol=1e-12)
    npt.assert_allclose(result[3], expected[3], rtol=1e-12, atol=1e-14)
    assert np.ptp(result[6:]) < 1e-14


def test_debye_1_invalid_inputs_return_exact_zero(device: str) -> None:
    """Ensure Debye invalid-domain inputs take the safe sentinel branch."""
    values = np.array([-1.0, np.nan, np.inf, -np.inf], dtype=np.float64)
    npt.assert_array_equal(
        _evaluate_scalar_kernel(_debye_kernel, values, device), 0.0
    )


def test_slip_correction_safe_domain_contract(device: str) -> None:
    """Ensure Cunningham slip has its specified exact safe-domain limits."""
    values = np.array(
        [0.0, 1.0e-300, 1.0, 1.0e300, -1.0, np.nan, np.inf, -np.inf],
        dtype=np.float64,
    )
    result = _evaluate_scalar_kernel(_slip_correction_kernel, values, device)
    assert result[0] == 1.0
    assert np.all(np.isfinite(result[:4]))
    npt.assert_array_equal(result[4:], 0.0)


def test_x_coth_x_matches_series_and_direct_reference(device: str) -> None:
    """Ensure rectangular geometry factor preserves limits and switch accuracy."""
    threshold = 1.0e-3
    values = np.array(
        [
            0.0,
            1.0e-15,
            np.nextafter(threshold, 0.0),
            threshold,
            np.nextafter(threshold, np.inf),
            0.1,
            10.0,
        ],
        dtype=np.float64,
    )
    expected = np.ones_like(values)
    nonzero = values != 0.0
    expected[nonzero] = values[nonzero] / np.tanh(values[nonzero])
    result = _evaluate_scalar_kernel(_x_coth_x_kernel, values, device)
    npt.assert_allclose(result, expected, rtol=1e-12, atol=0.0)
    invalid = np.array([-1.0, np.nan, np.inf, -np.inf], dtype=np.float64)
    npt.assert_array_equal(
        _evaluate_scalar_kernel(_x_coth_x_kernel, invalid, device), 0.0
    )


def test_radius_and_diffusion_safe_contracts(device: str) -> None:
    """Ensure relocated radius and diffusion primitives retain their contracts."""
    volumes = np.array(
        [0.0, 1.0e-30, 1.0e-18, -1.0, np.nan, np.inf], dtype=np.float64
    )
    radii = _evaluate_scalar_kernel(_radius_kernel, volumes, device)
    expected_radii = np.array(
        [
            0.0,
            (3e-30 / (4 * np.pi)) ** (1 / 3),
            (3e-18 / (4 * np.pi)) ** (1 / 3),
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )
    npt.assert_allclose(radii, expected_radii, rtol=1e-12, atol=0.0)

    temperatures = np.array([250.0, 298.15, 350.0], dtype=np.float64)
    mobilities = np.array([1.0e8, 2.0e8, 3.0e8], dtype=np.float64)
    temperature_wp = wp.array(temperatures, dtype=wp.float64, device=device)
    mobility_wp = wp.array(mobilities, dtype=wp.float64, device=device)
    result_wp = wp.zeros(3, dtype=wp.float64, device=device)
    wp.launch(
        _diffusion_kernel,
        dim=3,
        inputs=[temperature_wp, mobility_wp],
        outputs=[result_wp],
        device=device,
    )
    wp.synchronize()
    npt.assert_allclose(
        result_wp.numpy(),
        BOLTZMANN_CONSTANT * temperatures * mobilities,
        rtol=1e-12,
        atol=0.0,
    )


def test_effective_density_safe_contract(device: str) -> None:
    """Ensure moved effective density preserves finite and safe-zero behavior."""
    masses = np.array(
        [1.0e-18, 1.0e-308, 0.0, -1.0, np.nan, np.inf],
        dtype=np.float64,
    )
    volumes = np.array(
        [1.0e-21, 1.0e10, 1.0e-21, 1.0e-21, 1.0e-21, 1.0e-21],
        dtype=np.float64,
    )
    expected = np.array([1000.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    result_wp = wp.zeros(len(masses), dtype=wp.float64, device=device)
    wp.launch(
        _effective_density_kernel,
        dim=len(masses),
        inputs=[
            wp.array(masses, dtype=wp.float64, device=device),
            wp.array(volumes, dtype=wp.float64, device=device),
        ],
        outputs=[result_wp],
        device=device,
    )
    wp.synchronize()

    npt.assert_allclose(result_wp.numpy(), expected, rtol=1e-12, atol=0.0)


def test_settling_velocity_from_transport_matches_reference_and_safe_zero(
    device: str,
) -> None:
    """Ensure moved transport-form settling velocity handles edge inputs."""
    radii = np.array([1.0e-7, 2.0e-7, 0.0, -1.0, np.nan], dtype=np.float64)
    densities = np.full(5, 1000.0, dtype=np.float64)
    viscosities = np.full(5, 1.8e-5, dtype=np.float64)
    mean_free_paths = np.full(5, 6.5e-8, dtype=np.float64)
    knudsen = mean_free_paths[:2] / radii[:2]
    slip = 1.0 + knudsen * (1.257 + 0.4 * np.exp(-1.1 / knudsen))
    expected = np.zeros(5, dtype=np.float64)
    expected[:2] = (
        2.0
        * radii[:2] ** 2
        * densities[:2]
        * slip
        * STANDARD_GRAVITY
        / (9.0 * viscosities[:2])
    )
    result_wp = wp.zeros(len(radii), dtype=wp.float64, device=device)
    wp.launch(
        _settling_from_transport_kernel,
        dim=len(radii),
        inputs=[
            wp.array(radii, dtype=wp.float64, device=device),
            wp.array(densities, dtype=wp.float64, device=device),
            wp.array(viscosities, dtype=wp.float64, device=device),
            wp.array(mean_free_paths, dtype=wp.float64, device=device),
        ],
        outputs=[result_wp],
        device=device,
    )
    wp.synchronize()

    observed = result_wp.numpy()
    npt.assert_allclose(observed[:2], expected[:2], rtol=1e-12, atol=0.0)
    npt.assert_array_equal(observed[2:], 0.0)


def test_moved_helpers_have_properties_only_export_surface() -> None:
    """Ensure neutral transport ownership is properties-only, not dynamics."""
    import particula.gpu.dynamics as dynamics
    import particula.gpu.properties as properties

    names = (
        "particle_radius_from_volume_wp",
        "diffusion_coefficient_wp",
        "effective_density_wp",
        "settling_velocity_stokes_wp",
        "settling_velocity_stokes_from_transport_wp",
    )
    for name in names:
        assert hasattr(properties, name)
        assert not hasattr(dynamics, name)
