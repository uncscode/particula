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
    knudsen_number_wp,
    mean_thermal_speed_wp,
)
from particula.particles.properties.aerodynamic_mobility_module import (  # noqa: E402
    get_aerodynamic_mobility,
)
from particula.particles.properties.friction_factor_module import (  # noqa: E402
    get_friction_factor,
)
from particula.particles.properties.knudsen_number_module import (  # noqa: E402
    get_knudsen_number,
)
from particula.particles.properties.mean_thermal_speed_module import (  # noqa: E402
    get_mean_thermal_speed,
)
from particula.particles.properties.slip_correction_module import (  # noqa: E402
    get_cunningham_slip_correction,
)
from particula.util.constants import (  # noqa: E402
    BOLTZMANN_CONSTANT,
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
        atol=1e4,
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
