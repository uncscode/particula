"""Tests for Warp gas property functions.

These tests validate parity between Warp @wp.func implementations and the
NumPy reference functions.
"""

from typing import Any

import numpy as np
import numpy.testing as npt
import pytest

wp = pytest.importorskip("warp")

from particula.gas.properties.dynamic_viscosity import (  # noqa: E402
    get_dynamic_viscosity,
)
from particula.gas.properties.mean_free_path import (  # noqa: E402
    get_molecule_mean_free_path,
)
from particula.gas.properties.pressure_function import (  # noqa: E402
    get_partial_pressure,
)
from particula.gpu.properties.gas_properties import (  # noqa: E402
    dynamic_viscosity_wp,
    molecule_mean_free_path_wp,
    partial_pressure_wp,
)
from particula.util.constants import (  # noqa: E402
    GAS_CONSTANT,
    REF_TEMPERATURE_STP,
    REF_VISCOSITY_AIR_STP,
    SUTHERLAND_CONSTANT,
)


@wp.kernel
def _dynamic_viscosity_kernel(
    temperatures: Any,
    ref_viscosity: Any,
    ref_temperature: Any,
    sutherland_constant: Any,
    result: Any,
) -> None:
    """Compute dynamic viscosity for each temperature input.

    Args:
        temperatures: Temperature values for each particle [K].
        ref_viscosity: Reference viscosity at ref_temperature [Pa·s].
        ref_temperature: Reference temperature [K].
        sutherland_constant: Sutherland constant [K].
        result: Output array for dynamic viscosity [Pa·s].
    """
    tid = wp.tid()
    result[tid] = dynamic_viscosity_wp(
        temperatures[tid],
        ref_viscosity,
        ref_temperature,
        sutherland_constant,
    )


@wp.kernel
def _mean_free_path_kernel(
    molar_masses: Any,
    temperatures: Any,
    pressures: Any,
    dynamic_viscosities: Any,
    gas_constant: float,
    result: Any,
) -> None:
    """Compute mean free path for each gas sample.

    Args:
        molar_masses: Molar masses for each sample [kg/mol].
        temperatures: Temperatures for each sample [K].
        pressures: Pressures for each sample [Pa].
        dynamic_viscosities: Dynamic viscosities for each sample [Pa·s].
        gas_constant: Universal gas constant [J/(mol·K)].
        result: Output array for mean free paths [m].
    """
    tid = wp.tid()
    result[tid] = molecule_mean_free_path_wp(
        molar_masses[tid],
        temperatures[tid],
        pressures[tid],
        dynamic_viscosities[tid],
        wp.float64(gas_constant),
    )


@wp.kernel
def _partial_pressure_kernel(
    concentrations: Any,
    molar_masses: Any,
    temperatures: Any,
    gas_constant: float,
    result: Any,
) -> None:
    """Compute partial pressure for each gas sample.

    Args:
        concentrations: Gas concentrations for each sample [kg/m³].
        molar_masses: Molar masses for each sample [kg/mol].
        temperatures: Temperatures for each sample [K].
        gas_constant: Universal gas constant [J/(mol·K)].
        result: Output array for partial pressures [Pa].
    """
    tid = wp.tid()
    result[tid] = partial_pressure_wp(
        concentrations[tid],
        molar_masses[tid],
        temperatures[tid],
        wp.float64(gas_constant),
    )


@pytest.fixture(params=["cpu"] + (["cuda"] if wp.is_cuda_available() else []))
def device(request) -> str:
    """Provide available Warp devices for testing."""
    return request.param


def test_dynamic_viscosity_matches_numpy(device: str) -> None:
    """Ensure dynamic_viscosity_wp matches NumPy reference values."""
    temperatures = np.array([250.0, 298.15, 350.0, 500.0], dtype=np.float64)
    expected = np.array(
        [get_dynamic_viscosity(value) for value in temperatures],
        dtype=np.float64,
    )

    temperatures_wp = wp.array(temperatures, dtype=wp.float64, device=device)
    result_wp = wp.zeros(len(temperatures), dtype=wp.float64, device=device)

    wp.launch(
        _dynamic_viscosity_kernel,
        dim=len(temperatures),
        inputs=[
            temperatures_wp,
            wp.float64(REF_VISCOSITY_AIR_STP),
            wp.float64(REF_TEMPERATURE_STP),
            wp.float64(SUTHERLAND_CONSTANT),
        ],
        outputs=[result_wp],
        device=device,
    )
    wp.synchronize()

    npt.assert_allclose(result_wp.numpy(), expected, rtol=1e-10)


def test_mean_free_path_matches_numpy(device: str) -> None:
    """Ensure molecule_mean_free_path_wp matches NumPy reference values."""
    molar_masses = np.array([0.018, 0.029, 0.044], dtype=np.float64)
    temperatures = np.array([250.0, 298.15, 350.0], dtype=np.float64)
    pressures = np.array([90000.0, 101325.0, 120000.0], dtype=np.float64)
    dynamic_viscosities = np.array(
        [get_dynamic_viscosity(value) for value in temperatures],
        dtype=np.float64,
    )
    expected = np.array(
        [
            get_molecule_mean_free_path(
                molar_mass=molar_mass,
                temperature=temperature,
                pressure=pressure,
                dynamic_viscosity=viscosity,
            )
            for molar_mass, temperature, pressure, viscosity in zip(
                molar_masses,
                temperatures,
                pressures,
                dynamic_viscosities,
                strict=True,
            )
        ],
        dtype=np.float64,
    )

    molar_masses_wp = wp.array(molar_masses, dtype=wp.float64, device=device)
    temperatures_wp = wp.array(temperatures, dtype=wp.float64, device=device)
    pressures_wp = wp.array(pressures, dtype=wp.float64, device=device)
    viscosities_wp = wp.array(
        dynamic_viscosities, dtype=wp.float64, device=device
    )
    result_wp = wp.zeros(len(molar_masses), dtype=wp.float64, device=device)

    wp.launch(
        _mean_free_path_kernel,
        dim=len(molar_masses),
        inputs=[
            molar_masses_wp,
            temperatures_wp,
            pressures_wp,
            viscosities_wp,
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
        atol=1e-14,
    )


def test_partial_pressure_matches_numpy(device: str) -> None:
    """Ensure partial_pressure_wp matches NumPy reference values."""
    concentrations = np.array([1.0, 1.2, 0.8], dtype=np.float64)
    molar_masses = np.array([0.018, 0.02897, 0.044], dtype=np.float64)
    temperatures = np.array([298.15, 310.0, 290.0], dtype=np.float64)
    expected = np.array(
        [
            get_partial_pressure(
                concentration=concentration,
                molar_mass=molar_mass,
                temperature=temperature,
            )
            for concentration, molar_mass, temperature in zip(
                concentrations,
                molar_masses,
                temperatures,
                strict=True,
            )
        ],
        dtype=np.float64,
    )

    concentrations_wp = wp.array(
        concentrations, dtype=wp.float64, device=device
    )
    molar_masses_wp = wp.array(molar_masses, dtype=wp.float64, device=device)
    temperatures_wp = wp.array(temperatures, dtype=wp.float64, device=device)
    result_wp = wp.zeros(len(concentrations), dtype=wp.float64, device=device)

    wp.launch(
        _partial_pressure_kernel,
        dim=len(concentrations),
        inputs=[
            concentrations_wp,
            molar_masses_wp,
            temperatures_wp,
            wp.float64(GAS_CONSTANT),
        ],
        outputs=[result_wp],
        device=device,
    )
    wp.synchronize()

    npt.assert_allclose(result_wp.numpy(), expected, rtol=1e-10)
