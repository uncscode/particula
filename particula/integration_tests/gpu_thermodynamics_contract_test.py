"""Integration contracts for GPU condensation thermodynamic sidecars."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.testing as npt
import pytest
from particula.gas.gas_data import GasData
from particula.gas.properties.vapor_pressure_module import (
    get_buck_vapor_pressure,
)
from particula.gpu import to_warp_gas_data, to_warp_particle_data
from particula.particles.particle_data import ParticleData

pytestmark = pytest.mark.warp

_N_BOXES = 2
_N_SPECIES = 2
_TRANSFER_SENTINEL = -1.0
_VAPOR_PRESSURE_SENTINEL = 12345.0
_CONSTANT_VAPOR_PRESSURE = 725.0
_TIME_STEP = 1.0e-6


def _warp() -> Any:
    """Import Warp at test runtime to preserve marker deselection."""
    return pytest.importorskip("warp")


def _make_gpu_data() -> tuple[Any, Any]:
    """Create deterministic CPU-resident particle and gas GPU containers."""
    n_particles = 2
    particles = ParticleData(
        masses=np.array(
            [
                [[1.0e-18, 2.0e-18], [1.2e-18, 2.4e-18]],
                [[1.1e-18, 2.2e-18], [1.3e-18, 2.6e-18]],
            ],
            dtype=np.float64,
        ),
        concentration=np.ones((_N_BOXES, n_particles), dtype=np.float64),
        charge=np.zeros((_N_BOXES, n_particles), dtype=np.float64),
        density=np.array([1000.0, 1200.0], dtype=np.float64),
        volume=np.full(_N_BOXES, 1.0e-6, dtype=np.float64),
    )
    gas = GasData(
        name=["constant", "buck"],
        molar_mass=np.array([0.018, 0.05], dtype=np.float64),
        concentration=np.array(
            [[1.0e-6, 1.2e-6], [1.1e-6, 1.3e-6]], dtype=np.float64
        ),
        partitioning=np.ones(_N_SPECIES, dtype=bool),
    )
    vapor_pressure = np.full(
        (_N_BOXES, _N_SPECIES), _VAPOR_PRESSURE_SENTINEL, dtype=np.float64
    )
    return (
        to_warp_particle_data(particles, device="cpu"),
        to_warp_gas_data(gas, device="cpu", vapor_pressure=vapor_pressure),
    )


def _make_thermodynamics_config(gpu_gas: Any, device: str = "cpu") -> Any:
    """Build a matching mixed constant/Buck thermodynamics sidecar."""
    wp = _warp()
    from particula.gpu.kernels.thermodynamics import (
        THERMODYNAMICS_MODE_BUCK,
        THERMODYNAMICS_MODE_CONSTANT,
        ThermodynamicsConfig,
    )

    parameters = np.zeros((_N_SPECIES, 4), dtype=np.float64)
    parameters[0, 0] = _CONSTANT_VAPOR_PRESSURE
    return ThermodynamicsConfig(
        modes=wp.array(
            [THERMODYNAMICS_MODE_CONSTANT, THERMODYNAMICS_MODE_BUCK],
            dtype=wp.int32,
            device=device,
        ),
        parameters=wp.array(parameters, dtype=wp.float64, device=device),
        molar_mass_reference=wp.array(
            gpu_gas.molar_mass.numpy(), dtype=wp.float64, device=device
        ),
    )


def _snapshot(
    gpu_particles: Any, gpu_gas: Any, mass_transfer: Any
) -> tuple[np.ndarray, ...]:
    """Copy CPU-backed caller-owned outputs that must be atomic on failure."""
    caller_owned_arrays = (
        gpu_gas.vapor_pressure,
        gpu_gas.concentration,
        gpu_particles.masses,
        mass_transfer,
    )
    assert all(
        getattr(array.device, "is_cpu", False)
        or str(array.device).startswith("cpu")
        for array in caller_owned_arrays
    )
    snapshots = (*(array.numpy().copy() for array in caller_owned_arrays),)
    n_particles = gpu_particles.masses.shape[1]
    assert snapshots[0].shape == (_N_BOXES, _N_SPECIES)
    assert snapshots[1].shape == (_N_BOXES, _N_SPECIES)
    assert snapshots[2].shape == (_N_BOXES, n_particles, _N_SPECIES)
    assert snapshots[3].shape == (_N_BOXES, n_particles, _N_SPECIES)
    return snapshots


def _assert_snapshot_unchanged(
    snapshot: tuple[np.ndarray, ...],
    gpu_particles: Any,
    gpu_gas: Any,
    mass_transfer: Any,
) -> None:
    """Assert caller-owned outputs retain exact pre-exception values."""
    for expected, actual in zip(
        snapshot,
        _snapshot(gpu_particles, gpu_gas, mass_transfer),
        strict=True,
    ):
        npt.assert_array_equal(actual, expected)


def _expected_vapor_pressure(temperatures: np.ndarray) -> np.ndarray:
    """Return CPU constant/Buck reference vapor pressures in Pa."""
    return np.column_stack(
        (
            np.full(temperatures.size, _CONSTANT_VAPOR_PRESSURE),
            get_buck_vapor_pressure(temperatures),
        )
    )


def _call_step(
    gpu_particles: Any,
    gpu_gas: Any,
    temperature: Any,
    pressure: Any,
    mass_transfer: Any,
    thermodynamics: Any | None = None,
) -> tuple[Any, Any]:
    """Call the public condensation API through its legacy positional layout."""
    from particula.gpu.kernels import condensation_step_gpu

    kwargs = (
        {} if thermodynamics is None else {"thermodynamics": thermodynamics}
    )
    return condensation_step_gpu(
        gpu_particles,
        gpu_gas,
        temperature,
        pressure,
        _TIME_STEP,
        None,
        None,
        None,
        mass_transfer,
        **kwargs,
    )


def test_condensation_step_gpu_reuses_thermodynamics_and_transfer_buffer() -> (
    None
):
    """One sidecar and transfer buffer can be reused for two public calls."""
    wp = _warp()
    gpu_particles, gpu_gas = _make_gpu_data()
    config = _make_thermodynamics_config(gpu_gas)
    n_particles = gpu_particles.masses.shape[1]
    transfer = wp.full(
        (_N_BOXES, n_particles, _N_SPECIES),
        _TRANSFER_SENTINEL,
        dtype=wp.float64,
        device="cpu",
    )
    first_temperatures = np.array([273.15, 280.0], dtype=np.float64)
    second_temperatures = np.array([290.0, 305.0], dtype=np.float64)
    pressure = wp.array([101325.0, 101325.0], dtype=wp.float64, device="cpu")

    _, first_transfer = _call_step(
        gpu_particles,
        gpu_gas,
        wp.array(first_temperatures, dtype=wp.float64, device="cpu"),
        pressure,
        transfer,
        config,
    )
    assert first_transfer is transfer
    first_transfer_values = transfer.numpy()
    first_vapor_pressure = gpu_gas.vapor_pressure.numpy().copy()
    assert first_transfer_values.shape == (_N_BOXES, n_particles, _N_SPECIES)
    assert np.all(np.isfinite(first_transfer_values))
    assert np.all(first_transfer_values != _TRANSFER_SENTINEL)
    assert np.any(first_transfer_values != 0.0)
    npt.assert_allclose(
        first_vapor_pressure[:, 0],
        _expected_vapor_pressure(first_temperatures)[:, 0],
        rtol=0.0,
        atol=0.0,
    )
    npt.assert_allclose(
        first_vapor_pressure[:, 1],
        _expected_vapor_pressure(first_temperatures)[:, 1],
        rtol=1e-12,
        atol=0.0,
    )
    wp.copy(
        transfer,
        wp.full(
            (_N_BOXES, n_particles, _N_SPECIES),
            _TRANSFER_SENTINEL,
            dtype=wp.float64,
            device="cpu",
        ),
    )
    npt.assert_array_equal(
        transfer.numpy(),
        np.full(
            (_N_BOXES, n_particles, _N_SPECIES),
            _TRANSFER_SENTINEL,
        ),
    )

    _, second_transfer = _call_step(
        gpu_particles,
        gpu_gas,
        wp.array(second_temperatures, dtype=wp.float64, device="cpu"),
        pressure,
        transfer,
        config,
    )
    second_vapor_pressure = gpu_gas.vapor_pressure.numpy()
    assert second_transfer is transfer
    assert np.all(np.isfinite(second_transfer.numpy()))
    assert np.all(second_transfer.numpy() != _TRANSFER_SENTINEL)
    assert np.any(second_transfer.numpy() != 0.0)
    npt.assert_allclose(
        second_vapor_pressure[:, 0],
        _expected_vapor_pressure(second_temperatures)[:, 0],
        rtol=0.0,
        atol=0.0,
    )
    npt.assert_allclose(
        second_vapor_pressure[:, 1],
        _expected_vapor_pressure(second_temperatures)[:, 1],
        rtol=1e-12,
        atol=0.0,
    )
    assert not np.array_equal(
        second_vapor_pressure[:, 1], first_vapor_pressure[:, 1]
    )
    assert np.all(second_vapor_pressure != _VAPOR_PRESSURE_SENTINEL)


def test_condensation_step_gpu_missing_thermodynamics_is_atomic() -> None:
    """Missing required sidecar fails before mutating public caller buffers."""
    wp = _warp()
    gpu_particles, gpu_gas = _make_gpu_data()
    n_particles = gpu_particles.masses.shape[1]
    transfer = wp.full(
        (_N_BOXES, n_particles, _N_SPECIES),
        _TRANSFER_SENTINEL,
        dtype=wp.float64,
        device="cpu",
    )
    snapshot = _snapshot(gpu_particles, gpu_gas, transfer)

    with pytest.raises(ValueError, match="thermodynamics"):
        _call_step(
            gpu_particles,
            gpu_gas,
            wp.array([273.15, 280.0], dtype=wp.float64, device="cpu"),
            wp.array([101325.0, 101325.0], dtype=wp.float64, device="cpu"),
            transfer,
        )

    _assert_snapshot_unchanged(snapshot, gpu_particles, gpu_gas, transfer)


@pytest.mark.cuda
def test_condensation_step_gpu_cross_device_thermodynamics_is_atomic() -> None:
    """A CUDA sidecar fails before it mutates CPU-owned simulation buffers."""
    wp = _warp()
    from particula.gpu.tests.cuda_availability import (
        CUDA_SKIP_REASON,
        cuda_available,
    )

    if not cuda_available(wp):
        pytest.skip(CUDA_SKIP_REASON)
    gpu_particles, gpu_gas = _make_gpu_data()
    n_particles = gpu_particles.masses.shape[1]
    transfer = wp.full(
        (_N_BOXES, n_particles, _N_SPECIES),
        _TRANSFER_SENTINEL,
        dtype=wp.float64,
        device="cpu",
    )
    snapshot = _snapshot(gpu_particles, gpu_gas, transfer)
    config = _make_thermodynamics_config(gpu_gas, device="cuda")

    with pytest.raises(ValueError, match="thermodynamics.*device"):
        _call_step(
            gpu_particles,
            gpu_gas,
            wp.array([273.15, 280.0], dtype=wp.float64, device="cpu"),
            wp.array([101325.0, 101325.0], dtype=wp.float64, device="cpu"),
            transfer,
            config,
        )

    _assert_snapshot_unchanged(snapshot, gpu_particles, gpu_gas, transfer)
