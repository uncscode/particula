"""End-to-end tests for GPU condensation kernels."""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import numpy.testing as npt
import pytest

wp = pytest.importorskip("warp")

import particula.gpu.kernels.condensation as condensation_module  # noqa: E402
from particula.dynamics.condensation.mass_transfer import (  # noqa: E402
    get_first_order_mass_transport_k,
    get_mass_transfer_rate,
)
from particula.gas.environment_data import EnvironmentData  # noqa: E402
from particula.gas.gas_data import GasData  # noqa: E402
from particula.gas.properties.dynamic_viscosity import (  # noqa: E402
    get_dynamic_viscosity,
)
from particula.gas.properties.mean_free_path import (  # noqa: E402
    get_molecule_mean_free_path,
)
from particula.gas.properties.pressure_function import (  # noqa: E402
    get_partial_pressure,
)
from particula.gpu.conversion import (  # noqa: E402
    from_warp_particle_data,
    to_warp_environment_data,
    to_warp_gas_data,
    to_warp_particle_data,
)
from particula.gpu.dynamics.condensation_funcs import (  # noqa: E402
    particle_radius_from_volume_wp,
)
from particula.gpu.kernels.condensation import (  # noqa: E402
    _validate_mass_transfer_buffer,
    _validate_species_array,
    condensation_step_gpu,
)
from particula.gpu.tests.cuda_availability import (  # noqa: E402
    cuda_available,
    warp_devices,
)
from particula.particles.particle_data import ParticleData  # noqa: E402
from particula.particles.properties.aerodynamic_mobility_module import (  # noqa: E402
    get_aerodynamic_mobility,
)
from particula.particles.properties.diffusion_coefficient import (  # noqa: E402
    get_diffusion_coefficient,
)
from particula.particles.properties.kelvin_effect_module import (  # noqa: E402
    get_kelvin_radius,
    get_kelvin_term,
)
from particula.particles.properties.knudsen_number_module import (  # noqa: E402
    get_knudsen_number,
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
from particula.util import constants  # noqa: E402


@pytest.fixture(params=warp_devices(wp))
def device(request) -> str:
    """Provide available Warp devices for testing."""
    return request.param


def _make_particle_data(
    n_boxes: int,
    n_particles: int,
    n_species: int,
) -> ParticleData:
    """Create deterministic particle data for GPU tests."""
    base_masses = np.linspace(1.0e-18, 3.0e-18, n_species, dtype=np.float64)
    masses = np.empty((n_boxes, n_particles, n_species), dtype=np.float64)
    for box_idx in range(n_boxes):
        for particle_idx in range(n_particles):
            scale = 1.0 + 0.1 * particle_idx + 0.05 * box_idx
            masses[box_idx, particle_idx, :] = base_masses * scale
    concentration = np.ones((n_boxes, n_particles), dtype=np.float64)
    charge = np.zeros((n_boxes, n_particles), dtype=np.float64)
    density = np.linspace(1000.0, 1400.0, n_species, dtype=np.float64)
    volume = np.full((n_boxes,), 1.0e-6, dtype=np.float64)
    return ParticleData(
        masses=masses,
        concentration=concentration,
        charge=charge,
        density=density,
        volume=volume,
    )


def _make_gas_data(n_boxes: int, n_species: int) -> GasData:
    """Create deterministic gas data for GPU tests."""
    molar_mass = np.linspace(0.018, 0.05, n_species, dtype=np.float64)
    concentration = np.empty((n_boxes, n_species), dtype=np.float64)
    for box_idx in range(n_boxes):
        concentration[box_idx, :] = 1.0e-6 * (1.0 + 0.2 * box_idx)
    partitioning = np.ones((n_species,), dtype=bool)
    names = [f"species_{idx}" for idx in range(n_species)]
    return GasData(
        name=names,
        molar_mass=molar_mass,
        concentration=concentration,
        partitioning=partitioning,
    )


def _make_vapor_pressure(n_boxes: int, n_species: int) -> np.ndarray:
    """Create deterministic vapor pressure array."""
    vapor_pressure = np.empty((n_boxes, n_species), dtype=np.float64)
    for box_idx in range(n_boxes):
        vapor_pressure[box_idx, :] = 800.0 + 50.0 * box_idx
    return vapor_pressure


def _make_environment_data(
    n_boxes: int,
    n_species: int,
    temperature: float = 298.15,
    pressure: float = 101325.0,
) -> EnvironmentData:
    """Create deterministic environment data for contract tests."""
    return EnvironmentData(
        temperature=np.full((n_boxes,), temperature, dtype=np.float64),
        pressure=np.full((n_boxes,), pressure, dtype=np.float64),
        saturation_ratio=np.ones((n_boxes, n_species), dtype=np.float64),
    )


def _cpu_mass_transfer(
    particles: ParticleData,
    gas: GasData,
    vapor_pressure: np.ndarray,
    surface_tension: np.ndarray,
    mass_accommodation: np.ndarray,
    diffusion_coefficient_vapor: np.ndarray,
    temperature: float | np.ndarray,
    pressure: float | np.ndarray,
    time_step: float,
) -> np.ndarray:
    """Compute CPU mass transfer matching GPU kernel physics."""
    n_boxes, n_particles, n_species = particles.masses.shape
    mass_transfer = np.zeros_like(particles.masses)
    temperature_array = np.full((n_boxes,), temperature, dtype=np.float64)
    if isinstance(temperature, np.ndarray):
        temperature_array = np.asarray(temperature, dtype=np.float64)
    pressure_array = np.full((n_boxes,), pressure, dtype=np.float64)
    if isinstance(pressure, np.ndarray):
        pressure_array = np.asarray(pressure, dtype=np.float64)

    for box_idx in range(n_boxes):
        box_temperature = float(temperature_array[box_idx])
        box_pressure = float(pressure_array[box_idx])
        dynamic_viscosity = get_dynamic_viscosity(
            box_temperature,
            reference_viscosity=constants.REF_VISCOSITY_AIR_STP,
            reference_temperature=constants.REF_TEMPERATURE_STP,
        )
        mean_free_path = get_molecule_mean_free_path(
            molar_mass=constants.MOLECULAR_WEIGHT_AIR,
            temperature=box_temperature,
            pressure=box_pressure,
            dynamic_viscosity=dynamic_viscosity,
        )
        for particle_idx in range(n_particles):
            if particles.concentration[box_idx, particle_idx] == 0.0:
                continue
            total_volume = np.sum(
                particles.masses[box_idx, particle_idx, :] / particles.density
            )
            if total_volume <= 0.0:
                continue
            total_mass = np.sum(particles.masses[box_idx, particle_idx, :])
            radius = np.cbrt(3.0 * total_volume / (4.0 * np.pi))
            effective_density = (
                total_mass / total_volume if total_volume > 0.0 else 0.0
            )
            if effective_density <= 0.0:
                effective_density = particles.density[0]

            knudsen_number = get_knudsen_number(mean_free_path, radius)
            slip_correction = get_cunningham_slip_correction(knudsen_number)
            mobility = get_aerodynamic_mobility(
                particle_radius=radius,
                slip_correction_factor=slip_correction,
                dynamic_viscosity=dynamic_viscosity,
            )
            diffusion_particle = get_diffusion_coefficient(
                temperature=box_temperature,
                aerodynamic_mobility=mobility,
                boltzmann_constant=constants.BOLTZMANN_CONSTANT,
            )

            for species_idx in range(n_species):
                transition = get_vapor_transition_correction(
                    knudsen_number=knudsen_number,
                    mass_accommodation=mass_accommodation[species_idx],
                )
                diffusion_value = diffusion_coefficient_vapor[species_idx]
                if diffusion_value <= 0.0:
                    diffusion_value = diffusion_particle
                mass_transport = get_first_order_mass_transport_k(
                    particle_radius=radius,
                    vapor_transition=transition,
                    diffusion_coefficient=diffusion_value,
                )
                kelvin_radius = get_kelvin_radius(
                    effective_surface_tension=surface_tension[species_idx],
                    effective_density=effective_density,
                    molar_mass=gas.molar_mass[species_idx],
                    temperature=box_temperature,
                )
                kelvin_term = get_kelvin_term(radius, kelvin_radius)
                partial_pressure_gas = get_partial_pressure(
                    concentration=gas.concentration[box_idx, species_idx],
                    molar_mass=gas.molar_mass[species_idx],
                    temperature=box_temperature,
                )
                pressure_delta = get_partial_pressure_delta(
                    partial_pressure_gas=partial_pressure_gas,
                    partial_pressure_particle=vapor_pressure[
                        box_idx, species_idx
                    ],
                    kelvin_term=kelvin_term,
                )
                mass_rate = get_mass_transfer_rate(
                    pressure_delta=pressure_delta,
                    first_order_mass_transport=mass_transport,
                    temperature=box_temperature,
                    molar_mass=gas.molar_mass[species_idx],
                )
                mass_transfer[box_idx, particle_idx, species_idx] = (
                    mass_rate * time_step
                )
    return mass_transfer


def _run_gpu_step(
    particles: ParticleData,
    gas: GasData,
    vapor_pressure: np.ndarray,
    temperature: float | Any | None,
    pressure: float | Any | None,
    time_step: float,
    device: str,
    surface_tension: np.ndarray | None = None,
    mass_accommodation: np.ndarray | None = None,
    diffusion_coefficient_vapor: np.ndarray | None = None,
    mass_transfer: Any | None = None,
    environment: Any | None = None,
) -> tuple[ParticleData, Any]:
    """Run GPU condensation step and return CPU particle data."""
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas,
        device=device,
        vapor_pressure=vapor_pressure,
    )
    _, mass_transfer_buffer = condensation_step_gpu(
        gpu_particles,
        gpu_gas,
        temperature=temperature,
        pressure=pressure,
        time_step=time_step,
        surface_tension=surface_tension,
        mass_accommodation=mass_accommodation,
        diffusion_coefficient_vapor=diffusion_coefficient_vapor,
        mass_transfer=mass_transfer,
        environment=environment,
    )
    return (
        from_warp_particle_data(gpu_particles, sync=True),
        mass_transfer_buffer,
    )


def test_condensation_step_gpu_signature_keeps_environment_keyword_only() -> (
    None
):
    """The explicit environment input stays keyword-only."""
    parameter = inspect.signature(condensation_step_gpu).parameters[
        "environment"
    ]

    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_condensation_step_gpu_scalar_positional_call_remains_valid(
    device: str,
) -> None:
    """Legacy positional scalar callers remain source-compatible."""
    particles = _make_particle_data(n_boxes=1, n_particles=2, n_species=1)
    gas = _make_gas_data(n_boxes=1, n_species=1)
    vapor_pressure = _make_vapor_pressure(n_boxes=1, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas,
        device=device,
        vapor_pressure=vapor_pressure,
    )

    _, mass_transfer = condensation_step_gpu(
        gpu_particles,
        gpu_gas,
        298.15,
        101325.0,
        0.1,
    )
    wp.synchronize()

    assert mass_transfer.shape == (1, 2, 1)


@pytest.mark.parametrize(
    ("temperature", "pressure"),
    [
        (298.15, 101325.0),
        (298.15, None),
        (None, 101325.0),
    ],
)
def test_condensation_step_gpu_rejects_mixed_environment_inputs(
    device: str,
    temperature: float | None,
    pressure: float | None,
) -> None:
    """Mixed scalar and environment inputs raise a stable contract error."""
    particles = _make_particle_data(n_boxes=1, n_particles=1, n_species=1)
    gas = _make_gas_data(n_boxes=1, n_species=1)
    vapor_pressure = _make_vapor_pressure(n_boxes=1, n_species=1)
    environment = to_warp_environment_data(
        _make_environment_data(n_boxes=1, n_species=1),
        device=device,
    )
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas,
        device=device,
        vapor_pressure=vapor_pressure,
    )

    with pytest.raises(
        ValueError,
        match="direct temperature/pressure inputs with environment",
    ):
        condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature=temperature,
            pressure=pressure,
            time_step=0.1,
            environment=environment,
        )


def test_condensation_step_gpu_accepts_explicit_environment(
    device: str,
) -> None:
    """Pure ``environment=...`` execution succeeds when inputs are valid."""
    particles = _make_particle_data(n_boxes=1, n_particles=1, n_species=1)
    gas = _make_gas_data(n_boxes=1, n_species=1)
    vapor_pressure = _make_vapor_pressure(n_boxes=1, n_species=1)
    environment = to_warp_environment_data(
        _make_environment_data(n_boxes=1, n_species=1),
        device=device,
    )
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas,
        device=device,
        vapor_pressure=vapor_pressure,
    )

    _, scalar_mass_transfer = condensation_step_gpu(
        gpu_particles,
        gpu_gas,
        temperature=298.15,
        pressure=101325.0,
        time_step=0.1,
    )
    scalar_result = np.asarray(scalar_mass_transfer.numpy()).copy()

    gpu_particles = to_warp_particle_data(particles, device=device)
    _, environment_mass_transfer = condensation_step_gpu(
        gpu_particles,
        gpu_gas,
        temperature=None,
        pressure=None,
        time_step=0.1,
        environment=environment,
    )

    npt.assert_allclose(environment_mass_transfer.numpy(), scalar_result)


@pytest.mark.parametrize(
    ("temperature", "pressure"),
    [
        (298.15, None),
        (None, 101325.0),
        (None, None),
    ],
)
def test_condensation_step_gpu_rejects_missing_scalar_inputs_without_environment(
    device: str,
    temperature: float | None,
    pressure: float | None,
) -> None:
    """Scalar-mode calls require both temperature and pressure."""
    particles = _make_particle_data(n_boxes=1, n_particles=1, n_species=1)
    gas = _make_gas_data(n_boxes=1, n_species=1)
    vapor_pressure = _make_vapor_pressure(n_boxes=1, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas,
        device=device,
        vapor_pressure=vapor_pressure,
    )

    with pytest.raises(
        ValueError,
        match="temperature and pressure must both be provided",
    ):
        condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature=temperature,
            pressure=pressure,
            time_step=0.1,
        )


@pytest.mark.parametrize(
    ("temperature", "pressure", "message"),
    [
        (
            298.15,
            101325.0,
            "direct temperature/pressure inputs with environment",
        ),
        (
            None,
            None,
            r"\(n_boxes,\)",
        ),
    ],
)
def test_condensation_step_gpu_contract_errors_short_circuit_before_helpers(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    temperature: float | None,
    pressure: float | None,
    message: str,
) -> None:
    """Contract errors fire before buffer preparation or Warp launch work."""
    particles = _make_particle_data(n_boxes=1, n_particles=1, n_species=1)
    gas = _make_gas_data(n_boxes=1, n_species=1)
    vapor_pressure = _make_vapor_pressure(n_boxes=1, n_species=1)
    environment_data = _make_environment_data(n_boxes=1, n_species=1)
    if temperature is None and pressure is None:
        environment_data.temperature = np.array([298.15, 299.15])
    environment = to_warp_environment_data(environment_data, device=device)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas,
        device=device,
        vapor_pressure=vapor_pressure,
    )
    calls: list[str] = []

    def _unexpected_launch(*args: Any, **kwargs: Any) -> None:
        calls.append("launch")
        raise AssertionError("wp.launch should not be called")

    monkeypatch.setattr(condensation_module.wp, "launch", _unexpected_launch)

    with pytest.raises(ValueError, match=message):
        condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature=temperature,
            pressure=pressure,
            time_step=0.1,
            environment=environment,
        )

    assert calls == []


@pytest.mark.parametrize(
    ("temperature", "pressure"),
    [
        (298.15, None),
        (None, 101325.0),
        (None, None),
    ],
)
def test_condensation_step_gpu_missing_scalar_inputs_short_circuit_before_helpers(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    temperature: float | None,
    pressure: float | None,
) -> None:
    """Missing direct inputs fail before buffer preparation or launch work."""
    particles = _make_particle_data(n_boxes=1, n_particles=1, n_species=1)
    gas = _make_gas_data(n_boxes=1, n_species=1)
    vapor_pressure = _make_vapor_pressure(n_boxes=1, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas,
        device=device,
        vapor_pressure=vapor_pressure,
    )
    calls: list[str] = []

    def _unexpected_launch(*args: Any, **kwargs: Any) -> None:
        calls.append("launch")
        raise AssertionError("wp.launch should not be called")

    monkeypatch.setattr(condensation_module.wp, "launch", _unexpected_launch)

    with pytest.raises(
        ValueError,
        match="temperature and pressure must both be provided",
    ):
        condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature=temperature,
            pressure=pressure,
            time_step=0.1,
        )

    assert calls == []


def test_condensation_step_gpu_accepts_direct_environment_arrays(
    device: str,
) -> None:
    """Direct ``(n_boxes,)`` Warp-array inputs match scalar results."""
    particles = _make_particle_data(n_boxes=2, n_particles=2, n_species=1)
    gas = _make_gas_data(n_boxes=2, n_species=1)
    vapor_pressure = _make_vapor_pressure(n_boxes=2, n_species=1)
    temperature_values = np.array([298.15, 301.15], dtype=np.float64)
    pressure_values = np.array([101325.0, 100800.0], dtype=np.float64)
    expected = _cpu_mass_transfer(
        particles,
        gas,
        vapor_pressure,
        np.array([0.072], dtype=np.float64),
        np.array([1.0], dtype=np.float64),
        np.array([2.0e-5], dtype=np.float64),
        temperature_values,
        pressure_values,
        0.1,
    )

    gpu_result, mass_transfer = _run_gpu_step(
        particles,
        gas,
        vapor_pressure,
        temperature=wp.array(
            temperature_values, dtype=wp.float64, device=device
        ),
        pressure=wp.array(pressure_values, dtype=wp.float64, device=device),
        time_step=0.1,
        device=device,
    )

    npt.assert_allclose(mass_transfer.numpy(), expected, rtol=1.0e-10)
    npt.assert_allclose(
        gpu_result.masses, np.maximum(particles.masses + expected, 0.0)
    )


@pytest.mark.parametrize(
    ("temperature_input", "pressure_input"),
    [
        (
            298.15,
            np.array([101325.0, 100800.0], dtype=np.float64),
        ),
        (
            np.array([298.15, 301.15], dtype=np.float64),
            101325.0,
        ),
    ],
)
def test_condensation_step_gpu_accepts_hybrid_scalar_and_array_inputs(
    device: str,
    temperature_input: float | np.ndarray,
    pressure_input: float | np.ndarray,
) -> None:
    """Hybrid direct inputs match the CPU reference path."""
    particles = _make_particle_data(n_boxes=2, n_particles=2, n_species=1)
    gas = _make_gas_data(n_boxes=2, n_species=1)
    vapor_pressure = _make_vapor_pressure(n_boxes=2, n_species=1)
    temperature_values = (
        np.full((2,), temperature_input, dtype=np.float64)
        if isinstance(temperature_input, float)
        else temperature_input
    )
    pressure_values = (
        np.full((2,), pressure_input, dtype=np.float64)
        if isinstance(pressure_input, float)
        else pressure_input
    )
    expected = _cpu_mass_transfer(
        particles,
        gas,
        vapor_pressure,
        np.array([0.072], dtype=np.float64),
        np.array([1.0], dtype=np.float64),
        np.array([2.0e-5], dtype=np.float64),
        temperature_values,
        pressure_values,
        0.1,
    )

    temperature = temperature_input
    if isinstance(temperature_input, np.ndarray):
        temperature = wp.array(
            temperature_input, dtype=wp.float64, device=device
        )
    pressure = pressure_input
    if isinstance(pressure_input, np.ndarray):
        pressure = wp.array(pressure_input, dtype=wp.float64, device=device)

    _, mass_transfer = _run_gpu_step(
        particles,
        gas,
        vapor_pressure,
        temperature=temperature,
        pressure=pressure,
        time_step=0.1,
        device=device,
    )

    npt.assert_allclose(mass_transfer.numpy(), expected, rtol=1.0e-10)


def test_condensation_step_gpu_environment_shape_mismatch_raises_value_error(
    device: str,
) -> None:
    """Environment arrays must match ``(n_boxes,)`` before launch work."""
    particles = _make_particle_data(n_boxes=1, n_particles=1, n_species=1)
    gas = _make_gas_data(n_boxes=1, n_species=1)
    vapor_pressure = _make_vapor_pressure(n_boxes=1, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas, device=device, vapor_pressure=vapor_pressure
    )
    environment = to_warp_environment_data(
        _make_environment_data(1, 1), device=device
    )
    environment.temperature = wp.array(
        [298.15, 299.15], dtype=wp.float64, device=device
    )

    with pytest.raises(ValueError, match=r"\(n_boxes,\)"):
        condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature=None,
            pressure=None,
            time_step=0.1,
            environment=environment,
        )


def test_condensation_step_gpu_environment_device_mismatch_raises_value_error(
    device: str,
) -> None:
    """Environment arrays on the wrong device fail before launch work."""
    wrong_device = "cpu" if device == "cuda" else "cuda"
    if wrong_device == "cuda" and not cuda_available(wp):
        pytest.skip("CUDA not available for mismatch test")

    particles = _make_particle_data(n_boxes=1, n_particles=1, n_species=1)
    gas = _make_gas_data(n_boxes=1, n_species=1)
    vapor_pressure = _make_vapor_pressure(n_boxes=1, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas, device=device, vapor_pressure=vapor_pressure
    )
    environment = to_warp_environment_data(
        _make_environment_data(1, 1),
        device=wrong_device,
    )

    with pytest.raises(ValueError, match="environment.temperature device"):
        condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature=None,
            pressure=None,
            time_step=0.1,
            environment=environment,
        )


def test_condensation_step_gpu_direct_temperature_shape_mismatch_raises(
    device: str,
) -> None:
    """Direct temperature arrays must match ``(n_boxes,)`` before launch."""
    particles = _make_particle_data(n_boxes=2, n_particles=1, n_species=1)
    gas = _make_gas_data(n_boxes=2, n_species=1)
    vapor_pressure = _make_vapor_pressure(n_boxes=2, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas, device=device, vapor_pressure=vapor_pressure
    )
    temperature = wp.array([298.15], dtype=wp.float64, device=device)

    with pytest.raises(ValueError, match=r"temperature shape .*\(n_boxes,\)"):
        condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature=temperature,
            pressure=101325.0,
            time_step=0.1,
        )


def test_condensation_step_gpu_direct_pressure_device_mismatch_raises(
    device: str,
) -> None:
    """Direct pressure arrays on the wrong device fail before launch."""
    wrong_device = "cpu" if device == "cuda" else "cuda"
    if wrong_device == "cuda" and not cuda_available(wp):
        pytest.skip("CUDA not available for mismatch test")

    particles = _make_particle_data(n_boxes=1, n_particles=1, n_species=1)
    gas = _make_gas_data(n_boxes=1, n_species=1)
    vapor_pressure = _make_vapor_pressure(n_boxes=1, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas, device=device, vapor_pressure=vapor_pressure
    )
    pressure = wp.array([101325.0], dtype=wp.float64, device=wrong_device)

    with pytest.raises(ValueError, match="pressure device"):
        condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature=298.15,
            pressure=pressure,
            time_step=0.1,
        )


def test_condensation_step_gpu_prepares_box_properties_once_per_call(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
) -> None:
    """Condensation precomputes box properties once per entry-point call."""
    particles = _make_particle_data(n_boxes=2, n_particles=2, n_species=1)
    gas = _make_gas_data(n_boxes=2, n_species=1)
    vapor_pressure = _make_vapor_pressure(n_boxes=2, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas, device=device, vapor_pressure=vapor_pressure
    )
    launch_names: list[str] = []
    original_launch = condensation_module.wp.launch

    def _tracking_launch(kernel: Any, *args: Any, **kwargs: Any) -> Any:
        launch_names.append(getattr(kernel, "key", str(kernel)))
        return original_launch(kernel, *args, **kwargs)

    monkeypatch.setattr(condensation_module.wp, "launch", _tracking_launch)

    condensation_step_gpu(
        gpu_particles,
        gpu_gas,
        temperature=298.15,
        pressure=101325.0,
        time_step=0.1,
    )

    assert launch_names.count("_prepare_environment_properties_kernel") == 1


def test_condensation_step_gpu_matches_cpu_single_box(device: str) -> None:
    """GPU condensation matches CPU for a single box."""
    temperature = 298.15
    pressure = 101325.0
    time_step = 1.0
    particles = _make_particle_data(n_boxes=1, n_particles=2, n_species=2)
    gas = _make_gas_data(n_boxes=1, n_species=2)
    vapor_pressure = _make_vapor_pressure(n_boxes=1, n_species=2)
    surface_tension = np.array([0.072, 0.09], dtype=np.float64)
    mass_accommodation = np.array([1.0, 0.8], dtype=np.float64)
    diffusion = np.array([2.0e-5, 1.5e-5], dtype=np.float64)

    cpu_mass_transfer = _cpu_mass_transfer(
        particles,
        gas,
        vapor_pressure,
        surface_tension,
        mass_accommodation,
        diffusion,
        temperature,
        pressure,
        time_step,
    )
    expected_masses = np.maximum(particles.masses + cpu_mass_transfer, 0.0)

    gpu_result, _ = _run_gpu_step(
        particles,
        gas,
        vapor_pressure,
        temperature,
        pressure,
        time_step,
        device,
        surface_tension=wp.array(
            surface_tension, dtype=wp.float64, device=device
        ),
        mass_accommodation=wp.array(
            mass_accommodation, dtype=wp.float64, device=device
        ),
        diffusion_coefficient_vapor=wp.array(
            diffusion, dtype=wp.float64, device=device
        ),
    )

    npt.assert_allclose(gpu_result.masses, expected_masses, rtol=1.0e-10)


def test_condensation_step_gpu_multi_box_matches_cpu(device: str) -> None:
    """GPU condensation matches CPU for multiple boxes."""
    temperature = 300.0
    pressure = 100000.0
    time_step = 0.5
    particles = _make_particle_data(n_boxes=3, n_particles=2, n_species=2)
    gas = _make_gas_data(n_boxes=3, n_species=2)
    vapor_pressure = _make_vapor_pressure(n_boxes=3, n_species=2)
    surface_tension = np.array([0.072, 0.09], dtype=np.float64)
    mass_accommodation = np.array([0.9, 0.7], dtype=np.float64)
    diffusion = np.array([2.0e-5, 1.7e-5], dtype=np.float64)

    cpu_mass_transfer = _cpu_mass_transfer(
        particles,
        gas,
        vapor_pressure,
        surface_tension,
        mass_accommodation,
        diffusion,
        temperature,
        pressure,
        time_step,
    )
    expected_masses = np.maximum(particles.masses + cpu_mass_transfer, 0.0)

    gpu_result, _ = _run_gpu_step(
        particles,
        gas,
        vapor_pressure,
        temperature,
        pressure,
        time_step,
        device,
        surface_tension=wp.array(
            surface_tension, dtype=wp.float64, device=device
        ),
        mass_accommodation=wp.array(
            mass_accommodation, dtype=wp.float64, device=device
        ),
        diffusion_coefficient_vapor=wp.array(
            diffusion, dtype=wp.float64, device=device
        ),
    )

    npt.assert_allclose(gpu_result.masses, expected_masses, rtol=1.0e-10)


def test_apply_mass_transfer_kernel_clamps_negative(device: str) -> None:
    """Masses clamp to non-negative when evaporation exceeds mass."""
    temperature = 298.15
    pressure = 101325.0
    time_step = 10.0
    particles = _make_particle_data(n_boxes=1, n_particles=1, n_species=1)
    gas = _make_gas_data(n_boxes=1, n_species=1)
    vapor_pressure = np.full((1, 1), 1.0e6, dtype=np.float64)
    surface_tension = np.array([0.072], dtype=np.float64)
    mass_accommodation = np.array([1.0], dtype=np.float64)
    diffusion = np.array([2.0e-5], dtype=np.float64)

    gpu_result, _ = _run_gpu_step(
        particles,
        gas,
        vapor_pressure,
        temperature,
        pressure,
        time_step,
        device,
        surface_tension=wp.array(
            surface_tension, dtype=wp.float64, device=device
        ),
        mass_accommodation=wp.array(
            mass_accommodation, dtype=wp.float64, device=device
        ),
        diffusion_coefficient_vapor=wp.array(
            diffusion, dtype=wp.float64, device=device
        ),
    )

    assert np.all(gpu_result.masses >= 0.0)


def test_condensation_skips_inactive_particles(device: str) -> None:
    """Inactive particles retain their masses."""
    temperature = 298.15
    pressure = 101325.0
    time_step = 1.0
    particles = _make_particle_data(n_boxes=1, n_particles=2, n_species=1)
    particles.concentration[0, 1] = 0.0
    gas = _make_gas_data(n_boxes=1, n_species=1)
    vapor_pressure = _make_vapor_pressure(n_boxes=1, n_species=1)

    initial_mass = particles.masses[0, 1, 0]
    gpu_result, _ = _run_gpu_step(
        particles,
        gas,
        vapor_pressure,
        temperature,
        pressure,
        time_step,
        device,
    )

    assert gpu_result.masses[0, 1, 0] == pytest.approx(initial_mass)


def test_condensation_multi_species_parity(device: str) -> None:
    """Multi-species GPU condensation matches CPU."""
    temperature = 295.0
    pressure = 100500.0
    time_step = 0.8
    particles = _make_particle_data(n_boxes=1, n_particles=2, n_species=3)
    gas = _make_gas_data(n_boxes=1, n_species=3)
    vapor_pressure = _make_vapor_pressure(n_boxes=1, n_species=3)
    surface_tension = np.array([0.072, 0.08, 0.1], dtype=np.float64)
    mass_accommodation = np.array([1.0, 0.9, 0.7], dtype=np.float64)
    diffusion = np.array([2.0e-5, 1.7e-5, 1.2e-5], dtype=np.float64)

    cpu_mass_transfer = _cpu_mass_transfer(
        particles,
        gas,
        vapor_pressure,
        surface_tension,
        mass_accommodation,
        diffusion,
        temperature,
        pressure,
        time_step,
    )
    expected_masses = np.maximum(particles.masses + cpu_mass_transfer, 0.0)

    gpu_result, _ = _run_gpu_step(
        particles,
        gas,
        vapor_pressure,
        temperature,
        pressure,
        time_step,
        device,
        surface_tension=wp.array(
            surface_tension, dtype=wp.float64, device=device
        ),
        mass_accommodation=wp.array(
            mass_accommodation, dtype=wp.float64, device=device
        ),
        diffusion_coefficient_vapor=wp.array(
            diffusion, dtype=wp.float64, device=device
        ),
    )

    npt.assert_allclose(gpu_result.masses, expected_masses, rtol=1.0e-10)


def test_condensation_step_gpu_reuses_mass_transfer_buffer(
    device: str,
) -> None:
    """Preallocated mass transfer buffer is reused."""
    temperature = 298.15
    pressure = 101325.0
    time_step = 1.0
    particles = _make_particle_data(n_boxes=1, n_particles=2, n_species=2)
    gas = _make_gas_data(n_boxes=1, n_species=2)
    vapor_pressure = _make_vapor_pressure(n_boxes=1, n_species=2)

    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas,
        device=device,
        vapor_pressure=vapor_pressure,
    )
    mass_transfer = wp.zeros(
        (1, 2, 2),
        dtype=wp.float64,
        device=device,
    )
    _, returned_buffer = condensation_step_gpu(
        gpu_particles,
        gpu_gas,
        temperature=temperature,
        pressure=pressure,
        time_step=time_step,
        mass_transfer=mass_transfer,
    )
    assert returned_buffer is mass_transfer
    assert np.any(returned_buffer.numpy() != 0.0)


def test_condensation_step_gpu_rejects_mismatched_mass_transfer_shape(
    device: str,
) -> None:
    """Mismatched mass transfer shape raises ValueError."""
    particles = _make_particle_data(n_boxes=1, n_particles=2, n_species=2)
    gas = _make_gas_data(n_boxes=1, n_species=2)
    vapor_pressure = _make_vapor_pressure(n_boxes=1, n_species=2)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas,
        device=device,
        vapor_pressure=vapor_pressure,
    )
    mass_transfer = wp.zeros(
        (1, 2, 3),
        dtype=wp.float64,
        device=device,
    )

    with pytest.raises(ValueError, match="mass_transfer shape"):
        condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature=298.15,
            pressure=101325.0,
            time_step=1.0,
            mass_transfer=mass_transfer,
        )


def test_validate_species_array_rejects_length_mismatch(device: str) -> None:
    """Validation helper rejects arrays with wrong length."""
    array = wp.zeros(3, dtype=wp.float64, device=device)
    with pytest.raises(ValueError, match="length 3 does not match n_species"):
        _validate_species_array("surface_tension", array, 2, array.device)


def test_validate_species_array_rejects_device_mismatch(device: str) -> None:
    """Validation helper rejects arrays on a different device."""
    array = wp.zeros(2, dtype=wp.float64, device=device)
    wrong_device = "cpu" if device == "cuda" else "cuda"
    if wrong_device == "cuda" and not cuda_available(wp):
        pytest.skip("CUDA not available for mismatch test")
    with pytest.raises(ValueError, match="device does not match particle"):
        _validate_species_array(
            "surface_tension",
            array,
            2,
            wp.get_device(wrong_device),
        )


def test_validate_species_array_rejects_rank_mismatch(device: str) -> None:
    """Validation helper rejects arrays with more than one dimension."""
    array = wp.zeros((1, 2), dtype=wp.float64, device=device)
    with pytest.raises(ValueError, match="must be a 1D array"):
        _validate_species_array("surface_tension", array, 2, array.device)


def test_validate_mass_transfer_buffer_rejects_shape(device: str) -> None:
    """Validation helper rejects mass transfer buffers with bad shape."""
    buffer = wp.zeros((1, 2, 3), dtype=wp.float64, device=device)
    with pytest.raises(ValueError, match="mass_transfer shape"):
        _validate_mass_transfer_buffer(buffer, (1, 2, 2), buffer.device)


def test_validate_mass_transfer_buffer_rejects_device(device: str) -> None:
    """Validation helper rejects mass transfer buffers on wrong device."""
    buffer = wp.zeros((1, 2, 2), dtype=wp.float64, device=device)
    wrong_device = "cpu" if device == "cuda" else "cuda"
    if wrong_device == "cuda" and not cuda_available(wp):
        pytest.skip("CUDA not available for mismatch test")
    with pytest.raises(ValueError, match="buffer device does not match"):
        _validate_mass_transfer_buffer(
            buffer,
            (1, 2, 2),
            wp.get_device(wrong_device),
        )


def test_condensation_validation_helpers_accept_valid_inputs(
    device: str,
) -> None:
    """Validation helpers accept correctly shaped on-device buffers."""
    species_array = wp.zeros(2, dtype=wp.float64, device=device)
    mass_transfer = wp.zeros((1, 2, 2), dtype=wp.float64, device=device)

    _validate_species_array(
        "surface_tension",
        species_array,
        2,
        species_array.device,
    )
    _validate_mass_transfer_buffer(
        mass_transfer,
        (1, 2, 2),
        mass_transfer.device,
    )


def test_condensation_step_gpu_rejects_particle_length_mismatch(
    device: str,
) -> None:
    """Condensation rejects particle arrays with incorrect lengths."""
    particles = _make_particle_data(n_boxes=1, n_particles=2, n_species=2)
    gas = _make_gas_data(n_boxes=1, n_species=2)
    vapor_pressure = _make_vapor_pressure(n_boxes=1, n_species=2)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas,
        device=device,
        vapor_pressure=vapor_pressure,
    )

    bad_density = wp.zeros(3, dtype=wp.float64, device=device)
    gpu_particles.density = bad_density

    with pytest.raises(ValueError, match="particle density length"):
        condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature=298.15,
            pressure=101325.0,
            time_step=1.0,
        )


def test_condensation_step_gpu_rejects_particle_concentration_shape(
    device: str,
) -> None:
    """Condensation rejects particle concentration shape mismatches."""
    particles = _make_particle_data(n_boxes=1, n_particles=2, n_species=2)
    gas = _make_gas_data(n_boxes=1, n_species=2)
    vapor_pressure = _make_vapor_pressure(n_boxes=1, n_species=2)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas,
        device=device,
        vapor_pressure=vapor_pressure,
    )

    gpu_particles.concentration = wp.zeros(
        (1, 3),
        dtype=wp.float64,
        device=device,
    )

    with pytest.raises(ValueError, match="particle concentration shape"):
        condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature=298.15,
            pressure=101325.0,
            time_step=1.0,
        )


def test_condensation_step_gpu_rejects_gas_molar_mass_length(
    device: str,
) -> None:
    """Condensation rejects gas molar mass length mismatches."""
    particles = _make_particle_data(n_boxes=1, n_particles=2, n_species=2)
    gas = _make_gas_data(n_boxes=1, n_species=2)
    vapor_pressure = _make_vapor_pressure(n_boxes=1, n_species=2)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas,
        device=device,
        vapor_pressure=vapor_pressure,
    )

    gpu_gas.molar_mass = wp.zeros(3, dtype=wp.float64, device=device)

    with pytest.raises(ValueError, match="n_species mismatch"):
        condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature=298.15,
            pressure=101325.0,
            time_step=1.0,
        )


def test_condensation_step_gpu_rejects_gas_concentration_shape(
    device: str,
) -> None:
    """Condensation rejects gas concentration shape mismatches."""
    particles = _make_particle_data(n_boxes=1, n_particles=2, n_species=2)
    gas = _make_gas_data(n_boxes=1, n_species=2)
    vapor_pressure = _make_vapor_pressure(n_boxes=1, n_species=2)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas,
        device=device,
        vapor_pressure=vapor_pressure,
    )

    gpu_gas.concentration = wp.zeros(
        (1, 3),
        dtype=wp.float64,
        device=device,
    )

    with pytest.raises(ValueError, match="gas concentration shape"):
        condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature=298.15,
            pressure=101325.0,
            time_step=1.0,
        )


def test_condensation_step_gpu_rejects_vapor_pressure_shape(
    device: str,
) -> None:
    """Condensation rejects vapor pressure shape mismatches."""
    particles = _make_particle_data(n_boxes=1, n_particles=2, n_species=2)
    gas = _make_gas_data(n_boxes=1, n_species=2)
    vapor_pressure = _make_vapor_pressure(n_boxes=1, n_species=2)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas,
        device=device,
        vapor_pressure=vapor_pressure,
    )

    gpu_gas.vapor_pressure = wp.zeros(
        (1, 3),
        dtype=wp.float64,
        device=device,
    )

    with pytest.raises(ValueError, match="vapor pressure shape"):
        condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature=298.15,
            pressure=101325.0,
            time_step=1.0,
        )


def test_condensation_step_gpu_rejects_gas_device_mismatch(
    device: str,
) -> None:
    """Condensation rejects gas arrays on a different device."""
    particles = _make_particle_data(n_boxes=1, n_particles=2, n_species=2)
    gas = _make_gas_data(n_boxes=1, n_species=2)
    vapor_pressure = _make_vapor_pressure(n_boxes=1, n_species=2)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas,
        device=device,
        vapor_pressure=vapor_pressure,
    )

    wrong_device = "cpu" if device == "cuda" else "cuda"
    if wrong_device == "cuda" and not cuda_available(wp):
        pytest.skip("CUDA not available for mismatch test")

    gpu_gas.molar_mass = wp.zeros(2, dtype=wp.float64, device=wrong_device)

    with pytest.raises(ValueError, match="gas molar mass device mismatch"):
        condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature=298.15,
            pressure=101325.0,
            time_step=1.0,
        )


def test_condensation_step_gpu_rejects_particle_concentration_device_mismatch(
    device: str,
) -> None:
    """Condensation rejects particle concentration on wrong device."""
    particles = _make_particle_data(n_boxes=1, n_particles=2, n_species=2)
    gas = _make_gas_data(n_boxes=1, n_species=2)
    vapor_pressure = _make_vapor_pressure(n_boxes=1, n_species=2)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas,
        device=device,
        vapor_pressure=vapor_pressure,
    )

    wrong_device = "cpu" if device == "cuda" else "cuda"
    if wrong_device == "cuda" and not cuda_available(wp):
        pytest.skip("CUDA not available for mismatch test")

    gpu_particles.concentration = wp.zeros(
        (1, 2),
        dtype=wp.float64,
        device=wrong_device,
    )

    with pytest.raises(
        ValueError,
        match="particle concentration device mismatch",
    ):
        condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature=298.15,
            pressure=101325.0,
            time_step=1.0,
        )


def test_condensation_step_gpu_rejects_particle_density_device_mismatch(
    device: str,
) -> None:
    """Condensation rejects particle density on wrong device."""
    particles = _make_particle_data(n_boxes=1, n_particles=2, n_species=2)
    gas = _make_gas_data(n_boxes=1, n_species=2)
    vapor_pressure = _make_vapor_pressure(n_boxes=1, n_species=2)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas,
        device=device,
        vapor_pressure=vapor_pressure,
    )

    wrong_device = "cpu" if device == "cuda" else "cuda"
    if wrong_device == "cuda" and not cuda_available(wp):
        pytest.skip("CUDA not available for mismatch test")

    gpu_particles.density = wp.zeros(2, dtype=wp.float64, device=wrong_device)

    with pytest.raises(ValueError, match="particle density device mismatch"):
        condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature=298.15,
            pressure=101325.0,
            time_step=1.0,
        )


def test_condensation_step_gpu_rejects_gas_concentration_device_mismatch(
    device: str,
) -> None:
    """Condensation rejects gas concentration on wrong device."""
    particles = _make_particle_data(n_boxes=1, n_particles=2, n_species=2)
    gas = _make_gas_data(n_boxes=1, n_species=2)
    vapor_pressure = _make_vapor_pressure(n_boxes=1, n_species=2)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas,
        device=device,
        vapor_pressure=vapor_pressure,
    )

    wrong_device = "cpu" if device == "cuda" else "cuda"
    if wrong_device == "cuda" and not cuda_available(wp):
        pytest.skip("CUDA not available for mismatch test")

    gpu_gas.concentration = wp.zeros(
        (1, 2),
        dtype=wp.float64,
        device=wrong_device,
    )

    with pytest.raises(ValueError, match="gas concentration device mismatch"):
        condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature=298.15,
            pressure=101325.0,
            time_step=1.0,
        )


def test_condensation_step_gpu_rejects_vapor_pressure_device_mismatch(
    device: str,
) -> None:
    """Condensation rejects vapor pressure on wrong device."""
    particles = _make_particle_data(n_boxes=1, n_particles=2, n_species=2)
    gas = _make_gas_data(n_boxes=1, n_species=2)
    vapor_pressure = _make_vapor_pressure(n_boxes=1, n_species=2)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas,
        device=device,
        vapor_pressure=vapor_pressure,
    )

    wrong_device = "cpu" if device == "cuda" else "cuda"
    if wrong_device == "cuda" and not cuda_available(wp):
        pytest.skip("CUDA not available for mismatch test")

    gpu_gas.vapor_pressure = wp.zeros(
        (1, 2),
        dtype=wp.float64,
        device=wrong_device,
    )

    with pytest.raises(ValueError, match="gas vapor pressure device mismatch"):
        condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature=298.15,
            pressure=101325.0,
            time_step=1.0,
        )


def test_particle_radius_from_volume_wp_matches_numpy(device: str) -> None:
    """Warp helper for radius matches NumPy calculation."""
    volumes = np.array([1.0e-18, 8.0e-18], dtype=np.float64)
    expected = np.cbrt(3.0 * volumes / (4.0 * np.pi))
    volumes_wp = wp.array(volumes, dtype=wp.float64, device=device)
    radii_wp = wp.zeros(len(volumes), dtype=wp.float64, device=device)

    @wp.kernel
    def _radius_kernel(
        total_volume: Any,
        radii_out: Any,
    ) -> None:
        idx = wp.tid()
        radii_out[idx] = particle_radius_from_volume_wp(total_volume[idx])

    wp.launch(
        _radius_kernel,
        dim=len(volumes),
        inputs=[volumes_wp, radii_wp],
        device=device,
    )
    wp.synchronize()

    npt.assert_allclose(radii_wp.numpy(), expected, rtol=1.0e-8)
