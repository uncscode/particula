"""End-to-end tests for GPU condensation kernels."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
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


@dataclass(frozen=True)
class CondensationStiffnessCase:
    """Define a deterministic fixed-shape condensation stress case.

    Attributes:
        name: Recorded case name used by the stiffness sweep helpers.
        n_boxes: Number of spatial boxes in the deterministic case.
        n_particles: Number of particles per box.
        n_species: Number of condensable species.
        time_step: Baseline timestep used by legacy single-step checks.
        temperature: Baseline scalar temperature in K.
        pressure: Baseline scalar pressure in Pa.
        particle_mass_scale: Multiplier applied to the seed particle masses.
        gas_concentration_scale: Multiplier applied to the seed gas field.
        vapor_pressure_scale: Multiplier applied to the seed vapor pressure.
        box_temperature_step: Per-box temperature increment for multi-box cases.
        box_pressure_step: Per-box pressure increment for multi-box cases.
        zero_mass_particles: Particle indices forced to zero initial mass.
        zero_concentration_particles: Particle indices forced inactive.
    """

    name: str
    n_boxes: int
    n_particles: int
    n_species: int
    time_step: float
    temperature: float = 298.15
    pressure: float = 101325.0
    particle_mass_scale: float = 1.0
    gas_concentration_scale: float = 1.0
    vapor_pressure_scale: float = 1.0
    box_temperature_step: float = 0.0
    box_pressure_step: float = 0.0
    zero_mass_particles: tuple[tuple[int, int], ...] = ()
    zero_concentration_particles: tuple[tuple[int, int], ...] = ()

    def build_particle_data(self) -> ParticleData:
        """Build deterministic particle data for the configured case."""
        particles = _make_particle_data(
            n_boxes=self.n_boxes,
            n_particles=self.n_particles,
            n_species=self.n_species,
        )
        particles.masses *= self.particle_mass_scale
        for box_idx, particle_idx in self.zero_mass_particles:
            particles.masses[box_idx, particle_idx, :] = 0.0
        for box_idx, particle_idx in self.zero_concentration_particles:
            particles.concentration[box_idx, particle_idx] = 0.0
        return particles

    def build_gas_data(self) -> GasData:
        """Build deterministic gas data for the configured case."""
        gas = _make_gas_data(self.n_boxes, self.n_species)
        gas.concentration *= self.gas_concentration_scale
        return gas

    def build_vapor_pressure(self) -> np.ndarray:
        """Build deterministic vapor pressure data for the configured case."""
        vapor_pressure = _make_vapor_pressure(self.n_boxes, self.n_species)
        return vapor_pressure * self.vapor_pressure_scale

    def build_environment_data(self) -> EnvironmentData:
        """Build deterministic environment data for the configured case."""
        environment = _make_environment_data(
            self.n_boxes,
            self.n_species,
            temperature=self.temperature,
            pressure=self.pressure,
        )
        if self.n_boxes > 1:
            environment.temperature = (
                self.temperature
                + self.box_temperature_step
                * np.arange(
                    self.n_boxes,
                    dtype=np.float64,
                )
            )
            environment.pressure = (
                self.pressure
                + self.box_pressure_step
                * np.arange(
                    self.n_boxes,
                    dtype=np.float64,
                )
            )
        return environment

    def temperature_array(self) -> np.ndarray:
        """Return the per-box temperature inputs for the case."""
        return self.build_environment_data().temperature

    def pressure_array(self) -> np.ndarray:
        """Return the per-box pressure inputs for the case."""
        return self.build_environment_data().pressure


@dataclass(frozen=True)
class CondensationStiffnessClassification:
    """Store the particle-only stability classification for one trial.

    Attributes:
        label: Stability label derived from the recorded classification rule.
        mass_nonnegative: Whether all final particle masses stay non-negative.
        values_finite: Whether particle, gas, and vapor-pressure values are finite.
        metadata_valid: Whether shape and dtype checks passed for the case.
        zero_mass_change_stable: Whether zero-mass entries remain unchanged.
        max_fractional_mass_change: Largest positive-mass fractional change.
        threshold: Stability threshold applied to the trial.
        particle_only_update: Whether the contract only updates particle masses.
    """

    label: str
    mass_nonnegative: bool
    values_finite: bool
    metadata_valid: bool
    zero_mass_change_stable: bool
    max_fractional_mass_change: float
    threshold: float
    particle_only_update: bool = True


def _make_condensation_stiffness_cases() -> tuple[
    CondensationStiffnessCase, ...
]:
    """Return the compact reusable stiffness catalog."""
    return (
        CondensationStiffnessCase(
            name="nanometer",
            n_boxes=1,
            n_particles=4,
            n_species=2,
            time_step=0.05,
            particle_mass_scale=0.15,
            gas_concentration_scale=4.0,
            vapor_pressure_scale=0.6,
        ),
        CondensationStiffnessCase(
            name="accumulation_mode",
            n_boxes=1,
            n_particles=3,
            n_species=2,
            time_step=0.4,
            particle_mass_scale=15.0,
            gas_concentration_scale=1.5,
            vapor_pressure_scale=0.85,
        ),
        CondensationStiffnessCase(
            name="droplet_like",
            n_boxes=2,
            n_particles=2,
            n_species=2,
            time_step=4.0,
            particle_mass_scale=8.0e4,
            gas_concentration_scale=0.4,
            vapor_pressure_scale=1.25,
            box_temperature_step=4.0,
            box_pressure_step=-750.0,
        ),
    )


_RECORDED_TIMESTEP_GRID_BY_CASE: dict[str, tuple[float, ...]] = {
    "nanometer": (0.00005, 0.05, 50.0),
    "accumulation_mode": (0.004, 0.4, 40.0),
    "droplet_like": (0.04, 4.0, 400.0),
}


_RECORDED_STIFFNESS_THRESHOLD_BY_CASE: dict[str, tuple[float, ...]] = {
    "nanometer": (1.0, 0.5, 0.5),
    "accumulation_mode": (1.0, 0.5, 0.5),
    "droplet_like": (1.0, 0.5, 0.5),
}


@dataclass(frozen=True)
class CondensationStiffnessTrialRecord:
    """Capture one recorded-grid timestep trial for a stiffness case.

    Attributes:
        case_name: Name of the deterministic stiffness case.
        time_step: Executed timestep for the trial.
        configured_time_step: Matching timestep from the recorded grid.
        timestep_index: Recorded-grid position for the trial.
        environment_input_mode: Whether scalar or Warp-array inputs were used.
        classification: Particle-only stability classification for the result.
        gas_unchanged: Whether CPU-side gas concentration stayed unchanged.
        reuses_caller_mass_transfer_buffer: Whether the caller buffer was reused.
        mass_transfer_has_nonzero_values: Whether the buffer was populated.
        mass_transfer_changed_from_previous_trial: Whether reuse overwrote values.
        final_masses: Final particle masses copied back from Warp.
        initial_masses: Initial particle masses used for classification.
        mass_transfer_values: CPU copy of the reused mass-transfer buffer.
    """

    case_name: str
    time_step: float
    configured_time_step: float
    timestep_index: int
    environment_input_mode: str
    classification: CondensationStiffnessClassification
    gas_unchanged: bool
    reuses_caller_mass_transfer_buffer: bool
    mass_transfer_has_nonzero_values: bool
    mass_transfer_changed_from_previous_trial: bool
    final_masses: np.ndarray
    initial_masses: np.ndarray
    mass_transfer_values: np.ndarray


def _validate_stiffness_case_metadata(
    case: CondensationStiffnessCase,
    particles: ParticleData,
    gas: GasData,
    vapor_pressure: np.ndarray,
) -> None:
    """Validate fixed-shape and dtype expectations for a stiffness case."""
    expected_particle_shape = (case.n_boxes, case.n_particles, case.n_species)
    expected_concentration_shape = (case.n_boxes, case.n_particles)
    expected_gas_shape = (case.n_boxes, case.n_species)
    if particles.masses.shape != expected_particle_shape:
        raise ValueError(
            "particle masses shape does not match declared case metadata"
        )
    if particles.concentration.shape != expected_concentration_shape:
        raise ValueError(
            "particle concentration shape does not match declared case metadata"
        )
    if gas.concentration.shape != expected_gas_shape:
        raise ValueError(
            "gas concentration shape does not match declared case metadata"
        )
    if vapor_pressure.shape != expected_gas_shape:
        raise ValueError(
            "vapor pressure shape does not match declared case metadata"
        )

    dtype_expectations = (
        ("particle masses", particles.masses),
        ("particle concentration", particles.concentration),
        ("particle density", particles.density),
        ("particle volume", particles.volume),
        ("gas concentration", gas.concentration),
        ("gas molar mass", gas.molar_mass),
        ("vapor pressure", vapor_pressure),
    )
    for name, values in dtype_expectations:
        if values.dtype != np.float64:
            raise TypeError(f"{name} must use np.float64")


def _particle_mass_is_nonnegative(masses: np.ndarray) -> bool:
    """Return whether all particle masses are non-negative."""
    return bool(np.all(masses >= 0.0))


def _particle_values_are_finite(*arrays: np.ndarray) -> bool:
    """Return whether all provided arrays contain only finite values."""
    return all(bool(np.all(np.isfinite(array))) for array in arrays)


def _fractional_mass_change_per_bin(
    initial_masses: np.ndarray,
    final_masses: np.ndarray,
) -> np.ndarray:
    """Return per-bin fractional mass change for positive-mass entries only."""
    if initial_masses.shape != final_masses.shape:
        raise ValueError("initial and final masses must have matching shape")
    initial = np.asarray(initial_masses, dtype=np.float64)
    final = np.asarray(final_masses, dtype=np.float64)
    change = np.abs(final - initial)
    fractional_change = np.zeros_like(change)
    positive_mask = initial > 0.0
    fractional_change[positive_mask] = (
        change[positive_mask] / initial[positive_mask]
    )
    return fractional_change


def _zero_mass_entries_remain_stable(
    initial_masses: np.ndarray,
    final_masses: np.ndarray,
) -> bool:
    """Return whether zero-mass entries remain unchanged."""
    initial = np.asarray(initial_masses, dtype=np.float64)
    final = np.asarray(final_masses, dtype=np.float64)
    if initial.shape != final.shape:
        raise ValueError("initial and final masses must have matching shape")
    zero_mask = initial == 0.0
    return bool(np.all(final[zero_mask] == 0.0))


def _classify_particle_only_condensation_stiffness(
    case: CondensationStiffnessCase,
    initial_masses: np.ndarray,
    final_masses: np.ndarray,
    gas: GasData,
    vapor_pressure: np.ndarray,
    *,
    max_fractional_change: float,
) -> CondensationStiffnessClassification:
    """Classify particle-only condensation behavior as stable or unstable."""
    initial = np.asarray(initial_masses, dtype=np.float64)
    final = np.asarray(final_masses, dtype=np.float64)
    particles = case.build_particle_data()
    particles.masses = final
    _validate_stiffness_case_metadata(case, particles, gas, vapor_pressure)
    fractional_change = _fractional_mass_change_per_bin(
        initial,
        final,
    )
    zero_mass_change_stable = _zero_mass_entries_remain_stable(
        initial,
        final,
    )
    mass_nonnegative = _particle_mass_is_nonnegative(final)
    values_finite = _particle_values_are_finite(
        initial,
        final,
        gas.concentration,
        vapor_pressure,
    )
    max_change = (
        float(np.max(fractional_change)) if fractional_change.size else 0.0
    )
    label = "stable"
    if (
        max_change > max_fractional_change
        or not mass_nonnegative
        or not values_finite
        or not zero_mass_change_stable
    ):
        label = "unstable"
    return CondensationStiffnessClassification(
        label=label,
        mass_nonnegative=mass_nonnegative,
        values_finite=values_finite,
        metadata_valid=True,
        zero_mass_change_stable=zero_mass_change_stable,
        max_fractional_mass_change=max_change,
        threshold=max_fractional_change,
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


def _record_condensation_stiffness_trials(
    case: CondensationStiffnessCase,
    device: str,
) -> list[CondensationStiffnessTrialRecord]:
    """Run the recorded timestep grid for one deterministic stiffness case.

    The helper rebuilds fresh particle, gas, and vapor-pressure inputs for each
    recorded timestep, reuses a caller-owned Warp ``mass_transfer`` buffer
    across the full sweep, and records whether the particle-only path leaves the
    CPU gas concentration unchanged.

    Args:
        case: Deterministic stiffness case to execute.
        device: Warp device name used for the sweep.

    Returns:
        Ordered recorded-grid trial records for the requested case.
    """
    recorded_timesteps = _RECORDED_TIMESTEP_GRID_BY_CASE[case.name]
    stability_thresholds = _RECORDED_STIFFNESS_THRESHOLD_BY_CASE[case.name]
    mass_transfer = wp.zeros(
        (case.n_boxes, case.n_particles, case.n_species),
        dtype=wp.float64,
        device=device,
    )
    previous_mass_transfer_values: np.ndarray | None = None
    environment_input_mode = "direct_warp_arrays"
    temperature: float | Any = case.temperature
    pressure: float | Any = case.pressure
    if case.n_boxes == 1:
        environment_input_mode = "scalar_inputs"
    else:
        temperature = wp.array(
            case.temperature_array(),
            dtype=wp.float64,
            device=device,
        )
        pressure = wp.array(
            case.pressure_array(),
            dtype=wp.float64,
            device=device,
        )

    records: list[CondensationStiffnessTrialRecord] = []
    for timestep_index, time_step in enumerate(recorded_timesteps):
        particles = case.build_particle_data()
        gas = case.build_gas_data()
        vapor_pressure = case.build_vapor_pressure()
        initial_masses = particles.masses.copy()
        initial_gas_concentration = gas.concentration.copy()

        gpu_result, returned_mass_transfer = _run_gpu_step(
            particles,
            gas,
            vapor_pressure,
            temperature=temperature,
            pressure=pressure,
            time_step=time_step,
            device=device,
            mass_transfer=mass_transfer,
        )
        mass_transfer_values = returned_mass_transfer.numpy().copy()
        classification = _classify_particle_only_condensation_stiffness(
            case,
            initial_masses,
            gpu_result.masses,
            gas,
            vapor_pressure,
            max_fractional_change=stability_thresholds[timestep_index],
        )
        records.append(
            CondensationStiffnessTrialRecord(
                case_name=case.name,
                time_step=time_step,
                configured_time_step=recorded_timesteps[timestep_index],
                timestep_index=timestep_index,
                environment_input_mode=environment_input_mode,
                classification=classification,
                gas_unchanged=bool(
                    np.array_equal(gas.concentration, initial_gas_concentration)
                ),
                reuses_caller_mass_transfer_buffer=(
                    returned_mass_transfer is mass_transfer
                ),
                mass_transfer_has_nonzero_values=bool(
                    np.any(mass_transfer_values != 0.0)
                ),
                mass_transfer_changed_from_previous_trial=(
                    previous_mass_transfer_values is not None
                    and not np.array_equal(
                        mass_transfer_values,
                        previous_mass_transfer_values,
                    )
                ),
                final_masses=gpu_result.masses.copy(),
                initial_masses=initial_masses,
                mass_transfer_values=mass_transfer_values,
            )
        )
        previous_mass_transfer_values = mass_transfer_values

    return records


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


def test_condensation_step_gpu_uniform_direct_arrays_match_scalar_results(
    device: str,
) -> None:
    """Uniform per-box direct arrays preserve legacy scalar physics."""
    particles = _make_particle_data(n_boxes=2, n_particles=2, n_species=1)
    gas = _make_gas_data(n_boxes=2, n_species=1)
    vapor_pressure = _make_vapor_pressure(n_boxes=2, n_species=1)

    _, scalar_mass_transfer = _run_gpu_step(
        particles,
        gas,
        vapor_pressure,
        temperature=298.15,
        pressure=101325.0,
        time_step=0.1,
        device=device,
    )
    _, uniform_mass_transfer = _run_gpu_step(
        particles,
        gas,
        vapor_pressure,
        temperature=wp.array([298.15, 298.15], dtype=wp.float64, device=device),
        pressure=wp.array(
            [101325.0, 101325.0],
            dtype=wp.float64,
            device=device,
        ),
        time_step=0.1,
        device=device,
    )

    npt.assert_allclose(
        uniform_mass_transfer.numpy(),
        scalar_mass_transfer.numpy(),
        rtol=1.0e-10,
    )


def test_condensation_stiffness_case_builds_named_regimes() -> None:
    """Stiffness catalog exposes the expected deterministic baseline cases."""
    cases = _make_condensation_stiffness_cases()

    assert [case.name for case in cases] == [
        "nanometer",
        "accumulation_mode",
        "droplet_like",
    ]

    for case in cases:
        particles = case.build_particle_data()
        gas = case.build_gas_data()
        vapor_pressure = case.build_vapor_pressure()

        _validate_stiffness_case_metadata(case, particles, gas, vapor_pressure)
        assert particles.masses.shape == (
            case.n_boxes,
            case.n_particles,
            case.n_species,
        )
        assert particles.masses.dtype == np.float64
        assert gas.concentration.dtype == np.float64
        assert vapor_pressure.dtype == np.float64


def test_condensation_stiffness_case_zero_mass_preserves_fixed_shape() -> None:
    """Zero-mass helper coverage preserves deterministic fixed-shape outputs."""
    case = CondensationStiffnessCase(
        name="zero_mass_edge",
        n_boxes=1,
        n_particles=2,
        n_species=2,
        time_step=0.1,
        zero_mass_particles=((0, 1),),
    )

    particles = case.build_particle_data()
    gas = case.build_gas_data()
    vapor_pressure = case.build_vapor_pressure()

    _validate_stiffness_case_metadata(case, particles, gas, vapor_pressure)
    assert particles.masses.shape == (1, 2, 2)
    npt.assert_allclose(particles.masses[0, 1, :], 0.0)


def test_condensation_stiffness_case_direct_arrays_match_scalar_inputs(
    device: str,
) -> None:
    """Representative case supports scalar and direct ``(n_boxes,)`` inputs."""
    if device != "cpu":
        pytest.skip("Stiffness baseline runs on Warp CPU")

    case = next(
        candidate
        for candidate in _make_condensation_stiffness_cases()
        if candidate.name == "droplet_like"
    )
    particles = case.build_particle_data()
    gas = case.build_gas_data()
    vapor_pressure = case.build_vapor_pressure()

    _, scalar_mass_transfer = _run_gpu_step(
        particles,
        gas,
        vapor_pressure,
        temperature=case.temperature,
        pressure=case.pressure,
        time_step=case.time_step,
        device=device,
    )
    _, array_mass_transfer = _run_gpu_step(
        particles,
        gas,
        vapor_pressure,
        temperature=wp.array(
            case.temperature_array(),
            dtype=wp.float64,
            device=device,
        ),
        pressure=wp.array(
            case.pressure_array(),
            dtype=wp.float64,
            device=device,
        ),
        time_step=case.time_step,
        device=device,
    )

    assert scalar_mass_transfer.shape == array_mass_transfer.shape
    assert array_mass_transfer.dtype == wp.float64
    assert array_mass_transfer.shape == (
        case.n_boxes,
        case.n_particles,
        case.n_species,
    )
    npt.assert_allclose(
        array_mass_transfer.numpy()[0],
        scalar_mass_transfer.numpy()[0],
        rtol=1.0e-10,
    )
    assert (
        np.max(
            np.abs(
                array_mass_transfer.numpy()[1] - scalar_mass_transfer.numpy()[1]
            )
        )
        > 0.0
    )


def test_stiffness_metrics_reject_shape_metadata_mismatch() -> None:
    """Metadata validation fails clearly on declared shape mismatches."""
    case = CondensationStiffnessCase(
        name="shape_mismatch",
        n_boxes=1,
        n_particles=2,
        n_species=2,
        time_step=0.1,
    )
    particles = case.build_particle_data()
    gas = case.build_gas_data()
    vapor_pressure = case.build_vapor_pressure()
    particles.masses = np.zeros((1, 3, 2), dtype=np.float64)

    with pytest.raises(
        ValueError,
        match="particle masses shape does not match declared case metadata",
    ):
        _validate_stiffness_case_metadata(case, particles, gas, vapor_pressure)


def test_stiffness_metrics_reject_dtype_metadata_mismatch() -> None:
    """Metadata validation fails clearly on dtype mismatches."""
    case = CondensationStiffnessCase(
        name="dtype_mismatch",
        n_boxes=1,
        n_particles=2,
        n_species=2,
        time_step=0.1,
    )
    particles = case.build_particle_data()
    gas = case.build_gas_data()
    vapor_pressure = case.build_vapor_pressure().astype(np.float32)

    with pytest.raises(TypeError, match="vapor pressure must use np.float64"):
        _validate_stiffness_case_metadata(case, particles, gas, vapor_pressure)


def test_fractional_mass_change_zero_mass_returns_documented_result() -> None:
    """Zero-initial-mass entries report zero fractional change when unchanged."""
    initial = np.array([[[0.0, 2.0e-18]]], dtype=np.float64)
    final = np.array([[[0.0, 3.0e-18]]], dtype=np.float64)

    fractional_change = _fractional_mass_change_per_bin(initial, final)

    npt.assert_allclose(fractional_change, [[[0.0, 0.5]]])
    assert _zero_mass_entries_remain_stable(initial, final)


def test_fractional_mass_change_rejects_shape_mismatch() -> None:
    """Fractional mass change helper rejects mismatched shapes."""
    initial = np.zeros((1, 1, 1), dtype=np.float64)
    final = np.zeros((1, 2, 1), dtype=np.float64)

    with pytest.raises(
        ValueError,
        match="initial and final masses must have matching shape",
    ):
        _fractional_mass_change_per_bin(initial, final)


def test_zero_mass_entries_remain_stable_rejects_shape_mismatch() -> None:
    """Zero-mass stability helper rejects mismatched shapes."""
    initial = np.zeros((1, 1, 1), dtype=np.float64)
    final = np.zeros((1, 2, 1), dtype=np.float64)

    with pytest.raises(
        ValueError,
        match="initial and final masses must have matching shape",
    ):
        _zero_mass_entries_remain_stable(initial, final)


def test_condensation_stiffness_metric_helpers_detect_invalid_values() -> None:
    """Metric helpers expose non-negativity and finiteness checks."""
    good = np.array([0.0, 1.0], dtype=np.float64)
    bad = np.array([1.0, np.inf], dtype=np.float64)

    assert _particle_mass_is_nonnegative(good)
    assert not _particle_mass_is_nonnegative(np.array([-1.0], dtype=np.float64))
    assert _particle_values_are_finite(good)
    assert not _particle_values_are_finite(good, bad)


def test_condensation_stiffness_classification_threshold_boundary() -> None:
    """Exact threshold equality remains classified as stable."""
    case = CondensationStiffnessCase(
        name="threshold_boundary",
        n_boxes=1,
        n_particles=1,
        n_species=1,
        time_step=0.1,
    )
    initial = np.array([[[1.0e-18]]], dtype=np.float64)
    final = np.array([[[1.5e-18]]], dtype=np.float64)
    gas = case.build_gas_data()
    vapor_pressure = case.build_vapor_pressure()

    classification = _classify_particle_only_condensation_stiffness(
        case,
        initial,
        final,
        gas,
        vapor_pressure,
        max_fractional_change=0.5,
    )

    assert classification.label == "stable"
    assert classification.max_fractional_mass_change == pytest.approx(0.5)
    assert classification.particle_only_update


def test_condensation_stiffness_classification_marks_large_change_unstable() -> (
    None
):
    """Large fractional mass changes are classified as unstable."""
    case = CondensationStiffnessCase(
        name="large_change",
        n_boxes=1,
        n_particles=1,
        n_species=1,
        time_step=1.0,
    )
    initial = np.array([[[1.0e-18]]], dtype=np.float64)
    final = np.array([[[3.0e-18]]], dtype=np.float64)

    classification = _classify_particle_only_condensation_stiffness(
        case,
        initial,
        final,
        case.build_gas_data(),
        case.build_vapor_pressure(),
        max_fractional_change=0.5,
    )

    assert classification.label == "unstable"
    assert classification.max_fractional_mass_change > classification.threshold
    assert classification.particle_only_update


def test_condensation_stiffness_classification_marks_zero_mass_growth_unstable() -> (
    None
):
    """Zero-initial-mass growth is unstable even if fractional change is zero."""
    case = CondensationStiffnessCase(
        name="zero_mass_growth",
        n_boxes=1,
        n_particles=1,
        n_species=1,
        time_step=0.1,
    )
    initial = np.array([[[0.0]]], dtype=np.float64)
    final = np.array([[[1.0e-21]]], dtype=np.float64)

    classification = _classify_particle_only_condensation_stiffness(
        case,
        initial,
        final,
        case.build_gas_data(),
        case.build_vapor_pressure(),
        max_fractional_change=0.0,
    )

    assert classification.label == "unstable"
    assert classification.zero_mass_change_stable is False
    assert classification.max_fractional_mass_change == 0.0


def test_condensation_stiffness_classification_marks_nonfinite_values_unstable() -> (
    None
):
    """Non-finite particle results are unstable regardless of threshold."""
    case = CondensationStiffnessCase(
        name="nonfinite_values",
        n_boxes=1,
        n_particles=1,
        n_species=1,
        time_step=0.1,
    )
    initial = np.array([[[1.0e-18]]], dtype=np.float64)
    final = np.array([[[np.nan]]], dtype=np.float64)

    classification = _classify_particle_only_condensation_stiffness(
        case,
        initial,
        final,
        case.build_gas_data(),
        case.build_vapor_pressure(),
        max_fractional_change=10.0,
    )

    assert classification.label == "unstable"
    assert classification.values_finite is False


def test_condensation_stiffness_classification_explicitly_marks_particle_only(
    device: str,
) -> None:
    """Classification exposes the particle-only caveat without gas claims."""
    if device != "cpu":
        pytest.skip("Stiffness baseline runs on Warp CPU")

    case = next(
        candidate
        for candidate in _make_condensation_stiffness_cases()
        if candidate.name == "accumulation_mode"
    )
    particles = case.build_particle_data()
    gas = case.build_gas_data()
    vapor_pressure = case.build_vapor_pressure()
    initial_gas = gas.concentration.copy()
    initial_masses = particles.masses.copy()

    gpu_result, _ = _run_gpu_step(
        particles,
        gas,
        vapor_pressure,
        temperature=case.temperature,
        pressure=case.pressure,
        time_step=case.time_step,
        device=device,
    )
    classification = _classify_particle_only_condensation_stiffness(
        case,
        initial_masses,
        gpu_result.masses,
        gas,
        vapor_pressure,
        max_fractional_change=10.0,
    )

    assert classification.particle_only_update is True
    npt.assert_allclose(gas.concentration, initial_gas)


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


@pytest.mark.parametrize(
    ("temperature", "pressure", "message"),
    [
        (0.0, 101325.0, "temperature must be finite and > 0"),
        (298.15, 0.0, "pressure must be finite and > 0"),
        (float("nan"), 101325.0, "temperature must be finite and > 0"),
    ],
)
def test_condensation_step_gpu_invalid_scalar_domains_short_circuit_before_launch(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    temperature: float,
    pressure: float,
    message: str,
) -> None:
    """Invalid scalar domains fail before any Warp launch work."""
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

    with pytest.raises(ValueError, match=message):
        condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature=temperature,
            pressure=pressure,
            time_step=0.1,
        )

    assert calls == []


def test_condensation_step_gpu_invalid_environment_domains_short_circuit_before_launch(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
) -> None:
    """Invalid environment arrays fail before any Warp launch work."""
    particles = _make_particle_data(n_boxes=1, n_particles=1, n_species=1)
    gas = _make_gas_data(n_boxes=1, n_species=1)
    vapor_pressure = _make_vapor_pressure(n_boxes=1, n_species=1)
    environment = to_warp_environment_data(
        _make_environment_data(n_boxes=1, n_species=1),
        device=device,
    )
    environment.pressure = wp.array([0.0], dtype=wp.float64, device=device)
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
        match="environment.pressure must be finite and > 0",
    ):
        condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature=None,
            pressure=None,
            time_step=0.1,
            environment=environment,
        )

    assert calls == []


@pytest.mark.parametrize("field_name", ["temperature", "pressure"])
def test_condensation_step_gpu_missing_environment_field_short_circuits_before_launch(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    field_name: str,
) -> None:
    """Malformed environment payloads raise stable errors before launch."""
    particles = _make_particle_data(n_boxes=1, n_particles=1, n_species=1)
    gas = _make_gas_data(n_boxes=1, n_species=1)
    vapor_pressure = _make_vapor_pressure(n_boxes=1, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas,
        device=device,
        vapor_pressure=vapor_pressure,
    )
    original_masses = np.asarray(gpu_particles.masses.numpy()).copy()
    mass_transfer = wp.full(
        (1, 1, 1),
        wp.float64(5.0),
        dtype=wp.float64,
        device=device,
    )
    original_mass_transfer = np.asarray(mass_transfer.numpy()).copy()

    class _MalformedEnvironment:
        def __init__(self) -> None:
            self.temperature = wp.array(
                [298.15], dtype=wp.float64, device=device
            )
            self.pressure = wp.array(
                [101325.0], dtype=wp.float64, device=device
            )

    environment = _MalformedEnvironment()
    delattr(environment, field_name)
    calls: list[str] = []

    def _unexpected_launch(*args: Any, **kwargs: Any) -> None:
        calls.append("launch")
        raise AssertionError("wp.launch should not be called")

    monkeypatch.setattr(condensation_module.wp, "launch", _unexpected_launch)

    with pytest.raises(
        ValueError,
        match=rf"environment\.{field_name} must be a Warp array with shape "
        r"\(n_boxes,\)",
    ):
        condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature=None,
            pressure=None,
            time_step=0.1,
            mass_transfer=mass_transfer,
            environment=environment,
        )

    assert calls == []
    npt.assert_allclose(gpu_particles.masses.numpy(), original_masses)
    npt.assert_allclose(mass_transfer.numpy(), original_mass_transfer)


@pytest.mark.parametrize(
    ("temperature_values", "pressure_values", "message"),
    [
        (
            np.array([298.15, 0.0], dtype=np.float64),
            np.array([101325.0, 101325.0], dtype=np.float64),
            "temperature must be finite and > 0",
        ),
        (
            np.array([298.15, 298.15], dtype=np.float64),
            np.array([101325.0, np.nan], dtype=np.float64),
            "pressure must be finite and > 0",
        ),
    ],
)
def test_condensation_step_gpu_invalid_direct_array_domains_short_circuit_before_launch(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    temperature_values: np.ndarray,
    pressure_values: np.ndarray,
    message: str,
) -> None:
    """Invalid direct array domains fail before any Warp launch work."""
    particles = _make_particle_data(n_boxes=2, n_particles=1, n_species=1)
    gas = _make_gas_data(n_boxes=2, n_species=1)
    vapor_pressure = _make_vapor_pressure(n_boxes=2, n_species=1)
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
            temperature=wp.array(
                temperature_values,
                dtype=wp.float64,
                device=device,
            ),
            pressure=wp.array(
                pressure_values,
                dtype=wp.float64,
                device=device,
            ),
            time_step=0.1,
        )

    assert calls == []


@pytest.mark.parametrize(
    ("temperature", "pressure", "message"),
    [
        (
            np.array([298.15], dtype=np.float64),
            101325.0,
            "temperature must be a scalar or Warp array with shape",
        ),
        (
            298.15,
            np.array([101325.0], dtype=np.float64),
            "pressure must be a scalar or Warp array with shape",
        ),
    ],
)
def test_condensation_step_gpu_rejects_direct_non_warp_arrays_before_launch(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    temperature: float | np.ndarray,
    pressure: float | np.ndarray,
    message: str,
) -> None:
    """Unsupported direct non-Warp arrays fail before launch or mutation."""
    particles = _make_particle_data(n_boxes=1, n_particles=1, n_species=1)
    gas = _make_gas_data(n_boxes=1, n_species=1)
    vapor_pressure = _make_vapor_pressure(n_boxes=1, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas,
        device=device,
        vapor_pressure=vapor_pressure,
    )
    original_masses = np.asarray(gpu_particles.masses.numpy()).copy()
    mass_transfer = wp.full(
        (1, 1, 1),
        wp.float64(9.0),
        dtype=wp.float64,
        device=device,
    )
    original_mass_transfer = np.asarray(mass_transfer.numpy()).copy()
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
            mass_transfer=mass_transfer,
        )

    assert calls == []
    npt.assert_allclose(gpu_particles.masses.numpy(), original_masses)
    npt.assert_allclose(mass_transfer.numpy(), original_mass_transfer)


@pytest.mark.parametrize(
    ("field_name", "field_values", "message"),
    [
        (
            "temperature",
            np.array([np.nan, 298.15], dtype=np.float64),
            "environment.temperature must be finite and > 0",
        ),
        (
            "pressure",
            np.array([101325.0, 0.0], dtype=np.float64),
            "environment.pressure must be finite and > 0",
        ),
    ],
)
def test_condensation_step_gpu_invalid_environment_array_domains_short_circuit_before_launch(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    field_name: str,
    field_values: np.ndarray,
    message: str,
) -> None:
    """Invalid environment arrays fail before any Warp launch work."""
    particles = _make_particle_data(n_boxes=2, n_particles=1, n_species=1)
    gas = _make_gas_data(n_boxes=2, n_species=1)
    vapor_pressure = _make_vapor_pressure(n_boxes=2, n_species=1)
    environment = to_warp_environment_data(
        _make_environment_data(n_boxes=2, n_species=1),
        device=device,
    )
    setattr(
        environment,
        field_name,
        wp.array(field_values, dtype=wp.float64, device=device),
    )
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
            temperature=None,
            pressure=None,
            time_step=0.1,
            environment=environment,
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


def test_condensation_step_gpu_explicit_environment_matches_direct_arrays(
    device: str,
) -> None:
    """Accepted ``environment=...`` arrays match accepted direct-array inputs."""
    particles = _make_particle_data(n_boxes=2, n_particles=2, n_species=1)
    gas = _make_gas_data(n_boxes=2, n_species=1)
    vapor_pressure = _make_vapor_pressure(n_boxes=2, n_species=1)
    temperature_values = np.array([298.15, 301.15], dtype=np.float64)
    pressure_values = np.array([101325.0, 100800.0], dtype=np.float64)
    environment = to_warp_environment_data(
        _make_environment_data(n_boxes=2, n_species=1),
        device=device,
    )
    environment.temperature = wp.array(
        temperature_values,
        dtype=wp.float64,
        device=device,
    )
    environment.pressure = wp.array(
        pressure_values,
        dtype=wp.float64,
        device=device,
    )

    _, direct_mass_transfer = _run_gpu_step(
        particles,
        gas,
        vapor_pressure,
        temperature=wp.array(
            temperature_values,
            dtype=wp.float64,
            device=device,
        ),
        pressure=wp.array(
            pressure_values,
            dtype=wp.float64,
            device=device,
        ),
        time_step=0.1,
        device=device,
    )
    _, environment_mass_transfer = _run_gpu_step(
        particles,
        gas,
        vapor_pressure,
        temperature=None,
        pressure=None,
        time_step=0.1,
        device=device,
        environment=environment,
    )

    npt.assert_allclose(
        environment_mass_transfer.numpy(),
        direct_mass_transfer.numpy(),
        rtol=1.0e-10,
    )


def test_condensation_step_gpu_success_does_not_mutate_environment_inputs(
    device: str,
) -> None:
    """Successful ``environment=...`` execution preserves caller-owned arrays."""
    particles = _make_particle_data(n_boxes=2, n_particles=2, n_species=1)
    gas = _make_gas_data(n_boxes=2, n_species=1)
    vapor_pressure = _make_vapor_pressure(n_boxes=2, n_species=1)
    temperature_values = np.array([298.15, 301.15], dtype=np.float64)
    pressure_values = np.array([101325.0, 100800.0], dtype=np.float64)
    environment = to_warp_environment_data(
        _make_environment_data(n_boxes=2, n_species=1),
        device=device,
    )
    environment.temperature = wp.array(
        temperature_values,
        dtype=wp.float64,
        device=device,
    )
    environment.pressure = wp.array(
        pressure_values,
        dtype=wp.float64,
        device=device,
    )

    _run_gpu_step(
        particles,
        gas,
        vapor_pressure,
        temperature=None,
        pressure=None,
        time_step=0.1,
        device=device,
        environment=environment,
    )

    npt.assert_allclose(environment.temperature.numpy(), temperature_values)
    npt.assert_allclose(environment.pressure.numpy(), pressure_values)


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


def test_condensation_step_gpu_preserves_direct_environment_array_dtypes(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
) -> None:
    """Direct Warp arrays are reused without dtype coercion."""
    particles = _make_particle_data(n_boxes=2, n_particles=2, n_species=1)
    gas = _make_gas_data(n_boxes=2, n_species=1)
    vapor_pressure = _make_vapor_pressure(n_boxes=2, n_species=1)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas, device=device, vapor_pressure=vapor_pressure
    )
    launch_dtypes: list[tuple[Any, Any]] = []
    original_launch = condensation_module.wp.launch

    def _tracking_launch(kernel: Any, *args: Any, **kwargs: Any) -> Any:
        inputs = kwargs.get("inputs", [])
        if (
            getattr(kernel, "key", "")
            == "_prepare_environment_properties_kernel"
        ):
            launch_dtypes.append((inputs[0].dtype, inputs[1].dtype))
        return original_launch(kernel, *args, **kwargs)

    monkeypatch.setattr(condensation_module.wp, "launch", _tracking_launch)

    condensation_step_gpu(
        gpu_particles,
        gpu_gas,
        temperature=wp.array([298.15, 301.15], dtype=wp.float64, device=device),
        pressure=wp.array(
            [101325.0, 100800.0], dtype=wp.float64, device=device
        ),
        time_step=0.1,
    )

    assert launch_dtypes == [(wp.float64, wp.float64)]


def test_condensation_step_gpu_non_uniform_environment_matches_cpu(
    device: str,
) -> None:
    """Non-uniform explicit environment inputs reach box-local physics."""
    particles = _make_particle_data(n_boxes=2, n_particles=2, n_species=1)
    gas = _make_gas_data(n_boxes=2, n_species=1)
    vapor_pressure = _make_vapor_pressure(n_boxes=2, n_species=1)
    temperature_values = np.array([298.15, 308.15], dtype=np.float64)
    pressure_values = np.array([101325.0, 98000.0], dtype=np.float64)
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
    environment = to_warp_environment_data(
        _make_environment_data(n_boxes=2, n_species=1),
        device=device,
    )
    environment.temperature = wp.array(
        temperature_values,
        dtype=wp.float64,
        device=device,
    )
    environment.pressure = wp.array(
        pressure_values,
        dtype=wp.float64,
        device=device,
    )

    _, mass_transfer = _run_gpu_step(
        particles,
        gas,
        vapor_pressure,
        temperature=None,
        pressure=None,
        time_step=0.1,
        device=device,
        environment=environment,
    )

    assert not np.allclose(expected[0], expected[1], rtol=1.0e-6, atol=0.0)
    npt.assert_allclose(mass_transfer.numpy(), expected, rtol=1.0e-10)


def test_condensation_step_gpu_preserves_environment_array_dtypes(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
) -> None:
    """Explicit environment arrays are reused without dtype coercion."""
    particles = _make_particle_data(n_boxes=2, n_particles=2, n_species=1)
    gas = _make_gas_data(n_boxes=2, n_species=1)
    vapor_pressure = _make_vapor_pressure(n_boxes=2, n_species=1)

    class _EnvironmentLike:
        def __init__(self) -> None:
            self.temperature = wp.array(
                [298.15, 301.15], dtype=wp.float64, device=device
            )
            self.pressure = wp.array(
                [101325.0, 100800.0], dtype=wp.float64, device=device
            )

    environment = _EnvironmentLike()
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas, device=device, vapor_pressure=vapor_pressure
    )
    launch_dtypes: list[tuple[Any, Any]] = []
    original_launch = condensation_module.wp.launch

    def _tracking_launch(kernel: Any, *args: Any, **kwargs: Any) -> Any:
        inputs = kwargs.get("inputs", [])
        if (
            getattr(kernel, "key", "")
            == "_prepare_environment_properties_kernel"
        ):
            launch_dtypes.append((inputs[0].dtype, inputs[1].dtype))
        return original_launch(kernel, *args, **kwargs)

    monkeypatch.setattr(condensation_module.wp, "launch", _tracking_launch)

    condensation_step_gpu(
        gpu_particles,
        gpu_gas,
        temperature=None,
        pressure=None,
        time_step=0.1,
        environment=environment,
    )

    assert launch_dtypes == [(wp.float64, wp.float64)]


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


@pytest.mark.parametrize("case", _make_condensation_stiffness_cases())
def test_condensation_stiffness_recorded_grid_contains_stable_and_unstable_trials(
    case: CondensationStiffnessCase,
    device: str,
) -> None:
    """Recorded timestep sweeps include stable and unstable results per case."""
    if device != "cpu":
        pytest.skip("Recorded stiffness sweeps run on Warp CPU")

    records = _record_condensation_stiffness_trials(case, device=device)
    labels = {record.classification.label for record in records}

    assert labels == {"stable", "unstable"}, [
        record.classification.max_fractional_mass_change for record in records
    ]
    for record in records:
        assert record.time_step == record.configured_time_step
        assert record.classification.mass_nonnegative
        assert record.classification.values_finite
        assert record.classification.particle_only_update
        assert record.classification.zero_mass_change_stable
        assert record.gas_unchanged
        assert record.reuses_caller_mass_transfer_buffer
        assert record.mass_transfer_has_nonzero_values
        assert np.all(np.isfinite(record.final_masses))
        assert np.all(record.final_masses >= 0.0)

    assert any(
        record.mass_transfer_changed_from_previous_trial
        for record in records[1:]
    )


@pytest.mark.parametrize("case", _make_condensation_stiffness_cases())
def test_condensation_stiffness_recorded_grid_matches_configured_timesteps(
    case: CondensationStiffnessCase,
    device: str,
) -> None:
    """Recorded sweeps preserve configured timestep count, order, and mode."""
    if device != "cpu":
        pytest.skip("Recorded stiffness sweeps run on Warp CPU")

    records = _record_condensation_stiffness_trials(case, device=device)
    configured_timesteps = _RECORDED_TIMESTEP_GRID_BY_CASE[case.name]

    assert len(records) == len(configured_timesteps)
    assert [record.timestep_index for record in records] == list(
        range(len(configured_timesteps))
    )
    assert [record.time_step for record in records] == list(
        configured_timesteps
    )
    assert [record.configured_time_step for record in records] == list(
        configured_timesteps
    )
    if case.n_boxes == 1:
        assert {
            record.environment_input_mode for record in records
        } == {"scalar_inputs"}
    else:
        assert {
            record.environment_input_mode for record in records
        } == {"direct_warp_arrays"}


def test_condensation_stiffness_recorded_grid_cuda_contract_parity(
    device: str,
) -> None:
    """CUDA, when available, preserves the recorded-grid result contract."""
    if device == "cpu":
        pytest.skip("CUDA parity only runs on CUDA devices")
    if not cuda_available(wp):
        pytest.skip("CUDA is unavailable")

    case = next(
        candidate
        for candidate in _make_condensation_stiffness_cases()
        if candidate.name == "nanometer"
    )
    records = _record_condensation_stiffness_trials(case, device=device)

    assert len(records) == len(_RECORDED_TIMESTEP_GRID_BY_CASE[case.name])
    assert all(record.reuses_caller_mass_transfer_buffer for record in records)
    assert all(record.gas_unchanged for record in records)
    assert all(record.case_name == case.name for record in records)


def test_condensation_stiffness_invalid_environment_inputs_do_not_mutate_case(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
) -> None:
    """Pre-launch failures leave deterministic case inputs unchanged."""
    if device != "cpu":
        pytest.skip("Stiffness baseline runs on Warp CPU")

    case = next(
        candidate
        for candidate in _make_condensation_stiffness_cases()
        if candidate.name == "droplet_like"
    )
    particles = case.build_particle_data()
    gas = case.build_gas_data()
    vapor_pressure = case.build_vapor_pressure()
    original_masses = particles.masses.copy()
    original_gas = gas.concentration.copy()
    original_vapor_pressure = vapor_pressure.copy()
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

    with pytest.raises(ValueError, match="temperature must be finite and > 0"):
        condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature=wp.array(
                [case.temperature, 0.0],
                dtype=wp.float64,
                device=device,
            ),
            pressure=wp.array(
                case.pressure_array(),
                dtype=wp.float64,
                device=device,
            ),
            time_step=case.time_step,
        )

    assert calls == []
    npt.assert_allclose(particles.masses, original_masses)
    npt.assert_allclose(gas.concentration, original_gas)
    npt.assert_allclose(vapor_pressure, original_vapor_pressure)


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


def test_condensation_step_gpu_rejected_inputs_leave_mass_transfer_buffer_unchanged(
    device: str,
) -> None:
    """Rejected inputs do not mutate caller-owned mass-transfer buffers."""
    particles = _make_particle_data(n_boxes=1, n_particles=2, n_species=2)
    gas = _make_gas_data(n_boxes=1, n_species=2)
    vapor_pressure = _make_vapor_pressure(n_boxes=1, n_species=2)
    gpu_particles = to_warp_particle_data(particles, device=device)
    gpu_gas = to_warp_gas_data(
        gas,
        device=device,
        vapor_pressure=vapor_pressure,
    )
    mass_transfer = wp.full(
        (1, 2, 2),
        wp.float64(7.0),
        dtype=wp.float64,
        device=device,
    )
    original = np.asarray(mass_transfer.numpy()).copy()
    bad_temperature = wp.array(
        [298.15, 299.15], dtype=wp.float64, device=device
    )

    with pytest.raises(ValueError, match=r"temperature shape .*\(n_boxes,\)"):
        condensation_step_gpu(
            gpu_particles,
            gpu_gas,
            temperature=bad_temperature,
            pressure=101325.0,
            time_step=1.0,
            mass_transfer=mass_transfer,
        )

    npt.assert_allclose(mass_transfer.numpy(), original)


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


def test_condensation_step_gpu_zero_volume_particle_short_circuits_with_arrays(
    device: str,
) -> None:
    """Zero-volume particles remain unchanged with per-box array inputs."""
    particles = _make_particle_data(n_boxes=2, n_particles=2, n_species=1)
    particles.masses[1, 0, :] = 0.0
    gas = _make_gas_data(n_boxes=2, n_species=1)
    vapor_pressure = _make_vapor_pressure(n_boxes=2, n_species=1)

    gpu_result, mass_transfer = _run_gpu_step(
        particles,
        gas,
        vapor_pressure,
        temperature=wp.array([298.15, 304.15], dtype=wp.float64, device=device),
        pressure=wp.array(
            [101325.0, 99500.0],
            dtype=wp.float64,
            device=device,
        ),
        time_step=0.1,
        device=device,
    )

    assert gpu_result.masses[1, 0, 0] == pytest.approx(0.0)
    assert mass_transfer.numpy()[1, 0, 0] == pytest.approx(0.0)


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
