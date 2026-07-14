"""GPU condensation kernels and orchestration utilities.

This module composes condensation ``@wp.func`` building blocks into end-to-end
kernels and provides the high-level ``condensation_step_gpu`` API. Entry-point
validation accepts scalar direct inputs, explicit ``(n_boxes,)`` Warp arrays,
or a ``WarpEnvironmentData`` container. Aggregate preflight completes before
buffer allocation, Warp launch, vapor-pressure refresh, or mutation of any
caller-owned buffer. A required keyword-only ``ThermodynamicsConfig`` then
refreshes the caller-owned pure-vapor-pressure buffer from the current
normalized temperature before mass transfer. Float32 temperatures are cast to
a step-owned float64 device buffer for that refresh. Kernel launches operate
on GPU-resident Warp arrays and update particle masses in-place.
"""

# pyright: basic
# pyright: reportArgumentType=false
# pyright: reportAssignmentType=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportOperatorIssue=false

from dataclasses import dataclass
from typing import Any

import numpy as np

import particula.util.constants as constants

try:
    import warp as wp
except ImportError as exc:  # pragma: no cover - handled via import guards
    raise ImportError(
        "Warp is required for GPU condensation kernels. "
        "Install with: pip install warp-lang"
    ) from exc

from particula.gpu.dynamics.condensation_funcs import (
    diffusion_coefficient_wp,
    effective_surface_tension_wp,
    first_order_mass_transport_k_wp,
    mass_transfer_rate_wp,
    particle_radius_from_volume_wp,
    water_activity_ideal_wp,
    water_activity_kappa_wp,
)
from particula.gpu.kernels.environment import (
    _ensure_environment_arrays,
    _is_warp_array_like,
    validate_environment_inputs,
)
from particula.gpu.kernels.thermodynamics import (
    ThermodynamicsConfig,
    _validate_array_metadata,
    refresh_vapor_pressure_gpu,
    validate_thermodynamics_config,
)
from particula.gpu.properties.gas_properties import (
    dynamic_viscosity_wp,
    molecule_mean_free_path_wp,
    partial_pressure_wp,
)
from particula.gpu.properties.particle_properties import (
    aerodynamic_mobility_wp,
    cunningham_slip_correction_wp,
    kelvin_radius_wp,
    kelvin_term_wp,
    knudsen_number_wp,
    partial_pressure_delta_wp,
    vapor_transition_correction_wp,
)

# type: ignore
_DEFAULT_SURFACE_TENSION = 0.072
_DEFAULT_MASS_ACCOMMODATION = 1.0
_DEFAULT_DIFFUSION_COEFFICIENT = 2.0e-5

ACTIVITY_MODE_IDEAL = wp.int32(0)
ACTIVITY_MODE_KAPPA = wp.int32(1)
SURFACE_TENSION_MODE_STATIC = wp.int32(0)
SURFACE_TENSION_MODE_COMPOSITION_WEIGHTED = wp.int32(1)


@dataclass(frozen=True)
class CondensationActivitySurfaceConfig:
    """Configure caller-owned activity and surface-tension sidecar inputs.

    The frozen dataclass prevents rebinding fields, but does not make its Warp
    arrays immutable. ``kappas`` and ``molar_mass_reference`` therefore remain
    caller-owned mutable ``wp.float64`` arrays with shape ``(n_species,)``;
    callers must retain their lifetime and must not modify them concurrently
    with a launch. Activity is evaluated only for ``water_species_index``;
    every non-water vapor uses unit activity. Static tension uses the current
    condensing-species index, while composition-weighted tension computes one
    particle-wide value for every condensing species.

    Attributes:
        activity_mode: Integer selector: ``0`` for ideal or ``1`` for kappa
            water activity.
        surface_tension_mode: Integer selector: ``0`` for static or ``1`` for
            composition-weighted surface tension.
        water_species_index: Index of the sole species receiving activity;
            all other vapor species use unit activity.
        kappas: Caller-owned finite, non-negative ``wp.float64`` array of
            per-species kappa parameters with shape ``(n_species,)``.
        molar_mass_reference: Caller-owned positive ``wp.float64`` array of
            per-species molar masses [kg/mol], shaped ``(n_species,)`` and
            exactly ordered as ``gas.molar_mass``.
    """

    activity_mode: Any
    surface_tension_mode: Any
    water_species_index: Any
    kappas: Any
    molar_mass_reference: Any


@dataclass(frozen=True)
class CondensationScratchBuffers:
    """Own reusable stable-shape buffers for one P1 condensation update.

    This concrete-module-only sidecar is intentionally not exported.
    Non-``None``
    fields are caller-owned, active-device ``wp.float64`` arrays with stable
    shapes: both transfer fields have shape
    ``(n_boxes, n_particles, n_species)`` and both property fields have shape
    ``(n_boxes,)``. Successful calls preserve each supplied object's identity.
    P1 performs exactly one update: work and total transfers contain the same
    raw pre-clamp transfer, and the supplied total buffer is returned by
    identity. Callers must keep arrays alive and unmodified until launched work
    completes; reuse them only after successful completion. Complete metadata
    validation precedes allocation, normalization, refresh, launch, and
    mutation, so validation failures leave caller-owned state unchanged.
    """

    work_mass_transfer: Any | None = None
    total_mass_transfer: Any | None = None
    dynamic_viscosity: Any | None = None
    mean_free_path: Any | None = None


def validate_condensation_scratch_buffers(
    candidate: CondensationScratchBuffers | Any,
    dimensions: tuple[int, int, int],
    device: Any,
    caller_name: str,
) -> CondensationScratchBuffers:
    """Validate scratch metadata without allocation, reads, or mutation.

    This is an atomic metadata-only gate. It neither reads nor writes supplied
    arrays, allocates fallback buffers, launches Warp work, or synchronizes.
    """
    if type(candidate) is not CondensationScratchBuffers:
        raise ValueError(
            "scratch_buffers must be a CondensationScratchBuffers "
            f"in {caller_name}."
        )
    n_boxes, n_particles, n_species = dimensions
    for name, shape in (
        ("work_mass_transfer", (n_boxes, n_particles, n_species)),
        ("total_mass_transfer", (n_boxes, n_particles, n_species)),
        ("dynamic_viscosity", (n_boxes,)),
        ("mean_free_path", (n_boxes,)),
    ):
        values = getattr(candidate, name)
        if values is not None:
            _validate_array_metadata(
                name,
                values,
                shape,
                wp.float64,
                device,
                caller_name,
                "scratch_buffers",
            )
    return candidate


def _validate_int32_selector(
    name: str, value: Any, allowed: tuple[int, ...]
) -> int:
    """Validate a non-boolean integer selector before Warp int32 conversion."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"activity_surface.{name} must be an integer.")
    result = int(value)
    if result < np.iinfo(np.int32).min or result > np.iinfo(np.int32).max:
        raise ValueError(f"activity_surface.{name} is outside int32 range.")
    if result not in allowed:
        raise ValueError(
            f"activity_surface.{name} contains an unsupported mode."
        )
    return result


def _read_float64_array(values: Any) -> np.ndarray:
    """Read a validated Warp array for validation-only domain checks."""
    return np.asarray(values.numpy(), dtype=np.float64)


def validate_condensation_activity_surface_config(
    activity_surface: CondensationActivitySurfaceConfig | Any,
    n_species: int,
    device: Any,
    gas_molar_mass: Any,
    caller_name: str,
) -> CondensationActivitySurfaceConfig:
    """Validate and return the identical activity/surface sidecar.

    Args:
        activity_surface: Candidate frozen sidecar to validate.
        n_species: Number of ordered gas and particle species.
        device: Active Warp device for all sidecar arrays.
        gas_molar_mass: Ordered caller-owned gas molar-mass array.
        caller_name: Entry-point name used in validation errors.

    Returns:
        The original validated ``CondensationActivitySurfaceConfig`` object.

    Raises:
        ValueError: If the sidecar identity, selectors, arrays, domains, or
            ordered molar-mass reference violates the contract.

    Notes:
        This read-only preflight runs before allocation and launch. A malformed
        sidecar therefore leaves particle, gas, environment, and supplied
        output buffers unchanged; it never falls back to legacy physics.
    """
    if type(activity_surface) is not CondensationActivitySurfaceConfig:
        raise ValueError(
            "activity_surface must be a CondensationActivitySurfaceConfig "
            f"in {caller_name}."
        )
    _validate_int32_selector(
        "activity_mode", activity_surface.activity_mode, (0, 1)
    )
    _validate_int32_selector(
        "surface_tension_mode", activity_surface.surface_tension_mode, (0, 1)
    )
    water_index = _validate_int32_selector(
        "water_species_index",
        activity_surface.water_species_index,
        tuple(range(n_species)),
    )
    if water_index >= n_species:
        raise ValueError(
            "activity_surface.water_species_index is out of range."
        )
    _validate_array_metadata(
        "kappas",
        activity_surface.kappas,
        (n_species,),
        wp.float64,
        device,
        caller_name,
        "activity_surface",
    )
    _validate_array_metadata(
        "molar_mass_reference",
        activity_surface.molar_mass_reference,
        (n_species,),
        wp.float64,
        device,
        caller_name,
        "activity_surface",
    )
    _validate_array_metadata(
        "gas.molar_mass",
        gas_molar_mass,
        (n_species,),
        wp.float64,
        device,
        caller_name,
        "",
    )
    kappas = _read_float64_array(activity_surface.kappas)
    references = _read_float64_array(activity_surface.molar_mass_reference)
    gas_masses = _read_float64_array(gas_molar_mass)
    if not np.all(np.isfinite(kappas)) or np.any(kappas < 0.0):
        raise ValueError(
            "activity_surface.kappas must be finite and non-negative."
        )
    if not np.all(np.isfinite(references)) or np.any(references <= 0.0):
        raise ValueError(
            "activity_surface.molar_mass_reference must be finite and positive."
        )
    if not np.array_equal(references, gas_masses):
        raise ValueError(
            "activity_surface.molar_mass_reference must exactly match "
            "gas.molar_mass."
        )
    return activity_surface


@wp.kernel
# type: ignore[misc]
def condensation_mass_transfer_kernel(  # noqa: C901
    masses: Any,
    concentration: Any,
    density: Any,
    gas_concentration: Any,
    vapor_pressure: Any,
    molar_mass: Any,
    surface_tension: Any,
    kappas: Any,
    molar_mass_reference: Any,
    effective_surface_tension: Any,
    activity_enabled: wp.int32,
    activity_mode: wp.int32,
    surface_tension_mode: wp.int32,
    water_species_index: wp.int32,
    mass_accommodation: Any,
    diffusion_coefficient_vapor: Any,
    dynamic_viscosity: Any,
    mean_free_path: Any,
    gas_constant: Any,
    boltzmann_constant: Any,
    temperature: Any,
    time_step: Any,
    mass_transfer: Any,
) -> None:
    """Compute condensation mass transfer for each particle species.

    Water activity is applied only when enabled and the current species equals
    ``water_species_index``. In configured weighted-tension mode, the supplied
    per-particle tension is shared by all condensing species.

    Args:
        masses: Particle masses array ``(n_boxes, n_particles, n_species)``.
        concentration: Particle number concentration array.
        density: Per-species particle density array.
        gas_concentration: Gas concentrations array.
        vapor_pressure: Gas-phase vapor pressure array.
        molar_mass: Gas-phase molar mass array.
        surface_tension: Per-species surface tension array.
        kappas: Per-species kappa activity parameters.
        molar_mass_reference: Per-species activity reference molar masses.
        effective_surface_tension: Per-particle weighted tension buffer.
        activity_enabled: Flag enabling configured water activity.
        activity_mode: Selector for ideal or kappa water activity.
        surface_tension_mode: Selector for static or weighted tension.
        water_species_index: Index of the sole activity-adjusted species.
        mass_accommodation: Per-species mass accommodation coefficients.
        diffusion_coefficient_vapor: Per-species vapor diffusion coefficients.
        dynamic_viscosity: Per-box gas dynamic viscosity [Pa·s].
        mean_free_path: Per-box gas mean free path [m].
        gas_constant: Universal gas constant [J/(mol·K)].
        boltzmann_constant: Boltzmann constant [J/K].
        temperature: Per-box gas temperature [K].
        time_step: Condensation time step [s].
        mass_transfer: Output mass transfer array.
    """  # type: ignore
    box_idx, particle_idx = wp.tid()  # type: ignore[misc]
    dynamic_viscosity_value = dynamic_viscosity[box_idx]
    mean_free_path_value = mean_free_path[box_idx]
    temperature_value = temperature[box_idx]

    if concentration[box_idx, particle_idx] == wp.float64(0.0):
        for species_idx in range(masses.shape[2]):
            mass_transfer[box_idx, particle_idx, species_idx] = wp.float64(0.0)
        return

    n_species = masses.shape[2]
    total_volume = wp.float64(0.0)
    total_mass = wp.float64(0.0)
    for species_idx in range(n_species):
        species_mass = masses[box_idx, particle_idx, species_idx]
        total_mass += species_mass
        total_volume += species_mass / density[species_idx]

    if total_volume <= wp.float64(0.0):  # type: ignore[operator]
        for species_idx in range(n_species):
            mass_transfer[box_idx, particle_idx, species_idx] = wp.float64(0.0)
        return

    radius = particle_radius_from_volume_wp(total_volume)
    effective_density = total_mass / total_volume
    if effective_density <= wp.float64(0.0):
        effective_density = density[0]

    knudsen_number = knudsen_number_wp(mean_free_path_value, radius)
    slip_correction = cunningham_slip_correction_wp(knudsen_number)
    mobility = aerodynamic_mobility_wp(
        radius, slip_correction, dynamic_viscosity_value
    )
    diffusion_coefficient_particle = diffusion_coefficient_wp(
        temperature_value,
        mobility,
        boltzmann_constant,
    )

    for species_idx in range(n_species):
        transition = vapor_transition_correction_wp(
            knudsen_number,
            mass_accommodation[species_idx],
        )
        diffusion_value = diffusion_coefficient_vapor[species_idx]
        if diffusion_value <= wp.float64(0.0):
            diffusion_value = diffusion_coefficient_particle
        mass_transport = first_order_mass_transport_k_wp(
            radius,
            transition,
            diffusion_value,
        )
        tension = surface_tension[species_idx]
        if surface_tension_mode == SURFACE_TENSION_MODE_COMPOSITION_WEIGHTED:
            tension = effective_surface_tension[box_idx, particle_idx]
        kelvin_radius = kelvin_radius_wp(
            tension,
            effective_density,
            molar_mass[species_idx],
            temperature_value,
            gas_constant,
        )
        kelvin_term = kelvin_term_wp(radius, kelvin_radius)
        partial_pressure_gas = partial_pressure_wp(
            gas_concentration[box_idx, species_idx],
            molar_mass[species_idx],
            temperature_value,
            gas_constant,
        )
        activity_factor = wp.float64(1.0)
        if (
            activity_enabled == wp.int32(1)
            and species_idx == water_species_index
        ):
            water_species_index_int = int(water_species_index)
            if activity_mode == ACTIVITY_MODE_IDEAL:
                activity_factor = water_activity_ideal_wp(
                    masses,
                    molar_mass_reference,
                    box_idx,
                    particle_idx,
                    water_species_index_int,
                )
            else:
                activity_factor = water_activity_kappa_wp(
                    masses,
                    density,
                    kappas,
                    box_idx,
                    particle_idx,
                    water_species_index_int,
                )
        pressure_delta = partial_pressure_delta_wp(
            partial_pressure_gas,
            activity_factor * vapor_pressure[box_idx, species_idx],
            kelvin_term,
        )
        mass_rate = mass_transfer_rate_wp(
            pressure_delta,
            mass_transport,
            temperature_value,
            molar_mass[species_idx],
            gas_constant,
        )
        mass_transfer[box_idx, particle_idx, species_idx] = (
            mass_rate * time_step
        )


@wp.kernel
def _effective_surface_tension_kernel(
    masses: Any,
    density: Any,
    surface_tension: Any,
    output: Any,
) -> None:
    """Compute one composition-weighted surface tension per particle.

    The output is step-owned and subsequently reused for every condensing
    species of that particle, avoiding a full composition reduction per species.
    """
    box_idx, particle_idx = wp.tid()  # type: ignore[misc]
    output[box_idx, particle_idx] = effective_surface_tension_wp(
        masses, density, surface_tension, box_idx, particle_idx, 0, True
    )


@wp.kernel
# type: ignore[misc]
def _copy_temperature_to_float64_kernel(
    temperature: Any,
    output: Any,
) -> None:
    """Copy normalized temperatures to a float64 device buffer.

    This device-only helper prepares non-float64 normalized temperatures for
    thermodynamic vapor-pressure refresh without changing the input buffer.

    Args:
        temperature: Normalized per-box temperature array ``(n_boxes,)`` [K].
        output: Float64 per-box output array ``(n_boxes,)`` [K].
    """
    box_idx = wp.tid()  # type: ignore[misc]
    output[box_idx] = wp.float64(temperature[box_idx])


@wp.kernel
# type: ignore[misc]
def _copy_pressure_to_float64_kernel(
    pressure: Any,
    output: Any,
) -> None:
    """Copy normalized pressures into a float64 device buffer.

    Args:
        pressure: Normalized per-box pressure array ``(n_boxes,)`` [Pa].
        output: Float64 per-box output array ``(n_boxes,)`` [Pa].
    """
    box_idx = wp.tid()  # type: ignore[misc]
    output[box_idx] = wp.float64(pressure[box_idx])


@wp.kernel
# type: ignore[misc]
def _prepare_environment_properties_kernel(
    temperature: Any,
    pressure: Any,
    dynamic_viscosity: Any,
    mean_free_path: Any,
) -> None:
    """Precompute box-level gas properties once per entry-point call.

    Args:
        temperature: Per-box gas temperatures ``(n_boxes,)`` [K].
        pressure: Per-box gas pressures ``(n_boxes,)`` [Pa].
        dynamic_viscosity: Output dynamic viscosity array ``(n_boxes,)``
            [Pa·s].
        mean_free_path: Output mean free path array ``(n_boxes,)`` [m].
    """
    box_idx = wp.tid()  # type: ignore[misc]
    temperature_value = temperature[box_idx]
    pressure_value = pressure[box_idx]
    viscosity_value = dynamic_viscosity_wp(
        temperature_value,
        wp.float64(constants.REF_VISCOSITY_AIR_STP),
        wp.float64(constants.REF_TEMPERATURE_STP),
        wp.float64(constants.SUTHERLAND_CONSTANT),
    )
    dynamic_viscosity[box_idx] = viscosity_value
    mean_free_path[box_idx] = molecule_mean_free_path_wp(
        wp.float64(constants.MOLECULAR_WEIGHT_AIR),
        temperature_value,
        pressure_value,
        viscosity_value,
        wp.float64(constants.GAS_CONSTANT),
    )


@wp.kernel
# type: ignore[misc]
def apply_mass_transfer_kernel(
    masses: Any,
    mass_transfer: Any,
) -> None:
    """Apply condensation mass transfer and clamp masses to non-negative.

    Args:
        masses: Particle masses array ``(n_boxes, n_particles, n_species)``.
        mass_transfer: Mass transfer array matching ``masses`` shape.
    """
    box_idx, particle_idx = wp.tid()  # type: ignore[misc]
    n_species = masses.shape[2]
    for species_idx in range(n_species):
        updated_mass = (
            masses[box_idx, particle_idx, species_idx]
            + mass_transfer[box_idx, particle_idx, species_idx]
        )
        if updated_mass < wp.float64(0.0):
            updated_mass = wp.float64(0.0)
        masses[box_idx, particle_idx, species_idx] = updated_mass


@wp.kernel
# type: ignore[misc]
def apply_mass_transfer_with_total_kernel(
    masses: Any,
    work_mass_transfer: Any,
    total_mass_transfer: Any,
) -> None:
    """Apply work transfer while storing the matching raw P1 total."""
    box_idx, particle_idx = wp.tid()  # type: ignore[misc]
    for species_idx in range(masses.shape[2]):
        transfer = work_mass_transfer[box_idx, particle_idx, species_idx]
        total_mass_transfer[box_idx, particle_idx, species_idx] = transfer
        updated_mass = masses[box_idx, particle_idx, species_idx] + transfer
        if updated_mass < wp.float64(0.0):
            updated_mass = wp.float64(0.0)
        masses[box_idx, particle_idx, species_idx] = updated_mass


def _validate_species_array(
    name: str,
    array: Any,
    n_species: int,
    expected_device: Any,
) -> None:
    """Validate per-species array length and device.

    Args:
        name: Array name for error messages.
        array: Array-like object with ``shape`` attribute.
        n_species: Expected length.
        expected_device: Expected Warp device.

    Raises:
        ValueError: If the array length or device mismatches expectations.
    """
    if len(array.shape) != 1:
        raise ValueError(f"{name} must be a 1D array")
    if array.shape[0] != n_species:
        raise ValueError(
            f"{name} length {array.shape[0]} does not match n_species "
            f"{n_species}"
        )
    device = getattr(array, "device", None)
    if device is None or str(device) != str(expected_device):
        raise ValueError(f"{name} device does not match particle device")


def _validate_float64_species_array(
    name: str,
    array: Any,
    n_species: int,
    expected_device: Any,
    *,
    nonnegative: bool = False,
) -> None:
    """Validate a caller-owned finite float64 per-species Warp array."""
    if not _is_warp_array_like(array):
        raise ValueError(f"{name} must be a Warp array")
    _validate_species_array(name, array, n_species, expected_device)
    if array.dtype != wp.float64:
        raise ValueError(f"{name} must use dtype {wp.float64}")
    values = _read_float64_array(array)
    if not np.all(np.isfinite(values)) or (
        nonnegative and np.any(values < 0.0)
    ):
        raise ValueError(f"{name} must be finite and non-negative")


def _validate_mass_transfer_buffer(
    mass_transfer: Any,
    expected_shape: tuple[int, int, int],
    expected_device: str,
) -> None:
    """Validate mass transfer buffer shape and device.

    Args:
        mass_transfer: Mass transfer array allocated on the GPU.
        expected_shape: Expected ``(n_boxes, n_particles, n_species)`` shape.
        expected_device: Expected Warp device name.

    Raises:
        ValueError: If the shape or device does not match expectations.
    """
    if mass_transfer.shape != expected_shape:
        raise ValueError(
            f"mass_transfer shape {mass_transfer.shape} does not match "
            f"expected {expected_shape}"
        )
    device = getattr(mass_transfer, "device", None)
    if device is None or str(device) != str(expected_device):
        raise ValueError(
            "mass_transfer buffer device does not match particle device"
        )
    if (
        not _is_warp_array_like(mass_transfer)
        or mass_transfer.dtype != wp.float64
    ):
        raise ValueError(f"mass_transfer must use dtype {wp.float64}")


def _validate_particle_arrays(
    particles: Any,
    n_boxes: int,
    n_particles: int,
    n_species: int,
) -> None:
    """Validate particle array shapes and lengths.

    Args:
        particles: Particle data containing density and concentration arrays.
        n_boxes: Expected number of spatial boxes.
        n_particles: Expected number of particles per box.
        n_species: Expected number of particle species.

    Raises:
        ValueError: If particle array shapes do not match expectations.
    """
    if particles.density.shape[0] != n_species:
        raise ValueError("particle density length does not match n_species")
    if particles.concentration.shape != (n_boxes, n_particles):
        raise ValueError(
            "particle concentration shape does not match (n_boxes, n_particles)"
        )


def _validate_gas_arrays(
    gas: Any,
    n_boxes: int,
    n_species: int,
) -> None:
    """Validate gas array shapes and lengths.

    Args:
        gas: Gas data containing molar mass, concentration, and vapor pressure.
        n_boxes: Expected number of spatial boxes.
        n_species: Expected number of gas species.

    Raises:
        ValueError: If gas array shapes do not match expectations.
    """
    if gas.molar_mass.shape[0] != n_species:
        raise ValueError(
            "n_species mismatch between particle masses and gas molar mass"
        )
    if gas.concentration.shape != (n_boxes, n_species):
        raise ValueError(
            "gas concentration shape does not match (n_boxes, n_species)"
        )
    if gas.vapor_pressure.shape != (n_boxes, n_species):
        raise ValueError(
            "vapor pressure shape does not match (n_boxes, n_species)"
        )


def _validate_device_match(name: str, array: Any, expected_device: Any) -> None:
    """Validate that a Warp array is on the expected device.

    Args:
        name: Array label for error messages.
        array: Warp array to validate.
        expected_device: Expected Warp device.

    Raises:
        ValueError: If the array is not on the expected device.
    """
    device = getattr(array, "device", None)
    if device is None or str(device) != str(expected_device):
        raise ValueError(f"{name} device mismatch")


def _validate_device_arrays(particles: Any, gas: Any, device: Any) -> None:
    """Validate particle and gas arrays share the same Warp device.

    Args:
        particles: Particle data with GPU-backed arrays.
        gas: Gas data with GPU-backed arrays.
        device: Expected Warp device.

    Raises:
        ValueError: If any particle or gas array is on a different device.
    """
    _validate_device_match(
        "particle concentration", particles.concentration, device
    )
    _validate_device_match("particle density", particles.density, device)
    _validate_device_match("gas molar mass", gas.molar_mass, device)
    _validate_device_match("gas concentration", gas.concentration, device)
    _validate_device_match("gas vapor pressure", gas.vapor_pressure, device)


def condensation_step_gpu(  # noqa: C901
    particles: Any,
    gas: Any,
    temperature: float | Any | None,
    pressure: float | Any | None,
    time_step: float,
    surface_tension: Any | None = None,
    mass_accommodation: Any | None = None,
    diffusion_coefficient_vapor: Any | None = None,
    mass_transfer: Any | None = None,
    *,
    environment: Any | None = None,
    thermodynamics: ThermodynamicsConfig | Any | None = None,
    activity_surface: CondensationActivitySurfaceConfig | Any | None = None,
    scratch_buffers: CondensationScratchBuffers | Any | None = None,
) -> tuple[Any, Any]:
    """Execute one condensation timestep on the GPU.

    Args:
        particles: GPU-resident particle data.
        gas: GPU-resident gas data.
        temperature: Direct gas temperature as either a scalar or a Warp array
            with shape ``(n_boxes,)``. Use ``None`` only with
            ``environment=...``.
        pressure: Direct gas pressure as either a scalar or a Warp array with
            shape ``(n_boxes,)``. Use ``None`` only with ``environment=...``.
        time_step: Condensation time step in seconds.
        surface_tension: Optional per-species surface tension [N/m].
        mass_accommodation: Optional per-species accommodation coefficient.
        diffusion_coefficient_vapor: Optional per-species vapor diffusion
            coefficient [m^2/s].
        mass_transfer: Optional preallocated mass transfer buffer with shape
            ``(n_boxes, n_particles, n_species)``.
        environment: Optional ``WarpEnvironmentData`` with ``(n_boxes,)``
            temperature and pressure arrays on the same device as ``particles``
            and ``gas``. This mode is supported when both direct inputs are
            ``None``.
        thermodynamics: Required caller-owned, device-local thermodynamic
            sidecar with model arrays matching the ordered gas species.
        activity_surface: Optional frozen caller-owned sidecar with device-local
            ``wp.float64`` per-species arrays. It enables ideal or kappa
            activity only for ``water_species_index``; non-water species retain
            unit activity. Static tension uses the current condensing-species
            index, while weighted tension uses one particle-wide value. ``None``
            retains legacy unit activity and species-indexed static tension.
        scratch_buffers: Optional frozen caller-owned P1 buffers. Supplied
            transfer fields have shape ``(n_boxes, n_particles, n_species)``;
            supplied property fields have shape ``(n_boxes,)``. Every supplied
            field must be active-device ``wp.float64``. Fields may be omitted
            independently and use step-local fallback allocation. A supplied
            total transfer buffer is returned by identity; P1 work and total
            contain the same raw pre-clamp transfer. Validation is performed
            before allocation or mutation. Arrays must remain alive and
            unmodified until launched work completes and may be reused only
            after a successful completion.

    Returns:
        Tuple of the particle data with in-place updated masses and the raw,
        unclamped mass-transfer buffer [kg]. Gas concentration is unchanged.

    Raises:
        ValueError: If species counts, array lengths, or devices mismatch.
        ValueError: If direct ``temperature`` or ``pressure`` inputs are mixed
            with ``environment``.
        ValueError: If direct inputs are missing when ``environment`` is
            omitted.
        ValueError: If environment arrays do not match ``(n_boxes,)`` or the
            caller device.
        ValueError: If direct non-scalar, non-Warp-array temperature or
            pressure inputs are provided.
        ValueError: If ``thermodynamics`` is absent or does not match the gas
            species, active device, or required fixed schema.
        ValueError: If ``activity_surface`` violates its frozen-sidecar,
            selector, device-array, domain, or molar-mass ordering contract.

    Notes:
        Accepted environment sources are scalar direct inputs, direct
        ``(n_boxes,)`` Warp arrays, hybrid scalar-plus-Warp-array direct
        inputs, or keyword-only ``environment=...`` execution. NumPy arrays,
        Python lists, and other non-Warp direct array-likes are rejected.

        Particle masses are updated in-place on the GPU. Callers that require
        rollback should copy masses before invoking this function.

        The returned transfer buffer records the calculated transfer before
        mass clamping. Consequently, the final particle mass is
        ``maximum(initial_mass + mass_transfer, 0)`` and is not necessarily
        equal to ``initial_mass + mass_transfer`` for evaporation that would
        otherwise make a mass negative. This direct step does not couple the
        transfer to ``gas.concentration``.

        ``environment`` remains keyword-only so existing positional scalar
        callers stay source-compatible.

        Aggregate configuration and optional-buffer validation completes before
        allocation, launch, refresh, or mutation. Thus a validation failure
        leaves caller-owned particle, gas, environment, sidecar, and supplied
        output buffers unchanged and is retryable with corrected inputs.
        Successful calls overwrite ``gas.vapor_pressure`` from the normalized
        current temperature before preparing box-level properties and
        calculating mass transfer. Float32 temperature arrays are cast
        device-side to float64 for the refresh.

        Thermodynamic validation may synchronously read caller-owned device
        arrays, including on CUDA, without allocating, replacing, or mutating
        those buffers. Refresh evaluation and all production calculations stay
        device-resident.
    """
    n_boxes, n_particles, n_species = particles.masses.shape
    _validate_gas_arrays(gas, n_boxes, n_species)
    _validate_particle_arrays(particles, n_boxes, n_particles, n_species)

    device = particles.masses.device
    _validate_device_arrays(particles, gas, device)
    validate_environment_inputs(
        temperature=temperature,
        pressure=pressure,
        environment=environment,
        n_boxes=n_boxes,
        device=device,
        caller_name="condensation_step_gpu",
    )
    thermodynamics = validate_thermodynamics_config(
        thermodynamics,
        n_species,
        device,
        gas.molar_mass,
        "condensation_step_gpu",
    )

    if activity_surface is not None:
        activity_surface = validate_condensation_activity_surface_config(
            activity_surface,
            n_species,
            device,
            gas.molar_mass,
            "condensation_step_gpu",
        )

    if surface_tension is not None:
        _validate_float64_species_array(
            "surface_tension",
            surface_tension,
            n_species,
            device,
            nonnegative=True,
        )
    if mass_accommodation is not None:
        _validate_float64_species_array(
            "mass_accommodation",
            mass_accommodation,
            n_species,
            device,
            nonnegative=True,
        )
    if diffusion_coefficient_vapor is not None:
        _validate_float64_species_array(
            "diffusion_coefficient_vapor",
            diffusion_coefficient_vapor,
            n_species,
            device,
            nonnegative=True,
        )
    expected_shape = (n_boxes, n_particles, n_species)
    if mass_transfer is not None:
        _validate_mass_transfer_buffer(mass_transfer, expected_shape, device)

    if scratch_buffers is not None:
        scratch_buffers = validate_condensation_scratch_buffers(
            scratch_buffers,
            expected_shape,
            device,
            "condensation_step_gpu",
        )
        if mass_transfer is not None and (
            scratch_buffers.work_mass_transfer is not None
            or scratch_buffers.total_mass_transfer is not None
        ):
            raise ValueError(
                "mass_transfer conflicts with supplied scratch transfer "
                "buffers in condensation_step_gpu."
            )

    temperature_array, pressure_array = _ensure_environment_arrays(
        temperature=temperature,
        pressure=pressure,
        environment=environment,
        n_boxes=n_boxes,
        device=device,
        caller_name="condensation_step_gpu",
    )

    if surface_tension is None:
        surface_tension = wp.full(
            n_species,
            wp.float64(_DEFAULT_SURFACE_TENSION),
            dtype=wp.float64,
            device=device,
        )

    if mass_accommodation is None:
        mass_accommodation = wp.full(
            n_species,
            wp.float64(_DEFAULT_MASS_ACCOMMODATION),
            dtype=wp.float64,
            device=device,
        )

    if diffusion_coefficient_vapor is None:
        diffusion_coefficient_vapor = wp.full(
            n_species,
            wp.float64(_DEFAULT_DIFFUSION_COEFFICIENT),
            dtype=wp.float64,
            device=device,
        )

    scratch_transfer_mode = scratch_buffers is not None and (
        scratch_buffers.work_mass_transfer is not None
        or scratch_buffers.total_mass_transfer is not None
    )
    if mass_transfer is None and not scratch_transfer_mode:
        mass_transfer = wp.zeros(
            expected_shape,
            dtype=wp.float64,
            device=device,
        )
    work_mass_transfer = mass_transfer
    total_mass_transfer = mass_transfer
    if scratch_transfer_mode:
        work_mass_transfer = scratch_buffers.work_mass_transfer
        if work_mass_transfer is None:
            work_mass_transfer = wp.zeros(
                expected_shape,
                dtype=wp.float64,
                device=device,
            )
        total_mass_transfer = scratch_buffers.total_mass_transfer
        if total_mass_transfer is None:
            total_mass_transfer = wp.zeros(
                expected_shape,
                dtype=wp.float64,
                device=device,
            )
    dynamic_viscosity = (
        scratch_buffers.dynamic_viscosity
        if scratch_buffers is not None
        and scratch_buffers.dynamic_viscosity is not None
        else wp.zeros((n_boxes,), dtype=wp.float64, device=device)
    )
    mean_free_path = (
        scratch_buffers.mean_free_path
        if scratch_buffers is not None
        and scratch_buffers.mean_free_path is not None
        else wp.zeros((n_boxes,), dtype=wp.float64, device=device)
    )
    kappas = wp.zeros(n_species, dtype=wp.float64, device=device)
    molar_mass_reference = gas.molar_mass
    activity_enabled = wp.int32(0)
    activity_mode = ACTIVITY_MODE_IDEAL
    surface_tension_mode = SURFACE_TENSION_MODE_STATIC
    water_species_index = wp.int32(0)
    if activity_surface is not None:
        kappas = activity_surface.kappas
        molar_mass_reference = activity_surface.molar_mass_reference
        activity_enabled = wp.int32(1)
        activity_mode = wp.int32(activity_surface.activity_mode)
        surface_tension_mode = wp.int32(activity_surface.surface_tension_mode)
        water_species_index = wp.int32(activity_surface.water_species_index)
    effective_surface_tension = wp.zeros(
        (n_boxes, n_particles), dtype=wp.float64, device=device
    )
    if surface_tension_mode == SURFACE_TENSION_MODE_COMPOSITION_WEIGHTED:
        wp.launch(
            _effective_surface_tension_kernel,
            dim=(n_boxes, n_particles),
            inputs=[
                particles.masses,
                particles.density,
                surface_tension,
                effective_surface_tension,
            ],
            device=device,
        )

    refresh_temperature = temperature_array
    if temperature_array.dtype != wp.float64:
        refresh_temperature = wp.zeros(
            (n_boxes,),
            dtype=wp.float64,
            device=device,
        )
        wp.launch(
            _copy_temperature_to_float64_kernel,
            dim=n_boxes,
            inputs=[temperature_array, refresh_temperature],
            device=device,
        )

    kernel_pressure = pressure_array
    if pressure_array.dtype != wp.float64:
        kernel_pressure = wp.zeros(
            (n_boxes,),
            dtype=wp.float64,
            device=device,
        )
        wp.launch(
            _copy_pressure_to_float64_kernel,
            dim=n_boxes,
            inputs=[pressure_array, kernel_pressure],
            device=device,
        )

    refresh_vapor_pressure_gpu(thermodynamics, gas, refresh_temperature)

    wp.launch(
        _prepare_environment_properties_kernel,
        dim=n_boxes,
        inputs=[
            refresh_temperature,
            kernel_pressure,
            dynamic_viscosity,
            mean_free_path,
        ],
        device=device,
    )

    wp.launch(
        condensation_mass_transfer_kernel,
        dim=(n_boxes, n_particles),
        inputs=[
            particles.masses,
            particles.concentration,
            particles.density,
            gas.concentration,
            gas.vapor_pressure,
            gas.molar_mass,
            surface_tension,
            kappas,
            molar_mass_reference,
            effective_surface_tension,
            activity_enabled,
            activity_mode,
            surface_tension_mode,
            water_species_index,
            mass_accommodation,
            diffusion_coefficient_vapor,
            dynamic_viscosity,
            mean_free_path,
            wp.float64(constants.GAS_CONSTANT),
            wp.float64(constants.BOLTZMANN_CONSTANT),
            refresh_temperature,
            wp.float64(time_step),
            work_mass_transfer,
        ],
        device=device,
    )

    wp.launch(
        (
            apply_mass_transfer_with_total_kernel
            if scratch_transfer_mode
            else apply_mass_transfer_kernel
        ),
        dim=(n_boxes, n_particles),
        inputs=(
            [particles.masses, work_mass_transfer, total_mass_transfer]
            if scratch_transfer_mode
            else [particles.masses, mass_transfer]
        ),
        device=device,
    )

    return particles, total_mass_transfer
