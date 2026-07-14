"""GPU condensation kernels and orchestration utilities.

This module composes condensation ``@wp.func`` building blocks into end-to-end
kernels and provides the high-level ``condensation_step_gpu`` API. The private
P2 inventory finalizer is direct-test-only: it accepts an already gated
proposal, mutates particle masses and P2 scratch sidecars, and deliberately
leaves gas concentration unchanged while public gas coupling remains deferred.
Entry-point
validation accepts scalar direct inputs, explicit ``(n_boxes,)`` Warp arrays,
or a ``WarpEnvironmentData`` container. Aggregate preflight, including supplied
``CondensationScratchBuffers`` metadata, completes before buffer allocation,
Warp launch, vapor-pressure refresh, or mutation of caller-owned state. A
required keyword-only ``ThermodynamicsConfig`` then executes exactly four
equal substeps. Each substep refreshes the caller-owned pure-vapor-pressure
buffer, updates box-level environment properties, proposes transfer from the
current particle mass, gates disabled species and zero-concentration slots,
and applies a mass-clamped transfer. Float32
temperatures are cast to a step-owned float64 device buffer for refresh. Kernel
launches operate on GPU-resident Warp arrays and update particle masses
 in-place. Latent heat, when supplied, corrects each fixed substep. An optional
 caller-owned energy-transfer output records whole-call bounded-transfer energy
 on the active device, while thermal-work remains validated caller-owned state
 reserved for later work.
"""

# pyright: basic
# pyright: reportArgumentType=false
# pyright: reportAssignmentType=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportOperatorIssue=false

from dataclasses import dataclass
from numbers import Real
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
    _mass_transfer_rate_latent_heat_wp,
    _thermal_conductivity_wp,
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
    """Own reusable stable-shape buffers for a condensation-step update.

    This concrete-module-only sidecar is intentionally not exported. Every
    non-``None`` field is a caller-owned, active-device ``wp.float64`` array
    whose shape must remain stable: transfer fields have shape
    ``(n_boxes, n_particles, n_species)``, property fields have shape
    ``(n_boxes,)``, and P2-only limiting sidecars have shape
    ``(n_boxes, n_species)``. Fields may be omitted independently; omitted
    fields are
    step-local fallback allocations, while successful calls preserve every
    supplied object's identity. Transfer roles are deliberately separate: work
    retains the gated raw proposal from the final substep, while total records
    the applied transfer over the complete four-substep step.

    Each call performs four fixed updates. The total buffer is cleared once
    after preflight and accumulates the mass-clamped transfer applied in each
    substep; a supplied ``total_mass_transfer`` is returned by identity.
    Callers must keep supplied arrays alive and unmodified until launched work
    completes, and may reuse them only after a successful completion. Complete
    supplied-field metadata validation precedes allocation, environment
    normalization, refresh, launch, and mutation, so a validation failure
    leaves caller-owned state unchanged.

    Attributes:
        work_mass_transfer: Optional caller-owned work-transfer array with
            shape ``(n_boxes, n_particles, n_species)``.
        total_mass_transfer: Optional caller-owned applied-transfer accumulator
            with shape ``(n_boxes, n_particles, n_species)``.
        dynamic_viscosity: Optional caller-owned per-box dynamic-viscosity
            array with shape ``(n_boxes,)`` [Pa·s].
        mean_free_path: Optional caller-owned per-box mean-free-path array with
            shape ``(n_boxes,)`` [m].
        positive_mass_transfer_demand: Optional P2-only per-box, per-species
            reduction storage. P1 validates but does not read or write it.
        negative_mass_transfer_release: Optional P2-only per-box, per-species
            reduction storage. P1 validates but does not read or write it.
        positive_mass_transfer_scale: Optional P2-only per-box, per-species
            scale storage. P1 validates but does not read or write it.
    """

    work_mass_transfer: Any | None = None
    total_mass_transfer: Any | None = None
    dynamic_viscosity: Any | None = None
    mean_free_path: Any | None = None
    positive_mass_transfer_demand: Any | None = None
    negative_mass_transfer_release: Any | None = None
    positive_mass_transfer_scale: Any | None = None


def _read_float64_array(array: Any) -> np.ndarray:
    """Return a Warp array's values as a float64 NumPy array for validation."""
    return np.asarray(array.numpy(), dtype=np.float64)


def validate_condensation_scratch_buffers(
    candidate: CondensationScratchBuffers | Any,
    dimensions: tuple[int, int, int],
    device: Any,
    caller_name: str,
) -> CondensationScratchBuffers:
    """Validate scratch metadata without allocation, reads, or mutation.

    This atomic metadata-only gate requires the exact
    ``CondensationScratchBuffers`` type and validates every supplied field's
    stable shape, ``wp.float64`` dtype, and active device. It neither reads nor
    writes supplied arrays, allocates fallback buffers, launches Warp work, or
    synchronizes.

    Args:
        candidate: Scratch-buffer sidecar to validate.
        dimensions: Expected ``(n_boxes, n_particles, n_species)`` dimensions.
        device: Active Warp device required for supplied fields.
        caller_name: Entry-point name included in validation errors.

    Returns:
        The original validated ``CondensationScratchBuffers`` object.

    Raises:
        ValueError: If ``candidate`` is not the exact sidecar type or a
            supplied field has incompatible shape, dtype, or device.
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
        ("positive_mass_transfer_demand", (n_boxes, n_species)),
        ("negative_mass_transfer_release", (n_boxes, n_species)),
        ("positive_mass_transfer_scale", (n_boxes, n_species)),
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


@wp.kernel  # pragma: no cover - device kernels execute outside Python coverage
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
    latent_heat: Any,
    latent_heat_enabled: wp.int32,
    dynamic_viscosity: Any,
    mean_free_path: Any,
    gas_constant: Any,
    boltzmann_constant: Any,
    temperature: Any,
    time_step: Any,
    mass_transfer: Any,
) -> None:
    """Compute a raw condensation mass-transfer proposal per particle species.

    Water activity is applied only when enabled and the current species equals
    ``water_species_index``. In configured weighted-tension mode, the supplied
    per-particle tension is shared by all condensing species. For each species,
    the activity- and Kelvin-adjusted surface vapor pressure is used both for
    the gas-to-surface pressure difference and, when enabled, the latent-heat
    correction. Omitted latent heat and exactly zero per-species entries use
    the original isothermal rate path.

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
        latent_heat: Per-species latent heat [J/kg], unread when disabled.
        latent_heat_enabled: Flag selecting latent-heat rate correction.
        dynamic_viscosity: Per-box gas dynamic viscosity [Pa·s].
        mean_free_path: Per-box gas mean free path [m].
        gas_constant: Universal gas constant [J/(mol·K)].
        boltzmann_constant: Boltzmann constant [J/K].
        temperature: Per-box gas temperature [K].
        time_step: Condensation substep duration [s].
        mass_transfer: Output raw mass-transfer proposal array.
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
    thermal_conductivity = wp.float64(0.0)
    if latent_heat_enabled == wp.int32(1):
        thermal_conductivity = _thermal_conductivity_wp(temperature_value)

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
        surface_vapor_pressure = (
            activity_factor * vapor_pressure[box_idx, species_idx] * kelvin_term
        )
        pressure_delta = partial_pressure_gas - surface_vapor_pressure
        species_latent_heat = wp.float64(0.0)
        if latent_heat_enabled == wp.int32(1):
            species_latent_heat = latent_heat[species_idx]
        if species_latent_heat == wp.float64(0.0):
            mass_rate = mass_transfer_rate_wp(
                pressure_delta,
                mass_transport,
                temperature_value,
                molar_mass[species_idx],
                gas_constant,
            )
        else:
            mass_rate = _mass_transfer_rate_latent_heat_wp(
                pressure_delta,
                mass_transport,
                temperature_value,
                molar_mass[species_idx],
                species_latent_heat,
                thermal_conductivity,
                surface_vapor_pressure,
                diffusion_value,
                gas_constant,
            )
        mass_transfer[box_idx, particle_idx, species_idx] = (
            mass_rate * time_step
        )


@wp.kernel  # pragma: no cover - device kernels execute outside Python coverage
def _effective_surface_tension_kernel(
    masses: Any,
    density: Any,
    surface_tension: Any,
    output: Any,
) -> None:
    """Compute one composition-weighted surface tension per particle.

    The output is recalculated from the current mass in each substep and reused
    for every condensing species of that particle, avoiding a full composition
    reduction per species.
    """
    box_idx, particle_idx = wp.tid()  # type: ignore[misc]
    output[box_idx, particle_idx] = effective_surface_tension_wp(
        masses, density, surface_tension, box_idx, particle_idx, 0, True
    )


@wp.kernel  # pragma: no cover - device kernels execute outside Python coverage
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


@wp.kernel  # pragma: no cover - device kernels execute outside Python coverage
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


@wp.kernel  # pragma: no cover - device kernels execute outside Python coverage
# type: ignore[misc]
def _prepare_environment_properties_kernel(
    temperature: Any,
    pressure: Any,
    dynamic_viscosity: Any,
    mean_free_path: Any,
) -> None:
    """Precompute box-level gas properties for one condensation substep.

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


@wp.kernel  # pragma: no cover - device kernels execute outside Python coverage
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


@wp.kernel  # pragma: no cover - device kernels execute outside Python coverage
# type: ignore[misc]
def _clear_mass_transfer_kernel(
    total_mass_transfer: Any,
) -> None:
    """Clear the fixed-shape accumulator before the four-substep update."""
    box_idx, particle_idx = wp.tid()  # type: ignore[misc]
    for species_idx in range(total_mass_transfer.shape[2]):
        total_mass_transfer[box_idx, particle_idx, species_idx] = wp.float64(
            0.0
        )


@wp.kernel  # pragma: no cover - device kernels execute outside Python coverage
# type: ignore[misc]
def _clear_energy_transfer_kernel(
    energy_transfer: Any,
) -> None:
    """Clear caller-owned whole-call energy diagnostic storage."""
    box_idx = wp.tid()  # type: ignore[misc]
    for species_idx in range(energy_transfer.shape[1]):
        energy_transfer[box_idx, species_idx] = wp.float64(0.0)


@wp.kernel  # pragma: no cover - device kernels execute outside Python coverage
# type: ignore[misc]
def _accumulate_energy_transfer_kernel(
    total_mass_transfer: Any,
    latent_heat: Any,
    energy_transfer: Any,
) -> None:
    """Accumulate one particle's signed energy using parallel particle lanes."""
    box_idx, particle_idx = wp.tid()  # type: ignore[misc]
    for species_idx in range(total_mass_transfer.shape[2]):
        wp.atomic_add(
            energy_transfer,
            box_idx,
            species_idx,
            total_mass_transfer[box_idx, particle_idx, species_idx]
            * latent_heat[species_idx],
        )


@wp.kernel  # pragma: no cover - device kernels execute outside Python coverage
# type: ignore[misc]
def _accumulate_finalized_mass_transfer_kernel(
    finalized: Any,
    total_mass_transfer: Any,
) -> None:
    """Add one P2-finalized transfer to the whole-call total."""
    box_idx, particle_idx = wp.tid()  # type: ignore[misc]
    for species_idx in range(finalized.shape[2]):
        total_mass_transfer[box_idx, particle_idx, species_idx] += finalized[
            box_idx, particle_idx, species_idx
        ]


@wp.kernel  # pragma: no cover - device kernels execute outside Python coverage
# type: ignore[misc]
def _couple_finalized_transfer_to_gas_kernel(
    finalized: Any,
    concentration: Any,
    gas_concentration: Any,
) -> None:
    """Conservatively subtract finalized particle transfer from gas."""
    box_idx, species_idx = wp.tid()  # type: ignore[misc]
    gas_delta = wp.float64(0.0)
    for particle_idx in range(finalized.shape[1]):
        gas_delta += (
            finalized[box_idx, particle_idx, species_idx]
            * concentration[box_idx, particle_idx]
        )
    gas_concentration[box_idx, species_idx] -= gas_delta


@wp.kernel  # pragma: no cover - device kernels execute outside Python coverage
# type: ignore[misc]
def _bound_evaporation_candidate_kernel(
    masses: Any,
    gated_transfer: Any,
    candidate: Any,
) -> None:
    """Copy a gated proposal while bounding evaporation by owned mass.

    Args:
        masses: Particle masses ``(n_boxes, n_particles, n_species)`` [kg].
        gated_transfer: Already P1-gated transfer proposal matching ``masses``
            [kg].
        candidate: Output bounded proposal matching ``masses`` [kg].
    """
    box_idx, particle_idx = wp.tid()  # type: ignore[misc]
    for species_idx in range(masses.shape[2]):
        transfer = gated_transfer[box_idx, particle_idx, species_idx]
        mass = masses[box_idx, particle_idx, species_idx]
        if transfer < -mass:
            transfer = -mass
        candidate[box_idx, particle_idx, species_idx] = transfer


@wp.kernel  # pragma: no cover - device kernels execute outside Python coverage
# type: ignore[misc]
def _reduce_inventory_candidates_kernel(
    candidate: Any,
    concentration: Any,
    demand: Any,
    release: Any,
) -> None:
    """Reduce concentration-weighted demand and release in index order.

    Each ``(box, species)`` lane traverses particles in ascending index order,
    avoiding atomic accumulation. Positive candidates contribute to demand;
    negative candidates contribute their magnitude to release.

    Args:
        candidate: Bounded, already gated transfer proposal [kg].
        concentration: Particle number concentrations ``(n_boxes, n_particles)``
            [1/m^3].
        demand: Output positive transfer demand ``(n_boxes, n_species)``
            [kg/m^3].
        release: Output evaporation release ``(n_boxes, n_species)`` [kg/m^3].
    """
    box_idx, species_idx = wp.tid()  # type: ignore[misc]
    positive_demand = wp.float64(0.0)
    negative_release = wp.float64(0.0)
    for particle_idx in range(candidate.shape[1]):
        transfer = candidate[box_idx, particle_idx, species_idx]
        weighted_transfer = transfer * concentration[box_idx, particle_idx]
        if transfer > wp.float64(0.0):
            positive_demand += weighted_transfer
        elif transfer < wp.float64(0.0):
            negative_release -= weighted_transfer
    demand[box_idx, species_idx] = positive_demand
    release[box_idx, species_idx] = negative_release


@wp.kernel  # pragma: no cover - device kernels execute outside Python coverage
# type: ignore[misc]
def _scale_inventory_uptake_kernel(
    gas_concentration: Any,
    demand: Any,
    release: Any,
    scale: Any,
) -> None:
    """Compute each box-species uptake fraction from available inventory.

    Available inventory is gas concentration plus the bounded evaporation
    release. The output is one when demand is covered (including zero demand)
    and otherwise the available-to-demand ratio clamped to ``[0, 1]``.

    Args:
        gas_concentration: Gas concentrations ``(n_boxes, n_species)``
            [kg/m^3], read without mutation.
        demand: Positive transfer demand ``(n_boxes, n_species)`` [kg/m^3].
        release: Evaporation release ``(n_boxes, n_species)`` [kg/m^3].
        scale: Output positive-transfer fractions ``(n_boxes, n_species)``.
    """
    box_idx, species_idx = wp.tid()  # type: ignore[misc]
    demand_value = demand[box_idx, species_idx]
    available = (
        gas_concentration[box_idx, species_idx] + release[box_idx, species_idx]
    )
    fraction = wp.float64(1.0)
    if demand_value > available:
        fraction = available / demand_value
        if fraction < wp.float64(0.0):
            fraction = wp.float64(0.0)
        elif fraction > wp.float64(1.0):
            fraction = wp.float64(1.0)
    scale[box_idx, species_idx] = fraction


@wp.kernel  # pragma: no cover - device kernels execute outside Python coverage
# type: ignore[misc]
def _finalize_and_apply_inventory_transfer_kernel(
    masses: Any,
    candidate: Any,
    scale: Any,
    finalized: Any,
) -> None:
    """Apply bounded evaporation and inventory-scaled uptake to masses.

    Negative candidates are retained and positive candidates are multiplied by
    their box-species scale. This kernel has no gas-concentration argument and
    therefore does not perform gas coupling.

    Args:
        masses: Particle masses updated in place [kg].
        candidate: Bounded, already gated transfer proposal [kg].
        scale: Positive-transfer fractions by box and species.
        finalized: Output applied transfer matching ``masses`` [kg].
    """
    box_idx, particle_idx = wp.tid()  # type: ignore[misc]
    for species_idx in range(masses.shape[2]):
        transfer = candidate[box_idx, particle_idx, species_idx]
        if transfer > wp.float64(0.0):
            transfer *= scale[box_idx, species_idx]
        finalized[box_idx, particle_idx, species_idx] = transfer
        masses[box_idx, particle_idx, species_idx] += transfer


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
    maximum: float | None = None,
) -> None:
    """Validate metadata for a caller-owned float64 per-species Warp array.

    Numeric-domain validation runs before the step can clear or mutate any
    caller-owned output. CUDA values are checked by a device-side preflight
    kernel; only its one-element validation result is read back.
    """
    if not _is_warp_array_like(array):
        raise ValueError(f"{name} must be a Warp array")
    _validate_species_array(name, array, n_species, expected_device)
    if array.dtype != wp.float64:
        raise ValueError(f"{name} must use dtype {wp.float64}")
    if not getattr(expected_device, "is_cuda", False):
        values = np.asarray(array.numpy(), dtype=np.float64)
        if (
            not np.all(np.isfinite(values))
            or (nonnegative and np.any(values < 0.0))
            or (maximum is not None and np.any(values > maximum))
        ):
            raise ValueError(f"{name} must be finite and non-negative")
        return

    invalid = wp.zeros(1, dtype=wp.int32, device=expected_device)
    wp.launch(
        _validate_species_values_kernel,
        dim=n_species,
        inputs=[
            array,
            wp.int32(nonnegative),
            wp.float64(np.inf if maximum is None else maximum),
            invalid,
        ],
        device=expected_device,
    )
    if invalid.numpy()[0] != 0:
        raise ValueError(f"{name} must be finite and non-negative")


@wp.kernel  # pragma: no cover - device kernels execute outside Python coverage
# type: ignore[misc]
def _validate_species_values_kernel(
    values: Any,
    require_nonnegative: wp.int32,
    maximum: wp.float64,
    invalid: Any,
) -> None:
    """Record whether a species sidecar has an invalid numeric value."""
    species_idx = wp.tid()
    value = values[species_idx]
    if (
        not wp.isfinite(value)
        or (require_nonnegative != 0 and value < 0.0)
        or value > maximum
    ):
        wp.atomic_add(invalid, 0, 1)


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
    if not _is_warp_array_like(mass_transfer):
        raise ValueError("mass_transfer must be a Warp array")
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
    if mass_transfer.dtype != wp.float64:
        raise ValueError(f"mass_transfer must use dtype {wp.float64}")


def _validate_energy_transfer_buffer(
    energy_transfer: Any,
    expected_shape: tuple[int, int],
    expected_device: Any,
) -> None:
    """Validate metadata for caller-owned write-only energy output storage."""
    if not _is_warp_array_like(energy_transfer):
        raise ValueError("energy_transfer must be a Warp array")
    if energy_transfer.shape != expected_shape:
        raise ValueError(
            f"energy_transfer shape {energy_transfer.shape} does not match "
            f"expected {expected_shape}"
        )
    if getattr(energy_transfer, "device", None) is None or str(
        energy_transfer.device
    ) != str(expected_device):
        raise ValueError(
            "energy_transfer device does not match particle device"
        )
    if energy_transfer.dtype != wp.float64:
        raise ValueError(f"energy_transfer must use dtype {wp.float64}")


def _warp_array_memory_range(array: Any) -> tuple[int, int]:
    """Return a contiguous Warp array's byte range.

    Ownership checks reject strided views because a single address range cannot
    distinguish their gaps from overlapping storage.
    """
    dtype_sizes = {
        wp.float64: np.dtype(np.float64).itemsize,
        wp.float32: np.dtype(np.float32).itemsize,
        wp.int32: np.dtype(np.int32).itemsize,
    }
    itemsize = dtype_sizes.get(array.dtype)
    if itemsize is None:
        raise ValueError("overlap validation does not support this Warp dtype")
    strides = getattr(array, "strides", None)
    expected_strides: list[int] = []
    stride = itemsize
    for dimension in reversed(array.shape):
        expected_strides.insert(0, stride)
        stride *= dimension
    if strides is not None and tuple(strides) != tuple(expected_strides):
        raise ValueError(
            "overlap-checked Warp arrays must be contiguous, non-view arrays"
        )
    start = int(array.ptr)
    item_count = int(np.prod(array.shape, dtype=np.int64))
    return start, start + item_count * itemsize


def _validate_no_overlap(
    read_only_arrays: tuple[Any | None, ...],
    writable_arrays: tuple[Any | None, ...],
) -> None:
    """Reject thermal sidecars that alias writable scratch property storage."""
    for read_only in read_only_arrays:
        if read_only is None:
            continue
        read_start, read_end = _warp_array_memory_range(read_only)
        for writable in writable_arrays:
            if writable is None:
                continue
            write_start, write_end = _warp_array_memory_range(writable)
            if read_only is writable or (
                read_start < write_end and write_start < read_end
            ):
                raise ValueError(
                    "thermal sidecars must not overlap writable scratch "
                    "property buffers"
                )


def _validate_energy_transfer_ownership(
    energy_transfer: Any,
    mutable_arrays: tuple[Any, ...],
) -> None:
    """Reject energy output storage overlapping condensation state."""
    output_start, output_end = _warp_array_memory_range(energy_transfer)
    for array in mutable_arrays:
        if array is None or not _is_warp_array_like(array):
            continue
        if energy_transfer is array:
            raise ValueError(
                "energy_transfer must not overlap mutable or read-side "
                "condensation state"
            )
        array_start, array_end = _warp_array_memory_range(array)
        if output_start < array_end and array_start < output_end:
            raise ValueError(
                "energy_transfer must not overlap mutable or read-side "
                "condensation state"
            )


@wp.kernel  # pragma: no cover - device kernels execute outside Python coverage
# type: ignore[misc]
def _validate_p2_primary_values_kernel(
    masses: Any,
    concentration: Any,
    gas_concentration: Any,
    invalid: Any,
) -> None:
    """Record invalid primary P2 state without changing caller data."""
    box_idx, species_idx = wp.tid()  # type: ignore[misc]
    gas_value = gas_concentration[box_idx, species_idx]
    if not wp.isfinite(gas_value) or gas_value < 0.0:
        wp.atomic_add(invalid, 0, 1)
    for particle_idx in range(masses.shape[1]):
        concentration_value = concentration[box_idx, particle_idx]
        if not wp.isfinite(concentration_value) or concentration_value < 0.0:
            wp.atomic_add(invalid, 0, 1)
        mass = masses[box_idx, particle_idx, species_idx]
        if not wp.isfinite(mass) or mass < 0.0:
            wp.atomic_add(invalid, 0, 1)


@wp.kernel  # pragma: no cover - device kernels execute outside Python coverage
# type: ignore[misc]
def _validate_p2_proposal_values_kernel(proposal: Any, invalid: Any) -> None:
    """Record a non-finite freshly generated P1 proposal."""
    box_idx, particle_idx = wp.tid()  # type: ignore[misc]
    for species_idx in range(proposal.shape[2]):
        if not wp.isfinite(proposal[box_idx, particle_idx, species_idx]):
            wp.atomic_add(invalid, 0, 1)


def _validate_p2_primary_state(
    particles: Any,
    gas: Any,
    dimensions: tuple[int, int, int],
    device: Any,
) -> None:
    """Validate finite, non-negative primary P2 state before launch."""
    n_boxes, _, n_species = dimensions
    if not getattr(device, "is_cuda", False):
        values = (
            ("particles.masses", particles.masses, True),
            ("particles.concentration", particles.concentration, True),
            ("gas.concentration", gas.concentration, True),
        )
        for name, array, nonnegative in values:
            numeric_values = _read_float64_array(array)
            if not np.all(np.isfinite(numeric_values)) or (
                nonnegative and np.any(numeric_values < 0.0)
            ):
                requirement = (
                    "finite and non-negative" if nonnegative else "finite"
                )
                raise ValueError(f"{name} must be {requirement}")
        return

    invalid = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(
        _validate_p2_primary_values_kernel,
        dim=(n_boxes, n_species),
        inputs=[
            particles.masses,
            particles.concentration,
            gas.concentration,
            invalid,
        ],
        device=device,
    )
    if invalid.numpy()[0] != 0:
        raise ValueError(
            "P2 particle masses, concentrations, and gas concentration must be "
            "finite and non-negative"
        )


def _validate_p2_proposal_finiteness(
    proposal: Any, dimensions: tuple[int, int, int], device: Any
) -> None:
    """Validate a freshly written P1 proposal immediately before P2 mutation."""
    if not getattr(device, "is_cuda", False):
        if not np.all(np.isfinite(_read_float64_array(proposal))):
            raise ValueError("gated_mass_transfer must be finite")
        return
    invalid = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(
        _validate_p2_proposal_values_kernel,
        dim=dimensions[:2],
        inputs=[proposal, invalid],
        device=device,
    )
    if invalid.numpy()[0] != 0:
        raise ValueError("gated_mass_transfer must be finite")


def _validate_p2_sidecar_ownership(
    demand: Any | None,
    release: Any | None,
    scale: Any | None,
    protected_arrays: tuple[tuple[str, Any | None], ...],
) -> None:
    """Reject P2 reduction sidecars overlapping protected or peer storage."""
    sidecars = (
        ("positive_mass_transfer_demand", demand),
        ("negative_mass_transfer_release", release),
        ("positive_mass_transfer_scale", scale),
    )
    for index, (name, sidecar) in enumerate(sidecars):
        if sidecar is None:
            continue
        sidecar_start, sidecar_end = _warp_array_memory_range(sidecar)
        for other_name, other in (*sidecars[index + 1 :], *protected_arrays):
            if other is None:
                continue
            other_start, other_end = _warp_array_memory_range(other)
            if sidecar is other or (
                sidecar_start < other_end and other_start < sidecar_end
            ):
                raise ValueError(
                    f"{name} must not overlap {other_name} in direct P2 state"
                )


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


@wp.kernel  # pragma: no cover - device kernels execute outside Python coverage
# type: ignore[misc]
def _validate_partitioning_values_kernel(
    partitioning: Any, invalid: Any
) -> None:
    """Record non-binary entries in a per-box partitioning mask."""
    box_idx, species_idx = wp.tid()  # type: ignore[misc]
    value = partitioning[box_idx, species_idx]
    if value != wp.int32(0) and value != wp.int32(1):
        wp.atomic_add(invalid, 0, 1)


def _validate_partitioning_metadata(
    partitioning: Any,
    n_boxes: int,
    n_species: int,
    device: Any,
    caller_name: str,
) -> None:
    """Validate an active-device per-box partitioning mask's metadata."""
    _validate_array_metadata(
        "gas.partitioning",
        partitioning,
        (n_boxes, n_species),
        wp.int32,
        device,
        caller_name,
        "",
    )


def _validate_partitioning_values(
    partitioning: Any,
    n_boxes: int,
    n_species: int,
    device: Any,
) -> None:
    """Validate the binary values after all caller sidecars validate.

    The one-element status buffer is private disposable preflight state. Its
    readback is necessary to make arbitrary device-resident non-binary values
    observable before any caller-owned state is changed.
    """
    invalid = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(
        _validate_partitioning_values_kernel,
        dim=(n_boxes, n_species),
        inputs=[partitioning, invalid],
        device=device,
    )
    if invalid.numpy()[0] != 0:
        raise ValueError("gas.partitioning must contain only binary 0/1 values")


@wp.kernel  # pragma: no cover - device kernels execute outside Python coverage
# type: ignore[misc]
def _gate_mass_transfer_kernel(
    partitioning: Any,
    concentration: Any,
    work_mass_transfer: Any,
) -> None:
    """Zero proposals for disabled species and zero-concentration slots.

    A partitioning value of one permits a proposal; zero disables it. This P1
    gate is performed before the public step applies mass transfer. The P2
    inventory finalizer accepts the resulting pre-gated proposal and does not
    inspect this mask.

    Args:
        partitioning: Binary per-box, per-species mask where one permits
            partitioning.
        concentration: Particle number concentrations by box and particle.
        work_mass_transfer: Raw proposal overwritten with zeros where disabled.
    """
    box_idx, particle_idx = wp.tid()  # type: ignore[misc]
    for species_idx in range(work_mass_transfer.shape[2]):
        if partitioning[box_idx, species_idx] != wp.int32(1) or concentration[
            box_idx, particle_idx
        ] == wp.float64(0.0):
            work_mass_transfer[box_idx, particle_idx, species_idx] = wp.float64(
                0.0
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
    _validate_device_match("gas partitioning", gas.partitioning, device)


def _finalize_inventory_limited_mass_transfer(
    particles: Any,
    gas: Any,
    gated_mass_transfer: Any,
    scratch_buffers: CondensationScratchBuffers | Any | None = None,
    *,
    energy_transfer: Any | None = None,
) -> Any:
    """Finalize a pre-gated proposal under particle and gas inventory limits.

    This private, direct-test-only P2 primitive first bounds evaporation by
    owned particle mass. It then limits concentration-weighted positive uptake
    for each box and species to gas inventory plus permitted evaporation
    release. It mutates particle masses and any supplied P2 sidecars, but
    deliberately does not mutate ``gas.concentration``; public orchestration
    and gas coupling remain deferred.

    Args:
        particles: GPU particle data with float64 masses and concentrations.
        gas: GPU gas data whose concentration supplies read-only inventory.
        gated_mass_transfer: Float64 ``(n_boxes, n_particles, n_species)``
            proposal [kg] already P1-gated for partitioning and inactive slots.
        scratch_buffers: Optional validated scratch sidecar. Only its P2 demand,
            release, and scale fields are resolved or mutated.
        energy_transfer: Optional public-step energy output to protect from P2
            sidecar aliasing. It is never read or written by this helper.

    Returns:
        New float64 finalized applied-transfer array matching particle masses
        [kg].

    Raises:
        ValueError: If physical inputs are invalid, buffers have incompatible
            metadata, or P2 sidecars overlap protected or peer storage.

    Notes:
        Preflight validation finishes before this helper resolves fallback
        state, allocates output, or launches a mutating kernel. Validation
        failures leave caller-owned state unchanged; this does not promise
        rollback after a post-launch device-runtime failure.
    """
    masses = particles.masses
    if not _is_warp_array_like(masses) or len(masses.shape) != 3:
        raise ValueError("particles.masses must be a 3D Warp array")
    device = getattr(masses, "device", None)
    if device is None:
        raise ValueError("particles.masses must be on a Warp device")
    if masses.dtype != wp.float64:
        raise ValueError(f"particles.masses must use dtype {wp.float64}")

    dimensions = tuple(masses.shape)
    _validate_mass_transfer_buffer(gated_mass_transfer, dimensions, device)
    n_boxes, n_particles, n_species = dimensions
    _validate_particle_arrays(particles, n_boxes, n_particles, n_species)
    _validate_gas_arrays(gas, n_boxes, n_species)
    _validate_device_arrays(particles, gas, device)
    _validate_p2_primary_state(particles, gas, dimensions, device)
    _validate_p2_proposal_finiteness(gated_mass_transfer, dimensions, device)
    if energy_transfer is not None:
        _validate_energy_transfer_buffer(
            energy_transfer,
            (n_boxes, n_species),
            device,
        )
    if scratch_buffers is not None:
        validated_scratch = validate_condensation_scratch_buffers(
            scratch_buffers,
            dimensions,
            device,
            "_finalize_inventory_limited_mass_transfer",
        )
    else:
        validated_scratch = CondensationScratchBuffers()

    _validate_p2_sidecar_ownership(
        validated_scratch.positive_mass_transfer_demand,
        validated_scratch.negative_mass_transfer_release,
        validated_scratch.positive_mass_transfer_scale,
        (
            ("particles.masses", particles.masses),
            ("particles.concentration", particles.concentration),
            ("gas.concentration", gas.concentration),
            ("gated_mass_transfer", gated_mass_transfer),
            ("energy_transfer", energy_transfer),
            ("dynamic_viscosity", validated_scratch.dynamic_viscosity),
            ("mean_free_path", validated_scratch.mean_free_path),
        ),
    )
    reduction_shape = (n_boxes, n_species)
    demand = validated_scratch.positive_mass_transfer_demand
    if demand is None:
        demand = wp.zeros(reduction_shape, dtype=wp.float64, device=device)
    release = validated_scratch.negative_mass_transfer_release
    if release is None:
        release = wp.zeros(reduction_shape, dtype=wp.float64, device=device)
    scale = validated_scratch.positive_mass_transfer_scale
    if scale is None:
        scale = wp.zeros(reduction_shape, dtype=wp.float64, device=device)

    candidate = wp.zeros(dimensions, dtype=wp.float64, device=device)
    finalized = wp.zeros(dimensions, dtype=wp.float64, device=device)
    wp.launch(
        _bound_evaporation_candidate_kernel,
        dim=(n_boxes, n_particles),
        inputs=[masses, gated_mass_transfer, candidate],
        device=device,
    )
    wp.launch(
        _reduce_inventory_candidates_kernel,
        dim=reduction_shape,
        inputs=[candidate, particles.concentration, demand, release],
        device=device,
    )
    wp.launch(
        _scale_inventory_uptake_kernel,
        dim=reduction_shape,
        inputs=[gas.concentration, demand, release, scale],
        device=device,
    )
    wp.launch(
        _finalize_and_apply_inventory_transfer_kernel,
        dim=(n_boxes, n_particles),
        inputs=[masses, candidate, scale, finalized],
        device=device,
    )
    return finalized


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
    latent_heat: Any | None = None,
    energy_transfer: Any | None = None,
    thermal_work: Any | None = None,
) -> tuple[Any, Any]:
    """Execute a fixed four-substep condensation timestep on the GPU.

    Args:
        particles: GPU-resident particle data.
        gas: GPU-resident gas data.
            ``gas.partitioning`` must be an active-device binary ``wp.int32``
            mask shaped ``(n_boxes, n_species)``.
        temperature: Direct gas temperature as either a scalar or a Warp array
            with shape ``(n_boxes,)``. Use ``None`` only with
            ``environment=...``.
        pressure: Direct gas pressure as either a scalar or a Warp array with
            shape ``(n_boxes,)``. Use ``None`` only with ``environment=...``.
        time_step: Total condensation time step [s], divided into four equal
            substeps.
        surface_tension: Optional per-species surface tension [N/m].
        mass_accommodation: Optional per-species accommodation coefficient.
        diffusion_coefficient_vapor: Optional per-species vapor diffusion
            coefficient [m^2/s].
        mass_transfer: Optional preallocated accumulated applied-transfer buffer
            with shape ``(n_boxes, n_particles, n_species)``.
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
        scratch_buffers: Optional frozen caller-owned buffers. Supplied
            transfer fields have shape ``(n_boxes, n_particles, n_species)``;
            supplied property fields have shape ``(n_boxes,)``. Every supplied
            field must be active-device ``wp.float64``. Fields may be omitted
            independently and use step-local fallback allocation. A supplied
            total transfer buffer is returned by identity. Work retains the
            final gated raw proposal and total accumulates P2-finalized applied
             transfers.
             P2 limiting sidecars have shape ``(n_boxes, n_species)`` and are
             validated before launch and are reused by all four P2 finalization
             cycles.
            Validation is performed before allocation or mutation. Arrays must
            remain alive and unmodified until launched work completes and may
            be reused only after a successful completion.
        latent_heat: Optional caller-owned per-species latent heat [J/kg] as an
            active-device ``wp.float64`` array shaped ``(n_species,)``. When
            supplied, it is consumed in every fixed substep; zero entries use
            the exact isothermal rate path.
        energy_transfer: Optional caller-owned, active-device ``wp.float64``
            write-only energy output [J] shaped ``(n_boxes, n_species)``. It
            requires valid ``latent_heat``. After successful preflight, the
            same buffer is cleared and overwritten with signed whole-call
            bounded applied transfer times per-species ``latent_heat``; callers
            may reuse it on a later successful call. Its existing contents,
            including stale NaN/Inf, are deliberately not validated because it
            is output-only diagnostic storage. It is not returned as a third
            tuple item.
        thermal_work: Optional caller-owned per-species thermal-work sidecar
            [J/kg] as an active-device ``wp.float64`` array shaped
            ``(n_species,)``. It is validated but deferred and unused P3 state;
            this step does not allocate, initialize, modify, or consume it.

    Returns:
        Two-item tuple of the particle data with in-place updated masses and
        accumulated, P2-finalized mass transfer [kg]. ``energy_transfer``, when
        supplied, remains caller-owned output rather than a third tuple item.
        Gas concentration is coupled after each finalized substep transfer.

    Raises:
        ValueError: If species counts, array lengths, or devices mismatch.
        ValueError: If ``gas.partitioning`` is not an active-device binary
             ``wp.int32`` array shaped ``(n_boxes, n_species)``.
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
        ValueError: If ``scratch_buffers`` is not the exact frozen sidecar type,
            a supplied field lacks its stable active-device ``wp.float64``
            metadata, or ``mass_transfer`` overlaps a supplied scratch transfer
            field.
        ValueError: If ``latent_heat`` or ``thermal_work`` is not a finite,
            non-negative active-device ``wp.float64`` array shaped
            ``(n_species,)``. Latent heat above ``1e9`` J/kg is also rejected
            before launch to prevent non-isothermal arithmetic overflow.
        ValueError: If ``energy_transfer`` is supplied without valid
            ``latent_heat`` or lacks active-device ``wp.float64``
            ``(n_boxes, n_species)`` metadata.

    Notes:
        Accepted environment sources are scalar direct inputs, direct
        ``(n_boxes,)`` Warp arrays, hybrid scalar-plus-Warp-array direct
        inputs, or keyword-only ``environment=...`` execution. NumPy arrays,
        Python lists, and other non-Warp direct array-likes are rejected.

        Particle masses are updated in-place on the GPU. Callers that require
        rollback should copy masses before invoking this function.

        Exactly four equal substeps run for every valid timestep. The returned
        transfer buffer accumulates the P2-finalized applied transfer from all
        four substeps. A supplied scratch work buffer holds only the final
        gated raw proposal. Each proposal is zeroed before application for
        disabled ``gas.partitioning`` species and zero-concentration particle
        slots. Each finalized transfer is also subtracted from
        ``gas.concentration`` weighted by particle concentration, so the next
        substep proposal reads coupled gas inventory. Vapor-pressure refresh
        itself does not read gas concentration.

        ``environment`` remains keyword-only so existing positional scalar
        callers stay source-compatible.

        Aggregate configuration and optional-buffer validation completes before
        allocation, launch, refresh, or mutation. Thus a validation failure
        leaves caller-owned particle, gas, environment, sidecar, and supplied
        output buffers unchanged and is retryable with corrected inputs.
        In particular, partitioning metadata and its status-only binary
        validation complete before environment normalization, fallback
        allocation, output clearing, vapor-pressure refresh, or physical
        mutation.
        Each substep overwrites ``gas.vapor_pressure`` from the normalized
        current temperature, prepares box-level properties, calculates and
        gates a raw proposal from current particle mass, validates that fresh
        proposal, then P2-finalizes, applies, couples, and accumulates it.
        A later-substep fresh-proposal validation failure does not roll back
        earlier completed substeps: it may leave only that substep's raw work
        proposal written, while its P2, particle, gas, total, and energy state
        remains unchanged.
        Float32 temperature arrays are
        cast device-side to float64 for refresh.

        A supplied work and/or total scratch transfer field conflicts with
        ``mass_transfer`` because they overlap its legacy output role; a
        property-only sidecar can be used with ``mass_transfer``. The resolved
        total buffer is cleared once and returned by identity when supplied.
        Work is separate from total so it can retain the final gated raw
        proposal.
        With no scratch transfer fields, legacy ``mass_transfer`` remains the
        returned total by identity.

        Thermodynamic metadata and production calculations stay device-resident.
        CUDA numeric validation of each supplied thermal sidecar reads back one
        device validation flag before any caller-owned state is mutated.

        Supplied latent sidecars are caller-owned. Latent heat is consumed per
        fixed substep without mutation; omitting it or supplying zero for a
        species preserves that species' isothermal arithmetic. Thermal work is
        validated but unused. Invalid metadata or values leave physical state
        and all caller-owned work state unchanged. To make overflow detectable
        before launch, finite latent heat is limited to ``1e9`` J/kg.

        Energy transfer is caller-owned, allocation-stable diagnostic storage.
        It is cleared and overwritten only after successful preflight, is
        reconstructible on each successful whole call, and is deliberately not
        content-validated because the step only writes it. This write-only
        contract permits stale finite values and NaN/Inf prior to a call.
    """
    if (
        isinstance(time_step, bool)
        or not isinstance(time_step, Real)
        or not np.isfinite(time_step)
        or time_step < 0.0
    ):
        raise ValueError(
            "time_step must be a finite, nonnegative real value and not bool"
        )

    n_boxes, n_particles, n_species = particles.masses.shape
    _validate_gas_arrays(gas, n_boxes, n_species)
    _validate_particle_arrays(particles, n_boxes, n_particles, n_species)

    device = particles.masses.device
    _validate_device_arrays(particles, gas, device)
    _validate_partitioning_metadata(
        gas.partitioning,
        n_boxes,
        n_species,
        device,
        "condensation_step_gpu",
    )
    thermodynamics = validate_thermodynamics_config(
        thermodynamics,
        n_species,
        device,
        gas.molar_mass,
        "condensation_step_gpu",
    )

    if latent_heat is not None:
        _validate_float64_species_array(
            "latent_heat",
            latent_heat,
            n_species,
            device,
            nonnegative=True,
            maximum=1.0e9,
        )
    if energy_transfer is not None:
        if latent_heat is None:
            raise ValueError("energy_transfer requires latent_heat")
        _validate_energy_transfer_buffer(
            energy_transfer,
            (n_boxes, n_species),
            device,
        )
    if thermal_work is not None:
        _validate_float64_species_array(
            "thermal_work",
            thermal_work,
            n_species,
            device,
            nonnegative=True,
        )

    latent_heat_enabled = wp.int32(0)
    latent_heat_values = gas.molar_mass
    if latent_heat is not None:
        latent_heat_values = latent_heat
        latent_heat_enabled = wp.int32(1)

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
        _validate_no_overlap(
            (latent_heat, thermal_work),
            (
                scratch_buffers.dynamic_viscosity,
                scratch_buffers.mean_free_path,
            ),
        )

    _validate_partitioning_values(
        gas.partitioning,
        n_boxes,
        n_species,
        device,
    )

    validate_environment_inputs(
        temperature=temperature,
        pressure=pressure,
        environment=environment,
        n_boxes=n_boxes,
        device=device,
        caller_name="condensation_step_gpu",
    )
    # P2 primary state and caller-owned sidecars validate before environment
    # normalization or fallback allocation, so invalid state is fully atomic.
    _validate_p2_primary_state(particles, gas, expected_shape, device)
    if scratch_buffers is not None:
        _validate_p2_sidecar_ownership(
            scratch_buffers.positive_mass_transfer_demand,
            scratch_buffers.negative_mass_transfer_release,
            scratch_buffers.positive_mass_transfer_scale,
            (
                ("particles.masses", particles.masses),
                ("particles.concentration", particles.concentration),
                ("gas.concentration", gas.concentration),
                ("gas.vapor_pressure", gas.vapor_pressure),
                ("mass_transfer", mass_transfer),
                ("work_mass_transfer", scratch_buffers.work_mass_transfer),
                ("total_mass_transfer", scratch_buffers.total_mass_transfer),
                ("energy_transfer", energy_transfer),
                ("dynamic_viscosity", scratch_buffers.dynamic_viscosity),
                ("mean_free_path", scratch_buffers.mean_free_path),
            ),
        )

    temperature_array, pressure_array = _ensure_environment_arrays(
        temperature=temperature,
        pressure=pressure,
        environment=environment,
        n_boxes=n_boxes,
        device=device,
        caller_name="condensation_step_gpu",
    )

    if energy_transfer is not None:
        _validate_energy_transfer_ownership(
            energy_transfer,
            (
                particles.masses,
                particles.concentration,
                particles.density,
                particles.charge,
                particles.volume,
                gas.molar_mass,
                gas.concentration,
                gas.vapor_pressure,
                thermodynamics.modes,
                thermodynamics.parameters,
                thermodynamics.molar_mass_reference,
                temperature_array,
                pressure_array,
                latent_heat,
                thermal_work,
                (
                    activity_surface.kappas
                    if activity_surface is not None
                    else None
                ),
                activity_surface.molar_mass_reference
                if activity_surface is not None
                else None,
                surface_tension,
                mass_accommodation,
                diffusion_coefficient_vapor,
                mass_transfer,
                scratch_buffers.work_mass_transfer
                if scratch_buffers is not None
                else None,
                scratch_buffers.total_mass_transfer
                if scratch_buffers is not None
                else None,
                scratch_buffers.dynamic_viscosity
                if scratch_buffers is not None
                else None,
                scratch_buffers.mean_free_path
                if scratch_buffers is not None
                else None,
                scratch_buffers.positive_mass_transfer_demand
                if scratch_buffers is not None
                else None,
                scratch_buffers.negative_mass_transfer_release
                if scratch_buffers is not None
                else None,
                scratch_buffers.positive_mass_transfer_scale
                if scratch_buffers is not None
                else None,
            ),
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

    scratch_buffers_value = scratch_buffers
    scratch_transfer_mode = scratch_buffers_value is not None and (
        scratch_buffers_value.work_mass_transfer is not None
        or scratch_buffers_value.total_mass_transfer is not None
    )
    if scratch_transfer_mode:
        if scratch_buffers_value is None:
            raise ValueError("scratch_buffers unexpectedly missing")
        work_mass_transfer = scratch_buffers_value.work_mass_transfer
        total_mass_transfer = scratch_buffers_value.total_mass_transfer
    elif mass_transfer is not None:
        # The legacy output is the accumulated total, so proposals need private
        # step-local storage rather than overwriting the caller's total buffer.
        work_mass_transfer = None
        total_mass_transfer = mass_transfer
    else:
        work_mass_transfer = None
        total_mass_transfer = None
    if work_mass_transfer is None:
        work_mass_transfer = wp.zeros(
            expected_shape, dtype=wp.float64, device=device
        )
    if total_mass_transfer is None:
        total_mass_transfer = wp.zeros(
            expected_shape, dtype=wp.float64, device=device
        )
    demand = (
        scratch_buffers.positive_mass_transfer_demand
        if scratch_buffers is not None
        and scratch_buffers.positive_mass_transfer_demand is not None
        else wp.zeros((n_boxes, n_species), dtype=wp.float64, device=device)
    )
    release = (
        scratch_buffers.negative_mass_transfer_release
        if scratch_buffers is not None
        and scratch_buffers.negative_mass_transfer_release is not None
        else wp.zeros((n_boxes, n_species), dtype=wp.float64, device=device)
    )
    scale = (
        scratch_buffers.positive_mass_transfer_scale
        if scratch_buffers is not None
        and scratch_buffers.positive_mass_transfer_scale is not None
        else wp.zeros((n_boxes, n_species), dtype=wp.float64, device=device)
    )
    candidate = wp.zeros(expected_shape, dtype=wp.float64, device=device)
    finalized = wp.zeros(expected_shape, dtype=wp.float64, device=device)
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

    wp.launch(
        _clear_mass_transfer_kernel,
        dim=(n_boxes, n_particles),
        inputs=[total_mass_transfer],
        device=device,
    )
    if energy_transfer is not None:
        wp.launch(
            _clear_energy_transfer_kernel,
            dim=n_boxes,
            inputs=[energy_transfer],
            device=device,
        )
    substep_time_step = wp.float64(time_step / 4.0)
    for _ in range(4):
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
                latent_heat_values,
                latent_heat_enabled,
                dynamic_viscosity,
                mean_free_path,
                wp.float64(constants.GAS_CONSTANT),
                wp.float64(constants.BOLTZMANN_CONSTANT),
                refresh_temperature,
                substep_time_step,
                work_mass_transfer,
            ],
            device=device,
        )
        wp.launch(
            _gate_mass_transfer_kernel,
            dim=(n_boxes, n_particles),
            inputs=[
                gas.partitioning,
                particles.concentration,
                work_mass_transfer,
            ],
            device=device,
        )
        # Work storage is output-only at entry.  Validate its freshly written
        # P1-gated proposal before any P2 sidecar, particle, gas, or total
        # write.
        _validate_p2_proposal_finiteness(
            work_mass_transfer, expected_shape, device
        )
        wp.launch(
            _bound_evaporation_candidate_kernel,
            dim=(n_boxes, n_particles),
            inputs=[particles.masses, work_mass_transfer, candidate],
            device=device,
        )
        wp.launch(
            _reduce_inventory_candidates_kernel,
            dim=(n_boxes, n_species),
            inputs=[candidate, particles.concentration, demand, release],
            device=device,
        )
        wp.launch(
            _scale_inventory_uptake_kernel,
            dim=(n_boxes, n_species),
            inputs=[gas.concentration, demand, release, scale],
            device=device,
        )
        wp.launch(
            _finalize_and_apply_inventory_transfer_kernel,
            dim=(n_boxes, n_particles),
            inputs=[particles.masses, candidate, scale, finalized],
            device=device,
        )
        wp.launch(
            _accumulate_finalized_mass_transfer_kernel,
            dim=(n_boxes, n_particles),
            inputs=[finalized, total_mass_transfer],
            device=device,
        )
        wp.launch(
            _couple_finalized_transfer_to_gas_kernel,
            dim=(n_boxes, n_species),
            inputs=[finalized, particles.concentration, gas.concentration],
            device=device,
        )

    if energy_transfer is not None:
        wp.launch(
            _accumulate_energy_transfer_kernel,
            dim=(n_boxes, n_particles),
            inputs=[total_mass_transfer, latent_heat, energy_transfer],
            device=device,
        )

    return particles, total_mass_transfer
