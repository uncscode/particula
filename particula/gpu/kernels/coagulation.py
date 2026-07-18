"""GPU Brownian, charged, and sedimentation coagulation utilities.

This module composes Warp ``@wp.func`` building blocks into an end-to-end
coagulation pipeline. Its fixed-mask sampler executes particle-resolved
Brownian, charged hard-sphere, exact unit-efficiency SP2016 sedimentation, or
the supported Brownian-plus-charged configuration in either requested order.
It normalizes the combined configuration to one mask, accumulates a finite
positive additive rate and safe majorant, then makes one acceptance draw for
each valid candidate. Entry-point validation accepts scalar direct inputs,
explicit ``(n_boxes,)`` Warp arrays, or a ``WarpEnvironmentData`` container.
Particle metadata and device checks run before normalizing environment inputs,
setting up volume, initializing RNG state, or executing Brownian work. An
explicit opt-in finite-charge validation scan is available for Brownian
callers; every mode containing charged hard-sphere physics or exact SP2016
sedimentation always validates finite charge before resource allocation or
mutation. The kernels operate on
    GPU-resident particle data and produce collision pairs that are applied in
    place: recipient particles receive
donor mass and charge, while donor mass, concentration, and charge are cleared.

``CoagulationMechanismConfig`` and its resolver are concrete-module APIs:
import them from ``particula.gpu.kernels.coagulation``, not
``particula.gpu.kernels``. The immutable, keyword-only configuration is
host-side metadata, not device-resident simulation state. The public step
    defaults to Brownian, particle-resolved execution and also accepts
    charged-only, exact SP2016 sedimentation-only, and canonical
    Brownian-plus-charged particle-resolved execution. It rejects otherwise
    unsupported configurations during preflight, before runtime state is
    accessed or mutated. Supplied particles, collision outputs, and RNG sidecars
    are caller-owned same-device Warp resources.

The exact ``SEDIMENTATION_SP2016_MECHANISM_FLAG`` is a public direct-kernel
mode for particle-resolved, unit-efficiency SP2016 scheduling. It shares the
bounded sampler and persistent RNG path. Mixed masks containing sedimentation
remain non-executable. Sedimentation calls read-only validate finite,
nonnegative masses and concentrations and finite, positive densities before
any output, RNG, or particle-state mutation.
"""

# pyright: basic
# pyright: reportArgumentType=false
# pyright: reportAssignmentType=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportOperatorIssue=false

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast, no_type_check

import numpy as np

import particula.util.constants as constants

if TYPE_CHECKING:  # pragma: no cover - import for typing only
    import warp as wp
else:  # pragma: no cover - runtime import with helpful error
    try:
        import warp as wp
    except ImportError as exc:
        raise ImportError(
            "Warp is required for GPU coagulation kernels. "
            "Install with: pip install warp-lang"
        ) from exc

from particula.gpu.dynamics.coagulation_funcs import (
    brownian_diffusivity_wp,
    brownian_kernel_pair_wp,
    charged_hard_sphere_wp,
    effective_density_wp,
    g_collection_term_wp,
    particle_mean_free_path_wp,
    sedimentation_sp2016_pair_rate_wp,
    settling_velocity_stokes_from_transport_wp,
)
from particula.gpu.dynamics.condensation_funcs import (
    particle_radius_from_volume_wp,
)
from particula.gpu.kernels.environment import (
    _broadcast_scalar_array,
    _ensure_environment_arrays,
    _is_supported_warp_float_dtype,
    _is_warp_array_like,
    _validate_positive_finite_array,
)
from particula.gpu.properties.gas_properties import (
    dynamic_viscosity_wp,
    molecule_mean_free_path_wp,
)
from particula.gpu.properties.particle_properties import (
    aerodynamic_mobility_wp,
    cunningham_slip_correction_wp,
    knudsen_number_wp,
    mean_thermal_speed_wp,
)

MAX_COLLISION_PAIR_BUFFER_BYTES = 256 * 1024 * 1024
MAX_SCHEDULED_TRIALS_PER_BOX = 65_536
# Charged selection obtains an exact majorant by visiting every compact active
# pair in one thread. This limit caps that O(A²) pre-launch work at 32,640
# pairs per box and applies to every charged-containing execution mode.
MAX_CHARGED_ACTIVE_PARTICLES_PER_BOX = 256
# Exact SP2016 selection also visits every compact active pair in one thread.
# Keep its work bounded independently of the charged-mode public contract.
MAX_SEDIMENTATION_ACTIVE_PARTICLES_PER_BOX = 256

BROWNIAN_MECHANISM = "brownian"
CHARGED_HARD_SPHERE_MECHANISM = "charged_hard_sphere"
SEDIMENTATION_SP2016_MECHANISM = "sedimentation_sp2016"
TURBULENT_SHEAR_ST1956_MECHANISM = "turbulent_shear_st1956"

CANONICAL_COAGULATION_MECHANISMS = (
    BROWNIAN_MECHANISM,
    CHARGED_HARD_SPHERE_MECHANISM,
    SEDIMENTATION_SP2016_MECHANISM,
    TURBULENT_SHEAR_ST1956_MECHANISM,
)

BROWNIAN_MECHANISM_FLAG = 1
CHARGED_HARD_SPHERE_MECHANISM_FLAG = 2
SEDIMENTATION_SP2016_MECHANISM_FLAG = 4
TURBULENT_SHEAR_ST1956_MECHANISM_FLAG = 8

_COAGULATION_MECHANISM_FLAGS = MappingProxyType(
    {
        BROWNIAN_MECHANISM: BROWNIAN_MECHANISM_FLAG,
        CHARGED_HARD_SPHERE_MECHANISM: CHARGED_HARD_SPHERE_MECHANISM_FLAG,
        SEDIMENTATION_SP2016_MECHANISM: SEDIMENTATION_SP2016_MECHANISM_FLAG,
        TURBULENT_SHEAR_ST1956_MECHANISM: TURBULENT_SHEAR_ST1956_MECHANISM_FLAG,
    }
)


@dataclass(frozen=True)
class CoagulationMechanismConfig:
    """Configure host-side coagulation mechanism selection.

    The default selects Brownian, particle-resolved coagulation. This immutable,
    concrete-module-only API must be imported from
    ``particula.gpu.kernels.coagulation``, not ``particula.gpu.kernels``.
    ``coagulation_step_gpu`` accepts it only through its keyword-only
    ``mechanism_config`` argument. The configuration is host metadata and does
    not own, transfer, or synchronize Warp resources. Executable
    ``"particle_resolved"`` modes are Brownian-only,
    ``("sedimentation_sp2016",)``,
    ``("charged_hard_sphere",)``, and either requested order of
    ``("brownian", "charged_hard_sphere")``. The resolver normalizes the
    combined mode to the canonical fixed mask. Deferred mechanisms and other
    distribution types are rejected during host-side preflight before any
    runtime state access or mutation.

    Attributes:
        mechanisms: Requested mechanism identifiers, or ``None`` to select
            Brownian. The resolver normalizes supported identifiers to canonical
            order.
        distribution_type: Required distribution representation; only
            ``"particle_resolved"`` is structurally supported.
    """

    mechanisms: tuple[str, ...] | None = None
    distribution_type: str = "particle_resolved"


@dataclass(frozen=True)
class _ResolvedCoagulationMechanismConfig:
    """Store the normalized result of concrete-module configuration validation.

    This private result is not a public API or a ``coagulation_step_gpu``
    argument. Structural resolution retains recognized, but potentially
    deferred, mechanisms for later capability validation.

    Attributes:
        mechanisms: Structurally valid mechanism identifiers in canonical order.
        distribution_type: Validated particle distribution representation.
        mask: Bitwise OR of the fixed flags for ``mechanisms``.
    """

    mechanisms: tuple[str, ...]
    distribution_type: str
    mask: int


def resolve_coagulation_mechanism_config(
    config: CoagulationMechanismConfig,
) -> _ResolvedCoagulationMechanismConfig:
    """Resolve a configuration through structural validation.

    ``config.mechanisms=None`` defaults to Brownian; ``config`` itself remains
    required. Valid identifiers are normalized to canonical order and retained
    in the returned fixed-bit mask, including reserved terms that the separate
    capability validator rejects. This pure host-side,
    concrete-module-only helper neither allocates device storage nor mutates its
    input or runtime state. Import it from
    ``particula.gpu.kernels.coagulation``, not ``particula.gpu.kernels``.

    Args:
        config: Immutable host-side coagulation mechanism configuration.

    Returns:
        Private resolved configuration with canonical mechanisms, validated
        distribution type, and the combined fixed-bit mask.

    Raises:
        ValueError: If mechanisms are malformed, duplicate, or unknown, or if
            distribution_type is not exactly ``"particle_resolved"``.
    """
    if config.distribution_type != "particle_resolved":
        raise ValueError(
            "distribution_type must be exactly 'particle_resolved'."
        )

    mechanisms = config.mechanisms
    if mechanisms is None:
        mechanisms = (BROWNIAN_MECHANISM,)
    if not isinstance(mechanisms, tuple) or not mechanisms:
        raise ValueError("mechanisms must be a non-empty tuple of strings.")

    for mechanism in mechanisms:
        if not isinstance(mechanism, str):
            raise ValueError("mechanisms must contain only string identifiers.")

    seen_mechanisms: set[str] = set()
    for mechanism in mechanisms:
        if mechanism in seen_mechanisms:
            raise ValueError(f"Duplicate coagulation mechanism '{mechanism}'.")
        seen_mechanisms.add(mechanism)

    unknown = next(
        (
            mechanism
            for mechanism in mechanisms
            if mechanism not in _COAGULATION_MECHANISM_FLAGS
        ),
        None,
    )
    if unknown is not None:
        raise ValueError(f"Unknown coagulation mechanism '{unknown}'.")

    normalized_mechanisms = tuple(
        mechanism
        for mechanism in CANONICAL_COAGULATION_MECHANISMS
        if mechanism in mechanisms
    )
    mask = 0
    for mechanism in normalized_mechanisms:
        mask |= _COAGULATION_MECHANISM_FLAGS[mechanism]

    return _ResolvedCoagulationMechanismConfig(
        mechanisms=normalized_mechanisms,
        distribution_type=config.distribution_type,
        mask=mask,
    )


def validate_coagulation_mechanism_capabilities(
    resolved: _ResolvedCoagulationMechanismConfig,
) -> None:
    """Enforce the executable coagulation-mechanism boundary.

    This pure host-side, concrete-module-only validator accepts Brownian,
    exact SP2016 sedimentation, ``"charged_hard_sphere"``, or the supported
    Brownian-plus-charged ``"particle_resolved"`` execution. Deferred
    mechanisms are rejected during configuration preflight,
    before accessing or mutating runtime state. Import it from
    ``particula.gpu.kernels.coagulation``, not ``particula.gpu.kernels``.

    Args:
        resolved: Structurally validated, normalized mechanism configuration.

    Raises:
        ValueError: If a deferred mechanism is requested.
    """
    if resolved.mask in (
        BROWNIAN_MECHANISM_FLAG,
        CHARGED_HARD_SPHERE_MECHANISM_FLAG,
        SEDIMENTATION_SP2016_MECHANISM_FLAG,
        BROWNIAN_MECHANISM_FLAG | CHARGED_HARD_SPHERE_MECHANISM_FLAG,
    ):
        return
    reserved_messages = {
        TURBULENT_SHEAR_ST1956_MECHANISM: (
            "Coagulation mechanism 'turbulent_shear_st1956' is reserved for "
            "E5-F5."
        ),
    }
    for mechanism in resolved.mechanisms:
        if mechanism in reserved_messages:
            raise ValueError(reserved_messages[mechanism])
    raise ValueError("Unsupported coagulation mechanism configuration.")


@no_type_check
@wp.func
def _bound_scheduled_trials(expected_trials: Any) -> Any:
    """Clamp scheduled-trial counts before any int32 conversion."""
    if not (expected_trials > wp.float64(0.0)):
        return wp.float64(0.0)

    bounded_trials = expected_trials
    if bounded_trials > wp.float64(MAX_SCHEDULED_TRIALS_PER_BOX):
        bounded_trials = wp.float64(MAX_SCHEDULED_TRIALS_PER_BOX)
    return bounded_trials


@no_type_check
@wp.func
def _sanitize_positive_finite(value: Any) -> Any:
    """Return a finite, strictly positive term or zero for safe accumulation.

    Invalid, zero, and negative terms make no contribution to an additive
    pair-rate or majorant total.
    """
    if wp.isfinite(value) and value > wp.float64(0.0):
        return value
    return wp.float64(0.0)


@no_type_check
@wp.func
def _total_pair_rate(  # noqa: PLR0913
    mechanism_mask: Any,
    radius_i: Any,
    radius_j: Any,
    diffusivity_i: Any,
    diffusivity_j: Any,
    g_term_i: Any,
    g_term_j: Any,
    speed_i: Any,
    speed_j: Any,
    settling_velocity_i: Any,
    settling_velocity_j: Any,
    total_mass_i: Any,
    total_mass_j: Any,
    charge_i: Any,
    charge_j: Any,
    temperature: Any,
    pressure: Any,
    boltzmann_constant: Any,
    elementary_charge_value: Any,
    electric_permittivity: Any,
    gas_constant: Any,
    molecular_weight_air: Any,
    ref_viscosity: Any,
    ref_temperature: Any,
    sutherland_constant: Any,
) -> Any:
    """Return the enabled finite, positive pair rate.

    This helper dispatches fixed flag bits without dynamic mechanism iteration.
    Each enabled term is independently sanitized before its additive combined
    rate is returned. This helper's bit dispatch does not grant sampler
    support: the sampler executes sedimentation only for its exact public
    sedimentation-only mask. Invalid rate terms contribute zero.
    """
    total_rate = wp.float64(0.0)
    if mechanism_mask & wp.int32(BROWNIAN_MECHANISM_FLAG):
        total_rate += _sanitize_positive_finite(
            brownian_kernel_pair_wp(
                radius_i,
                radius_j,
                diffusivity_i,
                diffusivity_j,
                g_term_i,
                g_term_j,
                speed_i,
                speed_j,
                wp.float64(1.0),
            )
        )
    if mechanism_mask & wp.int32(CHARGED_HARD_SPHERE_MECHANISM_FLAG):
        total_rate += _sanitize_positive_finite(
            charged_hard_sphere_wp(
                radius_i,
                radius_j,
                total_mass_i,
                total_mass_j,
                charge_i,
                charge_j,
                temperature,
                pressure,
                boltzmann_constant,
                elementary_charge_value,
                electric_permittivity,
                gas_constant,
                molecular_weight_air,
                ref_viscosity,
                ref_temperature,
                sutherland_constant,
            )
        )
    if mechanism_mask & wp.int32(SEDIMENTATION_SP2016_MECHANISM_FLAG):
        total_rate += _sanitize_positive_finite(
            sedimentation_sp2016_pair_rate_wp(
                radius_i,
                radius_j,
                settling_velocity_i,
                settling_velocity_j,
            )
        )
    return total_rate


@no_type_check
@wp.func
def _charged_majorant_from_active_pairs(  # noqa: PLR0913
    active_indices: Any,
    box_idx: Any,
    active_count: Any,
    radii: Any,
    total_masses: Any,
    charges: Any,
    temperature: Any,
    pressure: Any,
    boltzmann_constant: Any,
    elementary_charge_value: Any,
    electric_permittivity: Any,
    gas_constant: Any,
    molecular_weight_air: Any,
    ref_viscosity: Any,
    ref_temperature: Any,
    sutherland_constant: Any,
) -> Any:
    """Return the finite charged-rate maximum over unique compact active pairs.

    This read-only helper resolves compact ranks through ``active_indices`` and
    sanitizes every charged hard-sphere candidate. Charge and mass dependence
    prevents a proved extrema-only bound, so the intentional O(n²) scan favors
    a correctness-first majorant. Fewer than two active ranks and invalid,
    nonpositive candidates contribute zero.

    Args:
        active_indices: Compact particle indices ``(n_boxes, n_particles)``.
        box_idx: Index of the simulation box to scan.
        active_count: Number of compact active ranks in ``box_idx``.
        radii: Particle radii ``(n_boxes, n_particles)`` [m].
        total_masses: Total particle masses ``(n_boxes, n_particles)`` [kg].
        charges: Particle charges ``(n_boxes, n_particles)`` in elementary
            charge counts.
        temperature: Per-box gas temperatures ``(n_boxes,)`` [K].
        pressure: Per-box gas pressures ``(n_boxes,)`` [Pa].
        boltzmann_constant: Boltzmann constant [J/K].
        elementary_charge_value: Elementary charge [C].
        electric_permittivity: Vacuum permittivity [F/m].
        gas_constant: Universal gas constant [J/(mol·K)].
        molecular_weight_air: Molecular weight of air [kg/mol].
        ref_viscosity: Reference gas viscosity [Pa·s].
        ref_temperature: Reference viscosity temperature [K].
        sutherland_constant: Sutherland viscosity constant [K].

    Returns:
        Finite, nonnegative maximum charged hard-sphere rate for the selected
        box, or zero when no positive finite selected-pair rate exists.
    """
    majorant = wp.float64(0.0)
    if active_count < wp.int32(2):
        return majorant

    # Charge and mass dependence prevents a proved extrema-only bound, so this
    # compact-active O(n²) scan intentionally prioritizes correctness.
    for first_rank in range(active_count - wp.int32(1)):
        first_idx = wp.int32(active_indices[box_idx, first_rank])
        for second_rank in range(first_rank + wp.int32(1), active_count):
            second_idx = wp.int32(active_indices[box_idx, second_rank])
            candidate = _sanitize_positive_finite(
                charged_hard_sphere_wp(
                    radii[box_idx, first_idx],
                    radii[box_idx, second_idx],
                    total_masses[box_idx, first_idx],
                    total_masses[box_idx, second_idx],
                    charges[box_idx, first_idx],
                    charges[box_idx, second_idx],
                    temperature[box_idx],
                    pressure[box_idx],
                    boltzmann_constant,
                    elementary_charge_value,
                    electric_permittivity,
                    gas_constant,
                    molecular_weight_air,
                    ref_viscosity,
                    ref_temperature,
                    sutherland_constant,
                )
            )
            if candidate > majorant:
                majorant = candidate
    return majorant


@no_type_check
@wp.func
def _sedimentation_majorant_from_active_pairs(
    active_indices: Any,
    box_idx: Any,
    active_count: Any,
    radii: Any,
    settling_velocities: Any,
) -> Any:
    """Return the finite sedimentation-rate maximum over active pairs.

    This helper exhaustively resolves compact active ranks, so sparse source
    slots cannot affect the exact SP2016 sedimentation sampler majorant.
    Invalid, zero, and nonpositive P1 rates contribute zero.

    Args:
        active_indices: Compact particle indices ``(n_boxes, n_particles)``.
        box_idx: Index of the simulation box to scan.
        active_count: Number of compact active ranks in ``box_idx``.
        radii: Particle radii ``(n_boxes, n_particles)`` [m].
        settling_velocities: Private particle settling velocities
            ``(n_boxes, n_particles)`` [m/s].

    Returns:
        Finite, nonnegative maximum sedimentation rate for the selected box,
        or zero when no positive finite selected-pair rate exists.
    """
    majorant = wp.float64(0.0)
    if active_count < wp.int32(2):
        return majorant

    for first_rank in range(active_count - wp.int32(1)):
        first_idx = wp.int32(active_indices[box_idx, first_rank])
        for second_rank in range(first_rank + wp.int32(1), active_count):
            second_idx = wp.int32(active_indices[box_idx, second_rank])
            candidate = _sanitize_positive_finite(
                sedimentation_sp2016_pair_rate_wp(
                    radii[box_idx, first_idx],
                    radii[box_idx, second_idx],
                    settling_velocities[box_idx, first_idx],
                    settling_velocities[box_idx, second_idx],
                )
            )
            if candidate > majorant:
                majorant = candidate
    return majorant


@no_type_check
@wp.func
def _total_majorant(  # noqa: PLR0913
    mechanism_mask: Any,
    radius_min: Any,
    radius_max: Any,
    diffusivity_min: Any,
    diffusivity_max: Any,
    g_term_min: Any,
    g_term_max: Any,
    speed_min: Any,
    speed_max: Any,
    active_indices: Any,
    box_idx: Any,
    active_count: Any,
    radii: Any,
    settling_velocities: Any,
    total_masses: Any,
    charges: Any,
    temperature: Any,
    pressure: Any,
    boltzmann_constant: Any,
    elementary_charge_value: Any,
    electric_permittivity: Any,
    gas_constant: Any,
    molecular_weight_air: Any,
    ref_viscosity: Any,
    ref_temperature: Any,
    sutherland_constant: Any,
) -> Any:
    """Accumulate enabled finite, positive terms for private staged tests.

    This helper dispatches fixed flag bits for internal tests and for exact
    charged-only or exact SP2016 sedimentation-only sampler execution. The
    sampler separately enforces its executable-mask boundary; a helper
    contribution does not make a mixed sedimentation mask executable.
    Brownian-containing
    production selection uses one exact active-pair scan through
    ``_total_pair_rate``. Other reserved bits contribute no term.
    """
    total_majorant = wp.float64(0.0)
    if mechanism_mask & wp.int32(BROWNIAN_MECHANISM_FLAG):
        total_majorant += _sanitize_positive_finite(
            brownian_kernel_pair_wp(
                radius_min,
                radius_max,
                diffusivity_min,
                diffusivity_max,
                g_term_min,
                g_term_max,
                speed_min,
                speed_max,
                wp.float64(1.0),
            )
        )
    if mechanism_mask & wp.int32(CHARGED_HARD_SPHERE_MECHANISM_FLAG):
        total_majorant += _sanitize_positive_finite(
            _charged_majorant_from_active_pairs(
                active_indices,
                box_idx,
                active_count,
                radii,
                total_masses,
                charges,
                temperature,
                pressure,
                boltzmann_constant,
                elementary_charge_value,
                electric_permittivity,
                gas_constant,
                molecular_weight_air,
                ref_viscosity,
                ref_temperature,
                sutherland_constant,
            )
        )
    if mechanism_mask & wp.int32(SEDIMENTATION_SP2016_MECHANISM_FLAG):
        total_majorant += _sanitize_positive_finite(
            _sedimentation_majorant_from_active_pairs(
                active_indices,
                box_idx,
                active_count,
                radii,
                settling_velocities,
            )
        )
    # Other reserved mechanism bits deliberately contribute no term.
    return total_majorant


@no_type_check
@wp.func
def _select_active_pair_by_rank(
    active_indices: Any,
    box_idx: Any,
    rank_i: Any,
    adjusted_rank_j: Any,
) -> Any:
    """Resolve two active-set ranks from a compact active-index buffer."""
    return wp.vec2i(
        wp.int32(active_indices[box_idx, rank_i]),
        wp.int32(active_indices[box_idx, adjusted_rank_j]),
    )


@no_type_check
@wp.func
def _remove_active_pair_by_rank_swap_pop(
    active_indices: Any,
    box_idx: Any,
    active_count: Any,
    rank_i: Any,
    adjusted_rank_j: Any,
) -> Any:
    """Remove two active ranks from a compact active-index buffer."""
    larger_rank = adjusted_rank_j
    smaller_rank = rank_i
    if rank_i > adjusted_rank_j:
        larger_rank = rank_i
        smaller_rank = adjusted_rank_j

    last_rank = active_count - wp.int32(1)
    active_indices[box_idx, larger_rank] = active_indices[box_idx, last_rank]
    active_indices[box_idx, last_rank] = wp.int32(-1)
    active_count -= wp.int32(1)

    last_rank = active_count - wp.int32(1)
    active_indices[box_idx, smaller_rank] = active_indices[box_idx, last_rank]
    active_indices[box_idx, last_rank] = wp.int32(-1)
    active_count -= wp.int32(1)
    return active_count


@no_type_check
@wp.func
def _resolve_active_pair_by_rank(
    active_indices: Any,
    box_idx: Any,
    n_particles: Any,
    rank_i: Any,
    adjusted_rank_j: Any,
) -> Any:
    """Resolve two active-set ranks to particle indices in one scan."""
    selected_i = wp.int32(-1)
    selected_j = wp.int32(-1)
    active_rank = wp.int32(0)

    for p_idx in range(n_particles):
        if active_indices[box_idx, p_idx] == wp.int32(0):
            continue

        if active_rank == rank_i:
            selected_i = wp.int32(p_idx)
        if active_rank == adjusted_rank_j:
            selected_j = wp.int32(p_idx)

        active_rank += wp.int32(1)
        if selected_i >= wp.int32(0) and selected_j >= wp.int32(0):
            break

    return wp.vec2i(selected_i, selected_j)


@no_type_check
@wp.kernel
# type: ignore[misc]
def _initialize_rng_states(seed: Any, rng_states: Any) -> None:
    """Initialize or reset per-box RNG states from a seed.

    Args:
        seed: RNG seed value.
        rng_states: RNG state buffer ``(n_boxes,)`` written in place.
    """  # type: ignore
    box_idx = wp.tid()  # type: ignore[misc]
    rng_states[box_idx] = wp.rand_init(wp.int32(seed), wp.int32(box_idx))


@no_type_check
@wp.kernel
# type: ignore[misc]
def _validate_charge_finite_kernel(charge: Any, invalid: Any) -> None:
    """Record non-finite charge in a private status buffer without mutation.

    Args:
        charge: Caller-owned particle charge array ``(n_boxes, n_particles)``
            read without modification.
        invalid: Private one-element ``wp.int32`` status buffer set to one
            when any charge value is non-finite.
    """  # type: ignore
    box_idx, particle_idx = wp.tid()  # type: ignore[misc]
    if not wp.isfinite(charge[box_idx, particle_idx]):
        wp.atomic_max(invalid, 0, 1)


@no_type_check
@wp.kernel
# type: ignore[misc]
def _validate_charged_particle_physics_kernel(
    masses: Any,
    concentration: Any,
    density: Any,
    invalid: Any,
    active_counts: Any,
) -> None:
    """Validate charged particle inputs and count selector-eligible slots.

    Masses and concentrations may be zero for inactive slots, but all particle
    state must be finite and nonnegative; species density must be finite and
    strictly positive.
    """  # type: ignore
    box_idx, particle_idx = wp.tid()  # type: ignore[misc]
    concentration_value = concentration[box_idx, particle_idx]
    if not wp.isfinite(concentration_value) or concentration_value < wp.float64(
        0.0
    ):
        wp.atomic_max(invalid, 0, 1)
    total_volume = wp.float64(0.0)
    for species_idx in range(masses.shape[2]):
        mass = masses[box_idx, particle_idx, species_idx]
        if not wp.isfinite(mass) or mass < wp.float64(0.0):
            wp.atomic_max(invalid, 0, 1)
        total_volume += mass / density[species_idx]

    # Match the selector eligibility condition exactly so the charged-work cap
    # covers precisely the slots that can enter its compact active set.
    if concentration_value > wp.float64(0.0) and total_volume > wp.float64(0.0):
        wp.atomic_add(active_counts, box_idx, 1)


@no_type_check
@wp.kernel
# type: ignore[misc]
def _validate_charged_density_kernel(density: Any, invalid: Any) -> None:
    """Record invalid charged species densities without mutation."""  # type: ignore
    species_idx = wp.tid()  # type: ignore[misc]
    density_value = density[species_idx]
    if not wp.isfinite(density_value) or density_value <= wp.float64(0.0):
        wp.atomic_max(invalid, 0, 1)


@no_type_check
@wp.kernel
# type: ignore[misc]
def _validate_sedimentation_particle_physics_kernel(  # noqa: C901
    masses: Any,
    concentration: Any,
    density: Any,
    charge: Any,
    active_limit: Any,
    invalid: Any,
) -> None:
    """Validate all SP2016 inputs with one box-local read-only reduction."""  # type: ignore
    box_idx = wp.tid()  # type: ignore[misc]
    for species_idx in range(density.shape[0]):
        density_value = density[species_idx]
        if not wp.isfinite(density_value) or density_value <= wp.float64(0.0):
            wp.atomic_max(invalid, 3, 1)

    reference_concentration = wp.float64(0.0)
    active_count = wp.int32(0)
    for particle_idx in range(masses.shape[1]):
        concentration_value = concentration[box_idx, particle_idx]
        if not wp.isfinite(charge[box_idx, particle_idx]):
            wp.atomic_max(invalid, 0, 1)
        if not wp.isfinite(concentration_value) or concentration_value < (
            wp.float64(0.0)
        ):
            wp.atomic_max(invalid, 2, 1)

        total_mass = wp.float64(0.0)
        total_volume = wp.float64(0.0)
        for species_idx in range(masses.shape[2]):
            mass = masses[box_idx, particle_idx, species_idx]
            if not wp.isfinite(mass) or mass < wp.float64(0.0):
                wp.atomic_max(invalid, 1, 1)
            total_mass += mass
            total_volume += mass / density[species_idx]

        if concentration_value > wp.float64(0.0):
            if (
                reference_concentration > wp.float64(0.0)
                and concentration_value != reference_concentration
            ):
                wp.atomic_max(invalid, 4, 1)
            reference_concentration = concentration_value
            if not (
                wp.isfinite(total_mass)
                and total_mass > wp.float64(0.0)
                and wp.isfinite(total_volume)
                and total_volume > wp.float64(0.0)
            ):
                wp.atomic_max(invalid, 6, 1)
            else:
                active_count += wp.int32(1)
    if active_count > active_limit:
        wp.atomic_max(invalid, 5, 1)


@no_type_check
@wp.kernel
# type: ignore[misc]
def brownian_coagulation_kernel(  # noqa: C901
    masses: Any,
    concentration: Any,
    density: Any,
    volume: Any,
    temperature: Any,
    pressure: Any,
    gas_constant: Any,
    boltzmann_constant: Any,
    molecular_weight_air: Any,
    ref_viscosity: Any,
    ref_temperature: Any,
    sutherland_constant: Any,
    elementary_charge_value: Any,
    electric_permittivity: Any,
    time_step: Any,
    radii: Any,
    diffusivities: Any,
    g_terms: Any,
    speeds: Any,
    settling_velocities: Any,
    total_masses: Any,
    charge: Any,
    active_indices: Any,
    collision_pairs: Any,
    n_collisions: Any,
    rng_states: Any,
    mechanism_mask: Any,
    collision_capacity: Any,
) -> None:
    """Select stochastic coagulation pairs for each box.

    Candidate pairs are drawn by rank within the current active set so each
    scheduled trial proposes two distinct active particles without retrying on
    inactive slots. The sampler computes one finite positive total majorant per
    box and one finite positive total pair rate per candidate. A valid candidate
    receives exactly one acceptance draw; invalid rates, invalid majorants, and
    rates above the majorant are skipped without collision-buffer or active-set
    mutation. The exact public SP2016 sedimentation-only mask joins the public
    Brownian and charged masks; unsupported masks return before scheduling or
    accessing the RNG state.

    Args:
        masses: Particle masses array ``(n_boxes, n_particles, n_species)``.
        concentration: Particle concentrations ``(n_boxes, n_particles)``.
        density: Species densities ``(n_species,)``.
        volume: Per-box volumes ``(n_boxes,)``.
        temperature: Per-box gas temperatures ``(n_boxes,)`` [K].
        pressure: Per-box gas pressures ``(n_boxes,)`` [Pa].
        gas_constant: Universal gas constant [J/(mol·K)].
        boltzmann_constant: Boltzmann constant [J/K].
        molecular_weight_air: Molecular weight of air [kg/mol].
        ref_viscosity: Reference viscosity at STP [Pa·s].
        ref_temperature: Reference temperature [K].
        sutherland_constant: Sutherland constant [K].
        elementary_charge_value: Elementary-charge magnitude [C].
        electric_permittivity: Vacuum electric permittivity [F/m].
        time_step: Coagulation time step [s].
        radii: Output particle radii ``(n_boxes, n_particles)``.
        diffusivities: Output diffusivities ``(n_boxes, n_particles)``.
        g_terms: Output collection terms ``(n_boxes, n_particles)``.
        speeds: Output mean thermal speeds ``(n_boxes, n_particles)``.
        settling_velocities: Private fp64 settling-velocity scratch
            ``(n_boxes, n_particles)``. The kernel clears every slot on entry
            and populates this call-local scratch only for valid active slots
            of the exact SP2016 sedimentation-only mask.
        active_indices: Output compact active indices
            ``(n_boxes, n_particles)``.
        collision_pairs: Output collision indices
            ``(n_boxes, max_collisions, 2)``.
        n_collisions: Output collision counts ``(n_boxes,)``.
        rng_states: Per-box RNG states ``(n_boxes,)`` mutated in place during
            pair selection. Reusing this buffer across calls preserves
            caller-owned persistent state unless it is reset before launch.
        total_masses: Private ``wp.float64`` total-mass scratch
            ``(n_boxes, n_particles)``. The kernel clears each slot and stores
            the sum over species only for active particles.
        charge: Caller-owned signed elementary-charge counts
            ``(n_boxes, n_particles)``.
        mechanism_mask: Fixed internal sampler mask. In addition to public
            Brownian and charged masks, only the exact SP2016 sedimentation-
            only mask is executable.
        collision_capacity: Maximum accepted collisions per box for this call.
    """  # type: ignore
    box_idx = wp.tid()  # type: ignore[misc]
    n_particles = masses.shape[1]

    executable_brownian_mask = mechanism_mask == wp.int32(
        BROWNIAN_MECHANISM_FLAG
    ) or mechanism_mask == wp.int32(
        BROWNIAN_MECHANISM_FLAG | CHARGED_HARD_SPHERE_MECHANISM_FLAG
    )
    executable_sedimentation_mask = mechanism_mask == wp.int32(
        SEDIMENTATION_SP2016_MECHANISM_FLAG
    )
    if not (
        executable_brownian_mask
        or mechanism_mask == wp.int32(CHARGED_HARD_SPHERE_MECHANISM_FLAG)
        or executable_sedimentation_mask
    ):
        return

    # One thread per box keeps the pair selection sequential and avoids
    # cross-thread races when writing collision pairs.
    n_species = masses.shape[2]

    temperature_value = temperature[box_idx]
    pressure_value = pressure[box_idx]

    dynamic_viscosity = wp.float64(0.0)
    mean_free_path = wp.float64(0.0)
    if executable_brownian_mask or executable_sedimentation_mask:
        dynamic_viscosity = dynamic_viscosity_wp(
            temperature_value,
            ref_viscosity,
            ref_temperature,
            sutherland_constant,
        )
        mean_free_path = molecule_mean_free_path_wp(
            molecular_weight_air,
            temperature_value,
            pressure_value,
            dynamic_viscosity,
            gas_constant,
        )

    active_count = wp.int32(0)
    for particle_idx in range(n_particles):
        total_masses[box_idx, particle_idx] = wp.float64(0.0)
        if concentration[box_idx, particle_idx] <= wp.float64(0.0):
            active_indices[box_idx, particle_idx] = wp.int32(-1)
            radii[box_idx, particle_idx] = wp.float64(0.0)
            if executable_brownian_mask:
                diffusivities[box_idx, particle_idx] = wp.float64(0.0)
                g_terms[box_idx, particle_idx] = wp.float64(0.0)
                speeds[box_idx, particle_idx] = wp.float64(0.0)
            continue

        total_mass = wp.float64(0.0)
        total_volume = wp.float64(0.0)
        for species_idx in range(n_species):
            species_mass = masses[box_idx, particle_idx, species_idx]
            total_mass += species_mass
            total_volume += species_mass / density[species_idx]

        if total_volume <= wp.float64(0.0) or total_mass <= wp.float64(0.0):
            active_indices[box_idx, particle_idx] = wp.int32(-1)
            radii[box_idx, particle_idx] = wp.float64(0.0)
            if executable_brownian_mask:
                diffusivities[box_idx, particle_idx] = wp.float64(0.0)
                g_terms[box_idx, particle_idx] = wp.float64(0.0)
                speeds[box_idx, particle_idx] = wp.float64(0.0)
            continue

        radius = particle_radius_from_volume_wp(total_volume)
        diffusivity = wp.float64(0.0)
        speed = wp.float64(0.0)
        g_term = wp.float64(0.0)
        if executable_brownian_mask:
            knudsen = knudsen_number_wp(mean_free_path, radius)
            slip = cunningham_slip_correction_wp(knudsen)
            mobility = aerodynamic_mobility_wp(radius, slip, dynamic_viscosity)
            diffusivity = brownian_diffusivity_wp(
                temperature_value, mobility, boltzmann_constant
            )
            speed = mean_thermal_speed_wp(
                total_mass, temperature_value, boltzmann_constant
            )
            particle_mean_free_path = particle_mean_free_path_wp(
                diffusivity, speed
            )
            g_term = g_collection_term_wp(particle_mean_free_path, radius)
        settling_velocity = wp.float64(0.0)
        if executable_sedimentation_mask:
            settling_velocity = settling_velocity_stokes_from_transport_wp(
                radius,
                effective_density_wp(total_mass, total_volume),
                dynamic_viscosity,
                mean_free_path,
            )

        active_indices[box_idx, active_count] = wp.int32(particle_idx)
        radii[box_idx, particle_idx] = radius
        total_masses[box_idx, particle_idx] = total_mass
        if executable_brownian_mask:
            diffusivities[box_idx, particle_idx] = diffusivity
            g_terms[box_idx, particle_idx] = g_term
            speeds[box_idx, particle_idx] = speed
        if executable_sedimentation_mask:
            settling_velocities[box_idx, particle_idx] = settling_velocity
        active_count += wp.int32(1)

    if active_count < wp.int32(2):
        n_collisions[box_idx] = wp.int32(0)
        return

    # Charged-only and exact SP2016 sedimentation-only rates require exact
    # compact active-pair maxima. Brownian masks use the shared scan below.
    majorant_total = wp.float64(0.0)
    if mechanism_mask == wp.int32(
        CHARGED_HARD_SPHERE_MECHANISM_FLAG
    ) or mechanism_mask == wp.int32(SEDIMENTATION_SP2016_MECHANISM_FLAG):
        majorant_total = _total_majorant(
            mechanism_mask,
            wp.float64(0.0),
            wp.float64(0.0),
            wp.float64(0.0),
            wp.float64(0.0),
            wp.float64(0.0),
            wp.float64(0.0),
            wp.float64(0.0),
            wp.float64(0.0),
            active_indices,
            box_idx,
            active_count,
            radii,
            settling_velocities,
            total_masses,
            charge,
            temperature,
            pressure,
            boltzmann_constant,
            elementary_charge_value,
            electric_permittivity,
            gas_constant,
            molecular_weight_air,
            ref_viscosity,
            ref_temperature,
            sutherland_constant,
        )
    # Brownian-containing masks use one compact active-pair scan. For the
    # combined mask _total_pair_rate includes the charged term, so do not add a
    # separate charged-only maximum that would double-count the majorant.
    if executable_brownian_mask:
        majorant_total = wp.float64(0.0)
        for first_rank in range(active_count - wp.int32(1)):
            first_idx = wp.int32(active_indices[box_idx, first_rank])
            for second_rank in range(first_rank + wp.int32(1), active_count):
                second_idx = wp.int32(active_indices[box_idx, second_rank])
                pair_rate = _total_pair_rate(
                    mechanism_mask,
                    radii[box_idx, first_idx],
                    radii[box_idx, second_idx],
                    diffusivities[box_idx, first_idx],
                    diffusivities[box_idx, second_idx],
                    g_terms[box_idx, first_idx],
                    g_terms[box_idx, second_idx],
                    speeds[box_idx, first_idx],
                    speeds[box_idx, second_idx],
                    settling_velocities[box_idx, first_idx],
                    settling_velocities[box_idx, second_idx],
                    total_masses[box_idx, first_idx],
                    total_masses[box_idx, second_idx],
                    charge[box_idx, first_idx],
                    charge[box_idx, second_idx],
                    temperature_value,
                    pressure_value,
                    boltzmann_constant,
                    elementary_charge_value,
                    electric_permittivity,
                    gas_constant,
                    molecular_weight_air,
                    ref_viscosity,
                    ref_temperature,
                    sutherland_constant,
                )
                if pair_rate > majorant_total:
                    majorant_total = pair_rate

    if not (wp.isfinite(majorant_total) and majorant_total > wp.float64(0.0)):
        n_collisions[box_idx] = wp.int32(0)
        return

    possible_pairs = (
        wp.float64(active_count)
        * wp.float64(active_count - 1)
        / (wp.float64(2.0))
    )
    scheduling_concentration = wp.float64(1.0)
    if executable_sedimentation_mask:
        scheduling_concentration = concentration[
            box_idx, active_indices[box_idx, 0]
        ]
    expected_trials = (
        majorant_total
        * possible_pairs
        * scheduling_concentration
        * time_step
        / volume[box_idx]
    )
    # A finite-positive calculation can overflow to positive infinity. Let the
    # bounded scheduler cap that case rather than silently dropping all work.
    # This comparison also rejects NaN, zero, and negative schedules.
    if not (expected_trials > wp.float64(0.0)):
        n_collisions[box_idx] = wp.int32(0)
        return

    collision_count = wp.int32(0)
    state = rng_states[box_idx]

    bounded_trials = _bound_scheduled_trials(expected_trials)
    tests = wp.int32(bounded_trials)
    remainder = bounded_trials - wp.float64(tests)
    if wp.randf(state) < remainder:
        tests += wp.int32(1)
    rng_states[box_idx] = state

    if tests <= wp.int32(0):
        n_collisions[box_idx] = wp.int32(0)
        return

    for _ in range(tests):
        if collision_count >= collision_capacity or active_count < wp.int32(2):
            break

        rank_i = wp.randi(state, wp.int32(0), active_count)
        rank_j = wp.randi(state, wp.int32(0), active_count - wp.int32(1))
        adjusted_rank_j = rank_j
        if adjusted_rank_j >= rank_i:
            # Skip the first chosen active rank so the second proposal maps to
            # one of the remaining active particles.
            adjusted_rank_j += wp.int32(1)

        selected_pair = _select_active_pair_by_rank(
            active_indices,
            box_idx,
            rank_i,
            adjusted_rank_j,
        )
        selected_i = selected_pair[0]
        selected_j = selected_pair[1]

        if selected_i < wp.int32(0) or selected_j < wp.int32(0):
            # Guard an unexpected active-rank mismatch without silently
            # truncating the remaining scheduled pass.
            continue

        if selected_j < selected_i:
            temp_idx = selected_i
            selected_i = selected_j
            selected_j = temp_idx

        total_rate = _total_pair_rate(
            mechanism_mask,
            radii[box_idx, selected_i],
            radii[box_idx, selected_j],
            diffusivities[box_idx, selected_i],
            diffusivities[box_idx, selected_j],
            g_terms[box_idx, selected_i],
            g_terms[box_idx, selected_j],
            speeds[box_idx, selected_i],
            speeds[box_idx, selected_j],
            settling_velocities[box_idx, selected_i],
            settling_velocities[box_idx, selected_j],
            total_masses[box_idx, selected_i],
            total_masses[box_idx, selected_j],
            charge[box_idx, selected_i],
            charge[box_idx, selected_j],
            temperature_value,
            pressure_value,
            boltzmann_constant,
            elementary_charge_value,
            electric_permittivity,
            gas_constant,
            molecular_weight_air,
            ref_viscosity,
            ref_temperature,
            sutherland_constant,
        )
        if not (wp.isfinite(total_rate) and total_rate > wp.float64(0.0)):
            continue
        # Every valid candidate gets exactly one acceptance draw.
        if wp.randf(state) < total_rate / majorant_total:
            merged_charge = (
                charge[box_idx, selected_i] + charge[box_idx, selected_j]
            )
            merge_is_representable = wp.isfinite(merged_charge)
            for species_idx in range(n_species):
                merged_mass = (
                    masses[box_idx, selected_i, species_idx]
                    + masses[box_idx, selected_j, species_idx]
                )
                if not (
                    wp.isfinite(merged_mass) and merged_mass >= wp.float64(0.0)
                ):
                    merge_is_representable = False
            if not merge_is_representable:
                # Preserve both particles for later representable proposals.
                # The apply kernel retains this check as defense in depth.
                continue
            collision_pairs[box_idx, collision_count, 0] = selected_i
            collision_pairs[box_idx, collision_count, 1] = selected_j
            collision_count += wp.int32(1)
            active_count = _remove_active_pair_by_rank_swap_pop(
                active_indices,
                box_idx,
                active_count,
                rank_i,
                adjusted_rank_j,
            )

    n_collisions[box_idx] = collision_count
    rng_states[box_idx] = state


@no_type_check
@wp.kernel
# type: ignore[misc]
def apply_coagulation_kernel(
    masses: Any,
    concentration: Any,
    charge: Any,
    collision_pairs: Any,
    n_collisions: Any,
) -> None:
    """Apply coagulation collisions by merging particle state in place.

    Args:
        masses: Particle masses array ``(n_boxes, n_particles, n_species)``.
        concentration: Particle concentrations ``(n_boxes, n_particles)``.
        charge: Particle charges with ``wp.float64`` dtype and shape
            ``(n_boxes, n_particles)``. Donor charge is added to its recipient
            and then cleared.
        collision_pairs: Collision index buffer
            ``(n_boxes, max_collisions, 2)``.
        n_collisions: Collision counts ``(n_boxes,)``.
    """  # type: ignore
    box_idx, collision_idx = wp.tid()  # type: ignore[misc]
    if collision_idx >= n_collisions[box_idx]:
        return

    idx_i = collision_pairs[box_idx, collision_idx, 0]
    idx_j = collision_pairs[box_idx, collision_idx, 1]
    if idx_i == idx_j:
        collision_pairs[box_idx, collision_idx, 0] = wp.int32(-1)
        collision_pairs[box_idx, collision_idx, 1] = wp.int32(-1)
        return

    merged_charge = charge[box_idx, idx_i] + charge[box_idx, idx_j]
    # An unrepresentable charge merge must not partially apply mass state or
    # turn finite caller state into infinity. Skipping preserves inventories.
    if not wp.isfinite(merged_charge):
        collision_pairs[box_idx, collision_idx, 0] = wp.int32(-1)
        collision_pairs[box_idx, collision_idx, 1] = wp.int32(-1)
        return

    n_species = masses.shape[2]
    for species_idx in range(n_species):
        merged_mass = (
            masses[box_idx, idx_i, species_idx]
            + masses[box_idx, idx_j, species_idx]
        )
        # Validate every component before changing any state. A skipped pair
        # preserves caller-owned inventories and avoids partial merges.
        if not wp.isfinite(merged_mass) or merged_mass < wp.float64(0.0):
            collision_pairs[box_idx, collision_idx, 0] = wp.int32(-1)
            collision_pairs[box_idx, collision_idx, 1] = wp.int32(-1)
            return

    for species_idx in range(n_species):
        masses[box_idx, idx_i, species_idx] = (
            masses[box_idx, idx_i, species_idx]
            + masses[box_idx, idx_j, species_idx]
        )
        masses[box_idx, idx_j, species_idx] = wp.float64(0.0)

    charge[box_idx, idx_i] = merged_charge
    charge[box_idx, idx_j] = wp.float64(0.0)
    concentration[box_idx, idx_j] = wp.float64(0.0)


@no_type_check
@wp.kernel
# type: ignore[misc]
def _compact_applied_collision_pairs_kernel(
    collision_pairs: Any,
    n_collisions: Any,
) -> None:
    """Remove rejected collision entries after the apply pass."""  # type: ignore
    box_idx = wp.tid()  # type: ignore[misc]
    original_count = n_collisions[box_idx]
    applied_count = wp.int32(0)
    for collision_idx in range(collision_pairs.shape[1]):
        if collision_idx >= original_count:
            break
        idx_i = collision_pairs[box_idx, collision_idx, 0]
        idx_j = collision_pairs[box_idx, collision_idx, 1]
        if idx_i >= wp.int32(0) and idx_j >= wp.int32(0):
            collision_pairs[box_idx, applied_count, 0] = idx_i
            collision_pairs[box_idx, applied_count, 1] = idx_j
            applied_count += wp.int32(1)
    n_collisions[box_idx] = applied_count


def _validate_particle_arrays(
    particles: Any,
    n_boxes: int,
    n_particles: int,
    n_species: int,
) -> None:
    """Validate particle array shapes, including the charge schema.

    Args:
        particles: GPU particle data container.
        n_boxes: Expected number of boxes.
        n_particles: Expected number of particles per box.
        n_species: Expected number of species.

    Raises:
        ValueError: If particle arrays do not match expected shapes, or charge
            is not ``wp.float64``.
    """
    if particles.masses.shape != (n_boxes, n_particles, n_species):
        raise ValueError("particle masses shape does not match expected")
    if particles.density.shape[0] != n_species:
        raise ValueError("particle density length does not match n_species")
    if particles.concentration.shape != (n_boxes, n_particles):
        raise ValueError(
            "particle concentration shape does not match (n_boxes, n_particles)"
        )
    if particles.charge.shape != (n_boxes, n_particles):
        raise ValueError(
            "particle charge shape does not match (n_boxes, n_particles)"
        )
    if particles.charge.dtype != wp.float64:
        raise ValueError("particle charge must use dtype float64")
    if particles.volume.shape != (n_boxes,):
        raise ValueError("particle volume shape does not match (n_boxes,)")


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


def _validate_device_arrays(particles: Any, device: Any) -> None:
    """Validate particle arrays, including charge, share one Warp device.

    Args:
        particles: GPU particle data container.
        device: Expected Warp device.

    Raises:
        ValueError: If any particle array is on a different device.
    """
    _validate_device_match("particle masses", particles.masses, device)
    _validate_device_match(
        "particle concentration", particles.concentration, device
    )
    _validate_device_match("particle charge", particles.charge, device)
    _validate_device_match("particle density", particles.density, device)
    _validate_device_match("particle volume", particles.volume, device)


def _validate_charge_finite(
    charge: Any,
    n_boxes: int,
    n_particles: int,
    device: Any,
) -> None:
    """Reject non-finite caller-owned charge through a read-only device scan.

    This helper runs only after the public entry point validates the charge
    shape, dtype, and device. It allocates a private status scalar, launches
    one ``O(n_boxes * n_particles)`` read-only scan, and reads back only that
    scalar. It neither copies nor mutates the caller-owned charge array.

    Args:
        charge: Caller-owned ``wp.float64`` charge array with shape
            ``(n_boxes, n_particles)``.
        n_boxes: Number of simulation boxes.
        n_particles: Number of particles per box.
        device: Warp device that owns ``charge`` and the private status buffer.

    Raises:
        ValueError: If any charge value is NaN or infinite.
    """
    invalid = wp.zeros((1,), dtype=wp.int32, device=device)
    wp.launch(
        _validate_charge_finite_kernel,
        dim=(n_boxes, n_particles),
        inputs=[charge, invalid],
        device=device,
    )
    wp.synchronize_device(device)
    if invalid.numpy()[0] != 0:
        raise ValueError("particle charge must contain only finite values")


def _validate_charged_particle_physics(
    particles: Any,
    n_boxes: int,
    n_particles: int,
    device: Any,
) -> None:
    """Reject invalid charged particle inputs and excessive active work.

    The exact charged majorant enumerates all unique active pairs in a single
    selector thread. This read-only preflight therefore bounds each box to
    ``MAX_CHARGED_ACTIVE_PARTICLES_PER_BOX`` active particles before any output
    allocation, RNG initialization, selector launch, or caller-state mutation.

    Raises:
        ValueError: If mass or concentration is non-finite/negative, density is
            non-finite/nonpositive, or a box exceeds the charged active limit.
    """
    invalid = wp.zeros((1,), dtype=wp.int32, device=device)
    wp.launch(
        _validate_charged_density_kernel,
        dim=(particles.density.shape[0],),
        inputs=[particles.density, invalid],
        device=device,
    )
    wp.synchronize_device(device)
    if invalid.numpy()[0] != 0:
        raise ValueError(
            "charged particle masses and concentrations must be finite and "
            "nonnegative, and density must be finite and > 0"
        )

    active_counts = wp.zeros((n_boxes,), dtype=wp.int32, device=device)
    wp.launch(
        _validate_charged_particle_physics_kernel,
        dim=(n_boxes, n_particles),
        inputs=[
            particles.masses,
            particles.concentration,
            particles.density,
            invalid,
            active_counts,
        ],
        device=device,
    )
    wp.synchronize_device(device)
    if invalid.numpy()[0] != 0:
        raise ValueError(
            "charged particle masses and concentrations must be finite and "
            "nonnegative, and density must be finite and > 0"
        )
    if np.any(active_counts.numpy() > MAX_CHARGED_ACTIVE_PARTICLES_PER_BOX):
        raise ValueError(
            "charged active particle count exceeds "
            "MAX_CHARGED_ACTIVE_PARTICLES_PER_BOX"
        )


def _validate_sedimentation_particle_physics(
    particles: Any,
    n_boxes: int,
    n_particles: int,
    device: Any,
) -> None:
    """Reject invalid or excessive SP2016 inputs through read-only scans.

    The fixed-size error state permits one device-to-host readback while
    retaining field-specific errors. A single box-local reduction validates
    charge, particle state, derived totals, and bounded exact-pair work.
    Positive concentrations must agree within a box because this
    particle-resolved merge representation stores one concentration per slot.
    Zero mass and concentration are valid, allowing inactive slots without
    applying charged-only constraints. This preflight runs before time-step,
    environment, output-buffer, or RNG work.

    Args:
        particles: Caller-owned Warp particle data with validated shape and
            device placement.
        n_boxes: Number of simulation boxes.
        n_particles: Number of particle slots per box.
        device: Warp device that owns the particle arrays and status buffer.

    Raises:
        ValueError: If mass or concentration is non-finite or negative, density
            is non-finite or nonpositive, concentrations are incompatible, or a
            box exceeds the sedimentation active limit.
    """
    invalid = wp.zeros((7,), dtype=wp.int32, device=device)
    wp.launch(
        _validate_sedimentation_particle_physics_kernel,
        dim=(n_boxes,),
        inputs=[
            particles.masses,
            particles.concentration,
            particles.density,
            particles.charge,
            wp.int32(MAX_SEDIMENTATION_ACTIVE_PARTICLES_PER_BOX),
            invalid,
        ],
        device=device,
    )
    wp.synchronize_device(device)
    invalid_values = invalid.numpy()
    if invalid_values[0] != 0:
        raise ValueError("particle charge must contain only finite values")
    if invalid_values[1] != 0:
        raise ValueError(
            "sedimentation particle masses must be finite and nonnegative"
        )
    if invalid_values[2] != 0:
        raise ValueError(
            "sedimentation particle concentration must be finite and "
            "nonnegative"
        )
    if invalid_values[3] != 0:
        raise ValueError(
            "sedimentation particle density must be finite and > 0"
        )
    if invalid_values[4] != 0:
        raise ValueError(
            "sedimentation particle concentrations must be equal when positive"
        )
    if invalid_values[5] != 0:
        raise ValueError(
            "sedimentation active particle count exceeds "
            "MAX_SEDIMENTATION_ACTIVE_PARTICLES_PER_BOX"
        )
    if invalid_values[6] != 0:
        raise ValueError(
            "sedimentation active particle total mass and volume must be "
            "finite and > 0"
        )


def _validate_collision_pairs(
    collision_pairs: Any,
    expected_shape: tuple[int, int, int],
    expected_device: Any,
) -> None:
    """Validate collision pair buffer shape and device.

    Args:
        collision_pairs: Collision pairs buffer.
        expected_shape: Expected ``(n_boxes, max_collisions, 2)`` shape.
        expected_device: Expected Warp device.

    Raises:
        ValueError: If the buffer shape or device mismatches.
    """
    if collision_pairs.shape != expected_shape:
        raise ValueError(
            f"collision_pairs shape {collision_pairs.shape} does not match "
            f"expected {expected_shape}"
        )
    if collision_pairs.dtype != wp.int32:
        raise ValueError("collision_pairs buffer must use dtype int32")
    _validate_device_match(
        "collision_pairs buffer", collision_pairs, expected_device
    )


def _validate_collision_counts(
    n_collisions: Any,
    expected_shape: tuple[int],
    expected_device: Any,
) -> None:
    """Validate collision count buffer shape and device.

    Args:
        n_collisions: Collision counts buffer.
        expected_shape: Expected ``(n_boxes,)`` shape.
        expected_device: Expected Warp device.

    Raises:
        ValueError: If the buffer shape or device mismatches.
    """
    if n_collisions.shape != expected_shape:
        raise ValueError(
            f"n_collisions shape {n_collisions.shape} does not match expected "
            f"{expected_shape}"
        )
    if n_collisions.dtype != wp.int32:
        raise ValueError("n_collisions buffer must use dtype int32")
    _validate_device_match("n_collisions buffer", n_collisions, expected_device)


def _validate_rng_states(
    rng_states: Any,
    expected_shape: tuple[int],
    expected_device: Any,
) -> None:
    """Validate RNG state buffer shape and device.

    Args:
        rng_states: RNG state buffer.
        expected_shape: Expected ``(n_boxes,)`` shape.
        expected_device: Expected Warp device.

    Raises:
        ValueError: If the buffer shape or device mismatches.
    """
    if rng_states.shape != expected_shape:
        raise ValueError(
            f"rng_states shape {rng_states.shape} does not match expected "
            f"{expected_shape}"
        )
    if rng_states.dtype != wp.uint32:
        raise ValueError("rng_states buffer must use dtype uint32")
    _validate_device_match("rng_states buffer", rng_states, expected_device)


def _validate_time_step(time_step: object) -> float:
    """Validate the coagulation time step before any mutation.

    Args:
        time_step: Proposed coagulation time step in seconds.

    Returns:
        The validated time step as a Python ``float``.

    Raises:
        ValueError: If ``time_step`` is not a finite nonnegative real value.
    """
    if isinstance(time_step, bool):
        raise ValueError("time_step must be finite and nonnegative")

    try:
        time_step_value = float(cast(Any, time_step))
    except (TypeError, ValueError) as exc:
        raise ValueError("time_step must be finite and nonnegative") from exc

    if not np.isfinite(time_step_value) or time_step_value < 0.0:
        raise ValueError("time_step must be finite and nonnegative")

    return time_step_value


def _validate_max_collisions(max_collisions: object) -> int:
    """Validate the supported collision-buffer length before allocation.

    Args:
        max_collisions: Proposed maximum accepted collisions per box.

    Returns:
        The validated collision limit as a Python ``int``.

    Raises:
        ValueError: If ``max_collisions`` is not a supported positive integer.
    """
    if isinstance(max_collisions, bool):
        raise ValueError(
            "max_collisions must be a positive integer <= 2147483647"
        )

    try:
        max_collisions_value = int(cast(Any, max_collisions))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "max_collisions must be a positive integer <= 2147483647"
        ) from exc

    if (
        max_collisions_value <= 0
        or max_collisions_value > np.iinfo(np.int32).max
        or max_collisions_value != max_collisions
    ):
        raise ValueError(
            "max_collisions must be a positive integer <= 2147483647"
        )

    return max_collisions_value


def _resolve_collision_capacity(
    max_collisions: object,
    n_boxes: int,
    n_particles: int,
) -> int:
    """Resolve a bounded accepted-collision capacity for one call.

    Args:
        max_collisions: Requested maximum collisions per box.
        n_boxes: Number of boxes in the current launch.
        n_particles: Number of particles per box.

    Returns:
        Effective accepted-collision capacity per box.

    Notes:
        Accepted collisions cannot exceed ``n_particles // 2`` because each
        accepted collision removes two active particles from the candidate set.
        Allocation is also bounded by an explicit buffer-byte budget so invalid
        requests cannot trigger unbounded internal allocation.
    """
    requested_capacity = _validate_max_collisions(max_collisions)
    physical_capacity = max(1, n_particles // 2)
    bytes_per_box_collision = 2 * np.dtype(np.int32).itemsize
    budget_capacity = max(
        1,
        MAX_COLLISION_PAIR_BUFFER_BYTES
        // max(1, n_boxes * bytes_per_box_collision),
    )
    return min(requested_capacity, physical_capacity, budget_capacity)


def initialize_coagulation_rng_states(
    rng_seed: int,
    rng_states: Any,
    *,
    device: Any | None = None,
) -> Any:
    """Initialize a caller-owned coagulation RNG state buffer from a seed.

    Args:
        rng_seed: Seed used to initialize the per-box Warp RNG state.
        rng_states: Caller-owned ``(n_boxes,)`` buffer with dtype ``wp.uint32``.
        device: Optional Warp device override. When omitted, uses
            ``rng_states.device``.

    Returns:
        The same ``rng_states`` buffer after in-place initialization.

    Raises:
        ValueError: If ``rng_states`` is not a one-dimensional ``wp.uint32``
            buffer or if the provided device mismatches the buffer device.
    """
    if len(rng_states.shape) != 1:
        raise ValueError("rng_states must have shape (n_boxes,)")
    if rng_states.dtype != wp.uint32:
        raise ValueError("rng_states buffer must use dtype uint32")

    resolved_device = getattr(rng_states, "device", None)
    if device is None:
        device = resolved_device
    elif isinstance(device, str):
        device = wp.get_device(device)
    _validate_device_match("rng_states buffer", rng_states, device)

    wp.launch(
        _initialize_rng_states,
        dim=rng_states.shape[0],
        inputs=[wp.uint32(rng_seed), rng_states],
        device=device,
    )
    wp.synchronize()
    return rng_states


def _ensure_volume_array(
    volume: float | Any,
    n_boxes: int,
    device: Any,
) -> Any:
    """Ensure a GPU volume array is available.

    Args:
        volume: Volume scalar or Warp array.
        n_boxes: Number of boxes.
        device: Warp device.

    Returns:
        Warp array of volumes ``(n_boxes,)``.

    Raises:
        ValueError: If volume array has mismatched shape, device, or domain.

    Notes:
        Valid Warp arrays are returned unchanged so launch code can reuse
        caller-owned per-box volume buffers.
    """
    volume_array: Any = volume
    if _is_warp_array_like(volume_array):
        if volume_array.shape != (n_boxes,):
            raise ValueError("volume shape does not match (n_boxes,)")
        _validate_device_match("volume", volume_array, device)
        if not _is_supported_warp_float_dtype(volume_array.dtype):
            raise ValueError("volume must use a supported Warp float dtype")
        _validate_positive_finite_array(
            "volume", volume_array, "coagulation_step_gpu"
        )
        return volume_array

    if isinstance(volume, bool) or not isinstance(volume, (float, np.floating)):
        if hasattr(volume, "shape"):
            raise ValueError(
                "volume must be a Warp array with shape (n_boxes,)"
            )
        raise ValueError("volume must be a floating scalar or Warp array")

    volume_scalar = float(volume)
    if not np.isfinite(volume_scalar) or volume_scalar <= 0.0:
        raise ValueError("volume must be finite and > 0")

    return _broadcast_scalar_array(volume_scalar, n_boxes, device)


def coagulation_step_gpu(  # noqa: C901
    particles: Any,
    temperature: float | Any | None,
    pressure: float | Any | None,
    time_step: object,
    volume: float | Any | None = None,
    max_collisions: object = 256,
    rng_seed: int = 0,
    collision_pairs: Any | None = None,
    n_collisions: Any | None = None,
    rng_states: Any | None = None,
    *,
    mechanism_config: CoagulationMechanismConfig | None = None,
    initialize_rng: bool = False,
    environment: Any | None = None,
    validate_charge_finite: bool = False,
) -> tuple[Any, Any, Any]:
    """Execute one direct, particle-resolved Warp coagulation timestep.

    This low-level API is not a CPU strategy-composition or ``Runnable`` API.
    It supports only ``"particle_resolved"`` execution: omission selects
    Brownian; the other exact executable modes are ``"charged_hard_sphere"``,
    Brownian-plus-charged, and unit-efficiency ``"sedimentation_sp2016"``.
    ``mechanism_config`` is immutable, keyword-only
    host metadata and is preflighted before any runtime input access,
    allocation, normalization, RNG work, or launch. The supported charged-only
    configuration is ``("charged_hard_sphere",)``; the combined mode accepts
    ``("brownian", "charged_hard_sphere")`` in either requested order and
    normalizes it to one fixed mask, one shared selector, and one additive
    majorant. Independently sanitized Brownian and charged pair-rate terms are
    summed before one candidate stream, acceptance stream, collision-buffer
    set, RNG stream, and apply pass. Its active-pair work is O(A²), while
    collision and selector buffers are O(A), for active count A; this is bounded
    implementation scope, not a throughput or scaling claim.
    Sedimentation is executable only as its exact singleton mask; its mixed
    variants, other charged variants, and non-particle-resolved distributions
    reject during preflight without runtime mutation. After particle metadata
    and device checks, direct temperature and pressure inputs are validated and
    normalized before volume setup or selector launches.
    ``particles`` and supplied ``collision_pairs``, ``n_collisions``,
    and ``rng_states`` are caller-owned same-device Warp resources; the step
    mutates them in place as applicable. The three-item return tuple contains
    no RNG state: supplied collision buffers are returned by identity, while
    ``rng_states`` remains caller-owned. Caller-owned RNG buffers are reset only
    when ``initialize_rng=True`` explicitly opts in.

    Args:
        particles: Caller-owned, Warp-resident particle data. All required Warp
            arrays, including ``charge``, must be on the same device. ``charge``
            must be ``wp.float64`` with shape ``(n_boxes, n_particles)``.
            Charged-containing calls scan it for finite values before output
            validation or allocation, RNG setup, or selector/apply work.
            Exact SP2016 sedimentation calls scan mass and concentration for
            finite nonnegative values and density for finite positive values
            before time-step, environment, output-buffer, or RNG work.
            Accepted collisions mutate mass, concentration, and charge in place:
            recipient particles receive donor mass and charge, and donor mass,
            concentration, and charge are cleared.
        temperature: Direct gas temperature as either a scalar or a Warp array
            with shape ``(n_boxes,)``. Use ``None`` only with
            ``environment=...``.
        pressure: Direct gas pressure as either a scalar or a Warp array with
            shape ``(n_boxes,)``. Use ``None`` only with ``environment=...``.
        time_step: Coagulation time step in seconds.
        volume: Per-box volume [m^3]. If None, uses ``particles.volume``.
        max_collisions: Maximum number of collisions per box.
        rng_seed: Seed used whenever this call initializes Warp RNG states.
            This always applies to the omitted-``rng_states`` convenience path.
            For caller-owned persistent buffers, the seed is only consumed when
            ``initialize_rng=True`` explicitly requests a reset.
        collision_pairs: Optional caller-owned, same-device Warp collision
            output buffer. The buffer is written in place. When omitted, the
            function allocates a call-local convenience buffer for this call.
        n_collisions: Optional caller-owned, same-device Warp per-box collision
            count output buffer. The buffer is written in place. When omitted,
            the function allocates a call-local convenience buffer for this
            call.
        rng_states: Optional caller-owned, same-device Warp RNG state buffer.
            The buffer is mutated in place. When omitted, this function
            allocates a call-local same-device ``(n_boxes,)`` buffer, seeds it
            from ``rng_seed``, and uses it only for the current call. When
            provided, the caller owns the persistent GPU-resident sidecar
            buffer and it is reused as-is across repeated calls unless
            ``initialize_rng=True`` explicitly requests a reset from
             ``rng_seed``.
        mechanism_config: Optional immutable host-side
            ``CoagulationMechanismConfig`` imported from
            ``particula.gpu.kernels.coagulation``; it is not re-exported by
            ``particula.gpu.kernels``. This keyword-only configuration does not
            transfer, synchronize, or own Warp state. Omission selects
            Brownian, particle-resolved execution. Charged-only and
            Brownian-plus-charged particle-resolved execution, and exact
            singleton SP2016 sedimentation are also supported; either requested
            Brownian-plus-charged order normalizes to the same execution.
            Malformed configurations, unsupported distributions, and reserved
            mechanism combinations fail before runtime inputs are accessed or
            mutable runtime state is changed. A
            wrong type raises exactly ``ValueError`` with the message
            ``"mechanism_config must be a CoagulationMechanismConfig."``;
            other errors are delegated to the resolver and capability gate.
        initialize_rng: Explicit reset flag for caller-provided
            ``rng_states``. The default ``False`` path validates the buffer and
            reuses its existing state without reseeding, even if ``rng_seed``
            matches an earlier call. Set ``True`` to launch
            ``_initialize_rng_states`` after validation and reset the
            persistent buffer from ``rng_seed``. This argument does not affect
            omitted ``rng_states`` convenience allocation, which always
            initializes the internal buffer for the current call.
        environment: Optional ``WarpEnvironmentData`` with ``(n_boxes,)``
            temperature and pressure arrays on the same device as ``particles``.
            This mode is supported when both direct inputs are ``None``.
        validate_charge_finite: When True, explicitly scan charge on device and
            synchronize for host-visible rejection of NaN or infinity. The
            default False avoids per-step allocation, synchronization, and host
            readback for Brownian execution. Every charged-containing mode
            performs this scan before caller output or RNG mutation.

    Returns:
        Tuple containing ``particles`` after in-place coagulation, the
        ``collision_pairs`` buffer for the step, and the ``n_collisions``
        buffer with per-box accepted collision counts.

    Raises:
        ValueError: If array shapes or devices mismatch expectations.
        ValueError: If direct ``temperature`` or ``pressure`` inputs are mixed
            with ``environment``.
        ValueError: If direct inputs are missing when ``environment`` is
            omitted.
        ValueError: If environment arrays do not match ``(n_boxes,)`` or the
            caller device.
        ValueError: If mechanism_config has the wrong type, is malformed, or
            requests an unsupported distribution or reserved mechanism.
        ValueError: If particle charge does not have shape
            ``(n_boxes, n_particles)``, dtype ``wp.float64``, or the particle
            device. Charged-containing and exact SP2016 sedimentation
            execution, and Brownian with ``validate_charge_finite=True``, also
            raise for NaN or infinity.
        ValueError: If charged-containing masses or concentrations are
            non-finite or negative, species density is non-finite or
            nonpositive, or a box exceeds
            ``MAX_CHARGED_ACTIVE_PARTICLES_PER_BOX`` active particles.
        ValueError: If SP2016 sedimentation masses or concentrations are
            non-finite or negative, positive concentrations differ within a
            box, species density is non-finite or nonpositive, or a box exceeds
            ``MAX_SEDIMENTATION_ACTIVE_PARTICLES_PER_BOX`` active particles.

    Notes:
        Accepted environment sources are scalar direct inputs, direct
        ``(n_boxes,)`` Warp arrays, hybrid scalar-plus-Warp-array direct
        inputs, or keyword-only ``environment=...`` execution.

        ``mechanism_config``, ``initialize_rng``, ``environment``, and
        ``validate_charge_finite`` remain
        keyword-only so existing positional scalar callers stay
        source-compatible. Configuration preflight occurs before all runtime
        input access, allocation, normalization, RNG work, and launches. It
        does not make unrelated later validation failures atomic.

        Validation runs before volume normalization, RNG setup, and Brownian or
        apply launches so invalid ``time_step``, shape, dtype, or device
        combinations fail without mutating particle state or allocating
        downstream work. The normalized environment arrays are forwarded
        directly into the launch path.

        Charged-containing execution read-only-validates finite/nonnegative
        masses and concentrations, finite/positive density, its bounded active
        count, and finite charge before time-step or environment normalization.
        The active bound caps its exact compact-active O(A²) majorant at 32,640
        pairs per box. Brownian validates charge only with
        ``validate_charge_finite=True`` and otherwise avoids this scan,
        allocation, synchronization, and host readback.

        Exact SP2016 sedimentation read-only-validates finite charge,
        finite/nonnegative masses and concentrations, finite/positive density,
        compatible positive concentrations, and its bounded active count before
        time-step, environment, output, or RNG work. Its exact compact-active
        O(A²) majorant is capped at 32,640 pairs per box. Zero masses and
        concentrations remain valid inactive-slot state; this mode does not
        impose charged-only constraints.

        A successful call allocates private, call-local selector scratch,
        including ``wp.float64`` ``(n_boxes, n_particles)`` total-mass and
        settling-velocity arrays. The selector clears every scratch slot and
        sums each active particle's species masses once for charged pair rates
        and their compact-active majorant. It populates settling velocities only
        for valid active slots of the exact SP2016 sedimentation-only mask.
        This scratch is not a caller-owned output, is not returned, and is
        allocated only after caller-resource preflight succeeds.

        Supported RNG setup cases are:

        - Omitted ``rng_states``: allocate a call-local internal buffer and
          initialize it from ``rng_seed`` for this call.
        - Provided ``rng_states`` with ``initialize_rng=False``: validate the
          caller-owned persistent buffer and reuse it without resetting.
        - Provided ``rng_states`` with ``initialize_rng=True``: validate the
          caller-owned persistent buffer, then reset it from ``rng_seed``.
        - Validation failure: raise before RNG initialization or other Warp
          launches mutate state.

        Reusing the same ``rng_seed`` alongside a persistent ``rng_states``
        buffer does not reseed that buffer on later calls unless
        ``initialize_rng=True`` is passed. Initialize caller-owned buffers once
        before repeated timesteps, then reuse them without hidden per-step
        resets. Graph-capture support remains deferred.
    """
    if mechanism_config is None:
        mechanism_config = CoagulationMechanismConfig()
    if not isinstance(mechanism_config, CoagulationMechanismConfig):
        raise ValueError(
            "mechanism_config must be a CoagulationMechanismConfig."
        )
    resolved_mechanism_config = resolve_coagulation_mechanism_config(
        mechanism_config
    )
    validate_coagulation_mechanism_capabilities(resolved_mechanism_config)

    n_boxes, n_particles, n_species = particles.masses.shape
    _validate_particle_arrays(particles, n_boxes, n_particles, n_species)

    device = particles.masses.device
    _validate_device_arrays(particles, device)
    charged_enabled = bool(
        resolved_mechanism_config.mask & CHARGED_HARD_SPHERE_MECHANISM_FLAG
    )
    sedimentation_enabled = (
        resolved_mechanism_config.mask == SEDIMENTATION_SP2016_MECHANISM_FLAG
    )
    if validate_charge_finite or charged_enabled:
        _validate_charge_finite(
            particles.charge,
            n_boxes,
            n_particles,
            device,
        )
    if charged_enabled:
        _validate_charged_particle_physics(
            particles,
            n_boxes,
            n_particles,
            device,
        )
    if sedimentation_enabled:
        _validate_sedimentation_particle_physics(
            particles,
            n_boxes,
            n_particles,
            device,
        )
    time_step_value = _validate_time_step(time_step)
    temperature_array, pressure_array = _ensure_environment_arrays(
        temperature=temperature,
        pressure=pressure,
        environment=environment,
        n_boxes=n_boxes,
        device=device,
        caller_name="coagulation_step_gpu",
    )

    if volume is None:
        volume = particles.volume
    volume_array = _ensure_volume_array(volume, n_boxes, device)

    max_collisions_value = _resolve_collision_capacity(
        max_collisions=max_collisions,
        n_boxes=n_boxes,
        n_particles=n_particles,
    )

    expected_pairs_shape = (n_boxes, max_collisions_value, 2)
    if collision_pairs is not None:
        if collision_pairs.shape[0] != n_boxes or collision_pairs.shape[2] != 2:
            raise ValueError("collision_pairs shape must match (n_boxes, *, 2)")
        if collision_pairs.shape[1] < max_collisions_value:
            raise ValueError(
                "collision_pairs capacity is smaller than the effective "
                "max_collisions bound"
            )
        if collision_pairs.dtype != wp.int32:
            raise ValueError("collision_pairs buffer must use dtype int32")
        _validate_device_match(
            "collision_pairs buffer", collision_pairs, device
        )

    expected_counts_shape = (n_boxes,)
    if n_collisions is not None:
        _validate_collision_counts(n_collisions, expected_counts_shape, device)

    initialize_rng_states_for_call = rng_states is None
    if rng_states is not None:
        _validate_rng_states(rng_states, expected_counts_shape, device)
        initialize_rng_states_for_call = initialize_rng

    if collision_pairs is None:
        collision_pairs = wp.zeros(
            expected_pairs_shape,
            dtype=wp.int32,
            device=device,
        )
    if n_collisions is None:
        n_collisions = wp.zeros(
            expected_counts_shape,
            dtype=wp.int32,
            device=device,
        )

    if rng_states is None:
        rng_states = wp.zeros(
            expected_counts_shape,
            dtype=wp.uint32,
            device=device,
        )

    radii = wp.zeros((n_boxes, n_particles), dtype=wp.float64, device=device)
    diffusivities = radii
    g_terms = radii
    speeds = radii
    if not sedimentation_enabled:
        diffusivities = wp.zeros(
            (n_boxes, n_particles), dtype=wp.float64, device=device
        )
        g_terms = wp.zeros(
            (n_boxes, n_particles), dtype=wp.float64, device=device
        )
        speeds = wp.zeros(
            (n_boxes, n_particles), dtype=wp.float64, device=device
        )
    # Settling velocities are meaningful only for exact singleton SP2016. For
    # all other modes pass an unused existing float64 scratch to avoid an
    # allocation. Sedimentation storage begins zeroed, so its kernel does not
    # need a redundant full-capacity clear.
    settling_velocities = radii
    if sedimentation_enabled:
        settling_velocities = wp.zeros(
            (n_boxes, n_particles), dtype=wp.float64, device=device
        )
    total_masses = wp.zeros(
        (n_boxes, n_particles), dtype=wp.float64, device=device
    )
    active_indices = wp.zeros(
        (n_boxes, n_particles), dtype=wp.int32, device=device
    )

    if initialize_rng_states_for_call:
        initialize_coagulation_rng_states(
            rng_seed=rng_seed,
            rng_states=rng_states,
            device=device,
        )

    wp.launch(
        brownian_coagulation_kernel,
        dim=(n_boxes,),
        inputs=[
            particles.masses,
            particles.concentration,
            particles.density,
            volume_array,
            temperature_array,
            pressure_array,
            wp.float64(constants.GAS_CONSTANT),
            wp.float64(constants.BOLTZMANN_CONSTANT),
            wp.float64(constants.MOLECULAR_WEIGHT_AIR),
            wp.float64(constants.REF_VISCOSITY_AIR_STP),
            wp.float64(constants.REF_TEMPERATURE_STP),
            wp.float64(constants.SUTHERLAND_CONSTANT),
            wp.float64(constants.ELEMENTARY_CHARGE_VALUE),
            wp.float64(constants.ELECTRIC_PERMITTIVITY),
            wp.float64(time_step_value),
            radii,
            diffusivities,
            g_terms,
            speeds,
            settling_velocities,
            total_masses,
            particles.charge,
            active_indices,
            collision_pairs,
            n_collisions,
            rng_states,
            wp.int32(resolved_mechanism_config.mask),
            wp.int32(max_collisions_value),
        ],
        device=device,
    )

    wp.launch(
        apply_coagulation_kernel,
        dim=(n_boxes, max_collisions_value),
        inputs=[
            particles.masses,
            particles.concentration,
            particles.charge,
            collision_pairs,
            n_collisions,
        ],
        device=device,
    )
    wp.launch(
        _compact_applied_collision_pairs_kernel,
        dim=(n_boxes,),
        inputs=[collision_pairs, n_collisions],
        device=device,
    )

    return particles, collision_pairs, n_collisions
