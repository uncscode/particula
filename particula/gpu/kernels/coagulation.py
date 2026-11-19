"""Native equal-count coagulation with tiled exact additive pair reductions.

All 15 nonempty mechanism subsets share one candidate and acceptance stream.
Work budgets reject before particle, collision-output or RNG writers. Direct
calls observe compact admission diagnostics; resident calls latch device errors.
Configuration and preflight helpers are concrete-module APIs. No CPU fallback,
hidden particle transfer, box subdivision or weight change is performed.
"""

# pyright: basic
# pyright: reportArgumentType=false
# pyright: reportAssignmentType=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportOperatorIssue=false

from dataclasses import dataclass, replace
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
    charged_kernel_wp,
    g_collection_term_wp,
    kinematic_viscosity_wp,
    particle_mean_free_path_wp,
    sedimentation_sp2016_pair_rate_wp,
    turbulent_shear_st1956_pair_rate_wp,
)
from particula.gpu.dynamics.ion_pair_data import (
    IonPairData,
    replace_ion_pair_rate,
)
from particula.gpu.dynamics.van_der_waals import (
    CaptureData,
    capture_pair_wp,
    prepare_capture,
)
from particula.gpu.kernels._coagulation_budget import (
    CoagulationWorkConfig,
    admit_coagulation_work,
    initialize_admitted_rng,
    raise_budget_error,
    validate_coagulation_layout,
)
from particula.gpu.kernels._coagulation_config import (
    _COAGULATION_MECHANISM_FLAGS,  # noqa: F401
    BROWNIAN_MECHANISM,  # noqa: F401
    BROWNIAN_MECHANISM_FLAG,
    CANONICAL_COAGULATION_MECHANISMS,  # noqa: F401
    CHARGED_HARD_SPHERE_MECHANISM,  # noqa: F401
    CHARGED_HARD_SPHERE_MECHANISM_FLAG,
    SEDIMENTATION_SP2016_MECHANISM,  # noqa: F401
    SEDIMENTATION_SP2016_MECHANISM_FLAG,
    TURBULENT_SHEAR_ST1956_MECHANISM,  # noqa: F401
    TURBULENT_SHEAR_ST1956_MECHANISM_FLAG,
    CoagulationMechanismConfig,
    _ResolvedCoagulationMechanismConfig,
    resolve_coagulation_mechanism_config,
    validate_coagulation_mechanism_capabilities,
)
from particula.gpu.kernels._coagulation_subcycling import (
    CoagulationSubcyclingConfig as CoagulationSubcyclingConfig,
)
from particula.gpu.kernels._coagulation_subcycling import (
    CoagulationSubcyclingError as CoagulationSubcyclingError,
)
from particula.gpu.kernels.environment import (
    _broadcast_scalar_array,
    _ensure_environment_arrays,
    _is_supported_warp_float_dtype,
    _is_warp_array_like,
    _validate_positive_finite_array,
)
from particula.gpu.population_metadata import (
    PARTICLE_RESOLVED,
    particle_distribution_code,
)
from particula.gpu.properties.gas_properties import (
    dynamic_viscosity_wp,
    molecule_mean_free_path_wp,
)
from particula.gpu.properties.particle_properties import (
    aerodynamic_mobility_wp,
    cunningham_slip_correction_wp,
    effective_density_wp,
    knudsen_number_wp,
    mean_thermal_speed_wp,
    particle_radius_from_volume_wp,
    settling_velocity_stokes_from_transport_wp,
)

# These configuration APIs remain publicly owned by this concrete module even
# though their host-only implementation is factored into a private module.
CoagulationMechanismConfig.__module__ = __name__
resolve_coagulation_mechanism_config.__module__ = __name__
validate_coagulation_mechanism_capabilities.__module__ = __name__

MAX_COLLISION_PAIR_BUFFER_BYTES = 256 * 1024 * 1024
MAX_SCHEDULED_TRIALS_PER_BOX = 65_536
# The legacy clamp below is retained for diagnostic test oracles only.
# Production admission uses CoagulationWorkConfig and never clips demand.


@no_type_check
@wp.func
def _bound_scheduled_trials(expected_trials: Any) -> Any:
    """Retain the legacy trial clamp for diagnostic oracles, not production."""
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
def _checked_positive_finite_add(left: Any, right: Any) -> Any:
    """Add two sanitized fp64 terms, failing closed on invalid arithmetic.

    Additive mechanism terms are independently optional, but an overflowing
    aggregate cannot safely be used as a selector rate or majorant.
    """
    sanitized_left = _sanitize_positive_finite(left)
    sanitized_right = _sanitize_positive_finite(right)
    total = sanitized_left + sanitized_right
    return _sanitize_positive_finite(total)


@no_type_check
@wp.func
def _safe_acceptance_probability(total_rate: Any, total_majorant: Any) -> Any:
    """Return a safe acceptance ratio or zero before an acceptance RNG draw.

    A rate may exceed its majorant only by an eight-machine-epsilon relative
    tolerance; that narrow case maps exactly to one. Material violations and
    non-finite tolerance arithmetic fail closed before the caller draws an
    acceptance variate. Candidate-selection draws precede this calculation.
    """
    zero = wp.float64(0.0)
    one = wp.float64(1.0)
    if not (
        wp.isfinite(total_rate)
        and total_rate > zero
        and wp.isfinite(total_majorant)
        and total_majorant > zero
    ):
        return zero

    if total_rate > total_majorant:
        larger = total_rate
        if total_majorant > larger:
            larger = total_majorant
        tolerance = wp.float64(8.0) * wp.float64(2.220446049250313e-16) * larger
        if (
            not wp.isfinite(tolerance)
            or total_rate - total_majorant > tolerance
        ):
            return zero
        return one

    probability = total_rate / total_majorant
    if not wp.isfinite(probability) or probability <= zero or probability > one:
        return zero
    return probability


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
    turbulent_dissipation: Any,
    kinematic_viscosity: Any,
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
    Each enabled term is independently sanitized and added with checked fp64
    arithmetic. An invalid component contributes zero, while an invalid or
    overflowing aggregate returns zero. This helper's bit dispatch does not
    itself grant sampler support; the entry-point capability gate admits only
    approved singleton, two-way, and four-way masks.
    """
    total_rate = wp.float64(0.0)
    if mechanism_mask & wp.int32(BROWNIAN_MECHANISM_FLAG):
        component_rate = _sanitize_positive_finite(
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
        accumulated_rate = _checked_positive_finite_add(
            total_rate, component_rate
        )
        if (
            total_rate > wp.float64(0.0)
            and component_rate > wp.float64(0.0)
            and accumulated_rate <= wp.float64(0.0)
        ):
            return wp.float64(0.0)
        total_rate = accumulated_rate
    if mechanism_mask & wp.int32(CHARGED_HARD_SPHERE_MECHANISM_FLAG):
        component_rate = _sanitize_positive_finite(
            charged_kernel_wp(
                wp.int32(mechanism_mask) >> wp.int32(8),
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
        accumulated_rate = _checked_positive_finite_add(
            total_rate, component_rate
        )
        if (
            total_rate > wp.float64(0.0)
            and component_rate > wp.float64(0.0)
            and accumulated_rate <= wp.float64(0.0)
        ):
            return wp.float64(0.0)
        total_rate = accumulated_rate
    if mechanism_mask & wp.int32(SEDIMENTATION_SP2016_MECHANISM_FLAG):
        component_rate = _sanitize_positive_finite(
            sedimentation_sp2016_pair_rate_wp(
                radius_i,
                radius_j,
                settling_velocity_i,
                settling_velocity_j,
            )
        )
        accumulated_rate = _checked_positive_finite_add(
            total_rate, component_rate
        )
        if (
            total_rate > wp.float64(0.0)
            and component_rate > wp.float64(0.0)
            and accumulated_rate <= wp.float64(0.0)
        ):
            return wp.float64(0.0)
        total_rate = accumulated_rate
    if mechanism_mask & wp.int32(TURBULENT_SHEAR_ST1956_MECHANISM_FLAG):
        component_rate = _sanitize_positive_finite(
            turbulent_shear_st1956_pair_rate_wp(
                radius_i,
                radius_j,
                turbulent_dissipation,
                kinematic_viscosity,
            )
        )
        accumulated_rate = _checked_positive_finite_add(
            total_rate, component_rate
        )
        if (
            total_rate > wp.float64(0.0)
            and component_rate > wp.float64(0.0)
            and accumulated_rate <= wp.float64(0.0)
        ):
            return wp.float64(0.0)
        total_rate = accumulated_rate
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
    Invalid, zero, and nonpositive rates contribute zero.

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
def _turbulent_majorant_from_active_radii(
    active_indices: Any,
    box_idx: Any,
    active_count: Any,
    radii: Any,
    turbulent_dissipation: Any,
    kinematic_viscosity: Any,
) -> Any:
    """Return an ST1956 majorant from the two largest compact active radii.

    The ST1956 prefactor is constant within a box and its diameter-sum cubed
    term is monotonic, making this O(A) compact-rank scan a safe majorant.

    Args:
        active_indices: Compact particle indices ``(n_boxes, n_particles)``.
        box_idx: Index of the simulation box to scan.
        active_count: Number of compact active ranks in ``box_idx``.
        radii: Particle radii ``(n_boxes, n_particles)`` [m].
        turbulent_dissipation: Per-box turbulent kinetic-energy dissipation
            rate [m^2/s^3].
        kinematic_viscosity: Per-box kinematic viscosity [m^2/s].

    Returns:
        Finite, nonnegative ST1956 majorant for the selected box, or zero when
        fewer than two active particles exist or no finite positive rate exists.
    """
    if active_count < wp.int32(2):
        return wp.float64(0.0)
    largest = wp.float64(0.0)
    second_largest = wp.float64(0.0)
    for rank in range(active_count):
        radius = radii[box_idx, wp.int32(active_indices[box_idx, rank])]
        if radius > largest:
            second_largest = largest
            largest = radius
        elif radius > second_largest:
            second_largest = radius
    return _sanitize_positive_finite(
        turbulent_shear_st1956_pair_rate_wp(
            largest,
            second_largest,
            turbulent_dissipation,
            kinematic_viscosity,
        )
    )


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
    turbulent_dissipation: Any,
    kinematic_viscosity: Any,
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
    """Return a checked sum of enabled finite, positive majorant components.

    Component maxima can occur on different active pairs, so each enabled
    component is bounded independently and then added with checked fp64
    arithmetic. An invalid or overflowing aggregate returns zero. This helper
    supports deterministic internal probes and public approved-mask execution;
    its dispatch does not make three-way masks executable. Brownian-containing
    production selection instead uses one exact active-pair scan through
    ``_total_pair_rate``.
    """
    total_majorant = wp.float64(0.0)
    if mechanism_mask & wp.int32(BROWNIAN_MECHANISM_FLAG):
        component_majorant = _sanitize_positive_finite(
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
        accumulated_majorant = _checked_positive_finite_add(
            total_majorant, component_majorant
        )
        if (
            total_majorant > wp.float64(0.0)
            and component_majorant > wp.float64(0.0)
            and accumulated_majorant <= wp.float64(0.0)
        ):
            return wp.float64(0.0)
        total_majorant = accumulated_majorant
    if mechanism_mask & wp.int32(CHARGED_HARD_SPHERE_MECHANISM_FLAG):
        component_majorant = _sanitize_positive_finite(
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
        accumulated_majorant = _checked_positive_finite_add(
            total_majorant, component_majorant
        )
        if (
            total_majorant > wp.float64(0.0)
            and component_majorant > wp.float64(0.0)
            and accumulated_majorant <= wp.float64(0.0)
        ):
            return wp.float64(0.0)
        total_majorant = accumulated_majorant
    if mechanism_mask & wp.int32(SEDIMENTATION_SP2016_MECHANISM_FLAG):
        component_majorant = _sanitize_positive_finite(
            _sedimentation_majorant_from_active_pairs(
                active_indices,
                box_idx,
                active_count,
                radii,
                settling_velocities,
            )
        )
        accumulated_majorant = _checked_positive_finite_add(
            total_majorant, component_majorant
        )
        if (
            total_majorant > wp.float64(0.0)
            and component_majorant > wp.float64(0.0)
            and accumulated_majorant <= wp.float64(0.0)
        ):
            return wp.float64(0.0)
        total_majorant = accumulated_majorant
    if mechanism_mask & wp.int32(TURBULENT_SHEAR_ST1956_MECHANISM_FLAG):
        component_majorant = _turbulent_majorant_from_active_radii(
            active_indices,
            box_idx,
            active_count,
            radii,
            turbulent_dissipation,
            kinematic_viscosity,
        )
        accumulated_majorant = _checked_positive_finite_add(
            total_majorant, component_majorant
        )
        if (
            total_majorant > wp.float64(0.0)
            and component_majorant > wp.float64(0.0)
            and accumulated_majorant <= wp.float64(0.0)
        ):
            return wp.float64(0.0)
        total_majorant = accumulated_majorant
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
def _validate_turbulent_particle_physics_kernel(
    masses: Any,
    concentration: Any,
    density: Any,
    charge: Any,
    invalid: Any,
) -> None:
    """Record invalid ST1956 particle state without mutating caller inputs."""  # type: ignore
    box_idx = wp.tid()
    for species_idx in range(density.shape[0]):
        density_value = density[species_idx]
        if not wp.isfinite(density_value) or density_value <= wp.float64(0.0):
            wp.atomic_max(invalid, 0, 1)

    for particle_idx in range(masses.shape[1]):
        concentration_value = concentration[box_idx, particle_idx]
        if not wp.isfinite(charge[box_idx, particle_idx]):
            wp.atomic_max(invalid, 0, 1)
        if not wp.isfinite(concentration_value) or concentration_value < (
            wp.float64(0.0)
        ):
            wp.atomic_max(invalid, 0, 1)
        total_mass = wp.float64(0.0)
        total_volume = wp.float64(0.0)
        for species_idx in range(masses.shape[2]):
            mass = masses[box_idx, particle_idx, species_idx]
            if not wp.isfinite(mass) or mass < wp.float64(0.0):
                wp.atomic_max(invalid, 0, 1)
            total_mass += mass
            total_volume += mass / density[species_idx]
        if concentration_value > wp.float64(0.0) and not (
            wp.isfinite(total_mass)
            and total_mass > wp.float64(0.0)
            and wp.isfinite(total_volume)
            and total_volume > wp.float64(0.0)
        ):
            wp.atomic_max(invalid, 0, 1)


@no_type_check
@wp.kernel
# type: ignore[misc]
def _validate_sedimentation_particle_physics_kernel(  # noqa: C901
    masses: Any,
    concentration: Any,
    density: Any,
    charge: Any,
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


@no_type_check
@wp.func
# type: ignore[misc]
def _coagulation_selector_impl(  # noqa: C901
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
    turbulent_dissipation: Any,
    fluid_density: Any,
    total_masses: Any,
    charge: Any,
    active_indices: Any,
    collision_pairs: Any,
    n_collisions: Any,
    rng_states: Any,
    mechanism_mask: Any,
    collision_capacity: Any,
    box_idx: int,
) -> None:
    """Select stochastic coagulation pairs for each box.

    Candidate pairs are drawn by rank within the current active set so each
    scheduled trial proposes two distinct active particles without retrying on
    inactive slots. The sampler computes one finite positive total majorant per
    box and one finite positive total pair rate per candidate. A valid candidate
    receives exactly one acceptance draw; invalid rates, invalid majorants, and
    rates above the majorant are skipped without collision-buffer or active-set
    mutation. All fifteen nonempty mechanism masks share this diagnostic
    candidate/acceptance path. Production obtains maxima from tiled reductions.

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
            ``(n_boxes, n_particles)``. When separately allocated for SP2016
            sedimentation, this scratch is zero-initialized before selector
            launch. The selector populates valid active slots and does not
            clear this scratch.
        turbulent_dissipation: Per-box turbulent kinetic-energy dissipation
            rate ``(n_boxes,)`` [m^2/s^3]. Read when ST1956 turbulent shear is
            enabled.
        fluid_density: Per-box carrier-fluid density ``(n_boxes,)`` [kg/m^3].
            Read when ST1956 turbulent shear is enabled to derive kinematic
            viscosity.
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
        mechanism_mask: Fixed internal sampler mask. Public singleton,
            two-way, and four-way approved masks are executable.
        collision_capacity: Maximum accepted collisions per box for this call.
        box_idx: Box index supplied by the kernel launch entry point.
    """  # type: ignore
    n_particles = masses.shape[1]

    # Low byte retains the mechanism flags; high byte selects the charged fit.
    base_mask = mechanism_mask & wp.int32(255)
    executable_mask = base_mask > wp.int32(0) and base_mask <= wp.int32(15)
    if not executable_mask:
        return
    brownian_enabled = mechanism_mask & wp.int32(BROWNIAN_MECHANISM_FLAG)
    sedimentation_enabled = mechanism_mask & wp.int32(
        SEDIMENTATION_SP2016_MECHANISM_FLAG
    )
    turbulent_enabled = mechanism_mask & wp.int32(
        TURBULENT_SHEAR_ST1956_MECHANISM_FLAG
    )

    # One thread per box keeps the pair selection sequential and avoids
    # cross-thread races when writing collision pairs.
    n_species = masses.shape[2]

    temperature_value = temperature[box_idx]
    pressure_value = pressure[box_idx]

    dynamic_viscosity = wp.float64(0.0)
    mean_free_path = wp.float64(0.0)
    kinematic_viscosity = wp.float64(0.0)
    if brownian_enabled or sedimentation_enabled or turbulent_enabled:
        dynamic_viscosity = dynamic_viscosity_wp(
            temperature_value,
            ref_viscosity,
            ref_temperature,
            sutherland_constant,
        )
        if brownian_enabled or sedimentation_enabled:
            mean_free_path = molecule_mean_free_path_wp(
                molecular_weight_air,
                temperature_value,
                pressure_value,
                dynamic_viscosity,
                gas_constant,
            )
        if turbulent_enabled:
            kinematic_viscosity = kinematic_viscosity_wp(
                dynamic_viscosity,
                fluid_density[box_idx],
            )

    active_count = wp.int32(0)
    for particle_idx in range(n_particles):
        total_masses[box_idx, particle_idx] = wp.float64(0.0)
        if concentration[box_idx, particle_idx] <= wp.float64(0.0):
            active_indices[box_idx, particle_idx] = wp.int32(-1)
            radii[box_idx, particle_idx] = wp.float64(0.0)
            if brownian_enabled:
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
            if brownian_enabled:
                diffusivities[box_idx, particle_idx] = wp.float64(0.0)
                g_terms[box_idx, particle_idx] = wp.float64(0.0)
                speeds[box_idx, particle_idx] = wp.float64(0.0)
            continue

        radius = particle_radius_from_volume_wp(total_volume)
        diffusivity = wp.float64(0.0)
        speed = wp.float64(0.0)
        g_term = wp.float64(0.0)
        if brownian_enabled:
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
        if sedimentation_enabled:
            settling_velocity = settling_velocity_stokes_from_transport_wp(
                radius,
                effective_density_wp(total_mass, total_volume),
                dynamic_viscosity,
                mean_free_path,
            )

        active_indices[box_idx, active_count] = wp.int32(particle_idx)
        radii[box_idx, particle_idx] = radius
        total_masses[box_idx, particle_idx] = total_mass
        if brownian_enabled:
            diffusivities[box_idx, particle_idx] = diffusivity
            g_terms[box_idx, particle_idx] = g_term
            speeds[box_idx, particle_idx] = speed
        if sedimentation_enabled:
            settling_velocities[box_idx, particle_idx] = settling_velocity
        active_count += wp.int32(1)

    if active_count < wp.int32(2):
        n_collisions[box_idx] = wp.int32(0)
        return

    if mechanism_mask == wp.int32(TURBULENT_SHEAR_ST1956_MECHANISM_FLAG):
        # ST1956 is monotonic in the pair diameter sum, so the two largest
        # active radii provide an analytical O(A) majorant.
        majorant_total = _turbulent_majorant_from_active_radii(
            active_indices,
            box_idx,
            active_count,
            radii,
            turbulent_dissipation[box_idx],
            kinematic_viscosity,
        )
    else:
        # The exact compact-pair scan bounds the complete additive rate,
        # including combinations whose component maxima occur on different
        # pairs.
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
                    turbulent_dissipation[box_idx],
                    kinematic_viscosity,
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

    _select_coagulation_pairs_impl(
        masses,
        concentration,
        density,
        volume,
        temperature,
        pressure,
        gas_constant,
        boltzmann_constant,
        molecular_weight_air,
        ref_viscosity,
        ref_temperature,
        sutherland_constant,
        elementary_charge_value,
        electric_permittivity,
        time_step,
        radii,
        diffusivities,
        g_terms,
        speeds,
        settling_velocities,
        turbulent_dissipation,
        fluid_density,
        total_masses,
        charge,
        active_indices,
        collision_pairs,
        n_collisions,
        rng_states,
        mechanism_mask,
        collision_capacity,
        box_idx,
        majorant_total,
        active_count,
        kinematic_viscosity,
        IonPairData(),
        CaptureData(),
    )


@no_type_check
@wp.func
def _select_coagulation_pairs_impl(  # noqa: C901
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
    turbulent_dissipation: Any,
    fluid_density: Any,
    total_masses: Any,
    charge: Any,
    active_indices: Any,
    collision_pairs: Any,
    n_collisions: Any,
    rng_states: Any,
    mechanism_mask: Any,
    collision_capacity: Any,
    box_idx: int,
    majorant_total: wp.float64,
    active_count: wp.int32,
    kinematic_viscosity: wp.float64,
    ion_data: IonPairData,
    capture: CaptureData,
) -> wp.vec2i:
    """Preserve sequential pair selection and the exact persistent RNG path."""
    n_species = masses.shape[2]
    temperature_value = temperature[box_idx]
    pressure_value = pressure[box_idx]
    if not (wp.isfinite(majorant_total) and majorant_total > wp.float64(0.0)):
        n_collisions[box_idx] = wp.int32(0)
        return wp.vec2i(0, 0)

    possible_pairs = (
        wp.float64(active_count)
        * wp.float64(active_count - 1)
        / (wp.float64(2.0))
    )
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
    # Production admission rejects overflow before invoking this selector.
    # This comparison also rejects NaN, zero, and negative schedules.
    if not (expected_trials > wp.float64(0.0)):
        n_collisions[box_idx] = wp.int32(0)
        return wp.vec2i(0, 0)

    # Public admission reports resource errors before this internal function.
    # Protect raw diagnostic callers from undefined float-to-int conversion.
    if not wp.isfinite(expected_trials) or expected_trials > wp.float64(
        2147483647
    ):
        n_collisions[box_idx] = wp.int32(0)
        return wp.vec2i(0, 0)

    collision_count = wp.int32(0)
    state = rng_states[box_idx]

    bounded_trials = expected_trials
    tests = wp.int32(bounded_trials)
    remainder = bounded_trials - wp.float64(tests)
    if wp.randf(state) < remainder:
        tests += wp.int32(1)
    rng_states[box_idx] = state

    if tests <= wp.int32(0):
        n_collisions[box_idx] = wp.int32(0)
        return wp.vec2i(0, 0)

    attempted = wp.int32(0)
    for _ in range(tests):
        if collision_count >= collision_capacity or active_count < wp.int32(2):
            break

        attempted += wp.int32(1)
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
            turbulent_dissipation[box_idx],
            kinematic_viscosity,
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
        if mechanism_mask & 16:
            total_rate += capture_pair_wp(
                capture,
                radii[box_idx, selected_i],
                radii[box_idx, selected_j],
                total_masses[box_idx, selected_i],
                total_masses[box_idx, selected_j],
                charge[box_idx, selected_i],
                charge[box_idx, selected_j],
                diffusivities[box_idx, selected_i]
                + diffusivities[box_idx, selected_j],
                temperature_value,
            )
        total_rate = replace_ion_pair_rate(
            ion_data, box_idx, selected_i, selected_j, total_rate
        )
        acceptance_probability = _safe_acceptance_probability(
            total_rate, majorant_total
        )
        if acceptance_probability <= wp.float64(0.0):
            continue
        # Every valid candidate gets exactly one acceptance draw.
        if wp.randf(state) < acceptance_probability:
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

    return wp.vec2i(tests, attempted)


@no_type_check
@wp.kernel
def _native_coagulation_kernel(
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
    turbulent_dissipation: Any,
    fluid_density: Any,
    total_masses: Any,
    charge: Any,
    active_indices: Any,
    collision_pairs: Any,
    n_collisions: Any,
    rng_states: Any,
    mechanism_mask: Any,
    collision_capacity: Any,
) -> None:
    """Schedule every native mechanism with its common raw particle count."""
    _coagulation_selector_impl(
        masses,
        concentration,
        density,
        volume,
        temperature,
        pressure,
        gas_constant,
        boltzmann_constant,
        molecular_weight_air,
        ref_viscosity,
        ref_temperature,
        sutherland_constant,
        elementary_charge_value,
        electric_permittivity,
        time_step,
        radii,
        diffusivities,
        g_terms,
        speeds,
        settling_velocities,
        turbulent_dissipation,
        fluid_density,
        total_masses,
        charge,
        active_indices,
        collision_pairs,
        n_collisions,
        rng_states,
        mechanism_mask,
        collision_capacity,
        wp.tid(),
    )


@no_type_check
@wp.kernel
def _native_coagulation_kernel_guarded(
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
    turbulent_dissipation: Any,
    fluid_density: Any,
    total_masses: Any,
    charge: Any,
    active_indices: Any,
    collision_pairs: Any,
    n_collisions: Any,
    rng_states: Any,
    mechanism_mask: Any,
    collision_capacity: Any,
    failure: wp.array(dtype=wp.int32),
) -> None:
    """Skip this resident writer after a latched simulation failure."""
    if failure[0] != 0:
        return
    _coagulation_selector_impl(
        masses,
        concentration,
        density,
        volume,
        temperature,
        pressure,
        gas_constant,
        boltzmann_constant,
        molecular_weight_air,
        ref_viscosity,
        ref_temperature,
        sutherland_constant,
        elementary_charge_value,
        electric_permittivity,
        time_step,
        radii,
        diffusivities,
        g_terms,
        speeds,
        settling_velocities,
        turbulent_dissipation,
        fluid_density,
        total_masses,
        charge,
        active_indices,
        collision_pairs,
        n_collisions,
        rng_states,
        mechanism_mask,
        collision_capacity,
        wp.tid(),
    )


@no_type_check
@wp.func
# type: ignore[misc]
def apply_coagulation_kernel_body(
    masses: Any,
    concentration: Any,
    charge: Any,
    collision_pairs: Any,
    n_collisions: Any,
    box_idx: wp.int32,
    collision_idx: wp.int32,
) -> None:
    """Apply representable coagulation merges to caller-owned state in place.

    A pair is applied only when the summed charge and every summed species mass
    are finite and nonnegative. Applied merges add donor mass and charge to the
    recipient, then clear the donor mass, concentration, and charge. A rejected
    unrepresentable pair leaves both particle inventories unchanged and is
    removed from the collision output during the following compaction pass.

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
def apply_coagulation_kernel(
    masses: Any,
    concentration: Any,
    charge: Any,
    collision_pairs: Any,
    n_collisions: Any,
) -> None:
    """Run the shared device operation through the direct kernel API."""
    box_idx, collision_idx = wp.tid()
    apply_coagulation_kernel_body(
        masses,
        concentration,
        charge,
        collision_pairs,
        n_collisions,
        box_idx,
        collision_idx,
    )


@no_type_check
@wp.kernel
def apply_coagulation_kernel_guarded(
    masses: Any,
    concentration: Any,
    charge: Any,
    collision_pairs: Any,
    n_collisions: Any,
    failure: wp.array(dtype=wp.int32),
) -> None:
    """Skip this resident writer after a latched simulation failure."""
    if failure[0] != 0:
        return
    box_idx, collision_idx = wp.tid()
    apply_coagulation_kernel_body(
        masses,
        concentration,
        charge,
        collision_pairs,
        n_collisions,
        box_idx,
        collision_idx,
    )


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


@no_type_check
@wp.kernel
def _compact_applied_collision_pairs_guarded(
    collision_pairs: Any,
    n_collisions: Any,
    failure: wp.array(dtype=wp.int32),
) -> None:
    """Remove rejected collision entries after the apply pass."""  # type: ignore
    if failure[0] != 0:
        return
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
    """Reject invalid charged particle state before allocating writer resources.

    Mass and concentration must be finite and nonnegative; density must be
    finite and positive. Physical population size is independent of work tiles.
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


def _validate_sedimentation_particle_physics(
    particles: Any,
    n_boxes: int,
    n_particles: int,
    device: Any,
) -> None:
    """Validate sedimentation state through a read-only device scan.

    Reject nonfinite charge, invalid mass/density/counts, unequal positive
    counts and invalid derived mass or volume. No active-population cap is
    imposed. Only compact error status is transferred to the host.
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
    if invalid_values[6] != 0:
        raise ValueError(
            "sedimentation active particle total mass and volume must be "
            "finite and > 0"
        )


def _validate_turbulent_particle_physics(
    particles: Any,
    n_boxes: int,
    device: Any,
) -> None:
    """Reject invalid ST1956 particle physics before mutable runtime work.

    Turbulent-shear selection uses the same physical particle fields as the
    other direct mechanisms, but does not impose their concentration uniformity
    or active-count constraints. It validates finite charge because the shared
    merge application consumes it, and rejects overflowed active mass/volume
    totals before rate sanitization could turn them into no-collision behavior.
    This read-only scan runs before time, environment, output, or RNG
    processing.
    """
    invalid = wp.zeros((1,), dtype=wp.int32, device=device)
    wp.launch(
        _validate_turbulent_particle_physics_kernel,
        dim=(n_boxes,),
        inputs=[
            particles.masses,
            particles.concentration,
            particles.density,
            particles.charge,
            invalid,
        ],
        device=device,
    )
    wp.synchronize_device(device)
    if invalid.numpy()[0] != 0:
        raise ValueError(
            "turbulent particle charge, masses, and concentrations must be "
            "finite; masses and concentrations must be nonnegative; density "
            "must be finite and > 0; active mass and volume must be finite "
            "and > 0"
        )


def _preflight_enabled_coagulation_terms(
    resolved: _ResolvedCoagulationMechanismConfig,
    particles: Any,
    n_boxes: int,
    n_particles: int,
    device: Any,
    turbulent_dissipation: Any,
    fluid_density: Any,
    validate_charge_finite: bool,
) -> None:
    """Run enabled-term validation before executable resource preparation."""
    mask = resolved.mask
    turbulent_enabled = bool(mask & TURBULENT_SHEAR_ST1956_MECHANISM_FLAG)
    charged_enabled = bool(mask & (CHARGED_HARD_SPHERE_MECHANISM_FLAG | 16))
    sedimentation_enabled = bool(mask & SEDIMENTATION_SP2016_MECHANISM_FLAG)

    if turbulent_enabled:
        _validate_turbulent_input(
            "turbulent_dissipation", turbulent_dissipation, n_boxes, device
        )
        _validate_turbulent_input(
            "fluid_density", fluid_density, n_boxes, device
        )
        _validate_turbulent_particle_physics(particles, n_boxes, device)
    if charged_enabled:
        if not sedimentation_enabled:
            _validate_charge_finite(
                particles.charge, n_boxes, n_particles, device
            )
        _validate_charged_particle_physics(
            particles, n_boxes, n_particles, device
        )
    if sedimentation_enabled:
        _validate_sedimentation_particle_physics(
            particles,
            n_boxes,
            n_particles,
            device,
        )
    if (
        validate_charge_finite
        and not charged_enabled
        and not sedimentation_enabled
    ):
        _validate_charge_finite(particles.charge, n_boxes, n_particles, device)


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
    byte_budget: int = MAX_COLLISION_PAIR_BUFFER_BYTES,
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
    capacity = min(requested_capacity, physical_capacity)
    required = n_boxes * capacity * bytes_per_box_collision
    if required > byte_budget:
        raise MemoryError(
            f"collision outputs require {required} bytes; budget is "
            f"{byte_budget} bytes. Increase the budget or reduce "
            "max_collisions."
        )
    return capacity


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


def _ensure_turbulent_input_array(
    name: str,
    value: Any,
    n_boxes: int,
    device: Any,
) -> Any:
    """Validate and normalize a turbulent-shear numerical validation per-box
        input.

    This private helper validates explicitly supplied state only; it neither
    infers turbulent properties from shared containers nor transfers host data.
    ``coagulation_step_gpu`` uses it for every executable mask containing
    ST1956 turbulent shear.

    Args:
        name: Input name used in contract errors.
        value: Positive finite floating scalar or Warp array. It represents
            turbulent dissipation [m^2/s^3] or fluid density [kg/m^3], according
            to ``name``.
        n_boxes: Number of boxes.
        device: Active particle device.

    Returns:
        A same-device Warp array with shape ``(n_boxes,)``. Valid supplied
        arrays are returned by identity; scalar broadcasts use private storage.

    Raises:
        ValueError: If the value is not a supported floating scalar or a
            same-device ``wp.float64`` Warp array with shape ``(n_boxes,)``
            and positive finite values.
    """
    _validate_turbulent_input(name, value, n_boxes, device)
    return _normalize_turbulent_input_array(value, n_boxes, device)


def _normalize_turbulent_input_array(
    value: Any,
    n_boxes: int,
    device: Any,
) -> Any:
    """Normalize an already-validated turbulent numerical validation input.

    This helper does not repeat validation.
    """
    if _is_warp_array_like(value):
        return value
    return _broadcast_scalar_array(float(value), n_boxes, device)


def _validate_turbulent_input(
    name: str,
    value: Any,
    n_boxes: int,
    device: Any,
) -> None:
    """Validate a turbulent-shear numerical validation input without scalar
        normalization.

    This validation-only boundary checks turbulent requests before allocating
    private broadcast storage.
    """
    if _is_warp_array_like(value):
        if value.shape != (n_boxes,):
            raise ValueError(f"{name} shape does not match (n_boxes,)")
        _validate_device_match(name, value, device)
        if value.dtype != wp.float64:
            raise ValueError(f"{name} must use dtype float64")
        _validate_positive_finite_array(name, value, "coagulation_step_gpu")
        return

    if isinstance(value, bool) or not isinstance(value, (float, np.floating)):
        if hasattr(value, "shape"):
            raise ValueError(
                f"{name} must be a Warp array with shape (n_boxes,)"
            )
        raise ValueError(f"{name} must be a floating scalar or Warp array")

    scalar_value = float(value)
    if not np.isfinite(scalar_value) or scalar_value <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")


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


@dataclass(frozen=True)
class _PreparedCoagulationCall:
    """Privately freeze validated coagulation inputs and scratch for enqueue.

    The carrier stores the exact caller-owned fields, outputs, device, and
    selector arguments established during preparation. Enqueue therefore does
    not repeat validation, allocation, normalization, RNG initialization, or
    resource lookup. Rebinding a source-container attribute cannot redirect an
    already prepared writer.

    Attributes:
        particles: Caller-owned particle container returned by the operation.
        masses: Frozen particle-mass array used by the apply kernel.
        concentration: Frozen particle-concentration array.
        charge: Frozen particle-charge array.
        collision_pairs: Collision-pair output buffer.
        n_collisions: Per-box collision-count output buffer.
        n_boxes: Number of simulation boxes.
        max_collisions: Effective collision capacity per box.
        device: Active Warp device for all launches.
        selector_inputs: Frozen arguments for the selector kernel.
    """

    particles: Any
    masses: Any
    concentration: Any
    charge: Any
    collision_pairs: Any
    n_collisions: Any
    n_boxes: int
    max_collisions: int
    device: Any
    selector_inputs: tuple[Any, ...]
    failure: Any = None
    brownian_parallel: Any = None
    work_config: Any = None
    adaptive: Any = None
    budget_report: Any = None
    budget_failure: Any = None
    rng_pending: Any = None
    rng_seed: int = 0
    work_counts: Any = None

    def check_admission(self) -> None:
        """Observe compact diagnostics at an explicit host boundary."""
        if (
            self.brownian_parallel.capture.x
            and self.budget_failure.numpy()[0] == 10
        ):
            raise ValueError("pair exceeds van der Waals capture table domain")
        raise_budget_error(self.budget_report)


def _prepare_coagulation_step_gpu(  # noqa: C901
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
    work_config: CoagulationWorkConfig | None = None,
    work_counts: Any | None = None,
    initialize_rng: bool = False,
    environment: Any | None = None,
    validate_charge_finite: bool = False,
    turbulent_dissipation: float | Any | None = None,
    fluid_density: float | Any | None = None,
) -> _PreparedCoagulationCall:
    """Prepare fixed resources for native equal-count resolved coagulation.

    All nonempty subsets of Brownian, charged Brownian, SP2016 sedimentation
    and ST1956 shear use tiled exact additive majorants. Particle counts have
    no algorithmic cap; slot indices remain signed int32 and pair ranks int64.
    Fixed buffers retain caller identity. This phase validates and allocates
    workspace but never resets supplied RNG or writes collision outputs.
    RNG initialization and all simulation writers follow per-step admission.

    The concrete work configuration controls trial and collision byte budgets.
    Direct calls report budget rejection after compact diagnostic readback;
    resident calls latch a device failure before all dependent writers.
    Particle concentration is represented count, volume is m^3, time is s,
    dissipation is m^2/s^3 and fluid density is kg/m^3. Positive represented
    counts must agree within each box. Turbulent inputs are required only for
    masks containing shear. Configuration validation precedes particle access.

    Returns:
        A frozen binding owning reusable property, reduction, admission and
        RNG-initialization scratch alongside caller-owned simulation arrays.
    """
    if work_config is None:
        work_config = CoagulationWorkConfig()
    if not isinstance(work_config, CoagulationWorkConfig):
        raise ValueError("work_config must be a CoagulationWorkConfig")
    if mechanism_config is None:
        mechanism_config = CoagulationMechanismConfig()
    if not isinstance(mechanism_config, CoagulationMechanismConfig):
        raise ValueError(
            "mechanism_config must be a CoagulationMechanismConfig."
        )
    resolved_mechanism_config = resolve_coagulation_mechanism_config(
        mechanism_config
    )
    turbulent_requested = bool(
        resolved_mechanism_config.mask & TURBULENT_SHEAR_ST1956_MECHANISM_FLAG
    )
    if not turbulent_requested:
        validate_coagulation_mechanism_capabilities(resolved_mechanism_config)

    population_code = particle_distribution_code(particles)
    if population_code != PARTICLE_RESOLVED:
        raise ValueError(
            "GPU coagulation supports only particle_resolved populations"
        )
    from . import _coagulation_native

    _coagulation_native.validate_schema(particles)
    if not isinstance(initialize_rng, (bool, np.bool_)):
        raise ValueError("initialize_rng must be Boolean")
    n_boxes, n_particles, n_species = particles.masses.shape
    validate_coagulation_layout(n_boxes, n_particles, n_species)
    _validate_particle_arrays(particles, n_boxes, n_particles, n_species)

    device = particles.masses.device
    _validate_device_arrays(particles, device)
    _preflight_enabled_coagulation_terms(
        resolved_mechanism_config,
        particles,
        n_boxes,
        n_particles,
        device,
        turbulent_dissipation,
        fluid_density,
        validate_charge_finite,
    )
    if turbulent_requested:
        validate_coagulation_mechanism_capabilities(resolved_mechanism_config)

    turbulent_enabled = bool(
        resolved_mechanism_config.mask & TURBULENT_SHEAR_ST1956_MECHANISM_FLAG
    )
    normalized_turbulent_dissipation = None
    normalized_fluid_density = None
    if turbulent_enabled:
        normalized_turbulent_dissipation = _normalize_turbulent_input_array(
            turbulent_dissipation,
            n_boxes,
            device,
        )
        normalized_fluid_density = _normalize_turbulent_input_array(
            fluid_density,
            n_boxes,
            device,
        )
    time_step_value = _validate_time_step(time_step)
    max_collisions_value = _resolve_collision_capacity(
        max_collisions=max_collisions,
        n_boxes=n_boxes,
        n_particles=n_particles,
        byte_budget=work_config.max_collision_buffer_bytes,
    )
    expected_counts_shape = (n_boxes,)
    initialize_rng_states_for_call = rng_states is None
    if rng_states is not None:
        _validate_rng_states(rng_states, expected_counts_shape, device)
        initialize_rng_states_for_call = initialize_rng
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
    _coagulation_native.validate_physics(particles, volume_array)

    expected_pairs_shape = (n_boxes, max_collisions_value, 2)
    if collision_pairs is not None:
        if len(collision_pairs.shape) != 3:
            raise ValueError("collision_pairs must have rank 3")
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

    if n_collisions is not None:
        _validate_collision_counts(n_collisions, expected_counts_shape, device)

    if work_counts is not None:
        if work_counts.shape != (n_boxes, 2) or work_counts.dtype != wp.int32:
            raise ValueError(
                "work_counts must be int32 with shape (n_boxes, 2)"
            )
        _validate_device_match("work_counts", work_counts, device)

    protected = [
        temperature_array,
        pressure_array,
        volume_array,
        normalized_turbulent_dissipation,
        normalized_fluid_density,
        temperature,
        pressure,
        volume,
    ]
    if turbulent_enabled:
        protected.extend((turbulent_dissipation, fluid_density))
    if environment is not None:
        protected.extend(
            (
                environment.temperature,
                environment.pressure,
                environment.saturation_ratio,
            )
        )
    _coagulation_native.validate_ownership(
        particles,
        collision_pairs,
        n_collisions,
        rng_states,
        tuple(protected),
        work_counts=work_counts,
    )

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

    if work_counts is None:
        work_counts = wp.zeros((n_boxes, 2), dtype=wp.int32, device=device)

    radii = wp.zeros((n_boxes, n_particles), dtype=wp.float64, device=device)
    # Parallel transport preparation writes each property independently.
    diffusivities = wp.zeros(
        (n_boxes, n_particles), dtype=wp.float64, device=device
    )
    g_terms = wp.zeros((n_boxes, n_particles), dtype=wp.float64, device=device)
    speeds = wp.zeros((n_boxes, n_particles), dtype=wp.float64, device=device)
    sedimentation_enabled = bool(
        resolved_mechanism_config.mask & SEDIMENTATION_SP2016_MECHANISM_FLAG
    )
    settling_velocities = radii
    if sedimentation_enabled:
        settling_velocities = wp.zeros(
            (n_boxes, n_particles), dtype=wp.float64, device=device
        )
    if turbulent_enabled:
        turbulent_dissipation_array = normalized_turbulent_dissipation
        fluid_density_array = normalized_fluid_density
    else:
        turbulent_dissipation_array = temperature_array
        fluid_density_array = pressure_array
    total_masses = wp.zeros(
        (n_boxes, n_particles), dtype=wp.float64, device=device
    )
    active_indices = wp.zeros(
        (n_boxes, n_particles), dtype=wp.int32, device=device
    )

    prepared = _PreparedCoagulationCall(
        particles,
        particles.masses,
        particles.concentration,
        particles.charge,
        collision_pairs,
        n_collisions,
        n_boxes,
        max_collisions_value,
        device,
        (
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
            turbulent_dissipation_array,
            fluid_density_array,
            total_masses,
            particles.charge,
            active_indices,
            collision_pairs,
            n_collisions,
            rng_states,
            wp.int32(
                resolved_mechanism_config.mask
                | (resolved_mechanism_config.charged_model << 8)
            ),
            wp.int32(max_collisions_value),
        ),
    )

    from particula.gpu.kernels._brownian_parallel import prepare_brownian_work

    capture = CaptureData()
    if resolved_mechanism_config.mask & 16:
        capture = prepare_capture(
            mechanism_config.hamaker_constant,
            mechanism_config.relative_permittivity,
            device,
        )
    return replace(
        prepared,
        work_config=work_config,
        budget_report=wp.zeros((n_boxes, 4), dtype=wp.float64, device=device),
        budget_failure=wp.zeros(1, dtype=wp.int32, device=device),
        rng_pending=wp.full(
            n_boxes,
            int(initialize_rng_states_for_call),
            dtype=wp.int32,
            device=device,
        ),
        rng_seed=rng_seed,
        work_counts=work_counts,
        brownian_parallel=prepare_brownian_work(
            prepared.selector_inputs, device, work_counts, capture
        ),
    )


def _enqueue_prepared_coagulation_call(
    prepared: _PreparedCoagulationCall,
) -> tuple[Any, Any, Any]:
    """Evaluate rates, admit all boxes, then enqueue guarded collision writers.

    Execution uses retained resources without allocation or host observation.
    The direct public wrapper observes compact budget diagnostics before
    returning; private prepared owners call check_admission explicitly.
    Failure preserves caller particle/output/RNG arrays for this operation;
    earlier successful operations are not rolled back.
    """
    failure = prepared.failure
    if failure is None:
        failure = prepared.budget_failure
    work = prepared.brownian_parallel
    if work is None:
        raise ValueError("coagulation requires prepared parallel resources")
    work.prepare_rates(failure)
    args = prepared.selector_inputs
    if prepared.adaptive is not None:
        prepared.adaptive.choose(prepared, failure)
    else:
        wp.launch(
            admit_coagulation_work,
            dim=prepared.n_boxes,
            inputs=[
                work.counts,
                args[24],
                args[1],
                args[3],
                work.majorants,
                args[14],
                prepared.work_config.max_trials_per_box,
                prepared.max_collisions,
                prepared.budget_report,
                failure,
            ],
            device=prepared.device,
        )
    wp.launch(
        initialize_admitted_rng,
        dim=prepared.n_boxes,
        inputs=[args[27], prepared.rng_pending, prepared.rng_seed, failure],
        device=prepared.device,
    )
    work.select(failure, prepared.adaptive.dt if prepared.adaptive else None)

    wp.launch(
        apply_coagulation_kernel_guarded,
        dim=(prepared.n_boxes, prepared.max_collisions),
        inputs=[
            prepared.masses,
            prepared.concentration,
            prepared.charge,
            prepared.collision_pairs,
            prepared.n_collisions,
            failure,
        ],
        device=prepared.device,
    )
    wp.launch(
        _compact_applied_collision_pairs_guarded,
        dim=prepared.n_boxes,
        inputs=[prepared.collision_pairs, prepared.n_collisions, failure],
        device=prepared.device,
    )
    return prepared.particles, prepared.collision_pairs, prepared.n_collisions


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
    work_config: CoagulationWorkConfig | None = None,
    work_counts: Any | None = None,
    initialize_rng: bool = False,
    environment: Any | None = None,
    validate_charge_finite: bool = False,
    turbulent_dissipation: float | Any | None = None,
    fluid_density: float | Any | None = None,
) -> tuple[Any, Any, Any]:
    """Prepare and enqueue one direct particle-resolved coagulation step.

    Preparation validates inputs and allocates workspace. Enqueue uses the
    frozen call record; optional RNG initialization follows successful device
    admission. This direct wrapper observes compact admission diagnostics.

    The optional ``work_counts`` buffer has shape (n_boxes, 2), dtype int32,
    and shares the particle device. Its columns contain scheduled and attempted
    candidate trials for this step. It must not overlap any other input/output.
    Admission rejection preserves it; successful calls overwrite every box.

    Returns:
        The particle container, collision-pair output, and per-box collision
        counts returned by the direct kernel.
    """
    prepared = _prepare_coagulation_step_gpu(
        particles,
        temperature,
        pressure,
        time_step,
        volume,
        max_collisions,
        rng_seed,
        collision_pairs,
        n_collisions,
        rng_states,
        mechanism_config=mechanism_config,
        work_config=work_config,
        work_counts=work_counts,
        initialize_rng=initialize_rng,
        environment=environment,
        validate_charge_finite=validate_charge_finite,
        turbulent_dissipation=turbulent_dissipation,
        fluid_density=fluid_density,
    )
    result = _enqueue_prepared_coagulation_call(prepared)
    prepared.check_admission()
    return result


def coagulation_preflight_gpu(
    particles: Any,
    temperature: float | Any | None,
    pressure: float | Any | None,
    time_step: object,
    **kwargs: Any,
) -> dict[str, Any]:
    """Inspect native capability, resources and trial demand without advancing.

    Accepts the direct step's keyword arguments. The report downloads compact
    per-box data, not particle arrays. Preparation allocates temporary workspace
    but leaves caller particle, collision-output and RNG buffers unchanged.
    Required bytes count unique bound collision, RNG, normalized environment,
    property, reduction and admission arrays (including caller-supplied
    sidecars). They exclude particle storage and runtime/compiler allocations;
    this is bound storage, not an estimate of incremental allocation.

    Returns:
        Report containing active counts, unordered pair evaluations, expected
        trials, conservative required trials, available trial capacity,
        budget-admissible timestep [s], per-box admission and workspace bytes.
        The admissible timestep is a resource bound, not an accuracy estimate.
    """
    prepared = _prepare_coagulation_step_gpu(
        particles, temperature, pressure, time_step, **kwargs
    )
    work = prepared.brownian_parallel
    work.prepare_rates(prepared.budget_failure)
    args = prepared.selector_inputs
    wp.launch(
        admit_coagulation_work,
        dim=prepared.n_boxes,
        inputs=[
            work.counts,
            args[24],
            args[1],
            args[3],
            work.majorants,
            args[14],
            prepared.work_config.max_trials_per_box,
            prepared.max_collisions,
            prepared.budget_report,
            prepared.budget_failure,
        ],
        device=prepared.device,
    )
    rows = prepared.budget_report.numpy()
    if work.capture.x and prepared.budget_failure.numpy()[0] == 10:
        raise ValueError("pair exceeds van der Waals capture table domain")
    counts = work.counts.numpy().astype(np.int64)
    arrays = [
        *args[3:6],
        *args[15:28],
        work.active,
        work.counts,
        work.partials,
        work.majorants,
        prepared.budget_report,
        prepared.budget_failure,
        prepared.rng_pending,
        prepared.work_counts,
    ]
    particle_pointers = {
        getattr(particles, name).ptr
        for name in ("masses", "concentration", "density", "volume", "charge")
    }
    if work.capture.x:
        arrays.append(work.capture.x)
        if work.capture.response:
            arrays.append(work.capture.response)
    unique = {
        array.ptr: array
        for array in arrays
        if array.ptr not in particle_pointers
    }
    return {
        "distribution_type": "particle_resolved",
        "requires_equal_active_counts": True,
        "active_counts": counts,
        "pair_evaluations": counts * (counts - 1) // 2,
        "expected_trials": rows[:, 0],
        "required_trials": rows[:, 1],
        "available_trials": rows[:, 2],
        "max_time_step": rows[:, 3],
        "admitted": np.isfinite(rows[:, 0]) & (rows[:, 1] <= rows[:, 2]),
        "workspace_and_collision_bytes": sum(
            array.capacity for array in unique.values()
        ),
    }


@dataclass(frozen=True)
class AdaptiveCoagulationResult:
    """Outer interval output with ordered pairs and per-box boundaries."""

    particles: Any
    collision_pairs: Any
    n_collisions: Any
    diagnostics: dict[str, Any]


def adaptive_coagulation_step_gpu(
    particles: Any,
    temperature: float | Any | None,
    pressure: float | Any | None,
    time_step: float,
    *,
    subcycling: CoagulationSubcyclingConfig,
    work_config: CoagulationWorkConfig | None = None,
    **kwargs: Any,
) -> AdaptiveCoagulationResult:
    """Run a synchronous, atomic direct interval on the particle device.

    Accepts the fixed call's optional inputs. Supplied collision and work
    buffers hold the final accepted pass. Returned device pairs span the whole
    interval in application order. Each diagnostics pair_end_offsets row gives
    the exclusive end of that pass per box. Reuse rng_states for continuation.
    """
    from functools import partial

    from particula.gpu.kernels._coagulation_subcycling import (
        AdaptiveInterval,
        adaptive_capacity,
        check_memory,
        interval_bytes,
        workspace_bytes,
    )

    if not isinstance(subcycling, CoagulationSubcyclingConfig):
        raise TypeError("expected CoagulationSubcyclingConfig")
    work_config = (
        work_config if work_config is not None else CoagulationWorkConfig()
    )
    if not isinstance(work_config, CoagulationWorkConfig):
        raise TypeError("expected CoagulationWorkConfig")
    from particula.gpu.kernels._coagulation_native import validate_schema

    validate_schema(particles)
    boxes, slots, _ = particles.masses.shape
    cap = adaptive_capacity(
        boxes, slots, work_config.max_collision_buffer_bytes
    )
    cap = min(cap, _validate_max_collisions(kwargs.pop("max_collisions", cap)))
    mechanism = kwargs.get("mechanism_config")
    sampler = workspace_bytes(
        boxes,
        slots,
        cap,
        getattr(mechanism, "relative_permittivity", None),
        getattr(mechanism, "hamaker_constant", None) is not None,
    )
    particle_bytes = sum(
        getattr(particles, name).capacity
        for name in ("masses", "charge", "concentration")
    )
    outputs = sum(
        getattr(kwargs.get(key), "capacity", 0)
        for key in (
            "collision_pairs",
            "work_counts",
            "n_collisions",
            "rng_states",
        )
    )
    categories = dict(
        sampler=sampler,
        controller=interval_bytes(boxes, slots, subcycling.max_subcycles),
        rollback=particle_bytes + outputs + boxes * (8 * cap + 16),
    )
    check_memory(subcycling, categories)
    prepared = _prepare_coagulation_step_gpu(
        particles,
        temperature,
        pressure,
        time_step,
        max_collisions=cap,
        work_config=work_config,
        **kwargs,
    )
    controller = AdaptiveInterval(prepared, subcycling, time_step)
    prepared = replace(prepared, adaptive=controller)
    outputs_to_restore = (
        prepared.masses,
        prepared.charge,
        prepared.concentration,
        prepared.collision_pairs,
        prepared.n_collisions,
        prepared.work_counts,
        prepared.selector_inputs[27],
    )
    backups = [(a, wp.clone(a)) for a in outputs_to_restore]
    failure = prepared.budget_failure
    controller.enqueue(
        prepared, failure, partial(_enqueue_prepared_coagulation_call, prepared)
    )
    code = int(failure.numpy()[0])
    if code:
        for array, backup in backups:
            wp.copy(array, backup)
        wp.synchronize_device(prepared.device)
        if code in (11, 13):
            raise controller.rejection(prepared, code)
        if code == 10:
            raise ValueError("pair exceeds van der Waals capture table domain")
        raise ValueError("invalid coagulation rate majorant")
    diagnostics = controller.diagnostics(prepared)
    diagnostics["reserved_bytes"] = categories
    return AdaptiveCoagulationResult(
        particles, controller.pairs, controller.offsets, diagnostics
    )
