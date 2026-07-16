"""GPU Brownian coagulation kernels and orchestration utilities.

This module composes Warp ``@wp.func`` building blocks into an end-to-end
coagulation pipeline. Entry-point validation accepts scalar direct inputs,
explicit ``(n_boxes,)`` Warp arrays, or a ``WarpEnvironmentData`` container.
Those sources are normalized into per-box Warp arrays before volume setup,
RNG initialization, or any Warp launch. The kernels operate on GPU-resident
particle data and produce collision pairs that are applied to merge particle
masses in-place.
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
    g_collection_term_wp,
    particle_mean_free_path_wp,
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
    """Configure P1 host-side coagulation mechanism validation.

    The default selects Brownian, particle-resolved coagulation. This frozen
    configuration is concrete-module-only and is not an argument of the public
    ``coagulation_step_gpu`` API in P1.

    Attributes:
        mechanisms: Requested canonical mechanism identifiers, or ``None`` to
            select Brownian.
        distribution_type: Required distribution representation; only
            ``"particle_resolved"`` is structurally supported in P1.
    """

    mechanisms: tuple[str, ...] | None = None
    distribution_type: str = "particle_resolved"


@dataclass(frozen=True)
class _ResolvedCoagulationMechanismConfig:
    """Store the normalized result of concrete-module P1 validation.

    This private result is not a public API or a ``coagulation_step_gpu``
    argument. Structural resolution retains recognized reserved mechanisms for
    later capability validation.

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
    """Resolve a configuration through concrete-module P1 structural validation.

    ``None`` defaults to Brownian. Valid identifiers are normalized to canonical
    order and retained in the returned fixed-bit mask, including reserved terms
    that the separate P1 capability gate rejects. This pure host-side helper
    neither allocates device storage nor mutates its input or runtime state; it
    is not part of the public ``coagulation_step_gpu`` API.

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
    """Enforce the concrete-module P1 executable-mechanism boundary.

    This pure host-side, concrete-module-only gate accepts Brownian execution
    only. It rejects structurally valid reserved terms while preserving their
    resolved flags for their owning implementation tracks. It neither mutates
    state nor integrates with the public ``coagulation_step_gpu`` API in P1.

    Args:
        resolved: Structurally validated, normalized mechanism configuration.

    Raises:
        ValueError: If ``charged_hard_sphere``, ``sedimentation_sp2016``, or
            ``turbulent_shear_st1956`` is requested before its owning track
            E5-F3, E5-F4, or E5-F5, respectively, is available.
    """
    reserved_messages = {
        CHARGED_HARD_SPHERE_MECHANISM: (
            "Coagulation mechanism 'charged_hard_sphere' is reserved for E5-F3."
        ),
        SEDIMENTATION_SP2016_MECHANISM: (
            "Coagulation mechanism 'sedimentation_sp2016' is reserved for "
            "E5-F4."
        ),
        TURBULENT_SHEAR_ST1956_MECHANISM: (
            "Coagulation mechanism 'turbulent_shear_st1956' is reserved for "
            "E5-F5."
        ),
    }
    for mechanism in resolved.mechanisms:
        if mechanism in reserved_messages:
            raise ValueError(reserved_messages[mechanism])


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
    """Return finite, strictly positive terms or zero for safe accumulation."""
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
) -> Any:
    """Sum enabled finite pair-rate terms using fixed mechanism branches."""
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
    # Reserved mechanism bits deliberately contribute no executable term.
    return total_rate


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
) -> Any:
    """Sum enabled finite majorant terms using fixed mechanism branches."""
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
    # Reserved mechanism bits deliberately contribute no executable term.
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
    time_step: Any,
    radii: Any,
    diffusivities: Any,
    g_terms: Any,
    speeds: Any,
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
    inactive slots.

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
        time_step: Coagulation time step [s].
        radii: Output particle radii ``(n_boxes, n_particles)``.
        diffusivities: Output diffusivities ``(n_boxes, n_particles)``.
        g_terms: Output collection terms ``(n_boxes, n_particles)``.
        speeds: Output mean thermal speeds ``(n_boxes, n_particles)``.
        active_indices: Output compact active indices
            ``(n_boxes, n_particles)``.
        collision_pairs: Output collision indices
            ``(n_boxes, max_collisions, 2)``.
        n_collisions: Output collision counts ``(n_boxes,)``.
        rng_states: Per-box RNG states ``(n_boxes,)`` mutated in place during
            pair selection. Reusing this buffer across calls preserves
            caller-owned persistent state unless it is reset before launch.
        mechanism_mask: Fixed internal mechanism-dispatch mask.
        collision_capacity: Maximum accepted collisions per box for this call.
    """  # type: ignore
    box_idx = wp.tid()  # type: ignore[misc]

    # One thread per box keeps the pair selection sequential and avoids
    # cross-thread races when writing collision pairs.
    n_particles = masses.shape[1]
    n_species = masses.shape[2]

    temperature_value = temperature[box_idx]
    pressure_value = pressure[box_idx]

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
        if concentration[box_idx, particle_idx] <= wp.float64(0.0):
            active_indices[box_idx, particle_idx] = wp.int32(-1)
            radii[box_idx, particle_idx] = wp.float64(0.0)
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
            diffusivities[box_idx, particle_idx] = wp.float64(0.0)
            g_terms[box_idx, particle_idx] = wp.float64(0.0)
            speeds[box_idx, particle_idx] = wp.float64(0.0)
            continue

        radius = particle_radius_from_volume_wp(total_volume)
        knudsen = knudsen_number_wp(mean_free_path, radius)
        slip = cunningham_slip_correction_wp(knudsen)
        mobility = aerodynamic_mobility_wp(radius, slip, dynamic_viscosity)
        diffusivity = brownian_diffusivity_wp(
            temperature_value, mobility, boltzmann_constant
        )
        speed = mean_thermal_speed_wp(
            total_mass, temperature_value, boltzmann_constant
        )
        particle_mean_free_path = particle_mean_free_path_wp(diffusivity, speed)
        g_term = g_collection_term_wp(particle_mean_free_path, radius)

        active_indices[box_idx, active_count] = wp.int32(particle_idx)
        radii[box_idx, particle_idx] = radius
        diffusivities[box_idx, particle_idx] = diffusivity
        g_terms[box_idx, particle_idx] = g_term
        speeds[box_idx, particle_idx] = speed
        active_count += wp.int32(1)

    if active_count < wp.int32(2):
        n_collisions[box_idx] = wp.int32(0)
        return

    # Find particles with min/max radius for the one total-rate majorant.
    # The Brownian kernel is maximized at the greatest size disparity
    # (small particle has high diffusivity, large particle has large
    # cross-section), so K(r_min, r_max) bounds all pairwise values.
    # This replaces the previous O(n^2) all-pairs scan with O(n).
    r_min = wp.float64(1.0e30)
    r_max = wp.float64(0.0)
    idx_min = wp.int32(-1)
    idx_max = wp.int32(-1)
    for active_rank in range(active_count):
        p_idx = wp.int32(active_indices[box_idx, active_rank])
        r_p = radii[box_idx, p_idx]
        if r_p < r_min:
            r_min = r_p
            idx_min = wp.int32(p_idx)
        if r_p > r_max:
            r_max = r_p
            idx_max = wp.int32(p_idx)

    majorant_total = wp.float64(0.0)
    if idx_min >= wp.int32(0) and idx_max >= wp.int32(0):
        if idx_min == idx_max:
            # All active particles have the same radius; use self-pair
            # kernel which is symmetric, so pick any two active indices.
            for active_rank in range(active_count):
                candidate_idx = wp.int32(active_indices[box_idx, active_rank])
                if candidate_idx != idx_min:
                    idx_max = candidate_idx
                    break
        majorant_total = _total_majorant(
            mechanism_mask,
            radii[box_idx, idx_min],
            radii[box_idx, idx_max],
            diffusivities[box_idx, idx_min],
            diffusivities[box_idx, idx_max],
            g_terms[box_idx, idx_min],
            g_terms[box_idx, idx_max],
            speeds[box_idx, idx_min],
            speeds[box_idx, idx_max],
        )

    if not (wp.isfinite(majorant_total) and majorant_total > wp.float64(0.0)):
        n_collisions[box_idx] = wp.int32(0)
        return

    possible_pairs = (
        wp.float64(active_count)
        * wp.float64(active_count - 1)
        / (wp.float64(2.0))
    )
    expected_trials = (
        majorant_total * possible_pairs * time_step / volume[box_idx]
    )
    if not (wp.isfinite(expected_trials) and expected_trials > wp.float64(0.0)):
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
        )
        if not (wp.isfinite(total_rate) and total_rate > wp.float64(0.0)):
            continue
        if total_rate > majorant_total:
            continue

        # Every valid candidate gets exactly one acceptance draw.
        if wp.randf(state) < total_rate / majorant_total:
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
    collision_pairs: Any,
    n_collisions: Any,
) -> None:
    """Apply coagulation collisions by merging particle masses.

    Args:
        masses: Particle masses array ``(n_boxes, n_particles, n_species)``.
        concentration: Particle concentrations ``(n_boxes, n_particles)``.
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
        return

    n_species = masses.shape[2]
    for species_idx in range(n_species):
        masses[box_idx, idx_i, species_idx] = (
            masses[box_idx, idx_i, species_idx]
            + masses[box_idx, idx_j, species_idx]
        )
        masses[box_idx, idx_j, species_idx] = wp.float64(0.0)

    concentration[box_idx, idx_j] = wp.float64(0.0)


def _validate_particle_arrays(
    particles: Any,
    n_boxes: int,
    n_particles: int,
    n_species: int,
) -> None:
    """Validate particle array shapes and lengths.

    Args:
        particles: GPU particle data container.
        n_boxes: Expected number of boxes.
        n_particles: Expected number of particles per box.
        n_species: Expected number of species.

    Raises:
        ValueError: If particle arrays do not match expected shapes.
    """
    if particles.masses.shape != (n_boxes, n_particles, n_species):
        raise ValueError("particle masses shape does not match expected")
    if particles.density.shape[0] != n_species:
        raise ValueError("particle density length does not match n_species")
    if particles.concentration.shape != (n_boxes, n_particles):
        raise ValueError(
            "particle concentration shape does not match (n_boxes, n_particles)"
        )
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
    """Validate particle arrays share the same Warp device.

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
    _validate_device_match("particle density", particles.density, device)
    _validate_device_match("particle volume", particles.volume, device)


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

    if hasattr(volume, "shape"):
        raise ValueError("volume must be a Warp array with shape (n_boxes,)")

    if isinstance(volume, bool) or not isinstance(volume, (float, np.floating)):
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
    initialize_rng: bool = False,
    environment: Any | None = None,
) -> tuple[Any, Any, Any]:
    """Execute one Brownian coagulation timestep on the GPU.

    Direct temperature and pressure inputs are validated and normalized before
    volume setup, RNG initialization, or any Warp kernel launch. Caller-owned
    RNG buffers are only reset when ``initialize_rng=True`` explicitly opts in
    to reinitialization.

    Args:
        particles: GPU-resident particle data.
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
        collision_pairs: Optional preallocated collision buffer.
        n_collisions: Optional preallocated collision count buffer.
        rng_states: Optional preallocated RNG state buffer. When omitted, this
            function allocates a call-local ``(n_boxes,)`` buffer, seeds it
            from ``rng_seed``, and uses it only for the current call. When
            provided, the caller owns the persistent GPU-resident sidecar
            buffer and it is reused as-is across repeated calls unless
            ``initialize_rng=True`` explicitly requests a reset from
            ``rng_seed``.
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

    Notes:
        Accepted environment sources are scalar direct inputs, direct
        ``(n_boxes,)`` Warp arrays, hybrid scalar-plus-Warp-array direct
        inputs, or keyword-only ``environment=...`` execution.

        ``initialize_rng`` and ``environment`` remain keyword-only so existing
        positional scalar callers stay source-compatible.

        Validation runs before volume normalization, RNG setup, and kernel
        launches so invalid ``time_step``, shape, dtype, or device
        combinations fail without mutating particle state or allocating
        downstream launch work. The normalized environment arrays are
        forwarded directly into the launch path.

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
        ``initialize_rng=True`` is passed. For repeated timesteps or graph
        capture setup, initialize caller-owned buffers once before the loop or
        before capture, then reuse them without hidden per-step resets.
    """
    n_boxes, n_particles, n_species = particles.masses.shape
    _validate_particle_arrays(particles, n_boxes, n_particles, n_species)

    device = particles.masses.device
    _validate_device_arrays(particles, device)
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
    diffusivities = wp.zeros(
        (n_boxes, n_particles), dtype=wp.float64, device=device
    )
    g_terms = wp.zeros((n_boxes, n_particles), dtype=wp.float64, device=device)
    speeds = wp.zeros((n_boxes, n_particles), dtype=wp.float64, device=device)
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
            wp.float64(time_step_value),
            radii,
            diffusivities,
            g_terms,
            speeds,
            active_indices,
            collision_pairs,
            n_collisions,
            rng_states,
            wp.int32(BROWNIAN_MECHANISM_FLAG),
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
            collision_pairs,
            n_collisions,
        ],
        device=device,
    )

    return particles, collision_pairs, n_collisions
