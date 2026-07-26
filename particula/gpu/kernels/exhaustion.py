"""Provide fixed-capacity direct Warp exhaustion helpers.

This concrete-module-only boundary consumes already-resolved per-box release
counts. It has no policy resolution, CPU fallback, host particle transfer,
resizing, or runnable wrapper. Callers own every particle-scale planning and
diagnostic buffer. Read-only preflight precedes planning; planning can replace
documented buffer contents, and one commit launches only after every box passes
its diagnostics.

The planner uses a deterministic bitonic sorting network and a linear
equal-weight interval sweep. Sorting costs ``O(B * N * log2(N)**2)``
compare-exchanges; all other planning and commit work costs ``O(B * N * S)``.
The caller supplies its entire particle-scale scratch footprint through
:class:`ResamplingBuffers`.

The concrete-only P4 representative-volume helper validates all caller-owned
state before writing its diagnostic or scaling selected rows. It writes the
resolved requested factor for selected rows and ``1.0`` otherwise. It has no
package export, policy, transfer, resizing, or runnable integration; callers
own device placement and explicitly synchronize before observing results.
"""

# mypy: disable-error-code="valid-type, misc, operator"

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

try:
    import warp as wp
except ImportError as exc:  # pragma: no cover - guarded by Warp tests
    raise ImportError(
        "Warp is required for GPU exhaustion helpers. "
        "Install with: pip install warp-lang"
    ) from exc

from particula.gpu.kernels.environment import _is_warp_array_like

PLANNING_SUCCESS = 0
PLANNING_EXACT_INVENTORY_FAILURE = 1
PLANNING_RADIUS_CUBED_FAILURE = 2
PLANNING_MEAN_RADIUS_FAILURE = 3
PLANNING_SURFACE_FAILURE = 4
PLANNING_DIVERSITY_FAILURE = 5
PLANNING_NONFINITE_FAILURE = 6


@dataclass(frozen=True)
class ResamplingBuffers:
    """Hold caller-owned fixed-shape Warp storage for resampling.

    The record is concrete-module-only and does not bind a reusable CPU plan
    to particle state. Every field must be a distinct, same-device Warp array
    with the shape implied by particle boxes ``B``, capacity ``N``, and species
    count ``S``. Successful nonzero-demand planning initializes every output
    and scratch lane deterministically; zero-demand buffer lanes are unchanged.

    Attributes:
        retained_counts: ``int32`` array shaped ``(B,)`` of planned retained
            active slots.
        released_counts: ``int32`` array shaped ``(B,)`` of planned releases.
        retained_indices: ``int32`` array shaped ``(B, N)`` of retained
            original slot indices, with unused lanes set to ``-1``.
        released_indices: ``int32`` array shaped ``(B, N)`` of released
            original slot indices, with unused lanes set to ``-1``.
        sorted_indices: ``int32`` array shaped ``(B, N)`` used for the sorted
            source sequence, with unused lanes set to ``-1``.
        replacement_masses: ``float64`` array shaped ``(B, N, S)`` containing
            planned per-particle species masses; unused rows are zero.
        replacement_concentration: ``float64`` array shaped ``(B, N)`` of
            planned concentrations; unused lanes are zero.
        replacement_charge: ``float64`` array shaped ``(B, N)`` of planned
            charges; unused lanes are zero.
        source_radii: ``float64`` array shaped ``(B, N)`` for planning radii.
        radius_cubed_relative_error: ``float64`` array shaped ``(B,)`` of
            radius-cubed diagnostic errors.
        mean_radius_relative_error: ``float64`` array shaped ``(B,)`` of
            mean-radius diagnostic errors.
        surface_relative_error: ``float64`` array shaped ``(B,)`` of surface
            area diagnostic errors.
        diversity_absolute_error: ``float64`` array shaped ``(B,)`` of
            Riemer-diversity diagnostic errors.
        planning_status: ``int32`` array shaped ``(B,)`` with ``0`` for
            success, ``1`` for inventory failure, ``2`` through ``5`` for
            radius-cubed, mean-radius, surface, and diversity failures, and
            ``6`` for non-finite planning results.
    """

    retained_counts: Any
    released_counts: Any
    retained_indices: Any
    released_indices: Any
    sorted_indices: Any
    replacement_masses: Any
    replacement_concentration: Any
    replacement_charge: Any
    source_radii: Any
    radius_cubed_relative_error: Any
    mean_radius_relative_error: Any
    surface_relative_error: Any
    diversity_absolute_error: Any
    planning_status: Any


@wp.func
def _radius(
    masses: wp.array3d(dtype=wp.float64),
    density: wp.array(dtype=wp.float64),
    box: Any,
    particle: Any,
    species_count: Any,
) -> wp.float64:
    volume = wp.float64(0.0)
    for species in range(species_count):
        volume += masses[box, particle, species] / density[species]
    return wp.pow(
        wp.float64(3.0) * volume / (wp.float64(4.0) * wp.float64(wp.pi)),
        wp.float64(1.0) / wp.float64(3.0),
    )


@wp.func
def _less_source(
    masses: wp.array3d(dtype=wp.float64),
    charge: wp.array2d(dtype=wp.float64),
    source_radii: wp.array2d(dtype=wp.float64),
    box: Any,
    left: Any,
    right: Any,
    species_count: Any,
) -> bool:
    """Compare the CPU P2 key, including original index tie breaking."""
    if left < 0:
        return False
    if right < 0:
        return True
    left_radius = source_radii[box, left]
    right_radius = source_radii[box, right]
    if left_radius != right_radius:
        return left_radius < right_radius
    left_total = wp.float64(0.0)
    right_total = wp.float64(0.0)
    for species in range(species_count):
        left_total += masses[box, left, species]
        right_total += masses[box, right, species]
    for species in range(species_count):
        left_fraction = masses[box, left, species] / left_total
        right_fraction = masses[box, right, species] / right_total
        if left_fraction != right_fraction:
            return left_fraction < right_fraction
    if charge[box, left] != charge[box, right]:
        return charge[box, left] < charge[box, right]
    return left < right


@wp.func
def _riemer_diversity_scattered(  # noqa: C901
    masses: wp.array3d(dtype=wp.float64),
    concentration: wp.array2d(dtype=wp.float64),
    indices: wp.array2d(dtype=wp.int32),
    box: Any,
    count: Any,
    species_count: Any,
) -> wp.float64:
    """Compute Riemer diversity over an indexed active source sequence."""
    total_mass = wp.float64(0.0)
    for item in range(count):
        particle = indices[box, item]
        particle_mass = wp.float64(0.0)
        for species in range(species_count):
            particle_mass += masses[box, particle, species]
        total_mass += concentration[box, particle] * particle_mass
    if not wp.isfinite(total_mass) or total_mass <= 0.0:
        return wp.float64(0.0)

    entropy = wp.float64(0.0)
    for item in range(count):
        particle = indices[box, item]
        particle_mass = wp.float64(0.0)
        for species in range(species_count):
            particle_mass += masses[box, particle, species]
        represented_mass = concentration[box, particle] * particle_mass
        if represented_mass <= 0.0:
            continue
        particle_fraction = represented_mass / total_mass
        if particle_fraction <= 0.0:
            continue
        per_particle_entropy = wp.float64(0.0)
        for species in range(species_count):
            species_mass = masses[box, particle, species]
            if species_mass <= 0.0 or particle_mass <= 0.0:
                continue
            species_fraction = species_mass / particle_mass
            if species_fraction > 0.0:
                per_particle_entropy -= species_fraction * wp.log(
                    species_fraction
                )
        entropy += particle_fraction * per_particle_entropy
    if not wp.isfinite(entropy):
        return wp.float64(0.0)
    return wp.exp(entropy)


@wp.func
def _riemer_diversity_dense(  # noqa: C901
    masses: wp.array3d(dtype=wp.float64),
    concentration: wp.array2d(dtype=wp.float64),
    box: Any,
    count: Any,
    species_count: Any,
) -> wp.float64:
    """Compute Riemer diversity over a dense replacement prefix."""
    total_mass = wp.float64(0.0)
    for item in range(count):
        particle_mass = wp.float64(0.0)
        for species in range(species_count):
            particle_mass += masses[box, item, species]
        total_mass += concentration[box, item] * particle_mass
    if not wp.isfinite(total_mass) or total_mass <= 0.0:
        return wp.float64(0.0)

    entropy = wp.float64(0.0)
    for item in range(count):
        particle_mass = wp.float64(0.0)
        for species in range(species_count):
            particle_mass += masses[box, item, species]
        represented_mass = concentration[box, item] * particle_mass
        if represented_mass <= 0.0:
            continue
        particle_fraction = represented_mass / total_mass
        if particle_fraction <= 0.0:
            continue
        per_particle_entropy = wp.float64(0.0)
        for species in range(species_count):
            species_mass = masses[box, item, species]
            if species_mass <= 0.0 or particle_mass <= 0.0:
                continue
            species_fraction = species_mass / particle_mass
            if species_fraction > 0.0:
                per_particle_entropy -= species_fraction * wp.log(
                    species_fraction
                )
        entropy += particle_fraction * per_particle_entropy
    if not wp.isfinite(entropy):
        return wp.float64(0.0)
    return wp.exp(entropy)


@wp.func
def _lane_read(
    primary: wp.array2d(dtype=wp.int32),
    secondary: wp.array2d(dtype=wp.int32),
    box: Any,
    lane: Any,
    primary_size: Any,
) -> int:
    """Read one logical sort lane from the combined index workspace."""
    if lane < primary_size:
        return primary[box, lane]
    return secondary[box, lane - primary_size]


@wp.func
def _lane_write(
    primary: wp.array2d(dtype=wp.int32),
    secondary: wp.array2d(dtype=wp.int32),
    box: Any,
    lane: Any,
    primary_size: Any,
    value: Any,
) -> None:
    """Write one logical sort lane into the combined index workspace."""
    if lane < primary_size:
        primary[box, lane] = value
    else:
        secondary[box, lane - primary_size] = value


@wp.kernel
def _scan_particle_values(  # noqa: C901
    masses: wp.array3d(dtype=wp.float64),
    concentration: wp.array2d(dtype=wp.float64),
    charge: wp.array2d(dtype=wp.float64),
    density: wp.array(dtype=wp.float64),
    volume: wp.array(dtype=wp.float64),
    invalid: wp.array(dtype=wp.int32),
) -> None:
    """Perform one read-only physical-state validation scan."""
    box, particle, species = wp.tid()
    if (
        not wp.isfinite(masses[box, particle, species])
        or masses[box, particle, species] < 0.0
    ):
        wp.atomic_add(invalid, 0, 1)
    if particle == 0 and (not wp.isfinite(volume[box]) or volume[box] <= 0.0):
        wp.atomic_add(invalid, 0, 1)
    if species == 0:
        if (
            not wp.isfinite(concentration[box, particle])
            or concentration[box, particle] < 0.0
        ):
            wp.atomic_add(invalid, 0, 1)
        if not wp.isfinite(charge[box, particle]):
            wp.atomic_add(invalid, 0, 1)
        if concentration[box, particle] == 0.0:
            if charge[box, particle] != 0.0:
                wp.atomic_add(invalid, 0, 1)


@wp.kernel
def _scan_density(
    density: wp.array(dtype=wp.float64),
    invalid: wp.array(dtype=wp.int32),
) -> None:
    """Validate global density when a batch has no particle rows."""
    density_index = wp.tid()
    if not wp.isfinite(density[density_index]) or density[density_index] <= 0.0:
        wp.atomic_add(invalid, 0, 1)


@wp.kernel
def _scan_slot_state(
    masses: wp.array3d(dtype=wp.float64),
    concentration: wp.array2d(dtype=wp.float64),
    density: wp.array(dtype=wp.float64),
    invalid: wp.array(dtype=wp.int32),
) -> None:
    """Validate inactive literal-zero state and active material volume."""
    box, particle = wp.tid()
    material_volume = wp.float64(0.0)
    any_mass = wp.float64(0.0)
    for species in range(density.shape[0]):
        material_volume += masses[box, particle, species] / density[species]
        any_mass += masses[box, particle, species]
    if concentration[box, particle] == 0.0 and any_mass != 0.0:
        wp.atomic_add(invalid, 0, 1)
    if concentration[box, particle] > 0.0 and (
        not wp.isfinite(material_volume) or material_volume <= 0.0
    ):
        wp.atomic_add(invalid, 0, 1)


@wp.kernel
def _validate_counts(
    concentration: wp.array2d(dtype=wp.float64),
    counts: wp.array(dtype=wp.int32),
    invalid: wp.array(dtype=wp.int32),
) -> None:
    """Validate each resolved nonzero release demand against active capacity."""
    box = cast(int, wp.tid())
    active = int(0)
    for particle in range(concentration.shape[1]):
        if concentration[box, particle] > 0.0:
            active += 1
    if counts[box] < 0 or (counts[box] != 0 and counts[box] >= active):
        wp.atomic_add(invalid, 0, 1)


@wp.kernel
def _plan_resampling(  # noqa: C901, PLR0915
    masses: wp.array3d(dtype=wp.float64),
    concentration: wp.array2d(dtype=wp.float64),
    charge: wp.array2d(dtype=wp.float64),
    density: wp.array(dtype=wp.float64),
    counts: wp.array(dtype=wp.int32),
    retained_counts: wp.array(dtype=wp.int32),
    released_counts: wp.array(dtype=wp.int32),
    retained_indices: wp.array2d(dtype=wp.int32),
    released_indices: wp.array2d(dtype=wp.int32),
    sorted_indices: wp.array2d(dtype=wp.int32),
    replacement_masses: wp.array3d(dtype=wp.float64),
    replacement_concentration: wp.array2d(dtype=wp.float64),
    replacement_charge: wp.array2d(dtype=wp.float64),
    source_radii: wp.array2d(dtype=wp.float64),
    radius_error: wp.array(dtype=wp.float64),
    mean_error: wp.array(dtype=wp.float64),
    surface_error: wp.array(dtype=wp.float64),
    diversity_error: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
    radius_bound: wp.float64,
    mean_bound: wp.float64,
    surface_bound: wp.float64,
    diversity_bound: wp.float64,
) -> None:
    """Plan one box sequentially; output rows remain caller-owned storage."""
    box = wp.tid()
    if counts[box] == 0:
        return
    n_particles = concentration.shape[1]
    n_species = density.shape[0]
    active = int(0)
    for particle in range(n_particles):
        retained_indices[box, particle] = -1
        released_indices[box, particle] = -1
        sorted_indices[box, particle] = -1
        source_radii[box, particle] = wp.float64(0.0)
        replacement_concentration[box, particle] = wp.float64(0.0)
        replacement_charge[box, particle] = wp.float64(0.0)
        for species in range(n_species):
            replacement_masses[box, particle, species] = wp.float64(0.0)
        if concentration[box, particle] > 0.0:
            _lane_write(
                sorted_indices,
                released_indices,
                box,
                active,
                n_particles,
                particle,
            )
            source_radii[box, particle] = _radius(
                masses, density, box, particle, n_species
            )
            active += 1
    kept = active - counts[box]
    retained_counts[box] = kept
    released_counts[box] = counts[box]
    # Bitonic compare-exchange network.  Inactive lanes are greater-than-all
    # sentinels, so active sources finish in ascending CPU P2 key order.
    padded = int(1)
    while padded < n_particles:
        padded *= 2
    span = int(2)
    while span <= padded:
        stride = span // 2
        while stride > 0:
            for position in range(padded):
                partner = position ^ stride
                if partner > position and partner < padded:
                    ascending = (position & span) == 0
                    left = _lane_read(
                        sorted_indices,
                        released_indices,
                        box,
                        position,
                        n_particles,
                    )
                    right = _lane_read(
                        sorted_indices,
                        released_indices,
                        box,
                        partner,
                        n_particles,
                    )
                    swap = _less_source(
                        masses,
                        charge,
                        source_radii,
                        box,
                        right if ascending else left,
                        left if ascending else right,
                        n_species,
                    )
                    if swap:
                        _lane_write(
                            sorted_indices,
                            released_indices,
                            box,
                            position,
                            n_particles,
                            right,
                        )
                        _lane_write(
                            sorted_indices,
                            released_indices,
                            box,
                            partner,
                            n_particles,
                            left,
                        )
            stride //= 2
        span *= 2
    for particle in range(n_particles):
        sorted_indices[box, particle] = _lane_read(
            sorted_indices,
            released_indices,
            box,
            particle,
            n_particles,
        )
    total_number = wp.float64(0.0)
    for source in range(active):
        total_number += concentration[box, sorted_indices[box, source]]
    equal = total_number / wp.float64(kept)
    source = int(0)
    source_end = concentration[box, sorted_indices[box, 0]]
    for target in range(kept):
        target_start = wp.float64(target) * equal
        target_end = wp.float64(target + 1) * equal
        while source < active and source_end <= target_start:
            source += 1
            if source < active:
                source_end += concentration[box, sorted_indices[box, source]]
        cursor = target_start
        while cursor < target_end and source < active:
            overlap_end = wp.min(target_end, source_end)
            overlap = overlap_end - cursor
            source_particle = sorted_indices[box, source]
            for species in range(n_species):
                replacement_masses[box, target, species] += (
                    overlap * masses[box, source_particle, species] / equal
                )
            replacement_charge[box, target] += (
                overlap * charge[box, source_particle] / equal
            )
            cursor = overlap_end
            if cursor >= source_end:
                source += 1
                if source < active:
                    source_end += concentration[
                        box, sorted_indices[box, source]
                    ]
        replacement_concentration[box, target] = equal
    current = int(0)
    for particle in range(n_particles):
        if concentration[box, particle] > 0.0:
            if current < kept:
                retained_indices[box, current] = particle
            else:
                released_indices[box, current - kept] = particle
            current += 1
    for particle in range(current, n_particles):
        released_indices[box, particle] = -1
    source_number = wp.float64(0.0)
    source_charge_total = wp.float64(0.0)
    replacement_number = wp.float64(0.0)
    replacement_charge_total = wp.float64(0.0)
    source_moment = wp.float64(0.0)
    source_mean = wp.float64(0.0)
    source_surface = wp.float64(0.0)
    replacement_moment = wp.float64(0.0)
    replacement_mean = wp.float64(0.0)
    replacement_surface = wp.float64(0.0)
    replacement_values_finite = bool(True)
    source_species_count = int(0)
    source_diversity = _riemer_diversity_scattered(
        masses, concentration, sorted_indices, box, active, n_species
    )
    for item in range(active):
        particle = sorted_indices[box, item]
        source_number += concentration[box, particle]
        source_charge_total += (
            concentration[box, particle] * charge[box, particle]
        )
        radius = source_radii[box, particle]
        source_moment += concentration[box, particle] * radius * radius * radius
        source_mean += concentration[box, particle] * radius
        source_surface += (
            concentration[box, particle]
            * wp.float64(4.0)
            * wp.float64(wp.pi)
            * radius
            * radius
        )
    for item in range(kept):
        if not wp.isfinite(
            replacement_concentration[box, item]
        ) or not wp.isfinite(replacement_charge[box, item]):
            replacement_values_finite = False
        replacement_number += replacement_concentration[box, item]
        replacement_charge_total += (
            replacement_concentration[box, item] * replacement_charge[box, item]
        )
        volume = wp.float64(0.0)
        for species in range(n_species):
            if not wp.isfinite(replacement_masses[box, item, species]):
                replacement_values_finite = False
            volume += replacement_masses[box, item, species] / density[species]
        radius = wp.pow(
            wp.float64(3.0) * volume / (wp.float64(4.0) * wp.float64(wp.pi)),
            wp.float64(1.0) / wp.float64(3.0),
        )
        replacement_moment += equal * radius * radius * radius
        replacement_mean += equal * radius
        replacement_surface += (
            equal * wp.float64(4.0) * wp.float64(wp.pi) * radius * radius
        )
    replacement_diversity = _riemer_diversity_dense(
        replacement_masses,
        replacement_concentration,
        box,
        kept,
        n_species,
    )
    radius_error[box] = wp.float64(0.0)
    mean_error[box] = wp.float64(0.0)
    surface_error[box] = wp.float64(0.0)
    if source_moment == 0.0:
        if replacement_moment != 0.0:
            radius_error[box] = wp.float64(wp.inf)
    else:
        radius_error[box] = wp.abs(replacement_moment - source_moment) / wp.abs(
            source_moment
        )
    if source_mean == 0.0:
        if replacement_mean != 0.0:
            mean_error[box] = wp.float64(wp.inf)
    else:
        mean_error[box] = wp.abs(replacement_mean - source_mean) / wp.abs(
            source_mean
        )
    if source_surface == 0.0:
        if replacement_surface != 0.0:
            surface_error[box] = wp.float64(wp.inf)
    else:
        surface_error[box] = wp.abs(
            replacement_surface - source_surface
        ) / wp.abs(source_surface)
    diversity_error[box] = wp.abs(replacement_diversity - source_diversity)
    status[box] = PLANNING_SUCCESS
    # The interval sweep preserves number, species inventory, and charge.  The
    # explicit check makes finite precision drift a planning (not commit) error.
    if status[box] == PLANNING_SUCCESS and wp.abs(
        replacement_number - source_number
    ) > (wp.float64(1e-12) * wp.abs(source_number) + wp.float64(1e-30)):
        status[box] = PLANNING_EXACT_INVENTORY_FAILURE
    if status[box] == PLANNING_SUCCESS and wp.abs(
        replacement_charge_total - source_charge_total
    ) > (wp.float64(1e-12) * wp.abs(source_charge_total) + wp.float64(1e-30)):
        status[box] = PLANNING_EXACT_INVENTORY_FAILURE
    for species in range(n_species):
        source_mass = wp.float64(0.0)
        replacement_mass = wp.float64(0.0)
        for item in range(active):
            particle = sorted_indices[box, item]
            source_mass += (
                concentration[box, particle] * masses[box, particle, species]
            )
        for item in range(kept):
            replacement_mass += equal * replacement_masses[box, item, species]
        if source_mass > 0.0:
            source_species_count += 1
        if not wp.isfinite(source_mass) or not wp.isfinite(replacement_mass):
            replacement_values_finite = False
        elif status[box] == PLANNING_SUCCESS and wp.abs(
            replacement_mass - source_mass
        ) > (wp.float64(1e-12) * wp.abs(source_mass) + wp.float64(1e-30)):
            status[box] = PLANNING_EXACT_INVENTORY_FAILURE
    if status[box] == PLANNING_SUCCESS and (
        not replacement_values_finite
        or not wp.isfinite(total_number)
        or not wp.isfinite(equal)
        or not wp.isfinite(source_number)
        or not wp.isfinite(source_charge_total)
        or not wp.isfinite(replacement_number)
        or not wp.isfinite(replacement_charge_total)
        or not wp.isfinite(source_moment)
        or not wp.isfinite(source_mean)
        or not wp.isfinite(source_surface)
        or not wp.isfinite(replacement_moment)
        or not wp.isfinite(replacement_mean)
        or not wp.isfinite(replacement_surface)
        or not wp.isfinite(source_diversity)
        or not wp.isfinite(replacement_diversity)
        or not wp.isfinite(radius_error[box])
        or not wp.isfinite(mean_error[box])
        or not wp.isfinite(surface_error[box])
        or not wp.isfinite(diversity_error[box])
    ):
        status[box] = PLANNING_NONFINITE_FAILURE
    elif status[box] == PLANNING_SUCCESS and (
        radius_error[box] > radius_bound or not wp.isfinite(radius_error[box])
    ):
        status[box] = PLANNING_RADIUS_CUBED_FAILURE
    elif mean_error[box] > mean_bound or not wp.isfinite(mean_error[box]):
        status[box] = PLANNING_MEAN_RADIUS_FAILURE
    elif surface_error[box] > surface_bound or not wp.isfinite(
        surface_error[box]
    ):
        status[box] = PLANNING_SURFACE_FAILURE
    elif source_species_count > 1 and diversity_error[box] > diversity_bound:
        status[box] = PLANNING_DIVERSITY_FAILURE


@wp.kernel
def _commit_resampling(
    masses: wp.array3d(dtype=wp.float64),
    concentration: wp.array2d(dtype=wp.float64),
    charge: wp.array2d(dtype=wp.float64),
    counts: wp.array(dtype=wp.int32),
    retained_counts: wp.array(dtype=wp.int32),
    retained_indices: wp.array2d(dtype=wp.int32),
    released_indices: wp.array2d(dtype=wp.int32),
    replacement_masses: wp.array3d(dtype=wp.float64),
    replacement_concentration: wp.array2d(dtype=wp.float64),
    replacement_charge: wp.array2d(dtype=wp.float64),
) -> None:
    """Commit all successful box plans in one fixed-shape launch."""
    box, particle, species = wp.tid()
    if counts[box] == 0:
        return
    if particle < retained_counts[box]:
        target = retained_indices[box, particle]
        masses[box, target, species] = replacement_masses[
            box, particle, species
        ]
        if species == 0:
            concentration[box, target] = replacement_concentration[
                box, particle
            ]
            charge[box, target] = replacement_charge[box, particle]
    if particle < counts[box]:
        target = released_indices[box, particle]
        masses[box, target, species] = wp.float64(0.0)
        if species == 0:
            concentration[box, target] = wp.float64(0.0)
            charge[box, target] = wp.float64(0.0)


@wp.kernel
def _any_nonzero_demand(
    counts: wp.array(dtype=wp.int32),
    result: wp.array(dtype=wp.int32),
) -> None:
    """Record whether at least one box requests a remap."""
    box = wp.tid()
    if counts[box] != 0:
        wp.atomic_add(result, 0, 1)


@wp.kernel
def _aggregate_planning_status(
    counts: wp.array(dtype=wp.int32),
    status: wp.array(dtype=wp.int32),
    result: wp.array(dtype=wp.int32),
) -> None:
    """Record whether any nonzero-demand box failed planning."""
    box = wp.tid()
    if counts[box] != 0 and status[box] != PLANNING_SUCCESS:
        wp.atomic_add(result, 0, 1)


@wp.kernel
def _scan_representative_volume_scaling(  # noqa: C901
    masses: wp.array3d(dtype=wp.float64),
    concentration: wp.array2d(dtype=wp.float64),
    charge: wp.array2d(dtype=wp.float64),
    density: wp.array(dtype=wp.float64),
    volume: wp.array(dtype=wp.float64),
    demand: wp.array(dtype=wp.float64),
    scaling_required: wp.array(dtype=wp.int32),
    requested_scale: wp.array(dtype=wp.float64),
    minimum_scale: wp.array(dtype=wp.float64),
    minimum_volume: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
) -> None:
    """Fuse physical and P4 sidecar checks into one read-only scan."""
    box, particle, species = wp.tid()
    if not wp.isfinite(masses[box, particle, species]) or (
        masses[box, particle, species] < 0.0
    ):
        wp.atomic_add(status, 0, 1)
    if concentration[box, particle] == 0.0 and (
        masses[box, particle, species] != 0.0
    ):
        wp.atomic_add(status, 0, 1)
    if species == 0:
        if not wp.isfinite(concentration[box, particle]) or (
            concentration[box, particle] < 0.0
        ):
            wp.atomic_add(status, 0, 1)
        if not wp.isfinite(charge[box, particle]):
            wp.atomic_add(status, 0, 1)
        if concentration[box, particle] == 0.0 and charge[box, particle] != 0.0:
            wp.atomic_add(status, 0, 1)
        if concentration[box, particle] > 0.0:
            total_mass = wp.float64(0.0)
            for mass_species in range(density.shape[0]):
                total_mass += masses[box, particle, mass_species]
            if not wp.isfinite(total_mass) or total_mass <= 0.0:
                wp.atomic_add(status, 0, 1)
        if scaling_required[box] == 1 and demand[box] > 0.0:
            scaled_concentration = (
                concentration[box, particle] * requested_scale[box]
            )
            if concentration[box, particle] > 0.0 and (
                not wp.isfinite(scaled_concentration)
                or scaled_concentration <= 0.0
            ):
                wp.atomic_add(status, 0, 1)
    if particle == 0 and species == 0:
        if not wp.isfinite(volume[box]) or volume[box] <= 0.0:
            wp.atomic_add(status, 0, 1)
        if not wp.isfinite(demand[box]) or demand[box] < 0.0:
            wp.atomic_add(status, 0, 1)
        if scaling_required[box] != 0 and scaling_required[box] != 1:
            wp.atomic_add(status, 0, 1)
        if (
            not wp.isfinite(requested_scale[box])
            or not wp.isfinite(minimum_scale[box])
            or not wp.isfinite(minimum_volume[box])
            or minimum_scale[box] <= 0.0
            or requested_scale[box] < minimum_scale[box]
            or requested_scale[box] > 1.0
            or minimum_volume[box] <= 0.0
        ):
            wp.atomic_add(status, 0, 1)
        if scaling_required[box] == 1 and demand[box] > 0.0:
            wp.atomic_add(status, 1, 1)
            if volume[box] * requested_scale[box] < minimum_volume[box]:
                wp.atomic_add(status, 0, 1)


@wp.kernel
def _scan_density_p4(
    density: wp.array(dtype=wp.float64),
    status: wp.array(dtype=wp.int32),
) -> None:
    """Validate global density state, including empty particle-box inputs."""
    density_index = wp.tid()
    if not wp.isfinite(density[density_index]) or density[density_index] <= 0.0:
        wp.atomic_add(status, 0, 1)


@wp.kernel
def _resolve_representative_volume_selection(
    demand: wp.array(dtype=wp.float64),
    scaling_required: wp.array(dtype=wp.int32),
    selected: wp.array(dtype=wp.int32),
) -> None:
    """Capture the immutable selected-row predicate before any writes."""
    box = wp.tid()
    if scaling_required[box] == 1 and demand[box] > 0.0:
        selected[box] = 1
    else:
        selected[box] = 0


@wp.kernel
def _write_resolved_representative_scale(
    selected: wp.array(dtype=wp.int32),
    requested_scale: wp.array(dtype=wp.float64),
    resolved_scale: wp.array(dtype=wp.float64),
) -> None:
    """Write the P4 diagnostic after complete read-only preflight."""
    box = wp.tid()
    if selected[box] == 1:
        resolved_scale[box] = requested_scale[box]
    else:
        resolved_scale[box] = wp.float64(1.0)


@wp.kernel
def _apply_representative_volume_scaling(
    concentration: wp.array2d(dtype=wp.float64),
    volume: wp.array(dtype=wp.float64),
    demand: wp.array(dtype=wp.float64),
    selected: wp.array(dtype=wp.int32),
    requested_scale: wp.array(dtype=wp.float64),
) -> None:
    """Scale only selected volume, concentration, and demand storage."""
    box, particle = wp.tid()
    if selected[box] == 1:
        factor = requested_scale[box]
        concentration[box, particle] *= factor
        if particle == 0:
            volume[box] *= factor
            demand[box] *= factor


def _require_exact_bound(name: str, value: Any) -> float:
    """Return a finite nonnegative exact Python-float diagnostic bound."""
    if type(value) is not float:
        raise TypeError(f"{name} must be an exact Python float")
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def _field(container: Any, name: str) -> Any:
    """Return a required particle field with a stable validation error."""
    try:
        return getattr(container, name)
    except AttributeError as exc:
        raise ValueError(f"particles must provide {name}") from exc


def _validate_array(
    value: Any,
    name: str,
    rank: int,
    shape: tuple[int, ...],
    dtype: Any,
    device: Any,
) -> None:
    """Validate caller-owned Warp metadata without reading array values."""
    if not _is_warp_array_like(value):
        raise TypeError(f"{name} must be a Warp array")
    if value.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if value.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}")
    if str(value.device) != str(device):
        raise ValueError(f"{name} must be on the particle device")
    _require_contiguous_array(value, name)


def _require_contiguous_array(value: Any, name: str) -> None:
    """Reject strided Warp views before pointer-range ownership checks."""
    strides = getattr(value, "strides", None)
    if strides is None:
        return
    dtype_sizes = {wp.float64: 8, wp.int32: 4}
    expected_strides: list[int] = []
    stride = dtype_sizes[value.dtype]
    for dimension in reversed(value.shape):
        expected_strides.insert(0, stride)
        stride *= dimension
    if tuple(strides) != tuple(expected_strides):
        raise ValueError(f"{name} must be a contiguous, non-view Warp array")


def _ranges_overlap(left: Any, right: Any) -> bool:
    """Return whether two caller allocations have overlapping byte ranges."""
    if int(np.prod(left.shape)) == 0 or int(np.prod(right.shape)) == 0:
        return False
    dtype_sizes = {wp.float64: 8, wp.int32: 4}
    left_itemsize = dtype_sizes[left.dtype]
    right_itemsize = dtype_sizes[right.dtype]
    left_start = int(left.ptr)
    right_start = int(right.ptr)
    left_end = left_start + int(np.prod(left.shape)) * left_itemsize
    right_end = right_start + int(np.prod(right.shape)) * right_itemsize
    return left_start < right_end and right_start < left_end


def _validate_buffers(
    buffers: ResamplingBuffers,
    b: int,
    n: int,
    s: int,
    device: Any,
    protected: list[Any],
) -> None:
    """Validate buffer metadata and enforce distinct non-aliasing storage."""
    if not isinstance(buffers, ResamplingBuffers):
        raise TypeError("buffers must be a ResamplingBuffers")
    schema = (
        ("retained_counts", 1, (b,), wp.int32),
        ("released_counts", 1, (b,), wp.int32),
        ("retained_indices", 2, (b, n), wp.int32),
        ("released_indices", 2, (b, n), wp.int32),
        ("sorted_indices", 2, (b, n), wp.int32),
        ("replacement_masses", 3, (b, n, s), wp.float64),
        ("replacement_concentration", 2, (b, n), wp.float64),
        ("replacement_charge", 2, (b, n), wp.float64),
        ("source_radii", 2, (b, n), wp.float64),
        ("radius_cubed_relative_error", 1, (b,), wp.float64),
        ("mean_radius_relative_error", 1, (b,), wp.float64),
        ("surface_relative_error", 1, (b,), wp.float64),
        ("diversity_absolute_error", 1, (b,), wp.float64),
        ("planning_status", 1, (b,), wp.int32),
    )
    values = []
    for name, rank, shape, dtype in schema:
        value = getattr(buffers, name)
        _validate_array(value, name, rank, shape, dtype, device)
        values.append(value)
    all_arrays = protected + values
    for index, left in enumerate(all_arrays):
        for right in all_arrays[index + 1 :]:
            if _ranges_overlap(left, right):
                raise ValueError("resampling arrays must not overlap")


def resampling_step_gpu(  # noqa: C901, PLR0913
    particles: Any,
    required_release_counts: Any,
    buffers: ResamplingBuffers,
    *,
    radius_cubed_relative_error: float = 1.0,
    mean_radius_relative_error: float = 1.0,
    surface_relative_error: float = 1.0,
    diversity_absolute_error: float = 1.0,
) -> Any:
    """Apply deterministic fixed-capacity P2 equal-weight remapping in place.

    A nonzero per-box count selects remapping and must not exceed active slots
    minus one; zero-demand boxes are write-free. The caller owns particles,
    counts, and all buffer arrays. Preflight validates schemas, physical state,
    and nonaliasing without writing caller storage. Planning may overwrite its
    documented buffer lanes. A planning failure raises before the single commit
    launch and therefore preserves particles; rollback is not promised after a
    commit launch.

    Args:
        particles: Fixed-capacity Warp particle container with masses,
            concentration, charge, density, and volume fields.
        required_release_counts: Same-device ``int32`` release counts shaped
            ``(B,)``. Counts are explicit inputs and are not modified.
        buffers: Caller-owned, nonaliasing :class:`ResamplingBuffers` storage.
        radius_cubed_relative_error: Exact Python ``float`` bound for the
            finite, nonnegative relative radius-cubed error.
        mean_radius_relative_error: Exact Python ``float`` bound for the
            finite, nonnegative relative mean-radius error.
        surface_relative_error: Exact Python ``float`` bound for the finite,
            nonnegative relative surface-area error.
        diversity_absolute_error: Exact Python ``float`` bound for the finite,
            nonnegative absolute Riemer-diversity error.

    Returns:
        The identical particle container after a successful in-place commit or
        a write-free empty/all-zero-demand call.

    Raises:
        TypeError: If a bound, caller array, or buffer record has an unsupported
            type.
        ValueError: If schemas, physical values, ownership, release counts, or
            planning diagnostics are invalid.
    """
    bounds = (
        _require_exact_bound(
            "radius_cubed_relative_error", radius_cubed_relative_error
        ),
        _require_exact_bound(
            "mean_radius_relative_error", mean_radius_relative_error
        ),
        _require_exact_bound("surface_relative_error", surface_relative_error),
        _require_exact_bound(
            "diversity_absolute_error", diversity_absolute_error
        ),
    )
    masses = _field(particles, "masses")
    if not _is_warp_array_like(masses):
        raise TypeError("masses must be a Warp array")
    if masses.ndim != 3:
        raise ValueError("masses must have rank 3")
    if masses.dtype != wp.float64:
        raise ValueError("masses must have dtype float64")
    _require_contiguous_array(masses, "masses")
    b, n, s = tuple(masses.shape)
    if n == 0:
        raise ValueError("particle capacity must be positive")
    if s == 0:
        raise ValueError("species capacity must be positive")
    device = masses.device
    concentration = _field(particles, "concentration")
    charge = _field(particles, "charge")
    density = _field(particles, "density")
    volume = _field(particles, "volume")
    _validate_array(
        concentration, "concentration", 2, (b, n), wp.float64, device
    )
    _validate_array(charge, "charge", 2, (b, n), wp.float64, device)
    _validate_array(density, "density", 1, (s,), wp.float64, device)
    _validate_array(volume, "volume", 1, (b,), wp.float64, device)
    _validate_array(
        required_release_counts,
        "required_release_counts",
        1,
        (b,),
        wp.int32,
        device,
    )
    _validate_buffers(
        buffers,
        b,
        n,
        s,
        device,
        [
            masses,
            concentration,
            charge,
            density,
            volume,
            required_release_counts,
        ],
    )
    invalid = wp.zeros(1, dtype=wp.int32, device=device)
    if b == 0:
        wp.launch(
            _scan_density,
            dim=s,
            inputs=[density, invalid],
            device=device,
        )
        if invalid.numpy()[0] != 0:
            raise ValueError(
                "particles or required_release_counts have invalid values"
            )
        return particles
    wp.launch(
        _scan_particle_values,
        dim=(b, n, s),
        inputs=[masses, concentration, charge, density, volume, invalid],
        device=device,
    )
    wp.launch(
        _scan_density,
        dim=s,
        inputs=[density, invalid],
        device=device,
    )
    wp.launch(
        _scan_slot_state,
        dim=(b, n),
        inputs=[masses, concentration, density, invalid],
        device=device,
    )
    wp.launch(
        _validate_counts,
        dim=b,
        inputs=[concentration, required_release_counts, invalid],
        device=device,
    )
    if invalid.numpy()[0] != 0:
        raise ValueError(
            "particles or required_release_counts have invalid values"
        )
    demand = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(
        _any_nonzero_demand,
        dim=b,
        inputs=[required_release_counts, demand],
        device=device,
    )
    if demand.numpy()[0] == 0:
        return particles
    wp.launch(
        _plan_resampling,
        dim=b,
        inputs=[
            masses,
            concentration,
            charge,
            density,
            required_release_counts,
            buffers.retained_counts,
            buffers.released_counts,
            buffers.retained_indices,
            buffers.released_indices,
            buffers.sorted_indices,
            buffers.replacement_masses,
            buffers.replacement_concentration,
            buffers.replacement_charge,
            buffers.source_radii,
            buffers.radius_cubed_relative_error,
            buffers.mean_radius_relative_error,
            buffers.surface_relative_error,
            buffers.diversity_absolute_error,
            buffers.planning_status,
            *bounds,
        ],
        device=device,
    )
    planning_failed = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(
        _aggregate_planning_status,
        dim=b,
        inputs=[
            required_release_counts,
            buffers.planning_status,
            planning_failed,
        ],
        device=device,
    )
    if planning_failed.numpy()[0] != 0:
        raise ValueError("resampling diagnostic exceeds bound")
    wp.launch(
        _commit_resampling,
        dim=(b, n, s),
        inputs=[
            masses,
            concentration,
            charge,
            required_release_counts,
            buffers.retained_counts,
            buffers.retained_indices,
            buffers.released_indices,
            buffers.replacement_masses,
            buffers.replacement_concentration,
            buffers.replacement_charge,
        ],
        device=device,
    )
    return particles


def representative_volume_scaling_step_gpu(  # noqa: C901, PLR0913
    particles: Any,
    provisional_source_demand: Any,
    scaling_required: Any,
    requested_scale: Any,
    minimum_scale: Any,
    minimum_volume: Any,
    resolved_scale: Any,
) -> tuple[Any, Any, Any]:
    """Apply P4 scaling to selected Warp particle-container rows in place.

    A row is selected exactly when its ``int32`` flag is ``1`` and its
    provisional demand is positive. Selected rows scale box volume, every
    particle concentration, and provisional demand by the requested factor;
    masses, charge, density, configuration sidecars, and unselected rows are
    unchanged. A read-only all-box preflight runs before the diagnostic or
    scaling launch, so rejected calls do not write caller-owned state.
    ``resolved_scale`` receives the requested factor for selected rows and
    ``1.0`` otherwise. Successful work is asynchronous: callers own all
    sidecars, device placement, and explicit Warp synchronization before
    reading results. This concrete-module-only boundary does not resolve
    policy, transfer data, resize storage, or provide a runnable or package
    export.

    Args:
        particles: Fixed-capacity Warp particle container whose masses,
            concentration, charge, density, and volume fields are same-device
            contiguous ``float64`` arrays.
        provisional_source_demand: Caller-owned same-device contiguous
            ``float64`` demand array shaped ``(B,)``.
        scaling_required: Caller-owned same-device contiguous ``int32`` flag
            array shaped ``(B,)`` containing only ``0`` or ``1``.
        requested_scale: Caller-owned same-device contiguous ``float64``
            requested factors shaped ``(B,)``.
        minimum_scale: Caller-owned same-device contiguous ``float64`` lower
            factor bounds shaped ``(B,)``.
        minimum_volume: Caller-owned same-device contiguous ``float64``
            post-scaling box-volume bounds in m³, shaped ``(B,)``.
        resolved_scale: Caller-owned same-device contiguous ``float64``
            diagnostic output shaped ``(B,)``.

    Returns:
        The identical particle container, demand sidecar, and resolved-scale
        sidecar after successful asynchronous diagnostic and, when selected,
        scaling launches.

    Raises:
        TypeError: If a particle field or sidecar is not a supported Warp array
            or has the required dtype.
        ValueError: If schemas, devices, contiguity, ownership, physical
            values, factor bounds, flags, or selected scaled-volume bounds are
            invalid. Rejection occurs before any diagnostic or scaling write.
    """
    masses = _field(particles, "masses")
    if not _is_warp_array_like(masses):
        raise TypeError("masses must be a Warp array")
    if masses.ndim != 3:
        raise ValueError("masses must have rank 3")
    if masses.dtype != wp.float64:
        raise ValueError("masses must have dtype float64")
    _require_contiguous_array(masses, "masses")
    b, n, s = tuple(masses.shape)
    if n == 0:
        raise ValueError("particle capacity must be positive")
    if s == 0:
        raise ValueError("species capacity must be positive")
    device = masses.device
    concentration = _field(particles, "concentration")
    charge = _field(particles, "charge")
    density = _field(particles, "density")
    volume = _field(particles, "volume")
    _validate_array(
        concentration, "concentration", 2, (b, n), wp.float64, device
    )
    _validate_array(charge, "charge", 2, (b, n), wp.float64, device)
    _validate_array(density, "density", 1, (s,), wp.float64, device)
    _validate_array(volume, "volume", 1, (b,), wp.float64, device)
    schema = (
        ("provisional_source_demand", provisional_source_demand, wp.float64),
        ("scaling_required", scaling_required, wp.int32),
        ("requested_scale", requested_scale, wp.float64),
        ("minimum_scale", minimum_scale, wp.float64),
        ("minimum_volume", minimum_volume, wp.float64),
        ("resolved_scale", resolved_scale, wp.float64),
    )
    sidecars: list[Any] = []
    for name, values, dtype in schema:
        if not _is_warp_array_like(values):
            raise TypeError(f"{name} must be a Warp array")
        if values.dtype != dtype:
            raise TypeError(f"{name} must have dtype {dtype}")
        _validate_array(values, name, 1, (b,), dtype, device)
        sidecars.append(values)
    protected = [masses, concentration, charge, density, volume]
    all_arrays = protected + sidecars
    for index, left in enumerate(all_arrays):
        for right in all_arrays[index + 1 :]:
            if _ranges_overlap(left, right):
                raise ValueError(
                    "representative scaling arrays must not overlap"
                )
    status = wp.zeros(2, dtype=wp.int32, device=device)
    wp.launch(
        _scan_density_p4,
        dim=s,
        inputs=[density, status],
        device=device,
    )
    if b == 0:
        if status.numpy()[0] != 0:
            raise ValueError(
                "particles or representative scaling sidecars have "
                "invalid values"
            )
        return particles, provisional_source_demand, resolved_scale
    wp.launch(
        _scan_representative_volume_scaling,
        dim=(b, n, s),
        inputs=[
            masses,
            concentration,
            charge,
            density,
            volume,
            provisional_source_demand,
            scaling_required,
            requested_scale,
            minimum_scale,
            minimum_volume,
            status,
        ],
        device=device,
    )
    status_values = status.numpy()
    if status_values[0] != 0:
        raise ValueError(
            "particles or representative scaling sidecars have invalid values"
        )
    selected = wp.zeros(b, dtype=wp.int32, device=device)
    wp.launch(
        _resolve_representative_volume_selection,
        dim=b,
        inputs=[provisional_source_demand, scaling_required, selected],
        device=device,
    )
    wp.launch(
        _write_resolved_representative_scale,
        dim=b,
        inputs=[
            selected,
            requested_scale,
            resolved_scale,
        ],
        device=device,
    )
    if status_values[1] != 0:
        wp.launch(
            _apply_representative_volume_scaling,
            dim=(b, n),
            inputs=[
                concentration,
                volume,
                provisional_source_demand,
                selected,
                requested_scale,
            ],
            device=device,
        )
    return particles, provisional_source_demand, resolved_scale
