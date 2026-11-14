"""Bounded-memory parallel properties and exact Brownian pair maxima.

Pair evaluation remains quadratic in the active cohort size. A fixed maximum
of 4096 reduction lanes per box bounds scratch independently of pair count.
Selection, acceptance, merge admission and random draws use the original
sequential device function.
"""

from dataclasses import dataclass
from typing import Any, no_type_check

import warp as wp

from particula.gpu.kernels.coagulation import (
    _sanitize_positive_finite,
    _select_coagulation_pairs_impl,
    aerodynamic_mobility_wp,
    brownian_diffusivity_wp,
    brownian_kernel_pair_wp,
    charged_kernel_wp,
    cunningham_slip_correction_wp,
    dynamic_viscosity_wp,
    g_collection_term_wp,
    knudsen_number_wp,
    mean_thermal_speed_wp,
    molecule_mean_free_path_wp,
    particle_mean_free_path_wp,
    particle_radius_from_volume_wp,
)


@no_type_check
@wp.kernel
def _properties(
    masses: wp.array3d(dtype=wp.float64),
    concentration: wp.array2d(dtype=wp.float64),
    density: wp.array(dtype=wp.float64),
    temperature: wp.array(dtype=wp.float64),
    pressure: wp.array(dtype=wp.float64),
    gas_constant: wp.float64,
    boltzmann_constant: wp.float64,
    molecular_weight_air: wp.float64,
    ref_viscosity: wp.float64,
    ref_temperature: wp.float64,
    sutherland_constant: wp.float64,
    radii: wp.array2d(dtype=wp.float64),
    diffusivities: wp.array2d(dtype=wp.float64),
    g_terms: wp.array2d(dtype=wp.float64),
    speeds: wp.array2d(dtype=wp.float64),
    total_masses: wp.array2d(dtype=wp.float64),
    active: wp.array2d(dtype=wp.int32),
    failure: wp.array(dtype=wp.int32),
):
    """Evaluate the original transport formulas independently per slot."""
    if failure:
        if failure[0] != 0:
            return
    box, slot = wp.tid()
    total_masses[box, slot] = wp.float64(0.0)
    radii[box, slot] = wp.float64(0.0)
    diffusivities[box, slot] = wp.float64(0.0)
    g_terms[box, slot] = wp.float64(0.0)
    speeds[box, slot] = wp.float64(0.0)
    active[box, slot] = 0
    if concentration[box, slot] <= wp.float64(0.0):
        return
    total_mass = wp.float64(0.0)
    total_volume = wp.float64(0.0)
    for species in range(masses.shape[2]):
        species_mass = masses[box, slot, species]
        total_mass += species_mass
        total_volume += species_mass / density[species]
    if total_volume <= wp.float64(0.0) or total_mass <= wp.float64(0.0):
        return
    temperature_value = temperature[box]
    viscosity = dynamic_viscosity_wp(
        temperature_value, ref_viscosity, ref_temperature, sutherland_constant
    )
    free_path = molecule_mean_free_path_wp(
        molecular_weight_air,
        temperature_value,
        pressure[box],
        viscosity,
        gas_constant,
    )
    radius = particle_radius_from_volume_wp(total_volume)
    knudsen = knudsen_number_wp(free_path, radius)
    slip = cunningham_slip_correction_wp(knudsen)
    mobility = aerodynamic_mobility_wp(radius, slip, viscosity)
    diffusivity = brownian_diffusivity_wp(
        temperature_value, mobility, boltzmann_constant
    )
    speed = mean_thermal_speed_wp(
        total_mass, temperature_value, boltzmann_constant
    )
    path = particle_mean_free_path_wp(diffusivity, speed)
    g_term = g_collection_term_wp(path, radius)
    radii[box, slot] = radius
    diffusivities[box, slot] = diffusivity
    speeds[box, slot] = speed
    g_terms[box, slot] = g_term
    total_masses[box, slot] = total_mass
    active[box, slot] = 1


@no_type_check
@wp.kernel
def _compact_active(
    active: wp.array2d(dtype=wp.int32),
    indices: wp.array2d(dtype=wp.int32),
    counts: wp.array(dtype=wp.int32),
    failure: wp.array(dtype=wp.int32),
):
    """Preserve ascending source indices, independent of GPU scheduling."""
    if failure:
        if failure[0] != 0:
            return
    box = wp.tid()
    count = int(0)
    for slot in range(active.shape[1]):
        if active[box, slot] != 0:
            indices[box, count] = slot
            count += 1
    for slot in range(count, active.shape[1]):
        indices[box, slot] = -1
    counts[box] = count


@no_type_check
@wp.kernel
def _pair_maxima(
    radii: wp.array2d(dtype=wp.float64),
    diffusivities: wp.array2d(dtype=wp.float64),
    g_terms: wp.array2d(dtype=wp.float64),
    speeds: wp.array2d(dtype=wp.float64),
    indices: wp.array2d(dtype=wp.int32),
    counts: wp.array(dtype=wp.int32),
    partials: wp.array2d(dtype=wp.float64),
    failure: wp.array(dtype=wp.int32),
):
    """Tile the upper pair matrix across bounded lanes without storing it."""
    if failure:
        if failure[0] != 0:
            return
    box, lane = wp.tid()
    count = wp.int64(counts[box])
    maximum = wp.float64(0.0)
    # Grid-stride tiles visit each valid upper-triangle pair exactly once.
    # int64 arithmetic avoids capacity-squared overflow before loop admission.
    pair = wp.int64(lane)
    while pair < count * count:
        first_rank = wp.int32(pair // count)
        second_rank = wp.int32(pair % count)
        if first_rank < second_rank:
            first = indices[box, first_rank]
            second = indices[box, second_rank]
            rate = _sanitize_positive_finite(
                brownian_kernel_pair_wp(
                    radii[box, first],
                    radii[box, second],
                    diffusivities[box, first],
                    diffusivities[box, second],
                    g_terms[box, first],
                    g_terms[box, second],
                    speeds[box, first],
                    speeds[box, second],
                    wp.float64(1.0),
                )
            )
            if rate > maximum:
                maximum = rate
        pair += wp.int64(partials.shape[1])
    partials[box, lane] = maximum


@no_type_check
@wp.kernel
def _charged_pair_maxima(
    radii: wp.array2d(dtype=wp.float64),
    masses: wp.array2d(dtype=wp.float64),
    charges: wp.array2d(dtype=wp.float64),
    temperature: wp.array(dtype=wp.float64),
    pressure: wp.array(dtype=wp.float64),
    gas_constant: wp.float64,
    boltzmann_constant: wp.float64,
    molecular_weight_air: wp.float64,
    ref_viscosity: wp.float64,
    ref_temperature: wp.float64,
    sutherland_constant: wp.float64,
    elementary_charge_value: wp.float64,
    electric_permittivity: wp.float64,
    mechanism_mask: wp.int32,
    indices: wp.array2d(dtype=wp.int32),
    counts: wp.array(dtype=wp.int32),
    partials: wp.array2d(dtype=wp.float64),
    failure: wp.array(dtype=wp.int32),
):
    """Tile the upper pair matrix across bounded lanes without storing it."""
    if failure:
        if failure[0] != 0:
            return
    box, lane = wp.tid()
    count = wp.int64(counts[box])
    maximum = wp.float64(0.0)
    # Grid-stride tiles visit each valid upper-triangle pair exactly once.
    # int64 arithmetic avoids capacity-squared overflow before loop admission.
    pair = wp.int64(lane)
    while pair < count * count:
        first_rank = wp.int32(pair // count)
        second_rank = wp.int32(pair % count)
        if first_rank < second_rank:
            first = indices[box, first_rank]
            second = indices[box, second_rank]
            rate = _sanitize_positive_finite(
                charged_kernel_wp(
                    mechanism_mask >> wp.int32(8),
                    radii[box, first],
                    radii[box, second],
                    masses[box, first],
                    masses[box, second],
                    charges[box, first],
                    charges[box, second],
                    temperature[box],
                    pressure[box],
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
            if rate > maximum:
                maximum = rate
        pair += wp.int64(partials.shape[1])
    partials[box, lane] = maximum


@no_type_check
@wp.kernel
def _finish_maxima(
    partials: wp.array2d(dtype=wp.float64),
    majorants: wp.array(dtype=wp.float64),
    failure: wp.array(dtype=wp.int32),
):
    """Reduce finite maxima without floating-point sum reordering."""
    if failure:
        if failure[0] != 0:
            return
    box = wp.tid()
    maximum = wp.float64(0.0)
    for lane in range(partials.shape[1]):
        maximum = wp.max(maximum, partials[box, lane])
    majorants[box] = maximum


@no_type_check
@wp.kernel
def _select_brownian(  # noqa: C901
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
    active_counts: wp.array(dtype=wp.int32),
    majorants: wp.array(dtype=wp.float64),
    failure: wp.array(dtype=wp.int32),
) -> None:
    """Select from parallel properties and maxima with the original RNG code."""
    if failure:
        if failure[0] != 0:
            return
    box_idx = wp.tid()
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
        majorants[box_idx],
        active_counts[box_idx],
        wp.float64(0.0),
    )


@dataclass(frozen=True)
class _PreparedBrownianWork:
    """Retain all parallel scratch and exact launch inputs during setup."""

    selector_inputs: tuple[Any, ...]
    property_inputs: tuple[Any, ...]
    active: Any
    counts: Any
    partials: Any
    majorants: Any
    device: Any

    def enqueue(self, failure):
        """Enqueue retained work without allocation or observation."""
        wp.launch(
            _properties,
            dim=self.active.shape,
            inputs=(*self.property_inputs, self.active, failure),
            device=self.device,
        )
        wp.launch(
            _compact_active,
            dim=self.active.shape[0],
            inputs=[
                self.active,
                self.selector_inputs[24],
                self.counts,
                failure,
            ],
            device=self.device,
        )
        if int(self.selector_inputs[28]) & 255 == 2:
            args = self.selector_inputs
            wp.launch(
                _charged_pair_maxima,
                dim=self.partials.shape,
                inputs=[
                    args[15],
                    args[22],
                    args[23],
                    *args[4:14],
                    args[28],
                    args[24],
                    self.counts,
                    self.partials,
                    failure,
                ],
                device=self.device,
            )
        else:
            wp.launch(
                _pair_maxima,
                dim=self.partials.shape,
                inputs=[
                    *self.selector_inputs[15:19],
                    self.selector_inputs[24],
                    self.counts,
                    self.partials,
                    failure,
                ],
                device=self.device,
            )
        wp.launch(
            _finish_maxima,
            dim=self.active.shape[0],
            inputs=[self.partials, self.majorants, failure],
            device=self.device,
        )
        wp.launch(
            _select_brownian,
            dim=self.active.shape[0],
            inputs=(
                *self.selector_inputs,
                self.counts,
                self.majorants,
                failure,
            ),
            device=self.device,
        )


def prepare_brownian_work(selector_inputs, device):
    """Allocate O(BN) flags and at most 4096 fp64 maxima per box."""
    boxes, slots = selector_inputs[1].shape
    lanes = max(1, min((slots * slots + 31) // 32, 4096))
    active = wp.empty((boxes, slots), dtype=wp.int32, device=device)
    counts = wp.empty(boxes, dtype=wp.int32, device=device)
    partials = wp.empty((boxes, lanes), dtype=wp.float64, device=device)
    majorants = wp.empty(boxes, dtype=wp.float64, device=device)
    positions = (0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 15, 16, 17, 18, 22)
    return _PreparedBrownianWork(
        selector_inputs,
        tuple(selector_inputs[index] for index in positions),
        active,
        counts,
        partials,
        majorants,
        device,
    )
