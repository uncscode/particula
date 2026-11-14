"""Validate explicit ions and cache attachment rates before collision draws."""

from typing import no_type_check

import warp as wp

from particula.gpu.dynamics.ion_attachment import (
    classify_ion_particle_wp,
    ion_attachment_wp,
)
from particula.gpu.dynamics.mobility_attachment import mobility_attachment_wp

_REFERENCE_TEMPERATURE = wp.constant(wp.float64(298.15))
_REFERENCE_PRESSURE = wp.constant(wp.float64(101325.0))


@no_type_check
@wp.kernel
def classify_ions(
    masses: wp.array3d(dtype=wp.float64),
    counts: wp.array2d(dtype=wp.float64),
    charges: wp.array2d(dtype=wp.float64),
    temperature: wp.array(dtype=wp.float64),
    pressure: wp.array(dtype=wp.float64),
    positive_species: wp.int32,
    negative_species: wp.int32,
    positive_mass: wp.float64,
    negative_mass: wp.float64,
    kinds: wp.array2d(dtype=wp.int32),
    presence: wp.array2d(dtype=wp.int32),
    failure: wp.array(dtype=wp.int32),
    attachment_model: wp.int32,
):
    """Validate unit particles and the fixed LF2013 air reference per box."""
    if failure[0] != 0:
        return
    box = wp.tid()
    presence[box, 0] = 0
    presence[box, 1] = 0
    if attachment_model == 0 and (
        temperature[box] != _REFERENCE_TEMPERATURE
        or pressure[box] != _REFERENCE_PRESSURE
    ):
        wp.atomic_max(failure, 0, 7)
        return
    for slot in range(counts.shape[1]):
        kinds[box, slot] = 2
        if counts[box, slot] > wp.float64(0.0):
            if counts[box, slot] != wp.float64(1.0):
                wp.atomic_max(failure, 0, 7)
                return
            kind = classify_ion_particle_wp(
                masses,
                charges,
                box,
                slot,
                positive_species,
                negative_species,
                positive_mass,
                negative_mass,
            )
            if kind == 3:
                wp.atomic_max(failure, 0, 7)
                return
            kinds[box, slot] = kind
            if kind == -1:
                presence[box, 0] = 1
            if kind == 1:
                presence[box, 1] = 1


@no_type_check
@wp.kernel
def cache_attachment_rates(
    kinds: wp.array2d(dtype=wp.int32),
    presence: wp.array2d(dtype=wp.int32),
    radii: wp.array2d(dtype=wp.float64),
    charges: wp.array2d(dtype=wp.float64),
    omitted: wp.int32,
    tables: wp.array3d(dtype=wp.float64),
    rates: wp.array3d(dtype=wp.float64),
    failure: wp.array(dtype=wp.int32),
):
    """Cache only needed polarities; reject invalid rates before sampling."""
    if failure[0] != 0:
        return
    box, slot, sign = wp.tid()
    rates[box, slot, sign] = wp.float64(0.0)
    if kinds[box, slot] == 0 and presence[box, sign] != 0:
        rate = ion_attachment_wp(
            radii[box, slot], charges[box, slot], 2 * sign - 1, omitted, tables
        )
        if rate < wp.float64(0.0):
            wp.atomic_max(failure, 0, 8)
            return
        rates[box, slot, sign] = rate


@no_type_check
@wp.kernel
def cache_mobility_rates(
    kinds: wp.array2d(dtype=wp.int32),
    presence: wp.array2d(dtype=wp.int32),
    radii: wp.array2d(dtype=wp.float64),
    charges: wp.array2d(dtype=wp.float64),
    masses: wp.array3d(dtype=wp.float64),
    temperature: wp.array(dtype=wp.float64),
    pressure: wp.array(dtype=wp.float64),
    positive_mass: wp.float64,
    negative_mass: wp.float64,
    positive_mobility: wp.float64,
    negative_mobility: wp.float64,
    ion_radius: wp.float64,
    rates: wp.array3d(dtype=wp.float64),
    failure: wp.array(dtype=wp.int32),
):
    """Refresh variable-T/P rates from current wet aerosol radii and charge."""
    if failure[0] != 0:
        return
    box, slot, sign = wp.tid()
    rates[box, slot, sign] = wp.float64(0.0)
    if kinds[box, slot] == 0 and presence[box, sign] != 0:
        ion_mass = negative_mass
        mobility = negative_mobility
        if sign == 1:
            ion_mass = positive_mass
            mobility = positive_mobility
        mass = wp.float64(0.0)
        for species in range(masses.shape[2]):
            mass += masses[box, slot, species]
        rate = mobility_attachment_wp(
            radii[box, slot],
            mass,
            charges[box, slot],
            temperature[box],
            pressure[box],
            ion_mass,
            mobility,
            wp.float64(2 * sign - 1),
            ion_radius,
        )
        if rate < wp.float64(0.0):
            wp.atomic_max(failure, 0, 8)
            return
        rates[box, slot, sign] = rate
