"""LF2013 attachment on the device using conditioned published coefficients."""

from typing import no_type_check

import warp as wp

_MIN_RADIUS = wp.constant(wp.float64(2e-10))
_MAX_RADIUS = wp.constant(wp.float64(1e-5))
_CUTOFF = wp.constant(wp.float64(1e-15))


@no_type_check
@wp.func
def ion_attachment_wp(
    radius: wp.float64,
    charge: wp.float64,
    polarity: wp.int32,
    omitted: wp.int32,
    tables: wp.array3d(dtype=wp.float64),
) -> wp.float64:
    """Return m^3/s, or -1 for invalid/unsupported input requiring a fault.

    Omitted policy: 0 rejects; 1 returns the zero lower bound; 2 returns the
    published cutoff upper bound. Bounds apply only to omitted repulsive
    rates. The caller must validate the fixed temperature/material contract.
    """
    invalid = wp.float64(-1.0)
    if (
        not wp.isfinite(radius)
        or radius < _MIN_RADIUS
        or radius > _MAX_RADIUS
        or not wp.isfinite(charge)
        or charge != wp.floor(charge)
        or wp.abs(charge) > wp.float64(100.0)
        or (polarity != -1 and polarity != 1)
        or omitted < 0
        or omitted > 2
    ):
        return invalid
    sign_index = wp.int32(0)
    if polarity > 0:
        sign_index = wp.int32(1)
    row = wp.int32(charge) + wp.int32(100)
    low = tables[sign_index, row, 26]
    high = tables[sign_index, row, 27]
    if radius > high:
        return invalid
    if radius < low:
        if charge * wp.float64(polarity) <= wp.float64(0.0) or omitted == 0:
            return invalid
        if omitted == 1:
            return wp.float64(0.0)
        return _CUTOFF
    x = (wp.log10(radius) - tables[sign_index, row, 24]) / tables[
        sign_index, row, 25
    ]
    exponent = tables[sign_index, row, 23]
    for index in range(22, -1, -1):
        exponent = exponent * x + tables[sign_index, row, index]
    rate = wp.pow(wp.float64(10.0), exponent)
    if not wp.isfinite(rate) or rate <= wp.float64(0.0):
        return invalid
    return rate


@no_type_check
@wp.func
def classify_ion_particle_wp(
    masses: wp.array3d(dtype=wp.float64),
    charges: wp.array2d(dtype=wp.float64),
    box: wp.int32,
    slot: wp.int32,
    positive_species: wp.int32,
    negative_species: wp.int32,
    positive_mass: wp.float64,
    negative_mass: wp.float64,
) -> wp.int32:
    """Classify active particles with reserved ion material lanes.

    Return 0 for aerosol, +1/-1 for a free ion, 2 for an inert neutral
    recombination product, or 3 for unsupported ion-only composition/charge.
    The caller must validate species indices, positive monomer masses and
    native particle arrays. This function is only called for active slots.
    """
    for species in range(masses.shape[2]):
        if species != positive_species and species != negative_species:
            if masses[box, slot, species] > wp.float64(0.0):
                return wp.int32(0)
    positive = masses[box, slot, positive_species]
    negative = masses[box, slot, negative_species]
    charge = charges[box, slot]
    if positive == positive_mass and negative == wp.float64(0.0):
        if charge == wp.float64(1.0):
            return wp.int32(1)
    if negative == negative_mass and positive == wp.float64(0.0):
        if charge == wp.float64(-1.0):
            return wp.int32(-1)
    if positive == positive_mass and negative == negative_mass:
        if charge == wp.float64(0.0):
            return wp.int32(2)
    return wp.int32(3)


@no_type_check
@wp.func
def ion_collision_rate_wp(
    first_kind: wp.int32,
    second_kind: wp.int32,
    first_radius: wp.float64,
    second_radius: wp.float64,
    first_charge: wp.float64,
    second_charge: wp.float64,
    recombination: wp.float64,
    omitted: wp.int32,
    tables: wp.array3d(dtype=wp.float64),
) -> wp.float64:
    """Rate for pairs with at least one ion/product; -1 requires a fault.

    Neutral ion-only products are inert; same-sign ions do not cluster.
    Opposite free ions use the configured recombination coefficient [m^3/s].
    Ion-aerosol pairs use LF2013. Aerosol-aerosol pairs must be handled by the
    caller's charged Brownian model and reject here to prevent double counting.
    """
    if first_kind == 3 or second_kind == 3:
        return wp.float64(-1.0)
    if first_kind == 2 or second_kind == 2:
        return wp.float64(0.0)
    if first_kind == 0 and second_kind == 0:
        return wp.float64(-1.0)
    if first_kind == 0:
        return ion_attachment_wp(
            first_radius, first_charge, second_kind, omitted, tables
        )
    if second_kind == 0:
        return ion_attachment_wp(
            second_radius, second_charge, first_kind, omitted, tables
        )
    if first_kind == second_kind:
        return wp.float64(0.0)
    if not wp.isfinite(recombination) or recombination < wp.float64(0.0):
        return wp.float64(-1.0)
    return recombination
