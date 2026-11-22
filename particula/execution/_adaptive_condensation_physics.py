"""Native material-resolved isothermal rates for adaptive device integration."""

import warp as wp

from particula.execution._condensation_ode_kernels import (
    FloatMatrix,
    FloatTensor,
    FloatVector,
    IntVector,
)
from particula.gpu.kernels import condensation as physics
from particula.util.constants import GAS_CONSTANT, MOLECULAR_WEIGHT_AIR

_R = wp.constant(wp.float64(GAS_CONSTANT))
_AIR = wp.constant(wp.float64(MOLECULAR_WEIGHT_AIR))


@wp.kernel
def material_rates(
    masses: FloatTensor,
    counts: FloatMatrix,
    density: FloatVector,
    gas: FloatMatrix,
    vapor: FloatMatrix,
    molar: FloatVector,
    tension: FloatVector,
    surface_density: FloatVector,
    surface_reference: FloatVector,
    phases: IntVector,
    references: FloatVector,
    activity_density: FloatVector,
    kappas: FloatVector,
    activity_parameters: FloatVector,
    activity_mode: wp.int32,
    water: wp.int32,
    accommodation: FloatVector,
    diffusion: FloatVector,
    mean_free_path: FloatVector,
    temperature: FloatVector,
    forward: IntVector,
    rate: FloatTensor,
):
    """Evaluate all exchanging lanes, including phase-weighted surface tension.

    Masses are kg per particle, gas kg/m3, densities kg/m3, diffusivity m2/s,
    tensions N/m, molar masses kg/mol, temperature K, and rates kg/s.
    The caller validates the physical trial before accepting it.
    """
    box, slot, material = wp.tid()  # type: ignore[misc]
    source = forward[material]
    rate[box, slot, material] = wp.float64(0.0)
    if source < 0 or counts[box, slot] == wp.float64(0.0):
        return
    volume = wp.float64(0.0)
    amount = wp.float64(0.0)
    weighted_tension = wp.float64(0.0)
    weight = wp.float64(0.0)
    unweighted_tension = wp.float64(0.0)
    members = wp.float64(0.0)
    for m in range(masses.shape[2]):
        value = masses[box, slot, m]
        volume += value / density[m]
        amount += value / references[m]
        if phases[m] == phases[material]:
            w = value / surface_reference[m]
            weighted_tension += w * tension[m]
            weight += w
            unweighted_tension += tension[m]
            members += wp.float64(1.0)
    if volume <= wp.float64(0.0) or amount <= wp.float64(0.0):  # type: ignore[operator]
        return
    sigma = unweighted_tension / members
    if weight > wp.float64(0.0):  # type: ignore[operator]
        sigma = weighted_tension / weight
    activity = masses[box, slot, material] / references[material] / amount
    if activity_mode == 1 and material == water:
        activity = physics.water_activity_kappa_wp(
            masses, activity_density, kappas, box, slot, int(water)
        )
    elif activity_mode == 3:
        activity = activity_parameters[
            (box * masses.shape[1] + slot) * masses.shape[2] + material
        ]
    elif activity_mode == 2:
        organic_fraction = masses[box, slot, 1] / references[1] / amount
        activity = physics.binary_activity_wp(
            activity_parameters, organic_fraction, material
        )
    radius = wp.max(
        physics.particle_radius_from_volume_wp(volume), wp.float64(1e-10)
    )
    knudsen = mean_free_path[box] / radius * wp.sqrt(_AIR / molar[source])
    transition = physics.vapor_transition_correction_wp(
        knudsen, accommodation[source]
    )
    transport = physics.first_order_mass_transport_k_wp(
        radius, transition, diffusion[source]
    )
    kelvin_radius = physics.kelvin_radius_wp(
        sigma, surface_density[material], molar[source], temperature[box], _R
    )
    equilibrium = (
        activity
        * vapor[box, source]
        * physics.kelvin_term_wp(radius, kelvin_radius)
    )
    partial = physics.partial_pressure_wp(
        gas[box, source], molar[source], temperature[box], _R
    )
    delta = partial - equilibrium
    # Matches native CPU nan_to_num pressure-difference semantics.
    if not wp.isfinite(delta):
        delta = wp.float64(0.0)
    rate[box, slot, material] = physics.mass_transfer_rate_wp(
        delta, transport, temperature[box], molar[source], _R
    )
