"""Helper routines for condensation and evaporation mass-transfer
calculations.

These helpers isolate common logic from :pymod:`mass_transfer` so that each
computational step can be unit-tested independently.  All functions operate
on NumPy arrays describing either mass change or particle properties and
enforce physical limits such as available gas mass or particle inventory.

References:
    - P. Hinds, *Aerosol Technology*, 2nd ed., Wiley-Interscience, 1999.
"""

import numpy as np
from numpy.typing import NDArray


def calc_mass_to_change(
    mass_rate: NDArray[np.float64],
    time_step: float,
    particle_concentration: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Calculate the requested mass change for every particle/species pair.

    The instantaneous mass-transfer rate (ṁ) is integrated over a time
    interval (Δt) and scaled by the particle number concentration (C) for
    each bin and species:

    Δm = ṁ × Δt × C

    Args:
        mass_rate: Mass transfer rate (ṁ) for each particle or ``(particle,
            species)`` pair in kg s⁻¹.
        time_step: Time step Δt in seconds.
        particle_concentration: Number concentration C in m⁻³.

    Returns:
        Requested mass change Δm for every particle/species combination in kg.

    Examples:
        ```py title="Scalar rate"
        import numpy as np
        from particula.dynamics.condensation import mass_transfer_utils as mtu
        mass_rate = np.array([1.0e-15])          # kg/s
        dm = mtu.calc_mass_to_change(mass_rate, 10.0, np.array([1.0e6]))
        # dm ≈ 1.0e-8 kg
        ```

    References:
        - "Mass transfer",
          [Wikipedia](https://en.wikipedia.org/wiki/Mass_transfer)
    """
    if mass_rate.ndim == 2:
        return mass_rate * time_step * particle_concentration[:, None]
    return mass_rate * time_step * particle_concentration


def apply_condensation_limit(
    mass_to_change: NDArray[np.float64],
    gas_mass: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Limit condensation so that total uptake does not exceed available gas.

    Positive mass changes (condensation) are summed per species and compared
    to the available gas mass. When the requested uptake exceeds the gas
    inventory, a scaling factor preserves conservation by clipping the
    condensation while leaving evaporation untouched.

    Args:
        mass_to_change: Requested mass change per bin and species in kg.
        gas_mass: Total gas mass available for condensation in kg.

    Returns:
        mass_to_change_scaled: Mass change array after applying the scaling
            factor for condensation.
        evap_sum: Column sum of evaporation (negative Δm) per species.
        neg_mask: Boolean mask identifying evaporation elements.

    Examples:
        ```py title="Insufficient gas example"
        scaled_dm, evap, mask = mtu.apply_condensation_limit(
            mass_to_change=np.array([[2.0e-9, -1.0e-9]]),
            gas_mass=np.array([5.0e-10])
        )
        ```
    """
    pos_mask = mass_to_change > 0.0
    neg_mask = mass_to_change < 0.0
    cond_sum = np.where(pos_mask, mass_to_change, 0.0).sum(axis=0)
    evap_sum = np.where(neg_mask, mass_to_change, 0.0).sum(axis=0)
    gas_mass_array = np.asarray(gas_mass, dtype=np.float64)
    cond_scale = np.ones_like(np.atleast_1d(cond_sum))
    if np.ndim(cond_sum) == 0:
        need_scale = (cond_sum > 0.0) & (cond_sum + evap_sum > gas_mass_array)
        if need_scale:
            gas_mass_scalar = float(gas_mass_array.reshape(-1)[0])
            cond_scale = np.array(
                [(gas_mass_scalar - float(evap_sum)) / float(cond_sum)]
            )
    else:
        gas_mass_aligned = np.broadcast_to(
            np.atleast_1d(gas_mass_array), cond_sum.shape
        )
        need_scale = (cond_sum > 0.0) & (cond_sum + evap_sum > gas_mass_aligned)
        cond_scale[need_scale] = (
            gas_mass_aligned[need_scale] - evap_sum[need_scale]
        ) / cond_sum[need_scale]
    cond_scale = np.clip(cond_scale, 0.0, 1.0)
    mass_to_change = np.where(
        pos_mask, mass_to_change * cond_scale, mass_to_change
    )
    return mass_to_change, evap_sum, neg_mask


def apply_evaporation_limit(
    mass_to_change: NDArray[np.float64],
    particle_mass: NDArray[np.float64],
    particle_concentration: NDArray[np.float64],
    evap_sum: NDArray[np.float64],
    neg_mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Limit evaporation so that mass loss honors the particle inventory.

    The available inventory (I) per species is the sum of particle mass scaled
    by particle concentration. When the requested evaporation exceeds the
    inventory, the negative mass changes are scaled to ensure no bin loses more
    mass than exists.

    Args:
        mass_to_change: Candidate mass change per bin/species in kg.
        particle_mass: Mass of one particle in each bin/species in kg.
        particle_concentration: Number concentration in m⁻³.
        evap_sum: Column sum of evaporation in kg (negative values).
        neg_mask: Boolean mask identifying evaporation entries.

    Returns:
        Mass change array with evaporation scaled to the available inventory.

    Examples:
        ```py title="Inventory-limited evaporation"
        limited_dm = mtu.apply_evaporation_limit(
            mass_to_change, particle_mass, particle_conc,
            evap_sum, neg_mask
        )
        ```
    """
    if particle_mass.ndim == 2:
        inventory = (particle_mass * particle_concentration[:, None]).sum(
            axis=0
        )
    else:
        inventory = (particle_mass * particle_concentration).sum()
    inventory_array = np.asarray(inventory, dtype=np.float64)
    evap_scale = np.ones_like(np.atleast_1d(evap_sum))
    if np.ndim(evap_sum) == 0:
        need_scale = -evap_sum > inventory_array
        if need_scale:
            inventory_scalar = float(inventory_array.reshape(-1)[0])
            evap_scale = np.array([inventory_scalar / (-evap_sum)])
    else:
        inventory_aligned = np.broadcast_to(
            np.atleast_1d(inventory_array), evap_sum.shape
        )
        need_scale = -evap_sum > inventory_aligned
        evap_scale[need_scale] = inventory_aligned[need_scale] / (
            -evap_sum[need_scale]
        )
    return np.where(neg_mask, mass_to_change * evap_scale, mass_to_change)


def apply_per_bin_limit(
    mass_to_change: NDArray[np.float64],
    particle_mass: NDArray[np.float64],
    particle_concentration: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Ensure no bin loses more mass than it contains.

    For each bin the maximum allowable evaporation is the particle mass per
    bin multiplied by the bin concentration. Any requested mass change that
    would deplete the bin beyond its inventory is clipped at that limit.

    Args:
        mass_to_change: Proposed mass change per bin/species in kg.
        particle_mass: Mass of a single particle per bin/species in kg.
        particle_concentration: Number concentration in m⁻³.

    Returns:
        Mass change array after applying the per-bin evaporation limit.

    Examples:
        ```py title="Per-bin clipping"
        clipped_dm = mtu.apply_per_bin_limit(
            mass_to_change, particle_mass, particle_conc
        )
        ```
    """
    if mass_to_change.ndim == 2:
        limit = -particle_mass * particle_concentration[:, None]
    else:
        limit = -particle_mass * particle_concentration
    return np.maximum(mass_to_change, limit)
