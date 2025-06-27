"""
Helper routines for condensation and evaporation mass-transfer
calculations.

These helpers isolate common logic from :pymod:`mass_transfer` so that each
computational step can be unit-tested independently.  All functions operate
on NumPy arrays describing either mass change or particle properties and
enforce physical limits such as available gas mass or particle inventory.

References:
    - P. Hinds, *Aerosol Technology*, 2nd ed., Wiley-Interscience, 1999.
"""

from numpy.typing import NDArray
import numpy as np


def calc_mass_to_change(
    mass_rate: NDArray[np.float64],
    time_step: float,
    particle_concentration: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Calculate the requested mass change for every particle/species pair.

    The instantaneous mass-transfer rate (ṁ) is integrated over a time
    interval (Δt) and scaled by the particle number concentration (C):

    - Δm = ṁ × Δt × C
        - Δm is the mass change per bin or per (bin, species) in kg,
        - ṁ is the mass rate in kg s⁻¹,
        - Δt is the time step in seconds,
        - C is the particle concentration in m⁻³.

    Arguments:
        - mass_rate : Mass transfer rate (ṁ) for each particle or
          ``(particle, species)`` pair in kg s⁻¹.
        - time_step : Time interval Δt in seconds.
        - particle_concentration : Number concentration C in m⁻³.

    Returns:
        - Requested mass change Δm for every particle/species combination
          in kg.

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
    """
    Limit condensation so that total uptake never exceeds available gas.

    For each chemical species the positive mass change (condensation) is
    summed (Σ_cond).  If Σ_cond + Σ_evap exceeds the available gas mass
    (M_g), a scaling factor is applied:

    - scale = (M_g − Σ_evap) / Σ_cond   if Σ_cond > 0
    - scale = 1                         otherwise

    Arguments:
        - mass_to_change : Requested mass change per bin and species in kg.
        - gas_mass : Total gas mass available for condensation in kg.

    Returns:
        - mass_to_change_scaled : Mass change array after scaling.
        - evap_sum : Column sum of evaporation (negative Δm) per species.
        - neg_mask : Boolean mask identifying evaporation elements.

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
    cond_scale = np.ones_like(np.atleast_1d(cond_sum))
    need_scale = (cond_sum > 0.0) & (cond_sum + evap_sum > gas_mass)
    if np.ndim(cond_sum) == 0:
        if need_scale:
            cond_scale = np.array(
                [(gas_mass.item() - evap_sum.item()) / cond_sum.item()]
            )
    else:
        cond_scale[need_scale] = (
            gas_mass[need_scale] - evap_sum[need_scale]
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
    """
    Limit evaporation so that total loss does not exceed particle inventory.

    The available inventory (I) per species is the sum of particle mass
    multiplied by particle concentration.  If −Σ_evap > I the evaporation
    terms are scaled:

    - scale = I / (−Σ_evap)

    Arguments:
        - mass_to_change : Candidate mass change per bin/species in kg.
        - particle_mass : Mass of a single particle in each bin/species in
          kg.
        - particle_concentration : Number concentration in m⁻³.
        - evap_sum : Column sum of evaporation in kg (negative values).
        - neg_mask : Boolean mask indicating evaporation entries.

    Returns:
        - Mass change array after enforcing the inventory limit.

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
    evap_scale = np.ones_like(np.atleast_1d(evap_sum))
    need_scale = -evap_sum > inventory
    if np.ndim(evap_sum) == 0:
        if need_scale:
            evap_scale = np.array([inventory / (-evap_sum)])
    else:
        evap_scale[need_scale] = inventory[need_scale] / (
            -evap_sum[need_scale]
        )
    return np.where(neg_mask, mass_to_change * evap_scale, mass_to_change)


def apply_per_bin_limit(
    mass_to_change: NDArray[np.float64],
    particle_mass: NDArray[np.float64],
    particle_concentration: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Ensure no single bin loses more mass than it contains.

    For every bin the maximum allowable evaporation is the total mass
    present in that bin:

    - limit = −m_p × C

    where *m_p* is the particle mass and *C* the particle concentration.
    Any requested mass change lower than this limit is clipped.

    Arguments:
        - mass_to_change : Proposed mass change per bin/species in kg.
        - particle_mass : Mass of one particle per bin/species in kg.
        - particle_concentration : Number concentration in m⁻³.

    Returns:
        - Mass change array after applying the per-bin limit.

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
