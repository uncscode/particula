"""Helper routines for condensation and evaporation calculations.

These functions isolate common logic from :mod:`mass_transfer` so that each step can be tested independently. Each helper operates on arrays of mass change or particle properties and enforces specific physical limitations such as available gas mass or particle inventory.
"""

from numpy.typing import NDArray
import numpy as np


def calc_mass_to_change(
    mass_rate: NDArray[np.float64],
    time_step: float,
    particle_concentration: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return the mass change each particle would experience during `time_step`.

    Parameters
    ----------
    mass_rate : NDArray[np.float64]
        Mass transfer rate for each particle or `(particle, species)` pair
        in units of `kg/s`.
    time_step : float
        Size of the time step `[s]`.
    particle_concentration : NDArray[np.float64]
        Number concentration of particles `[#/m^3]`.

    Returns
    -------
    NDArray[np.float64]
        Requested mass change for every particle/species combination.
    """
    if mass_rate.ndim == 2:
        return mass_rate * time_step * particle_concentration[:, None]
    return mass_rate * time_step * particle_concentration


def apply_condensation_limit(
    mass_to_change: NDArray[np.float64],
    gas_mass: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Scale condensation so that the column sum never exceeds the available gas mass.

    Parameters
    ----------
    mass_to_change : NDArray[np.float64]
        Requested mass change for each bin and species ``[kg]``.
    gas_mass : NDArray[np.float64]
        Total gas mass available for condensation ``[kg]``.

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]
        The scaled mass change, the evaporation sum for each species, and
        a mask identifying evaporation elements.
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
    mass_to_change = np.where(pos_mask, mass_to_change * cond_scale, mass_to_change)
    return mass_to_change, evap_sum, neg_mask


def apply_evaporation_limit(
    mass_to_change: NDArray[np.float64],
    particle_mass: NDArray[np.float64],
    particle_concentration: NDArray[np.float64],
    evap_sum: NDArray[np.float64],
    neg_mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Scale evaporation so the column sum never exceeds the particle inventory."

    Parameters
    ----------
    mass_to_change : NDArray[np.float64]
        Candidate mass change for each bin and species ``[kg]``.
    particle_mass : NDArray[np.float64]
        Particle mass for each bin and species ``[kg]``.
    particle_concentration : NDArray[np.float64]
        Concentration of particles in each bin ``[#/m^3]``.
    evap_sum : NDArray[np.float64]
        Sum of evaporation (negative mass change) per species ``[kg]``.
    neg_mask : NDArray[np.bool_]
        Boolean mask indicating evaporation entries.

    Returns
    -------
    NDArray[np.float64]
        Mass change limited by the available particle mass.
    """
    if particle_mass.ndim == 2:
        inventory = (particle_mass * particle_concentration[:, None]).sum(axis=0)
    else:
        inventory = (particle_mass * particle_concentration).sum()
    evap_scale = np.ones_like(np.atleast_1d(evap_sum))
    need_scale = -evap_sum > inventory
    if np.ndim(evap_sum) == 0:
        if need_scale:
            evap_scale = np.array([inventory / (-evap_sum)])
    else:
        evap_scale[need_scale] = inventory[need_scale] / (-evap_sum[need_scale])
    return np.where(neg_mask, mass_to_change * evap_scale, mass_to_change)


def apply_per_bin_limit(
    mass_to_change: NDArray[np.float64],
    particle_mass: NDArray[np.float64],
    particle_concentration: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Clip evaporation so no single bin loses more mass than it contains.

    Parameters
    ----------
    mass_to_change : NDArray[np.float64]
        Proposed mass change for each bin and species ``[kg]``.
    particle_mass : NDArray[np.float64]
        Mass of each particle bin and species ``[kg]``.
    particle_concentration : NDArray[np.float64]
        Number concentration of particles ``[#/m^3]``.

    Returns
    -------
    NDArray[np.float64]
        Mass change after enforcing the per-bin evaporation limit.
    """
    if mass_to_change.ndim == 2:
        limit = -particle_mass * particle_concentration[:, None]
    else:
        limit = -particle_mass * particle_concentration
    return np.maximum(mass_to_change, limit)

