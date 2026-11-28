"""Fixed-pivot condensation of pure-water droplets on a wet-volume grid."""

import math

import torch

# kg/m³ per (droplet/cm³ × µm³ of diameter cubed), for liquid water.
WATER_MASS = 1000 * math.pi / 6 * 1e-12


def remap(pop, new_d2, volume):
    """Return cell inventories and liquid uptake in kg/m³ of chamber volume.

    An evaporating drop below the first pivot exchanges mass with a zero-volume
    (disappeared) pivot. Growth stops at the last pivot, leaving excess vapor
    in the gas. Inside the grid both number and liquid volume are conserved.
    """
    target = new_d2.clamp_min(0).pow(1.5).clamp_max(volume[-1])
    hi = torch.searchsorted(volume, target).clamp_max(len(volume) - 1)
    lo = (hi - 1).clamp_min(0)
    has_lo = hi > 0
    lower = torch.where(has_lo, volume[lo], 0.0)
    upper_weight = (target - lower) / (volume[hi] - lower)
    wet = torch.zeros_like(pop).scatter_add(1, hi, pop * upper_weight)
    wet = wet.scatter_add(1, lo, pop * (1 - upper_weight) * has_lo)
    uptake = WATER_MASS * ((target - volume) * pop).sum(1)
    return wet, uptake


def budget_error(pop, wet, volume, gas, wetgas, cell_fraction):
    """Largest cell-local gas-plus-liquid residual, kg/m³ of cell volume."""
    liquid_change = WATER_MASS * ((wet - pop) * volume).sum(1) / cell_fraction
    return (liquid_change + (wetgas - gas)).abs().max()
