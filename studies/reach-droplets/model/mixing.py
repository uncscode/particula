"""Repack cell inventories for native Particula droplet condensation."""

import numpy as np
from simulate import par, RHO, MODE


def repack(groups, volume):
    n = max(1, max(map(len, groups)))
    mass = np.zeros((len(groups), n))
    count = np.zeros_like(mass)
    for j, g in enumerate(groups):
        mass[j, : len(g)] = g
        count[j, : len(g)] = 1.0
    return par.particles.ParticleData(
        masses=mass[:, :, None],
        density=np.array([RHO]),
        concentration=count,
        charge=np.zeros_like(mass),
        volume=volume,
        distribution_type=MODE,
    )
