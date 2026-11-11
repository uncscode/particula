"""Direct data fixtures shared by both native CPU wall-loss suites."""

import numpy as np
from particula.particles import ParticleData


def _particle(kind="discrete", volume=1.0, size=64):
    radius = np.geomspace(1e-8, 1e-5, size)
    mass = 4 * np.pi * 1000 * radius**3 / 3
    pdf = 1e10 * np.exp(-0.5 * (np.log(radius / 1e-7) / 0.5) ** 2)
    pdf /= radius * 0.5 * np.sqrt(2 * np.pi)
    if kind == "particle_resolved":
        concentration = np.ones(size)
    elif kind == "continuous_pdf":
        concentration = pdf
    else:
        concentration = pdf * np.diff(
            radius, append=2 * radius[-1] - radius[-2]
        )
    return ParticleData(
        masses=mass[None, :, None],
        concentration=concentration[None],
        charge=np.zeros((1, size)),
        density=np.array([1000.0]),
        volume=np.array([volume]),
        distribution_type=kind,
        radius_grid=radius if kind == "continuous_pdf" else None,
    )


def _concentration(particle):
    return particle.concentration[0] / particle.volume[0]


def _total(particle):
    concentration = _concentration(particle)
    if particle.distribution_type == "continuous_pdf":
        return np.sum(
            (concentration[1:] + concentration[:-1])
            * np.diff(particle.radius_grid)
            / 2
        )
    return concentration.sum()
