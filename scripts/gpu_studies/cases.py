"""Shared GPU study workloads and independent inventory diagnostics."""

import numpy as np
from particula.execution.tests.gpu_study_cases import (
    StudyCase,
    build_case,
    water_pressure,
)

__all__ = [
    "StudyCase",
    "build_case",
    "particle_inventory",
    "radius_histogram",
    "water_pressure",
]


def particle_inventory(aerosol):
    """Independent extensive mass [kg] and represented number per box."""
    p = aerosol.particles
    return np.sum(
        p.masses * p.concentration[..., None], axis=1
    ), p.concentration.sum(axis=1)


def radius_histogram(aerosol, edges=None):
    """Number concentration in logarithmic radius bins, resolved by box."""
    edges = np.geomspace(0.2e-9, 2e-6, 65) if edges is None else edges
    p = aerosol.particles
    radius = np.cbrt(np.sum(p.masses / p.density, axis=-1) * 3 / (4 * np.pi))
    rows = [
        np.histogram(r, bins=edges, weights=c / v)[0]
        for r, c, v in zip(radius, p.concentration, p.volume, strict=True)
    ]
    return edges, np.asarray(rows)
