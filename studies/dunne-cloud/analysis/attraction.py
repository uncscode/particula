"""Particula collision coefficient between a hydrated acid molecule and particles."""

from functools import lru_cache
import numpy as np
import particula as par


@lru_cache(None)
def acid_pair(
    temp, pressure, hamaker, hydrate=1.0, density=1500.0, upper=12.0, count=111
):
    diameters = np.geomspace(1.0, upper, count)
    masses = np.r_[
        (98.079 + hydrate * 18.01528) * 1.66053906660e-27,
        density * np.pi / 6 * (diameters * 1.0e-9) ** 3,
    ]
    pop = par.particles.ParticleData(
        masses=masses[None, :, None],
        density=np.array([density]),
        charge=np.zeros((1, len(masses))),
        concentration=np.ones((1, len(masses))),
        volume=np.ones(1),
        distribution_type="discrete",
    )
    strategy = par.dynamics.CoagulationFactory().get_strategy(
        "van_der_waals",
        dict(
            distribution_type="discrete",
            hamaker_constant=hamaker,
            relative_permittivity=1.0,
        ),
    )
    return diameters, strategy.kernel(pop, float(temp), float(pressure))[0, 1:]
