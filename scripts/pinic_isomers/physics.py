"""Closed-box absorption and its independent thermodynamic fixed point."""

import numpy as np
from particula import Aerosol
from particula.dynamics import CondensationIsothermal
from particula.dynamics.condensation.species_mapping import (
    CondensationSpeciesMap,
)
from particula.execution.gpu_condensation_ode import GPUCondensationODE
from particula.gas import EnvironmentData, GasData
from particula.gas.vapor_pressure_strategies import (
    ConstantVaporPressureStrategy,
)
from particula.particles import ParticleData
from particula.particles.activity_strategies import ActivityIdealMass
from particula.particles.surface_strategies import SurfaceStrategyVolume
from particula.util.constants import GAS_CONSTANT
from scipy.optimize import brentq

TEMP = 298.15
MOLAR = 0.18621
DENSITY = 1200.0
SIGMA = 0.03
RADIUS = 50e-9
TOTAL = 5.0  # micrograms per cubic metre, both exchangeable isomers
PINIC = 10 ** (-5691.7 / TEMP + 14.73) * MOLAR * 1e9 / (GAS_CONSTANT * TEMP)


def kelvin(seed, organic):
    """Kelvin correction for volume-additive uptake on equal 50 nm radii."""
    radius = RADIUS * np.cbrt(1 + organic / seed)
    return np.exp(2 * SIGMA * MOLAR / (GAS_CONSTANT * TEMP * DENSITY * radius))


def equilibrium(seed, totals, cstar):
    """Solve the ideal absorptive mass balance in micrograms per m³."""

    def residual(organic):
        absorber = seed + organic
        return organic - np.sum(
            totals * absorber / (absorber + cstar * kelvin(seed, organic))
        )

    return brentq(residual, 0.0, sum(totals), xtol=1e-13)


def effective_cstar(totals, cstar, seed=10.0):
    """Calibrate the one-surrogate model to the reference loading."""
    organic = equilibrium(seed, totals, cstar)
    return (
        (totals.sum() / organic - 1) * (seed + organic) / kelvin(seed, organic)
    )


def trajectory(seeds, totals, cstar, times, tolerance):
    """Advance a batch of monodisperse boxes using native CUDA condensation."""
    lanes = len(totals)
    boxes = len(seeds)
    molar = np.full(lanes, MOLAR)
    density = np.full(lanes + 1, DENSITY)
    core_mass = 4 * np.pi * DENSITY * RADIUS**3 / 3
    counts = seeds * 1e-9 / core_mass
    masses = np.zeros((boxes, 1, lanes + 1))
    masses[:, 0, 0] = core_mass
    names = [f"organic_{i}" for i in range(lanes)]
    particles = ParticleData(
        masses=masses,
        concentration=counts[:, None],
        charge=np.zeros((boxes, 1)),
        density=density,
        volume=np.ones(boxes),
        distribution_type="particle_resolved",
    )
    gas = GasData(
        name=names,
        molar_mass=molar,
        concentration=np.tile(totals * 1e-9, (boxes, 1)),
        partitioning=np.ones(lanes, dtype=bool),
    )
    env = EnvironmentData(
        temperature=np.full(boxes, TEMP),
        pressure=np.full(boxes, 101325.0),
        saturation_ratio=np.zeros((boxes, lanes)),
    )
    strategy = CondensationIsothermal(
        molar_mass=molar,
        diffusion_coefficient=5e-6,
        activity_strategy=ActivityIdealMass(),
        surface_strategy=SurfaceStrategyVolume(
            surface_tension=SIGMA, density=DENSITY
        ),
        vapor_pressure_strategy=ConstantVaporPressureStrategy(
            cstar * 1e-9 * GAS_CONSTANT * TEMP / MOLAR
        ),
        species_map=CondensationSpeciesMap(
            tuple(names), tuple((i, i + 1) for i in range(lanes))
        ),
    )
    owner = GPUCondensationODE(
        Aerosol(particles=particles, gas=gas, environment=env),
        strategy,
        device="cuda:0",
        method="ros3",
        rtol=tolerance,
        particle_atol=1e-28,
        gas_atol=1e-17,
        first_step=0.1,
        max_step=3600.0,
    )
    series = []
    budget = 0.0
    for end in times:
        while owner.time < end:
            owner.step(end)
        state = owner.snapshot()
        particle = state.particles.species_mass_density * 1e9
        vapor = state.gas.concentration * 1e9
        np.testing.assert_allclose(particle[:, 0], seeds, rtol=1e-12)
        budget = max(
            budget, float(np.max(abs((particle[:, 1:] + vapor) / totals - 1)))
        )
        if np.min(particle[:, 1:]) < -1e-12 or np.min(vapor) < -1e-12:
            raise ValueError("Negative organic inventory")
        series.append(particle[:, 1:].tolist())
    if budget > 1e-9:
        raise ValueError("Species conservation failed")
    return series, budget
