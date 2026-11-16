"""Independent radius-space quadrature reference for isothermal growth."""

import numpy as np
import numpy.testing as npt
import pytest
from particula.dynamics.condensation.condensation_strategies import (
    CondensationIsothermal,
)
from particula.dynamics.condensation.species_mapping import (
    CondensationSpeciesMap,
)
from particula.dynamics.dilution import get_dilution_step
from particula.gas import GasData
from particula.gas.vapor_pressure_strategies import (
    ConstantVaporPressureStrategy,
)
from particula.particles import create_resolved_particle_data
from particula.particles.activity_strategies import ActivityIdealMass
from particula.particles.surface_strategies import SurfaceStrategyVolume
from scipy.integrate import quad


@pytest.mark.parametrize(
    "closed,order",
    [(False, None), (True, None), (True, "before"), (True, "after")],
)
def test_condensation_converges_to_independent_radius_quadrature(
    closed, order, record_property
):
    """Time refinement approaches a separated-variable solution, with budgets.

    Zero surface vapor pressure isolates transport. For a monodisperse sphere,
    dm/dt=4*pi*r*D*f*C and m=4*pi*rho*r**3/3 imply
    dt/dx=rho*r0**2*x/(D*f*C), where x=r/r0. In a closed box,
    C=C0-N*m0*(x**3-1). Quadrature integrates this inverse equation, without
    calling the production rate or time integrator. Tolerances concern this
    smooth, non-depleting case only; they do not qualify stiff condensation.
    """
    radius, density, diffusion = 1e-6, 1000.0, 2e-5
    temperature, pressure, viscosity, molar_mass = (
        298.15,
        101325.0,
        1.8e-5,
        0.018,
    )
    gas_initial, number_density, target = 1e-5, 1e9, 1.2
    mass = 4 * np.pi * density * radius**3 / 3
    mean_free_path = (2 * viscosity / pressure) / np.sqrt(
        8 * molar_mass / (np.pi * 8.31446261815324 * temperature)
    )

    def inverse_growth(x):
        kn = mean_free_path / (radius * x)
        correction = 0.75 * (1 + kn) / (kn**2 + 1.283 * kn + 0.75)
        vapor = gas_initial - (
            number_density * mass * (x**3 - 1) if closed else 0
        )
        return density * radius**2 * x / (diffusion * correction * vapor)

    duration, quadrature_error = quad(
        inverse_growth, 1, target, epsabs=1e-12, epsrel=1e-12
    )
    assert quadrature_error < 1e-10 * duration
    # With common dilution hazard k, N and total material decay as exp(-kt).
    # The growth equation uses effective time (1-exp(-kt))/k.
    hazard = 0.2 / duration if order else 0.0
    if order:
        duration = -np.log1p(-hazard * duration) / hazard
    errors = []
    for steps in (16, 32, 64):
        particle = create_resolved_particle_data(
            masses=np.array([[mass]]),
            density=np.array([density]),
            concentration=number_density,
            volume=1.0,
        )
        gas = GasData(
            name=["vapor"],
            molar_mass=np.array([molar_mass]),
            concentration=np.array([[gas_initial]]),
            partitioning=np.array([True]),
        )
        strategy = CondensationIsothermal(
            molar_mass=molar_mass,
            diffusion_coefficient=diffusion,
            update_gases=closed,
            activity_strategy=ActivityIdealMass(),
            surface_strategy=SurfaceStrategyVolume(density=density),
            vapor_pressure_strategy=ConstantVaporPressureStrategy(0.0),
            species_map=CondensationSpeciesMap(("vapor",), ((0, 0),)),
        )

        def dilute(particle=particle, gas=gas, steps=steps):
            particle.concentration[:] = get_dilution_step(
                hazard, particle.concentration, duration / steps
            )
            gas.concentration[:] = get_dilution_step(
                hazard, gas.concentration, duration / steps
            )

        for _ in range(steps):
            if order == "before":
                dilute()
            strategy.step(
                particle,
                gas,
                temperature,
                pressure,
                duration / steps,
                viscosity,
            )
            if order == "after":
                dilute()
        errors.append(abs(particle.radii[0, 0] / radius - target))
        if closed:
            npt.assert_allclose(
                gas.concentration[0, 0]
                + particle.concentration[0, 0] * particle.masses[0, 0, 0],
                (gas_initial + number_density * mass)
                * np.exp(-hazard * duration),
                rtol=2e-14,
                atol=0,
            )
        else:
            npt.assert_array_equal(gas.concentration, [[gas_initial]])
    record_property("radius_errors_16_32_64", repr(errors))
    record_property("duration_seconds", duration)
    assert errors[-1] < 0.002
    assert np.all(np.array(errors[1:]) < 0.6 * np.array(errors[:-1]))
