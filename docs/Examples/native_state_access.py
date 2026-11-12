"""Inspect and replace native CPU aerosol state without facade conversions.

Run from the v0.3.0 development checkout with
``python docs/Examples/native_state_access.py``. This single-box example uses
resolved counts in 0.25 m³ and full ordered gas storage including inert gas.
Replacement selects supplied state; it neither conserves the old inventory
automatically nor rebinds a prepared GPU or resident session.
"""

import numpy as np
import particula as par
from particula.dynamics.condensation.species_mapping import (
    CondensationSpeciesMap,
)


def build_aerosol() -> par.Aerosol:
    """Build a two-material population with three independently ordered gases."""
    particles = par.particles.create_resolved_particle_data(
        masses=[[2e-18, 3e-18], [4e-18, 5e-18]],
        density=[1000.0, 1200.0],
        concentration=[2.0, 2.0],
        volume=0.25,
    )
    gas = par.gas.GasData(
        name=["inert", "organic", "water"],
        molar_mass=np.array([0.028, 0.046, 0.018]),
        concentration=np.array([[9e-15, 2e-15, 3e-15]]),
        partitioning=np.array([False, True, True]),
    )
    environment = par.gas.EnvironmentData(
        temperature=[298.15],
        pressure=[101325.0],
        saturation_ratio=[[0.0, 1.0, 1.0]],
    )
    return par.Aerosol(particles=particles, gas=gas, environment=environment)


def build_condensation(gas: par.gas.GasData) -> par.dynamics.MassCondensation:
    """Configure named water/organic lanes against the complete gas order."""
    mapping = CondensationSpeciesMap(
        tuple(gas.name),
        ((gas.name.index("water"), 0), (gas.name.index("organic"), 1)),
    )
    return par.dynamics.MassCondensation(
        par.dynamics.CondensationIsothermal(
            molar_mass=gas.molar_mass.copy(),
            activity_strategy=par.particles.ActivityIdealMass(),
            surface_strategy=par.particles.SurfaceStrategyVolume(
                density=1000.0
            ),
            vapor_pressure_strategy=par.gas.ConstantVaporPressureStrategy(0.0),
            species_map=mapping,
        )
    )


def replacement_layout(aerosol: par.Aerosol) -> tuple:
    """Prepare detached candidates adding one inert gas and environment lane.

    The supplied aerosol is unchanged. Gas and environment must be published
    together because either individual width change would be incompatible.
    """
    gas = aerosol.gas
    return (
        aerosol.particles.copy(),
        par.gas.GasData(
            name=[*gas.name, "tracer"],
            molar_mass=np.append(gas.molar_mass, 0.04),
            concentration=np.column_stack((gas.concentration, [7e-15])),
            partitioning=np.append(gas.partitioning, False),
        ),
        par.gas.EnvironmentData(
            temperature=aerosol.environment.temperature.copy(),
            pressure=aerosol.environment.pressure.copy(),
            saturation_ratio=np.column_stack(
                (aerosol.environment.saturation_ratio, [0.0])
            ),
        ),
    )


def run_example() -> par.Aerosol:
    """Exercise identity, detached properties, replacement and mapped physics."""
    aerosol = build_aerosol()
    held = aerosol.particles
    counts = held.concentration  # Writable raw array, not number density.
    assert counts is aerosol.particles.concentration
    np.testing.assert_array_equal(held.number_density, [16.0])
    detached_density = held.slot_concentration_density
    detached_density[:] = 0.0
    np.testing.assert_array_equal(counts, [[2.0, 2.0]])

    # Individual compatible assignments retain exactly the supplied objects.
    particles, gas, environment = (
        held.copy(),
        aerosol.gas.copy(),
        aerosol.environment.copy(),
    )
    aerosol.particles = particles
    aerosol.gas = gas
    aerosol.environment = environment
    assert aerosol.particles is particles and aerosol.gas is gas
    assert aerosol.environment is environment

    candidates = replacement_layout(aerosol)
    aerosol.replace_data(
        particles=candidates[0], gas=candidates[1], environment=candidates[2]
    )
    assert aerosol.particles is candidates[0]
    assert aerosol.gas is candidates[1] and aerosol.environment is candidates[2]
    # CPU replacement does not adapt old process configuration. Construct new
    # process-owned mapping explicitly for the expanded full gas order.
    process = build_condensation(aerosol.gas)
    before = (
        aerosol.particles.species_mass_inventory
        + aerosol.gas.concentration[:, [2, 1]]
        * aerosol.particles.volume[:, None]
    )
    assert process.execute(aerosol, time_step=4e8, sub_steps=4) is aerosol
    after = (
        aerosol.particles.species_mass_inventory
        + aerosol.gas.concentration[:, [2, 1]]
        * aerosol.particles.volume[:, None]
    )
    np.testing.assert_allclose(after, before, rtol=1e-12, atol=1e-30)
    return aerosol


if __name__ == "__main__":
    run_example()
    print("Native access, replacement and mapped condensation checks passed.")
