"""Complete native CPU runnable composition without facade construction."""

import numpy as np
from particula.dynamics import (
    Coagulation,
    Dilution,
    DilutionStrategy,
    MassCondensation,
    WallLoss,
)
from particula.dynamics.coagulation.coagulation_strategy.brownian_coagulation_strategy import (
    BrownianCoagulationStrategy,
)
from particula.dynamics.condensation.condensation_strategies import (
    CondensationIsothermal,
)
from particula.dynamics.condensation.species_mapping import (
    CondensationSpeciesMap,
)
from particula.dynamics.tests.native_nucleation_test import (
    fixture,
    inventory,
    runnable,
)
from particula.dynamics.wall_loss import ChargedWallLossStrategy
from particula.gas.vapor_pressure_strategies import (
    ConstantVaporPressureStrategy,
)
from particula.particles.activity_strategies import ActivityIdealMass
from particula.particles.surface_strategies import SurfaceStrategyVolume


def test_complete_native_cpu_sequence_conserves_each_stage(
    monkeypatch, forbid_legacy_facades
):
    """Real five-process composition accounts for dilution and wall sinks."""
    aerosol, config = fixture(capacity=8)
    particles, gas, environment = (
        aerosol.particles,
        aerosol.gas,
        aerosol.environment,
    )
    processes = (
        MassCondensation(
            CondensationIsothermal(
                molar_mass=gas.molar_mass.copy(),
                activity_strategy=ActivityIdealMass(),
                surface_strategy=SurfaceStrategyVolume(density=1000.0),
                vapor_pressure_strategy=ConstantVaporPressureStrategy(0.0),
                species_map=CondensationSpeciesMap(
                    tuple(gas.name), ((2, 0), (1, 1))
                ),
            )
        ),
        Coagulation(BrownianCoagulationStrategy("particle_resolved")),
        Dilution(DilutionStrategy(1e-7)),
        WallLoss(
            ChargedWallLossStrategy(
                1e-3,
                "spherical",
                chamber_radius=1e-5,
                wall_potential=0,
                distribution_type="particle_resolved",
            )
        ),
        runnable(aerosol, config),
    )
    events = []
    for index, process in enumerate(processes):
        execute = process.execute

        def observed(
            state, time_step, sub_steps=1, *, execute=execute, index=index
        ):
            before = inventory(state).copy()
            before_particles = (
                np.sum(
                    particles.masses * particles.concentration[..., None],
                    axis=1,
                )
                / particles.volume[:, None]
            )
            result = execute(state, time_step, sub_steps)
            assert result is aerosol
            if index in (0, 1, 4):
                np.testing.assert_allclose(
                    inventory(state), before, rtol=1e-12, atol=1e-30
                )
            elif index == 2:
                np.testing.assert_allclose(
                    inventory(state),
                    before * np.exp(-1e-7 * time_step),
                    rtol=1e-12,
                    atol=1e-30,
                )
            else:
                np.testing.assert_allclose(
                    inventory(state),
                    before - before_particles,
                    rtol=1e-12,
                    atol=1e-30,
                )
                np.testing.assert_array_equal(particles.concentration, 0)
                np.testing.assert_array_equal(particles.charge, 0)
            events.append(index)
            return result

        monkeypatch.setattr(process, "execute", observed)
    sequence = processes[0]
    for process in processes[1:]:
        sequence = sequence | process
    assert sequence.execute(aerosol, 1e6) is aerosol
    assert events == list(range(5))
    assert aerosol.particles is particles and aerosol.gas is gas
    assert aerosol.environment is environment
    assert np.any(particles.concentration > 0)
