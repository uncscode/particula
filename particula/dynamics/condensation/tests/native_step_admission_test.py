"""Current-state rejection, no-work paths and repeated native science steps."""

import numpy as np
import pytest
from particula.dynamics.condensation.condensation_strategies import (
    CondensationIsothermal,
    CondensationIsothermalStaggered,
    CondensationLatentHeat,
)
from particula.dynamics.condensation.tests.species_mapping_test import (
    make_state,
    native_strategy,
    snapshot,
)

FAMILIES = [
    CondensationIsothermal,
    CondensationIsothermalStaggered,
    CondensationLatentHeat,
]


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize(
    "fault",
    [
        "time_bool",
        "time_array",
        "temperature",
        "pressure",
        "viscosity",
        "map",
        "names",
        "flags",
        "mass",
        "density",
        "volume",
        "dtype",
        "readonly_mass",
        "readonly_gas",
        "alias_gas",
        "skip",
        "activity",
    ],
)
def test_native_step_rechecks_mutated_inputs_before_writes(family, fault):
    """All families revalidate physical state/configuration at the step boundary."""
    particle, gas, mapping = make_state()
    strategy = native_strategy(family, gas, mapping)
    parameters = dict(
        temperature=298.0,
        pressure=101325.0,
        time_step=0.01,
        dynamic_viscosity=1.8e-5,
    )
    controls = {
        "time_bool": {"time_step": True},
        "time_array": {"time_step": np.array([0.01])},
        "temperature": {"temperature": True},
        "pressure": {"pressure": True},
        "viscosity": {"dynamic_viscosity": True},
    }
    mutations = {
        "map": lambda: setattr(strategy, "species_map", None),
        "names": gas.name.reverse,
        "flags": lambda: setattr(
            gas, "partitioning", gas.partitioning.astype(np.int32)
        ),
        "mass": lambda: particle.masses.__setitem__((0, 0, 0), -1),
        "density": lambda: particle.density.__setitem__(0, 0),
        "volume": lambda: particle.volume.__setitem__(0, 0),
        "dtype": lambda: setattr(
            particle, "masses", particle.masses.astype(np.float32)
        ),
        "readonly_mass": lambda: setattr(
            particle.masses.flags, "writeable", False
        ),
        "readonly_gas": lambda: setattr(
            gas.concentration.flags, "writeable", False
        ),
        "alias_gas": lambda: setattr(
            gas, "concentration", particle.masses.ravel()[:3].reshape(1, 3)
        ),
        "skip": lambda: setattr(strategy, "skip_partitioning_indices", [3]),
        "activity": lambda: setattr(strategy, "activity_strategy", None),
    }
    if fault in controls:
        parameters.update(controls[fault])
    else:
        mutations[fault]()
    before = snapshot(particle, gas)
    if family is CondensationLatentHeat:
        strategy.last_latent_heat_energy = 123.0
    with pytest.raises((TypeError, ValueError)):
        strategy.step(particle, gas, **parameters)
    assert snapshot(particle, gas) == before
    if family is CondensationLatentHeat:
        assert strategy.last_latent_heat_energy == 123.0


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize(
    "case", ["zero_time", "zero_weights", "disabled", "empty"]
)
def test_valid_no_work_preserves_native_storage(family, case):
    """Empty resolved populations, inactive weights and flags remain unchanged."""
    particle, gas, mapping = make_state()
    dt = 0.01
    if case == "zero_time":
        dt = 0
    elif case == "zero_weights":
        particle.concentration[:] = 0
    elif case == "disabled":
        gas.partitioning[:] = False
    else:
        particle.masses = np.empty((1, 0, 2), dtype=np.float64)
        particle.concentration = np.empty((1, 0), dtype=np.float64)
        particle.charge = np.empty((1, 0), dtype=np.float64)
    strategy = native_strategy(family, gas, mapping)
    before = snapshot(particle, gas)
    result = strategy.step(particle, gas, 298, 101325, dt, 1.8e-5)
    assert result[0] is particle and result[1] is gas
    assert snapshot(particle, gas) == before


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("volume", [0.25, 1.0, 4.0])
def test_repeated_native_steps_conserve_each_species(family, volume):
    """Changing material state across calls remains finite and conservative."""
    particle, gas, mapping = make_state(volume=volume)
    gas.concentration[0, 1:] = [2e-18, 3e-18]
    counts = particle.concentration[0].copy()
    initial = np.sum(particle.masses[0] * counts[:, None], axis=0)
    initial += gas.concentration[0, [2, 1]] * volume
    strategy = native_strategy(family, gas, mapping)
    for _ in range(40):
        strategy.step(particle, gas, 298, 101325, 0.01, 1.8e-5)
        assert np.all(np.isfinite(particle.masses)) and np.all(
            particle.masses >= 0
        )
        assert np.all(np.isfinite(gas.concentration)) and np.all(
            gas.concentration >= 0
        )
        current = np.sum(particle.masses[0] * counts[:, None], axis=0)
        current += gas.concentration[0, [2, 1]] * volume
        np.testing.assert_allclose(current, initial, rtol=1e-12, atol=1e-30)
