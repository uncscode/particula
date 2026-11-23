"""Public activity, reservoir and equilibrium runnable contracts."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from particula.equilibria import (
    Equilibria,
    EquilibriaFactory,
    EquilibriumResult,
)
from particula.particles.activity_factories import ActivityFactory
from particula.particles.activity_strategies import ActivityIdealMolar

aiomfac = pytest.importorskip("particula_aiomfac")

GROUPS = {
    "water": {16: 1},
    "alcohol": {141: 2, 151: 1, 153: 1},
    "salt": {204: 2, 261: 1},
}


def provider(**kwargs):
    """Create a reordered provider with one inert material."""
    return ActivityFactory().get_strategy(
        "aiomfac",
        {
            "mixture": aiomfac.MixtureSpec(GROUPS),
            "material_mapping": {3: "alcohol", 1: "water", 0: "salt"},
            "inert_indices": [2],
            **kwargs,
        },
    )


def test_named_neutral_activities_pressure_and_legacy_identity():
    """Check target ordering, temperature shapes and checkpoint modes."""
    native = np.array([0.18, 0.4, 0.05, 0.42])
    legacy = provider()
    stable = provider(phase_mode="stable")
    assert legacy.phase_mode == "single_liquid"
    assert deepcopy(legacy).prepared_identity() == legacy.prepared_identity()
    assert deepcopy(stable).prepared_identity() == stable.prepared_identity()
    assert stable.prepared_identity() != legacy.prepared_identity()
    phases = stable.phase_equilibrium(native, 303.15)
    masses = np.broadcast_to(native, (2, 1, 4))
    pressure = stable.partial_pressures_at_temperature(
        masses, [[303.15], [303.15]], ["alcohol", "water"], [2.0, 3.0]
    )
    expected = phases.alpha.neutral_activities[::-1] * [2.0, 3.0]
    np.testing.assert_allclose(pressure, np.broadcast_to(expected, (2, 1, 2)))
    with pytest.raises(ValueError, match="volatile neutral"):
        stable.activities_at_temperature(native, 303.15, ["salt"])
    with pytest.raises(ValueError, match="direct"):
        provider(phase_mode="stable", evaluation="hybrid", table="unused")
    ideal = ActivityIdealMolar(molar_mass=np.array([0.018, 0.06]))
    fraction = ideal.activities_at_temperature(
        [0.018, 0.06], 303.15, ["water"], material_names=["alcohol", "water"]
    )
    np.testing.assert_allclose(fraction, [0.5])


def test_closed_and_prescribed_reservoirs_through_the_runnable():
    """Recover the prescribed humidity while closing both vapor budgets."""
    strategy = EquilibriaFactory().get_strategy(
        "liquid_vapor", {"activity_model": provider()}
    )
    inputs = dict(
        temperature=303.15,
        saturation_concentrations={"water": 3e4, "alcohol": 0.3},
    )
    state = SimpleNamespace(
        partitioning_inputs=dict(
            inputs, inventories=[0.0, 7.0, 3.0], relative_humidity=0.77
        )
    )
    open_result = Equilibria(strategy).execute(state, 1.0).equilibria_result
    assert isinstance(open_result, EquilibriumResult)
    assert open_result.beta_phase is not None
    recovered = (
        open_result.alpha_phase.species_concentrations
        + open_result.beta_phase.species_concentrations
        + open_result.gas_concentrations
    )
    np.testing.assert_allclose(
        recovered, [open_result.water_exchanged, 7.0, 3.0], rtol=1e-12
    )
    closed = strategy.solve(inventories=recovered, **inputs)
    assert closed.water_exchanged == 0
    assert closed.relative_humidity == pytest.approx(0.77, abs=1e-8)
    np.testing.assert_allclose(
        closed.gas_concentrations, open_result.gas_concentrations, rtol=1e-8
    )
    np.testing.assert_allclose(
        closed.liquids.beta_masses, open_result.liquids.beta_masses, rtol=1e-6
    )


def test_pure_water_condensation_and_disappearance():
    """Verify an analytic dew point and the all-vapor branch."""
    strategy = EquilibriaFactory().get_strategy(
        "liquid_vapor", {"activity_model": provider()}
    )
    params = dict(
        saturation_concentrations={"water": 100.0}, temperature=303.15
    )
    wet = strategy.solve(inventories=[250.0, 0.0, 0.0], **params)
    np.testing.assert_allclose(wet.gas_concentrations, [100.0, 0.0, 0.0])
    assert wet.relative_humidity == pytest.approx(1.0)
    assert wet.alpha_phase.water_concentration == pytest.approx(150.0)
    vapor = strategy.solve(inventories=[30.0, 0.0, 0.0], **params)
    assert vapor.liquids is None
    assert vapor.relative_humidity == pytest.approx(0.3)
    with pytest.raises(ValueError, match="no unique liquid"):
        strategy.solve(
            inventories=[100.0, 0.0, 0.0], relative_humidity=1.0, **params
        )
