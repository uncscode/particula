"""Independent composition accuracy, grouped clocks and atomic cycle tests."""

import numpy as np
import numpy.testing as npt
import pytest
from scipy.linalg import expm

from particula.execution.splitting import (
    Flow,
    SplitGroup,
    SplitSimulation,
    condensation_flow,
    dilution_flow,
)
from particula.execution.tests.condensation_ode_test import (
    case,
    reference_duration,
)


@pytest.mark.parametrize("method,order", [("sequential", 1), ("strang", 2)])
def test_noncommuting_native_flows_have_expected_order(method, order):
    """Exact exchange and removal maps expose scheduler-only trajectory error."""
    a = np.array([[-0.7, 0.2], [0.7, -0.2]])
    b = np.diag([-0.4, 0.0])

    def flow(name, matrix):
        def advance(state, time, dt):
            state.gas.concentration[0] = (
                expm(matrix * dt) @ state.gas.concentration[0]
            )
            return state

        return Flow(name, advance, kind="exact")

    group = SplitGroup((flow("exchange", a), flow("loss", b)), method)
    errors = []
    for steps in (16, 32, 64):
        initial, _ = case()
        initial.gas.concentration[0] = [1, 0]
        owner = SplitSimulation(initial, group, time_step=1 / steps)
        while owner.time < 1:
            owner.step(1)
        exact = expm(a + b) @ [1, 0]
        errors.append(
            np.max(np.abs(owner.snapshot().gas.concentration[0] - exact))
        )
    assert order - 0.1 < np.log2(errors[-2] / errors[-1]) < order + 0.1


def test_nested_groups_preserve_component_times_and_half_steps():
    """Nested group order remains explicit, with each component's own clock."""
    events = []

    def flow(name):
        def advance(state, time, dt):
            events.append((name, time, dt))
            return state

        return Flow(name, advance, kind="exact")

    group = SplitGroup((flow("a"), SplitGroup((flow("b"), flow("c")))))
    initial, _ = case()
    owner = SplitSimulation(initial, group, time=3, time_step=2)
    owner.step(5)
    assert events == [
        ("a", 3, 1),
        ("b", 3, 1),
        ("c", 3, 2),
        ("b", 4, 1),
        ("a", 4, 1),
    ]
    assert owner.time == 5
    assert owner.completed_steps == 1


def test_failed_later_component_cannot_publish_partial_cycle():
    """An earlier component's detached writes vanish when the next one fails."""

    def first(state, time, dt):
        state.gas.concentration *= 0.5
        return state

    def fail(state, time, dt):
        raise ValueError("scientific failure")

    initial, _ = case()
    owner = SplitSimulation(
        initial, SplitGroup((Flow("a", first), Flow("b", fail))), time_step=1
    )
    with pytest.raises(ValueError, match="scientific failure"):
        owner.step(1)
    npt.assert_array_equal(
        owner.snapshot().gas.concentration, initial.gas.concentration
    )
    assert owner.time == owner.completed_steps == 0


def test_ineligible_processes_reject_before_execution():
    """Neither stochastic declarations nor first-order maps qualify for Strang."""
    with pytest.raises(ValueError, match="deterministic"):
        Flow("random", lambda a, t, h: a, kind="stochastic")
    first = Flow("first", lambda a, t, h: a, order=1)
    with pytest.raises(ValueError, match="second-order"):
        SplitGroup((first,))
    with pytest.raises(ValueError, match="unique"):
        SplitGroup((Flow("x", lambda a, t, h: a), Flow("x", lambda a, t, h: a)))


def test_native_growth_dilution_strang_convergence_and_budget():
    """Shared Stage 1 coupled reference qualifies the native composed trajectory."""
    growth_time = reference_duration()
    hazard = 0.2 / growth_time
    duration = -np.log1p(-hazard * growth_time) / hazard
    errors = []
    for steps in (8, 16, 32):
        initial, strategy = case()
        total = (
            initial.gas.concentration[0, 1]
            + initial.particles.species_mass_density[0, 0]
        )
        group = SplitGroup(
            (
                dilution_flow("dilution", hazard),
                condensation_flow(
                    "growth",
                    strategy,
                    rtol=1e-9,
                    particle_atol=1e-25,
                    gas_atol=1e-15,
                    dynamic_viscosity=1.8e-5,
                ),
            )
        )
        owner = SplitSimulation(initial, group, time_step=duration / steps)
        while owner.time < duration:
            owner.step(duration)
        result = owner.snapshot()
        errors.append(abs(result.particles.radii[0, 0] / 1e-6 - 1.2))
        npt.assert_allclose(
            result.gas.concentration[0, 1]
            + result.particles.species_mass_density[0, 0],
            total * np.exp(-hazard * duration),
            rtol=3e-13,
        )
    assert errors[1] < 0.3 * errors[0]
    assert errors[2] < 0.3 * errors[1]
    assert errors[-1] < 1e-5
