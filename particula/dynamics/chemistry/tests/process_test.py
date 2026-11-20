"""Independent trajectories, conservation and same-owner chemistry restart."""

import numpy as np
import pytest
from particula.aerosol import Aerosol
from particula.dynamics.chemistry import ReactionMechanism
from particula.dynamics.chemistry.process import ChemistrySolver, GasChemistry
from particula.dynamics.chemistry.tests.mechanism_test import conversion
from particula.execution.cpu_simulation import CPUSimulation
from particula.gas import EnvironmentData, GasData
from particula.particles import ParticleData
from particula.util.constants import GAS_CONSTANT
from scipy.integrate import solve_ivp


def _reaction(name, left, right, k, units):
    return dict(
        id=name,
        reactants=left,
        products=right,
        rate=dict(type="constant", k=k, units=units),
    )


def robertson():
    """Encode the three extents from the dimensionless Robertson benchmark."""
    a, b, c = "gas:A", "gas:B", "gas:C"
    record = conversion()
    record["reactions"] = [
        _reaction("decay", {a: 1}, {b: 1}, 0.04, "1/s"),
        _reaction("exchange", {b: 1, c: 1}, {a: 1, c: 1}, 1e4, "m^3/mol/s"),
        _reaction("dimer", {b: 2}, {b: 1, c: 1}, 3e7, "m^3/mol/s"),
    ]
    return ReactionMechanism(record)


def robertson_reference(times, method):
    """Use an independently written normalized ODE with a stiff solver."""

    def rhs(t, y):
        a, b, c = y
        return [
            -0.04 * a + 1e4 * b * c,
            0.04 * a - 1e4 * b * c - 3e7 * b * b,
            3e7 * b * b,
        ]

    result = solve_ivp(
        rhs,
        (0, float(times[-1])),
        [1.0, 0.0, 0.0],
        t_eval=times,
        method=method,
        rtol=1e-11,
        atol=[1e-14, 1e-17, 1e-14],
    )
    assert result.success
    return result.y.T


def state_and_process(case, method):
    """Build independent boxes with different temperatures and an inert lane."""
    masses = np.array([0.01, 0.02, 0.03, 0.04])
    molar = np.array([[1.0, 1.0, 0.0, 0.01], [2.0, 2.0, 0.0, 0.02]])
    if case == "stiff":
        mechanism = robertson()
        masses[:3] = 0.1
        molar[:, :3] = [1, 0, 0]
    else:
        record = conversion()
        record["reactions"][0]["rate"] = dict(
            type="arrhenius",
            A=2,
            b=1,
            Ea_J_per_mol=1000,
            T_ref_K=298.0,
            units="m^3/mol/s",
        )
        mechanism = ReactionMechanism(record)
    state = Aerosol(
        particles=ParticleData(
            masses=np.full((2, 1, 1), 1e-18),
            concentration=np.ones((2, 1)),
            charge=np.zeros((2, 1)),
            density=np.array([1000.0]),
            volume=np.array([1.0, 2.0]),
            distribution_type="particle_resolved",
        ),
        gas=GasData(
            ["A", "B", "C", "inert"], masses, molar * masses, np.zeros(4, bool)
        ),
        environment=EnvironmentData(
            temperature=np.array([298.0, 318.0]),
            pressure=np.full(2, 101325.0),
            saturation_ratio=np.zeros((2, 4)),
        ),
    )
    solver = ChemistrySolver(method=method, rtol=1e-8, atol=1e-14, max_step=10)
    return state, GasChemistry(mechanism.bind(state.gas), solver)


@pytest.mark.parametrize("method", ["rk23", "ros3"])
def test_bimolecular_cpu(method):
    """Match two exact Arrhenius trajectories, mass inventories and restart."""
    state, process = state_and_process("arrhenius", method)
    original = state.gas.concentration.copy()
    owner = CPUSimulation(state, process, time_step=0.5)
    owner.step(0.5)
    resumed = CPUSimulation.resume(owner.checkpoint(), process)
    owner.step(1.0)
    resumed.step(1.0)
    actual = owner.snapshot().gas.concentration
    np.testing.assert_array_equal(actual, resumed.snapshot().gas.concentration)
    temperature = state.environment.temperature
    k = 2 * (temperature / 298) * np.exp(-1000 / (GAS_CONSTANT * temperature))
    initial = np.array([1.0, 2.0])
    reactant = initial / (1 + k * initial)
    expected = (
        np.column_stack([reactant, reactant, initial - reactant])
        * state.gas.molar_mass[:3]
    )
    np.testing.assert_allclose(actual[:, :3], expected, rtol=1e-7, atol=1e-13)
    np.testing.assert_allclose(
        actual.sum(axis=1), original.sum(axis=1), rtol=1e-12
    )
    np.testing.assert_array_equal(actual[:, 3], original[:, 3])


def test_robertson_cpu():
    """Resolve the short-lived intermediate and verify convergence to Radau and BDF."""
    state, process = state_and_process("stiff", "ros3")
    times = np.array([1e-4, 1e-2, 1.0, 100.0])
    history = []
    prior = 0.0
    for t in times:
        process.execute(state, float(t - prior))
        history.append(state.gas.concentration[0, :3] / 0.1)
        prior = t
    for method in ("Radau", "BDF"):
        np.testing.assert_allclose(
            history, robertson_reference(times, method), rtol=2e-6, atol=1e-12
        )
    np.testing.assert_allclose(np.sum(history, axis=1), 1.0, rtol=1e-12)
    assert np.min(history) >= 0
