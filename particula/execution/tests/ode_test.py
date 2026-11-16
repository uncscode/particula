"""Independent order, conservation, rejection and clock tests for ODE stepping."""

import numpy as np
import numpy.testing as npt
import pytest
from scipy.linalg import expm

from particula.execution.ode import AdaptiveSolver, ODEProblem


def linear_problem(matrix):
    """Describe an autonomous positive linear system with its exact Jacobian."""
    return ODEProblem(
        rate=lambda t, y: matrix @ y,
        admissible=lambda y: bool(np.all(y >= 0)),
        jacobian=lambda t, y: matrix,
        time_derivative=lambda t, y: np.zeros_like(y),
    )


@pytest.mark.parametrize("method", ["rk23", "ros2"])
def test_tolerance_refinement_and_closed_species_budget(method):
    """Tighter tolerance improves trajectory accuracy without losing material."""
    matrix = np.array([[-0.7, 0.2], [0.7, -0.2]])
    errors = []
    for tolerance in (1e-3, 1e-5, 1e-7):
        solver = AdaptiveSolver(
            linear_problem(matrix),
            [1.0, 0.0],
            method=method,
            rtol=tolerance,
            atol=tolerance * 1e-3,
            first_step=1.0,
        )
        error = 0.0
        while solver.time < 5:
            solver.step(5)
            exact = expm(matrix * solver.time) @ [1.0, 0.0]
            error = max(error, np.max(np.abs(solver.state - exact)))
            assert abs(solver.state.sum() - 1) < 2e-14
            assert np.min(solver.state) >= 0
        errors.append(error)
    assert errors[1] < errors[0] / 5
    assert errors[2] < errors[1] / 5
    assert errors[-1] < 1e-6


@pytest.mark.parametrize("method,order", [("rk23", 3), ("ros2", 2)])
def test_observed_order_and_absolute_stage_times(method, order):
    """A nonautonomous manufactured solution detects wrong ROS time terms."""
    # y(t)=exp(-t)+t^2 solves y'=-y+t^2+2t.
    problem = ODEProblem(
        lambda t, y: -y + t * t + 2 * t,
        lambda y: bool(np.all(y > 0)),
        lambda t, y: np.array([[-1.0]]),
        lambda t, y: np.array([2 * t + 2]),
    )
    errors = []
    for steps in (32, 64, 128):
        solver = AdaptiveSolver(
            problem, [np.exp(-2) + 4], time=2, method=method
        )
        value = solver.state
        for i in range(steps):
            step = solver._rk23 if method == "rk23" else solver._ros2
            value, _ = step(2 + i / steps, value, 1 / steps)
        errors.append(abs(value[0] - (np.exp(-3) + 9)))
    assert order - 0.15 < np.log2(errors[-2] / errors[-1]) < order + 0.15


def test_domain_rejection_preserves_accepted_state_clock_and_input():
    """A failed oversized trial cannot publish negative state or accepted time."""
    initial = np.array([1.0])
    solver = AdaptiveSolver(
        linear_problem(np.array([[-100.0]])),
        initial,
        first_step=1.0,
        max_trials=1,
    )
    with pytest.raises(RuntimeError, match="trial count"):
        solver.step(1)
    npt.assert_array_equal(initial, [1])
    npt.assert_array_equal(solver.state, initial)
    assert solver.time == 0
    assert solver.statistics.accepted == 0
    assert solver.statistics.domain_rejected == 1


def test_domain_rejection_recovers_without_clipping():
    """The same out-of-domain trial succeeds after reducing its timestep."""
    solver = AdaptiveSolver(
        linear_problem(np.array([[-100.0]])),
        [1.0],
        first_step=1.0,
    )
    solver.step(1)
    assert solver.statistics.domain_rejected > 0
    assert 0 < solver.time < 1
    npt.assert_allclose(solver.state, [np.exp(-100 * solver.time)], rtol=1e-5)


def test_callback_mutation_and_configuration_errors_propagate():
    """Callback bugs cannot be interpreted as a reason to silently retry."""

    def mutating(t, y):
        y[:] = 0
        return y

    solver = AdaptiveSolver(ODEProblem(mutating, lambda y: True), [1.0])
    with pytest.raises(ValueError, match="read-only"):
        solver.step(1)
    assert solver.statistics.rejected == 0
    npt.assert_array_equal(solver.state, [1])
    with pytest.raises(ValueError, match="requires"):
        AdaptiveSolver(
            ODEProblem(mutating, lambda y: True), [1.0], method="ros2"
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"atol": 0},
        {"rtol": -1},
        {"atol": [1, 2]},
        {"min_step": 0},
        {"max_trials": True},
        {"time": float("nan")},
        {"first_step": 2},
    ],
)
def test_bad_controls_reject_before_rates(kwargs):
    """Invalid numerical controls are configuration errors, not failed trials."""
    with pytest.raises(ValueError):
        AdaptiveSolver(linear_problem(np.array([[-1.0]])), [1], **kwargs)


def test_final_bound_and_detached_observation():
    """Final time lands exactly and passive inspection cannot alter state."""
    solver = AdaptiveSolver(linear_problem(np.array([[-1.0]])), [1.0])
    while solver.time < 0.12345:
        solver.state[:] = 99
        solver.step(0.12345)
    assert solver.time == 0.12345
    calls = solver.statistics.rate_calls
    solver.step(0.12345)
    assert solver.statistics.rate_calls == calls
    npt.assert_allclose(solver.state, np.exp(-solver.time), rtol=1e-5)
