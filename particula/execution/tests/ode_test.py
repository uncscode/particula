"""Independent order, conservation, rejection and clock tests for ODE stepping."""

import numpy as np
import numpy.testing as npt
import pytest
from scipy.linalg import expm
from scipy.sparse import csc_matrix

from particula.execution.ode import AdaptiveSolver, ODEProblem


def linear_problem(matrix):
    """Describe an autonomous positive linear system with its exact Jacobian."""
    return ODEProblem(
        rate=lambda t, y: matrix @ y,
        admissible=lambda y: bool(np.all(y >= 0)),
        jacobian=lambda t, y: matrix,
        time_derivative=lambda t, y: np.zeros_like(y),
    )


@pytest.mark.parametrize("method", ["rk23", "ros2", "ros3"])
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


@pytest.mark.parametrize(
    "method,order", [("rk23", 3), ("ros2", 2), ("ros3", 3)]
)
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
            step = getattr(solver, "_" + method)
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


def test_ros3_nonlinear_order():
    """A changing Jacobian retains order for y'=-y^2, y=1/(1+t)."""
    problem = ODEProblem(
        lambda t, y: -(y**2),
        lambda y: bool(np.all(y > 0)),
        lambda t, y: np.array([[-2 * y[0]]]),
        lambda t, y: np.zeros_like(y),
    )
    errors = []
    for steps in (16, 32, 64):
        solver = AdaptiveSolver(problem, [1.0], method="ros3")
        value = solver.state
        for i in range(steps):
            value, _ = solver._ros3(i / steps, value, 1 / steps)
        errors.append(abs(value[0] - 0.5))
    assert 2.9 < np.log2(errors[-2] / errors[-1]) < 3.1


@pytest.mark.parametrize("sparse", [False, True])
def test_ros3_stiff_transient_refinement(sparse):
    """Resolve the fast transient, rather than only its eventual equilibrium."""
    matrix = np.array([[-1000.0, 1.0], [1000.0, -1.0]])
    problem = ODEProblem(
        lambda t, y: matrix @ y,
        lambda y: bool(np.all(y >= 0)),
        lambda t, y: csc_matrix(matrix) if sparse else matrix,
        lambda t, y: np.zeros_like(y),
    )
    errors = []
    for tolerance in (1e-4, 1e-6, 1e-8):
        solver = AdaptiveSolver(
            problem,
            [1.0, 0.0],
            method="ros3",
            rtol=tolerance,
            atol=1e-12,
            first_step=1e-6,
            max_step=0.1,
        )
        maximum = 0.0
        for bound in np.geomspace(1e-5, 1.0, 50):
            while solver.time < bound:
                solver.step(float(bound))
            maximum = max(
                maximum,
                float(
                    np.max(abs(solver.state - expm(matrix * bound) @ [1, 0]))
                ),
            )
            assert abs(solver.state.sum() - 1) < 2e-12
        errors.append(maximum)
    assert errors[2] < errors[1] / 20 < errors[0] / 400
    assert errors[2] < 2e-9


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("method,stages", [("ros2", 2), ("ros3", 3)])
def test_rosenbrock_zero_rate_inventory_survives_pivoting(
    sparse, method, stages
):
    """A passive reservoir cannot grow from a pivoted linear solve."""
    a = 305.36647011984786
    b = 1.5806695870710714e-8
    c = 2.534891692370676e-5
    weight = 4.6556629518440924e-8
    matrix = np.array(
        [[0, 0, 0], [a, -b, c], [-weight * a, weight * b, -weight * c]]
    )
    initial = np.array([1.0, 0.4, 1.0])
    problem = ODEProblem(
        lambda t, y: matrix @ y,
        lambda y: True,
        lambda t, y: csc_matrix(matrix) if sparse else matrix,
        lambda t, y: np.zeros_like(y),
    )
    solver = AdaptiveSolver(problem, initial, method=method)
    candidate, _ = getattr(solver, f"_{method}")(0.0, initial, 1.0)
    assert candidate[0] == initial[0]
    np.testing.assert_allclose(
        candidate, expm(matrix) @ initial, rtol=2e-12, atol=1e-14
    )
    assert (
        abs(
            weight * candidate[1]
            + candidate[2]
            - (weight * initial[1] + initial[2])
        )
        < 3e-16
    )
    assert solver.statistics.factorizations == 1
    assert solver.statistics.linear_solves == 3 * stages


def test_ros3_domain_failure_does_not_publish_trial():
    """An inadmissible internal stage leaves state and accepted time untouched."""
    problem = ODEProblem(
        lambda t, y: y,
        lambda y: bool(np.all((y >= 0) & (y <= 1.000001))),
        lambda t, y: np.eye(1),
        lambda t, y: np.zeros_like(y),
    )
    solver = AdaptiveSolver(
        problem,
        [1.0],
        method="ros3",
        first_step=1.0,
        max_trials=1,
    )
    with pytest.raises(RuntimeError, match="trial count"):
        solver.step(1.0)
    npt.assert_array_equal(solver.state, [1.0])
    assert solver.time == 0
    assert solver.statistics.domain_rejected == 1
