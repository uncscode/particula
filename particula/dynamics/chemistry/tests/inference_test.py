"""Shared and run-specific calibration and conditional profile fits."""

import numpy as np
import pytest
from particula.dynamics.chemistry.inference import (
    ExperimentLoss,
    fit_experiments,
)


def experiment(indices, observed):
    """A quadratic measurement objective with known parameter sensitivities."""

    def loss(x, gradient):
        delta = x - observed
        return float(delta @ delta / 2), delta if gradient else None

    return ExperimentLoss(indices, loss)


@pytest.mark.parametrize("gradient", [True, False])
def test_shared_local_and_fixed(gradient):
    """Shared rates average two targets while individual doses remain local."""
    series = [experiment((0, 1), [0.2, 0.4]), experiment((0, 2), [0.6, 0.9])]
    result = fit_experiments(
        series, [(0, 1)] * 3, [np.array([0.0, 0.0, 0.0])], gradient=gradient
    )
    np.testing.assert_allclose(result.parameters, [0.4, 0.4, 0.9], atol=1e-7)
    np.testing.assert_allclose(result.loss, 0.04)
    profile = fit_experiments(
        series, [(0, 0), (0, 1), (0, 1)], [np.zeros(3)], gradient=gradient
    )
    np.testing.assert_allclose(profile.parameters, [0.0, 0.4, 0.9], atol=1e-7)
    assert profile.loss > result.loss


def test_rank_held_out_ties():
    """Both candidate paths predict the same conserved observable."""
    from particula.dynamics.chemistry.discovery import SearchResult
    from particula.dynamics.chemistry.inference import (
        CandidateFit,
        fit_networks,
    )

    direct = ((-1, 0, 1),)
    chain = ((-1, 1, 0), (0, -1, 1))
    search = SearchResult((chain, direct), 2, None)

    def prepare(net):
        size = len(net)
        measured = experiment((0,), [0.5])
        return CandidateFit(
            [measured], [(0, 1)] * size, [np.zeros(size)], [measured]
        )

    ranks = fit_networks(search, prepare, criterion="held_out", tolerance=1e-8)
    assert {r.network for r in ranks if r.competitive} == {chain, direct}
    aic = fit_networks(search, prepare, criterion="aic", tolerance=0.01)
    assert [r.network for r in aic if r.competitive] == [direct]


@pytest.mark.parametrize("gradient", [False, True])
def test_evaluation_budget(gradient):
    """Account for FD probes and retain the best evaluated feasible point."""
    fit = fit_experiments(
        [experiment((0,), [0.5])],
        [(0.0, 1.0)],
        [np.array([0.1])],
        gradient=gradient,
        max_evaluations=1,
    )
    assert fit.evaluations == (1,)
    assert fit.converged == (False,)
    np.testing.assert_allclose(fit.parameters, [0.1])


@pytest.mark.parametrize("gradient", [False, True])
def test_fixed_parameters_score_without_optimization(gradient):
    """A profile point with every parameter fixed needs one loss evaluation."""
    calls = []

    def measure(x, need_gradient):
        calls.append((x.copy(), need_gradient))
        return float(x @ x), 2 * x if need_gradient else None

    fit = fit_experiments(
        [ExperimentLoss((1, 0), measure)],
        [(2, 2), (3, 3)],
        [np.array([2, 3])],
        gradient=gradient,
    )
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0], [3, 2])
    assert calls[0][1] == gradient
    np.testing.assert_array_equal(fit.parameters, [2, 3])
    assert fit.loss == 13
    assert fit.evaluations == (1,)
    assert fit.converged == (True,)


@pytest.mark.parametrize(
    ("bounds", "start", "message"),
    [
        ([(1, 0)], [0.5], "ordered bounds"),
        ([(0, np.inf)], [1], "ordered bounds"),
        ([(0, 1)], [1.1], "feasible starts"),
    ],
)
def test_invalid_calibration_domain(bounds, start, message):
    """Reject infeasible starts and invalid parameter bounds before fitting."""
    with pytest.raises(ValueError, match=message):
        fit_experiments([experiment((0,), [0.5])], bounds, [np.array(start)])


@pytest.mark.parametrize(
    ("loss", "derivative", "message"),
    [
        (np.inf, [0], "Nonfinite experiment loss"),
        (0, [np.nan], "Invalid experiment gradient"),
        (0, [1, 2], "Invalid experiment gradient"),
    ],
)
def test_invalid_experiment_output(loss, derivative, message):
    """Reject nonfinite losses and malformed local sensitivities."""
    objective = ExperimentLoss((0,), lambda x, grad: (loss, derivative))
    with pytest.raises(ValueError, match=message):
        fit_experiments([objective], [(0, 1)], [np.array([0.5])])
