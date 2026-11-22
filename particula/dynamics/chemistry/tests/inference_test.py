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
