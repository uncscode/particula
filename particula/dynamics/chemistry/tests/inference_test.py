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
