"""Check observation adjoints with correlated channels and missing samples."""

import numpy as np
import pytest
from particula.dynamics.chemistry.observations import ObservationSeries


def test_masked_likelihood_gradient():
    """Compare a directional derivative including a wholly missing time."""
    series = ObservationSeries.create(
        [0, 0.7, 2], [[1, 2], [np.nan, 1], [np.nan, np.nan]], [[4, 1], [1, 1]]
    )
    prediction = np.array([[3.0, 4.0], [np.nan, 3.0], [np.nan, np.nan]])
    _, gradient = series.loss(prediction)
    direction = np.array([[0.5, -0.7], [0.0, 0.25], [0.0, 0.0]])
    eps = 1e-5
    difference = (
        series.loss(prediction + eps * direction)[0]
        - series.loss(prediction - eps * direction)[0]
    ) / (2 * eps)
    np.testing.assert_allclose(
        (direction * gradient).sum(), difference, atol=1e-9
    )
    np.testing.assert_array_equal(gradient[[1, 2], 0], 0)


def test_dependent_channels():
    """Exact duplicate channels require a reduced independent observation."""
    with pytest.raises(ValueError, match="positive definite"):
        ObservationSeries.create([0], [[1, 1]], [[1, 1], [1, 1]])


def test_time_specific_noise_scales_and_missing_channel():
    """Each observed time has its own variance and normalization term."""
    series = ObservationSeries.create(
        [0, 2], [[1, np.nan], [np.nan, 2]], [[[4, 0], [0, 1]], [[1, 0], [0, 9]]]
    )
    loss, derivative = series.loss([[3, 10], [20, 5]])
    np.testing.assert_allclose(loss, 1 + np.log(2 * np.pi * 6))
    np.testing.assert_allclose(derivative, [[0.5, 0], [0, 1 / 3]])


@pytest.mark.parametrize(
    ("times", "data", "covariance", "message"),
    [
        ([np.nan], [[1]], [[1]], "finite vector"),
        ([1, 1], [[1], [2]], [[1]], "must increase"),
        ([0], [[np.inf]], [[1]], "time by channel"),
        ([0], [[np.nan]], [[1]], "one observation"),
        ([0], [[1, 2]], [[1]], "channel covariance"),
        ([0], [[1, 2]], [[2, 1], [0, 2]], "symmetric"),
    ],
)
def test_invalid_measurement_series(times, data, covariance, message):
    """Reject measurement series that cannot define a Gaussian likelihood."""
    with pytest.raises(ValueError, match=message):
        ObservationSeries.create(times, data, covariance)
