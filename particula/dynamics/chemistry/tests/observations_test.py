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
