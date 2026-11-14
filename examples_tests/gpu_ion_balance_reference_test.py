"""Independent analytic and time-grid checks for the ion-balance reference."""

import runpy
from pathlib import Path

import numpy as np
import pytest

SOURCE = (
    Path(__file__).resolve().parents[1]
    / "docs/examples/gpu/prepared/gpu_ion_balance.py"
)


@pytest.fixture(scope="module")
def reference():
    """Load definitions without executing the CUDA example."""
    pytest.importorskip("warp")
    return runpy.run_path(str(SOURCE))


def test_two_pair_decay_matches_analytic_probabilities(reference):
    """For n=2, sequential death rates are 4 alpha/V then alpha/V."""
    times = np.linspace(0, 20000, 17)
    rate = reference["ALPHA"] / reference["VOLUME"]
    p2 = np.exp(-4 * rate * times)
    p1 = 4 / 3 * (np.exp(-rate * times) - p2)
    expected_mean = 2 * p2 + p1
    expected_variance = 4 * p2 + p1 - expected_mean**2
    mean, variance = reference["finite_population_reference"](
        0, 2, times, maximum_pairs=8
    )
    np.testing.assert_allclose(mean, expected_mean, rtol=2e-12, atol=1e-14)
    np.testing.assert_allclose(
        variance, expected_variance, rtol=2e-12, atol=1e-14
    )


@pytest.mark.parametrize(
    "times", [[0], [1, 2], [0, 1, 3], [0, 0], [0, np.nan], [[0, 1]]]
)
def test_reference_rejects_mislabeled_time_grids(reference, times):
    """The solver must not silently evaluate a different observation grid."""
    with pytest.raises(ValueError, match="times"):
        reference["finite_population_reference"](0, 2, times)


@pytest.mark.parametrize(
    "changes", [{"q": -1}, {"initial_pairs": -1}, {"maximum_pairs": 0}]
)
def test_reference_rejects_invalid_chain_parameters(reference, changes):
    """Reject negative production or invalid population states explicitly."""
    parameters = dict(q=0, initial_pairs=2, times=[0, 1], maximum_pairs=8)
    parameters.update(changes)
    with pytest.raises(ValueError):
        reference["finite_population_reference"](**parameters)
