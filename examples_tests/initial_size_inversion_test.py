"""Independent reference and identifiability checks for the inversion example."""

import numpy as np
from scripts.inverse_studies.coagulation_reference import assess
from scripts.inverse_studies.distribution import NUMBER, TRUTH, observe
from scripts.inverse_studies.scalar_campaign import independent_endpoint
from scripts.inverse_studies.scalar_reference import recover_radius


def test_independent_observations_recover_initial_radius():
    """Check the dimensional quadrature against an independent ODE solve."""
    for truth in (0.82, 0.987, 1.17):
        assert abs(recover_radius(independent_endpoint(truth)) - truth) < 2e-6


def test_vectorized_global_search_preserves_box_specific_response():
    """Population batching must retain distinct per-box count fractions."""
    trials = np.stack([TRUTH, TRUTH + 0.02, TRUTH - 0.03])
    batched, slopes = observe(trials, 0.04)
    for i, radii in enumerate(trials):
        expected, derivative = observe(radii, 0.04)
        np.testing.assert_array_equal(batched[i], expected)
        np.testing.assert_array_equal(slopes[i], derivative)
    assert len(NUMBER) == batched.shape[1]


def test_constant_kernel_counts_agree_in_the_ensemble_mean():
    """Compare each discrete size band against its own sampling error."""
    assert assess()["max_z"] < 4
