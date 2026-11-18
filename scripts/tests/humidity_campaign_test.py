"""Regression checks for the units and accuracy measures in the BAT campaign."""

import numpy as np
from scripts.gpu_studies.humidity_campaign import chamber
from scripts.gpu_studies.humidity_references import errors


def test_batch_preserves_number_and_mass_per_particle():
    """Vary only represented count across boxes at fixed volume and dry size."""
    baseline, _ = chamber(1)
    batch, _ = chamber(8)
    p = batch.particles
    np.testing.assert_allclose(
        p.concentration[:, 0] / p.volume, 1e11 * np.geomspace(0.5, 2.0, 8)
    )
    np.testing.assert_array_equal(
        p.masses, np.repeat(baseline.particles.masses, 8, axis=0)
    )
    np.testing.assert_array_equal(
        batch.gas.concentration,
        np.repeat(baseline.gas.concentration, 8, axis=0),
    )


def test_transfer_errors_use_each_leg_and_box_scale():
    """A large vapor offset must not conceal a small transfer error."""
    ref = np.ones((4, 2, 3))
    ref[:, :, 1] = [[1000, 2000], [990, 1900], [50, 80], [55, 100]]
    ref[:, :, 2] = [[10, 100], [20, 200], [20, 200], [15, 180]]
    candidate = ref.copy()
    candidate[1, 0, 1] += 1
    candidate[3, 1, 2] += 1
    candidate[2:, 1, 1] += 2
    candidate[0, 1, 0] += 0.01
    np.testing.assert_allclose(errors(candidate, ref), [[0, 10, 0], [1, 0, 5]])
