"""Integration checks at the two literature studies' percent and nm boundaries."""

from pathlib import Path
from runpy import run_path

import numpy as np
from scripts.antarctic_ccn.model import critical_k

ROOT = Path(__file__).resolve().parents[2]
SOLEDAD = (
    ROOT / "studies/mt-soledad/source/field-mixtures/comparison/response.py"
)


def test_antarctic_threshold_preserves_field_axes_and_units():
    """Two anchors, three growth bins and four counter settings stay distinct."""
    kappa = np.array([[0.0, 0.01, 0.1], [0.3, 0.6, 1.0]])
    settings = np.array([0.1, 0.2, 0.5, 1.0])
    result = critical_k(kappa, settings, 2.1)

    assert result.shape == (2, 3, 4)
    np.testing.assert_allclose(result[0, 0], 2.1 / np.log1p(settings / 100))
    # Independent forward maximum at κ=0.3 and S=1.002, in nm.
    np.testing.assert_allclose(result[1, 0, 1], 104.53745312, rtol=1e-9)
    assert np.all(np.diff(result, axis=-1) < 0)


def test_soledad_threshold_preserves_singleton_and_independent_grid_check():
    """The historical water-volume grid measures a nonzero discretization error."""
    model = run_path(str(SOLEDAD))
    threshold = model["koehler_threshold"]
    diameter = threshold(0.3, 0.2, 2.1)
    assert diameter.shape == (1,)
    np.testing.assert_allclose(diameter, [104.53745312], rtol=1e-9)
    kappa = np.array([0.0, 0.01, 0.1, 0.3, 0.6, 1.0])
    for percent in (0.1, 0.2, 0.5, 1.0):
        result = threshold(kappa, percent, 2.1)
        reference = model["_grid_reference_nm"](kappa, percent, 2.1)
        assert result.shape == reference.shape == kappa.shape
        np.testing.assert_allclose(result, reference, rtol=2.5e-6, atol=0)
        assert np.max(np.abs(result - reference)) > 0
