"""Test the gibbs_free_energy function."""

import numpy as np
import pytest

from particula.activity.gibbs import (
    gibbs_free_energy,
    gibbs_free_engery,  # Deprecated misspelled alias
)


def test_gibbs_free_energy_shapes_positive():
    """Outputs are finite and preserve shapes for vector input."""
    organic_mole_fraction = np.array([0.2, 0.4, 0.6, 0.8])
    gibbs_mix = np.array([0.1, 0.2, 0.3, 0.4])

    gibbs_ideal, gibbs_real = gibbs_free_energy(
        organic_mole_fraction, gibbs_mix
    )

    assert gibbs_ideal.shape == organic_mole_fraction.shape
    assert gibbs_real.shape == organic_mole_fraction.shape
    assert np.all(np.isfinite(gibbs_ideal))
    assert np.all(np.isfinite(gibbs_real))


def test_gibbs_free_energy_single_value():
    """Outputs are finite for single-value input."""
    organic_mole_fraction = np.array([0.5])
    gibbs_mix = np.array([0.2])

    gibbs_ideal, gibbs_real = gibbs_free_energy(
        organic_mole_fraction, gibbs_mix
    )

    assert gibbs_ideal.shape == (1,)
    assert gibbs_real.shape == (1,)
    assert np.all(np.isfinite(gibbs_ideal))
    assert np.all(np.isfinite(gibbs_real))


def test_gibbs_free_engery_deprecation_warning():
    """Deprecated alias emits DeprecationWarning."""
    organic_mole_fraction = np.array([0.5])
    gibbs_mix = np.array([0.2])

    with pytest.warns(
        DeprecationWarning, match="gibbs_free_engery is deprecated"
    ):
        gibbs_ideal, gibbs_real = gibbs_free_engery(
            organic_mole_fraction, gibbs_mix
        )

    # Verify it still works correctly
    assert gibbs_ideal.shape == (1,)
    assert gibbs_real.shape == (1,)
    assert np.all(np.isfinite(gibbs_ideal))
    assert np.all(np.isfinite(gibbs_real))
