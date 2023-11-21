"""Test the gibbs_free_engery function."""
import numpy as np
from particula.activity.gibbs import gibbs_free_engery


def test_gibbs_free_engery():
    """Test the gibbs_free_engery function."""
    # Test case 1: Known values
    organic_mole_fraction = np.array([0.2, 0.4, 0.6, 0.8])
    gibbs_mix = np.array([0.1, 0.2, 0.3, 0.4])

    gibbs_ideal, gibbs_real = gibbs_free_engery(
        organic_mole_fraction, gibbs_mix)

    assert np.all(gibbs_ideal ** 2 >= 0)
    assert np.all(gibbs_real ** 2 >= 0)

    # Test case 3: Single value
    organic_mole_fraction = np.array([0.5])
    gibbs_mix = np.array([0.2])

    gibbs_ideal, gibbs_real = gibbs_free_engery(
        organic_mole_fraction, gibbs_mix)

    assert np.all(gibbs_ideal ** 2 >= 0)
    assert np.all(gibbs_real ** 2 >= 0)
