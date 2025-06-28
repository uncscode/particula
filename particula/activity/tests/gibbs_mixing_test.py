"""Gibbs mixing test."""

import unittest

import numpy as np

from particula.activity.bat_coefficients import (
    G19_FIT_HIGH,
    G19_FIT_LOW,
    G19_FIT_MID,
)
from particula.activity.gibbs_mixing import (
    gibbs_mix_weight,
    gibbs_of_mixing,
)


# pylint: disable=too-many-instance-attributes
class TestGibbsMixing(unittest.TestCase):
    """Tests for Gibbs mixing functions."""

    def setUp(self) -> None:
        """Set up test variables."""
        self.molar_mass_ratio = 18.016 / 250
        self.organic_mole_fraction = np.linspace(0.1, 1, 10)
        self.oxygen2carbon = 0.3
        self.density = 2500
        self.molar_mass_ratio_array = np.array([18.016 / 250, 18.016 / 200])
        self.organic_mole_fraction_array = np.array([0.1, 0.2])
        self.oxygen2carbon_array = np.array([0.3, 0.4])
        self.density_array = np.array([2500, 2600])

    def test_gibbs_of_mixing(self) -> None:
        """Test for gibbs_of_mixing function."""
        for fit_dict in [G19_FIT_LOW, G19_FIT_MID, G19_FIT_HIGH]:
            gibbs_mix, derivative_gibbs = gibbs_of_mixing(
                molar_mass_ratio=self.molar_mass_ratio,
                organic_mole_fraction=self.organic_mole_fraction,
                oxygen2carbon=self.oxygen2carbon,
                density=self.density,
                fit_dict=fit_dict,
            )
            self.assertTrue(np.all(gibbs_mix >= 0))
            self.assertTrue(np.all(derivative_gibbs**2 >= 0))

        gibbs_mix, derivative_gibbs = gibbs_of_mixing(
            molar_mass_ratio=self.molar_mass_ratio_array,
            organic_mole_fraction=self.organic_mole_fraction_array,
            oxygen2carbon=self.oxygen2carbon_array,
            density=self.density_array,
            fit_dict=G19_FIT_LOW,
        )
        self.assertTrue(np.all(gibbs_mix >= 0))
        self.assertTrue(np.all(derivative_gibbs**2 >= 0))

    def test_gibbs_mix_weight(self) -> None:
        """Test for gibbs_mix_weight function."""
        gibbs_mix, derivative_gibbs = gibbs_mix_weight(
            molar_mass_ratio=self.molar_mass_ratio,
            organic_mole_fraction=self.organic_mole_fraction,
            oxygen2carbon=self.oxygen2carbon,
            density=self.density,
        )
        self.assertTrue(np.all(gibbs_mix >= 0))
        self.assertTrue(np.all(derivative_gibbs**2 >= 0))

        gibbs_mix, derivative_gibbs = gibbs_mix_weight(
            molar_mass_ratio=self.molar_mass_ratio,
            organic_mole_fraction=self.organic_mole_fraction,
            oxygen2carbon=self.oxygen2carbon,
            density=self.density,
            functional_group="alcohol",
        )
        self.assertTrue(np.all(gibbs_mix >= 0))
        self.assertTrue(np.all(derivative_gibbs**2 >= 0))

        gibbs_mix, derivative_gibbs = gibbs_mix_weight(
            molar_mass_ratio=self.molar_mass_ratio_array,
            organic_mole_fraction=self.organic_mole_fraction_array,
            oxygen2carbon=self.oxygen2carbon_array,
            density=self.density_array,
        )
        self.assertTrue(np.all(gibbs_mix >= 0))
        self.assertTrue(np.all(derivative_gibbs**2 >= 0))
