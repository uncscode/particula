"""Test to make sure the partitioning can run, does not test the results."""


import numpy as np
from particula.activity import species_density
from particula.equilibria import partitioning


def test_partitioning():
    """Evaluation of the partitioning function."""
    c_star_j_dry = [1e-6, 1e-4, 1e-1, 1e2, 1e4]  # ug/m3

    concentration_organic_matter = [1, 5, 10, 15, 10]  # ug/m3
    oxygen2carbon = np.array([0.2, 0.3, 0.5, 0.4, 0.4])

    molar_mass = np.array([200, 200, 200, 200, 200])  # g/mol

    water_activity_desired = np.array([0.8])

    density = species_density.organic_array(
        molar_mass=molar_mass,
        oxygen2carbon=oxygen2carbon,
        hydrogen2carbon=None,
        nitrogen2carbon=None,
    )

    gamma_organic_ab, mass_fraction_water_ab, q_ab = \
        partitioning.get_properties_for_liquid_vapor_partitioning(
            water_activity_desired=water_activity_desired,
            molar_mass=molar_mass,
            oxygen2carbon=oxygen2carbon,
            density=density,
        )

    # optimize the partition coefficients
    _, _, _, _ = \
        partitioning.liquid_vapor_partitioning(
            c_star_j_dry=c_star_j_dry,
            concentration_organic_matter=concentration_organic_matter,
            molar_mass=molar_mass,
            gamma_organic_ab=gamma_organic_ab,
            mass_fraction_water_ab=mass_fraction_water_ab,
            q_ab=q_ab,
            partition_coefficient_guess=None,
        )
