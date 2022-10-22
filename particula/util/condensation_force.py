""" calculate the condensation driving force

    https://www.nature.com/articles/nature18271
"""

import numpy as np


def condensation_force(
    vapor_concentraton,
    sat_vapor_concentration,
    full_particle_activity,
):
    """ calculate the condensation driving force

        Equation (9) in https://www.nature.com/articles/nature18271
    """
    return (
        vapor_concentraton -
        sat_vapor_concentration *
        full_particle_activity
    )


def particle_activity(
    mass_fraction,
    activity_coefficient,
    the_kelvin_term,
):
    """ calculate the particle activity

        Equation (9--10) in https://www.nature.com/articles/nature18271
    """
    return (
        mass_fraction *
        activity_coefficient *
        the_kelvin_term
    )


def kelvin_term(
    surface_tension,
    molar_weight,
    gas_constant,
    temperature,
    density,
    radius,
):
    """ calculate the kelvin term

        Equation (10) in https://www.nature.com/articles/nature18271
    """
    return np.exp(
        4 * surface_tension * molar_weight / (
            gas_constant * temperature * density * radius * 2
        )
    )
