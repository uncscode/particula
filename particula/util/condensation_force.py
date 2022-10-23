""" calculate the condensation driving force

    https://www.nature.com/articles/nature18271
"""

from particula.util.kelvin_correction import kelvin_term


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
    the_kelvin_term=kelvin_term(),
):
    """ calculate the particle activity

        Equation (9--10) in https://www.nature.com/articles/nature18271
    """

    return (
        mass_fraction *
        activity_coefficient *
        the_kelvin_term
    )
