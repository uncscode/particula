""" calculate the condensation driving force

    https://www.nature.com/articles/nature18271
"""

from particula.util.kelvin_correction import kelvin_term as ktfun


def condensation_force(
    vapor_concentraton,
    sat_vapor_concentration,
    particle_activity=None,
    **kwargs
):
    """ calculate the condensation driving force

        Equation (9) in https://www.nature.com/articles/nature18271
    """
    fpa = particle_activity_fun(**kwargs) if particle_activity is None \
        else particle_activity

    return (
        vapor_concentraton -
        sat_vapor_concentration *
        fpa
    )


def particle_activity_fun(
    mass_fraction,
    activity_coefficient,
    kelvin_term=None,
    **kwargs
):
    """ calculate the particle activity

        Equation (9--10) in https://www.nature.com/articles/nature18271
    """

    ktf = ktfun(**kwargs) if kelvin_term is None else kelvin_term

    return (
        mass_fraction *
        activity_coefficient *
        ktf
    )
