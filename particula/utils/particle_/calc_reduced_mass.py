""" Calculate the reducedm mass
"""


def reduced_mass(mass, other_mass) -> float:

    """ Returns the reduced mass of two particles.

    Parameters:
        mass_array      (np array)  [kg]
        mass_other      (float)     [kg]

    Returns:
        reduced_mass    (array)     [unitless]

    The reduced mass is an "effective inertial" mass.
    Allows a two-body problem to be solved as a one-body problem.
    """

    return mass * other_mass / (mass + other_mass)
