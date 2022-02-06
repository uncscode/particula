""" Calculate the reduced friction factor
"""

from particula.utils import friction_factor

def reduced_friction_factor(
    radius,
    other_radius,
    mean_free_path_air,
    dynamic_viscosity_air,
) -> float:

    """Returns the reduced friction factor between two particles.

    Parameters:
        radii_array                 (np array)  [m]
        radius_other                (float)     [m]
        mean_free_path_air          (float)     [m]
        dynamic_viscosity_air       (float)     [N*s/m]

    Returns:
        reduced_friction_factor     (array)     [unitless]

    Similar to the reduced mass. The reduced friction factor allows
    a two-body problem to be solved as a one-body problem.
    """

    a_friction_factor = friction_factor(
        radius, mean_free_path_air, dynamic_viscosity_air
    )
    b_friction_factor = friction_factor(
        other_radius, mean_free_path_air, dynamic_viscosity_air
    )

    return (a_friction_factor * b_friction_factor)/(
        a_friction_factor + b_friction_factor
    )
