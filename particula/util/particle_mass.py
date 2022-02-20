""" calculating the particle mass
"""

import numpy as np

from particula import u


def particle_mass(radius, density=1000):
    """ Returns particle's mass: 4/3 pi r^3 * density.

        Examples:
        ```
        >>> particle_mass(1*u.m)
        <Quantity(4188.7902, 'kilogram')>
        >>> particle_mass(1*u.nm, 1000*u.kg/u.m**3).m
        4.188790204786392e-24
        >>> particle_mass(1*u.nm, 1000*u.kg/u.m**3).m_as(u.g)
        4.188790204786392e-21
        >>> particle_mass([1, 2, 3]*u.nm, 1000*u.kg/u.m**3).m
        array([4.18879020e-24, 3.35103216e-23, 1.13097336e-22])
        >>> particle_mass([1, 2]*u.nm, [1, 2]*u.g/u.cm**3).m
        array([4.18879020e-24, 6.70206433e-23])
        >>> particle_mass(2*u.nm, 2*u.g/u.cm**3).m
        6.702064327658225e-23
        ```

        Parameters:
            radius  (float) [m]
            density (float) [kg/m^3] (default: 1000)

        Returns:
                    (float) [kg]
    """

    if isinstance(radius, u.Quantity):
        if radius.to_base_units().u == u.m:
            radius = radius.to_base_units()
        else:
            raise ValueError(
                f"input {radius} must be in meters."
            )
    else:
        radius = u.Quantity(radius, u.m)

    if isinstance(density, u.Quantity):
        if density.to_base_units().u == u.kg / u.m**3:
            density = density.to_base_units()
        else:
            raise ValueError(
                f"input {density} must be in kg/m^3."
            )
    else:
        density = u.Quantity(density, u.kg/u.m**3)

    return density * (4*np.pi/3) * (radius**3)
