""" calculating the particle mass
"""

import numpy as np
from particula.util.input_handling import in_density, in_radius, in_scalar


def mass(
    radius=None,
    density=1000,
    shape_factor=1,
    volume_void=0,
    **kwargs,
):
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
            radius       (float) [m]
            density      (float) [kg/m^3] (default: 1000)
            shape_factor (float) [ ]      (default: 1)
            volume_void  (float) [ ]      (default: 0)

        Returns:
                         (float) [kg]
    """
    _ = kwargs.get("something", None)
    radius = in_radius(radius)
    density = in_density(density)
    shape_factor = in_scalar(shape_factor)
    vol_void = in_scalar(volume_void)

    return (
        density * (4*np.pi/3) * (radius**3)
        * shape_factor * (1 - vol_void)
    ).to_base_units()
