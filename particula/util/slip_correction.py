""" calculate slip correction
"""

import numpy as np
from particula.util.input_handling import in_radius, in_scalar
from particula.util.knudsen_number import knu as knu_func


def scf(
    radius=None,
    knu=None,
    **kwargs
):
    """ Returns particle's Cunningham slip correction factor.

        Dimensionless quantity accounting for non-continuum effects
        on small particles. It is a deviation from Stokes' Law.
        Stokes assumes a no-slip condition that is not correct at
        high Knudsen numbers. The slip correction factor is used to
        calculate the friction factor.

        Thus, the slip correction factor is about unity (1) for larger
        particles (Kn -> 0). Its behavior on the other end of the
        spectrum (smaller particles; Kn -> inf) is more nuanced, though
        it tends to scale linearly on a log-log scale, log Cc vs log Kn.

        Examples:
        ```
        >>> from particula import u
        >>> from particula.util.slip_correction import scf
        >>> # with radius 1e-9 m
        >>> scf(radius=1e-9)
        <Quantity(110.720731, 'dimensionless')>
        >>> # with radius 1e-9 m and knu=1
        >>> scf(radius=1e-9*u.m, knu=1)
        <Quantity(2.39014843, 'dimensionless')>
        >>> # using knu(**kwargs)
        >>> scf(radius=1e-9*u.m, mfp=60*u.nm)
        <Quantity(99.9840088, 'dimensionless')>
        >>> # using mfp(**kwargs)
        >>> scf(
        ... radius=1e-9*u.m,
        ... temperature=300,
        ... pressure=1e5,
        ... molecular_weight=0.03
        ... )
        <Quantity(111.101591, 'dimensionless')>
        ```

        Parameters:
            radius  (float) [m]
            knu     (float) [ ] (default: util)

        Returns:
                    (float) [dimensionless]

        Notes:
            knu can be calculated using knu(**kwargs);
            refer to particula.util.knudsen_number.knu for more info.

    """
    radius = in_radius(radius)

    knu_val = knu_func(radius=radius, **kwargs) if knu is None \
        else in_scalar(knu)

    return 1 + knu_val * (
        1.257 + 0.4*np.exp(-1.1/knu_val)
    )
