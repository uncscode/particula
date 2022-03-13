""" calculating Knudsen number

    TODO:
        add a sanity check for mixing radius shapes and temperature shapes
        for example, someone may want to calculate knu for [1e-9, 2e-9, 3e-9]
        but only at two different temperatures [290, 300].
        A solution is to broadcast into a new shape, e.g. (3, 2) with the
        leading dim being the radius; or we could simply ban this entirely.

        -- implementing tranpose for now.
"""

import numpy as np
from particula.util.input_handling import in_length, in_radius
from particula.util.mean_free_path import mfp as mfp_func


def knu(
    radius=None,
    mfp=None,
    **kwargs
):
    """ Returns particle's Knudsen number.

        The Knudsen number reflects the relative length scales of
        the particle and the suspending fluid (air, water, etc.).
        This is calculated by the mean free path of the medium
        divided by the particle radius.

        The Knudsen number is a measure of continuum effects and
        deviation thereof. For larger particles, the Knudsen number
        goes towards 0. For smaller particles, the Knudsen number
        goes towards infinity.

        Examples:
        ```
        >>> from particula import u
        >>> from particula.util.knudsen_number import knu
        >>> # with radius 1e-9 m
        >>> knu(radius=1e-9)
        <Quantity(66.4798498, 'dimensionless')>
        >>> # with radius 1e-9 m and mfp 60 nm
        >>> knu(radius=1e-9*u.m, mfp=60*u.nm).m
        60.00000000000001
        >>> calculating via mfp(**kwargs)
        >>> knu(
        ... radius=1e-9*u.m,
        ... temperature=300,
        ... pressure=1e5,
        ... molecular_weight=0.03,
        ... )
        <Quantity(66.7097062, 'dimensionless')>
        ```

        Parameters:
            radius  (float) [m]
            mfp     (float) [m] (default: util)

        Returns:
                    (float) [dimensionless]

        Notes:
            mfp can be calculated using mfp(**kwargs);
            refer to particula.util.mean_free_path.mfp for more info.

    """

    mfp_val = mfp_func(**kwargs) if mfp is None else in_length(mfp)
    radius = in_radius(radius)

    return np.transpose([mfp_val.m])*mfp_val.u / radius
