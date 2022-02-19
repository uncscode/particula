""" calculating knudsen number
"""

from particula import u
from particula.util.mean_free_path import mean_free_path as mfp_def


def knudsen_number(radius, mfp=mfp_def()) -> float:
    """ Returns particle's Knudsen number.

        Parameters:
            radius  (float) [m]
            mfp     (float) [m] (default: mfp_def())

        Returns:
                    (float) [dimensionless]

        The Knudsen number reflects the relative length scales of
        the particle and the suspending fluid (air, water, etc.).
        This is calculated by the mean free path of the medium
        divided by the particle radius.

        The Knudsen number is a measure of continuum effects and
        deviation thereof. For larger particles, the Knudsen number
        goes towards 0. For smaller particles, the Knudsen number
        goes towards infinity.
    """

    if isinstance(radius, u.Quantity):
        if radius.to_base_units().u == "meter":
            radius = radius.to_base_units()
        else:
            raise ValueError(f"{radius} must be in meters!")
    else:
        radius = u.Quantity(radius, u.m)

    if isinstance(mfp, u.Quantity):
        if mfp.to_base_units().u == "meter":
            mfp = mfp.to_base_units()
        else:
            raise ValueError(f"{mfp} must be in meters!")
    else:
        mfp = u.Quantity(mfp, u.m)

    return mfp / radius
