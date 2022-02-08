""" calculate the knudsen number
"""

from particula import u

def knudsen_number(radius, mfp_air=66.4e-9) -> float:

    """ Returns particle's Knudsen number.

        Parameters:
            radius  (float) [m]
            mfp_air (float) [m] (default: 66.4e-9)

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
        radius = radius.to_base_units()
    else:
        radius = u.Quantity(radius, u.m)

    if isinstance(mfp_air, u.Quantity):
        mfp_air = mfp_air.to_base_units()
    else:
        mfp_air = u.Quantity(mfp_air, u.m)

    return mfp_air / radius
