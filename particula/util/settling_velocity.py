""" settling velocity
"""

from particula.util.input_handling import in_radius, in_density
from particula.constants import STANDARD_GRAVITY


def psv(
    rad=None,
    den=None,
    scf_val=None,
    sgc=STANDARD_GRAVITY,
    vis_val=None,
    **kwargs
):
    """ calculate the settling velocity
    """
    _ = kwargs
    rad = in_radius(rad)
    den = in_density(den)

    return (
        (2*rad)**2 *
        den *
        scf_val *
        sgc /
        (18 * vis_val)
    )
