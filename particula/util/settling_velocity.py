""" settling velocity
"""

from particula.util.input_handling import in_radius, in_scalar, in_viscosity, in_density
from particula.util.slip_correction import scf
from particula.util.dynamic_viscosity import dyn_vis
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
    rad = in_radius(rad)
    den = in_density(den)
    scf_val = in_scalar(
        scf_val) if scf_val is not None else scf(**kwargs)
    vis_val = in_viscosity(
        vis_val) if vis_val is not None else dyn_vis(**kwargs)

    return (
        (2*rad)**2 *
        den *
        scf_val *
        sgc /
        (18 * vis_val)
    )
