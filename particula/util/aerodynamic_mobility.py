""" mobility of particle
"""
from typing import Union
from numpy.typing import NDArray
import numpy as np

from particula.util.input_handling import in_radius, in_scalar, in_viscosity
from particula.util.slip_correction import scf
from particula.util.dynamic_viscosity import dyn_vis


def pam(
    radius=None,
    scf_val=None,
    vis_val=None,
    **kwargs
):
    """ particle aerodynamic mobility
    """
    rad = in_radius(radius)
    scf_val = in_scalar(
        scf_val) if scf_val is not None else scf(radius=rad, **kwargs)

    # dyn_vis has no radius, is this a bug waiting to happen?
    vis_val = in_viscosity(
        vis_val) if vis_val is not None else dyn_vis(radius=rad, **kwargs)

    return (
        scf_val / (3 * np.pi * vis_val * rad * 2)
    )


def particle_aerodynamic_mobility(
    radius: Union[float, NDArray[np.float_]],
    slip_correction_factor: Union[float, NDArray[np.float_]],
    dynamic_viscosity: float
) -> Union[float, NDArray[np.float_]]:
    """
    Calculate the aerodynamic mobility of a particle, defined as the ratio
    of the slip correction factor to the product of the dynamic viscosity of
    the fluid, the particle radius, and a slip correction constant derived.

    This mobility quantifies the ease with which a particle can move through
    a fluid.

    Args:
    -----
    - radius: The radius of the particle (m).
    - slip_correction_factor: The slip correction factor for the particle
    in the fluid (dimensionless).
    - dynamic_viscosity: The dynamic viscosity of the fluid (Pa.s).

    Returns:
    --------
    - The particle aerodynamic mobility (m^2/s).
    """
    return (
        slip_correction_factor / (6 * np.pi * dynamic_viscosity * radius)
    )
