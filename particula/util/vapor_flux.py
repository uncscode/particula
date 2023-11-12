""" calculate the vapor flux to the surface of particles
"""
import numpy as np

from particula import u
from particula.util.fuchs_sutugin import fsc as fsc_func
from particula.util.input_handling import (in_concentration, in_handling,
                                           in_scalar)
from particula.util.molecular_enhancement import mol_enh
from particula.util.particle_surface import area
from particula.util.rms_speed import cbar


def phi(  # pylint: disable=too-many-arguments
    particle_area=None,
    molecular_enhancement=None,
    vapor_attachment=1,
    vapor_speed=None,
    driving_force=1,
    fsc=None,
    **kwargs,
):
    """ vapor flux
    """
    particle_area = area(
        **kwargs) if particle_area is None else in_handling(
            particle_area, u.m**2)

    molecular_enhancement_val = mol_enh(
        **kwargs) if molecular_enhancement is None else in_scalar(
            molecular_enhancement)

    vapor_attachment = np.transpose(
        [in_scalar(in_scalar(vapor_attachment)).m]
    )*in_scalar(in_scalar(vapor_attachment)).u

    vapor_speed_val = cbar(
        **kwargs)/4 if vapor_speed is None else in_handling(
            vapor_speed, u.m/u.s)

    driving_force = in_concentration(driving_force)
    fsc_val = fsc_func(**kwargs) if fsc is None else in_scalar(fsc)

    rxd = particle_area.size
    return (
        particle_area.reshape(-1, 1) *
        molecular_enhancement_val *
        np.transpose(vapor_attachment) *
        vapor_speed_val *
        driving_force *
        fsc_val.reshape(rxd, -1)
    )
