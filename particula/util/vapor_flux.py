""" calculate the vapor flux to the surface of particles
"""

from particula import u
from particula.util.particle_surface import area
from particula.util.fuchs_sutugin import fsc as fsc_func
from particula.util.rms_speed import cbar
from particula.util.input_handling import in_radius, in_scalar, in_handling, in_concentration
from particula.util.molecular_enhancement import mol_enh

def phi(
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
    radius = in_radius(radius)
    molecular_enhancement_val = mol_enh(**kwargs) if molecular_enhancement is None else in_length(molecular_enhancement)
    vapor_attachment = in_scalar(vapor_attachment)
    vapor_speed_val = cbar(**kwargs)/4 if vapor_speed is None else in_handling(vapor_speed, u.m/u.s)
    driving_force = in_concentration(driving_force)
    fsc_val = fsc_func(**kwargs) if fsc is None else in_scalar(fsc)

    return (
        area(radius=self.particle_radius, area_factor=1) *
        self.molecular_enhancement() *
        self.vapor_attachment *
        self.vapor_speed() *
        self.driving_force() *
        fsc(knu_val=self.knudsen_number(), alpha=1)
    )