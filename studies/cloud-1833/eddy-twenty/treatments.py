"""Physical wall coefficients and equal-number, tagged 5 nm inlet histories."""
import numpy as np
from physics import par, T, P, t, stage
from functools import lru_cache

VOLUME = 26.1
DIAMETER = 3.0
HEIGHT = VOLUME / (np.pi * (DIAMETER / 2)**2)
TRANSITION = t[np.flatnonzero(stage == 12)[0]]
REFERENCE_HOUR = 10.15028

def coefficient(diameter_nm, cfg, original):
    if 'eddy' not in cfg:
        return original * cfg.get('scale', .35)
    radius = diameter_nm * 0.5e-9
    density = cfg.get('density', 1350.)
    kwargs = dict(wall_eddy_diffusivity=cfg['eddy'], particle_radius=radius,
                  particle_density=density, temperature=T, pressure=P)
    shape = cfg.get('shape', 'cylinder')
    if shape == 'sphere':
        return par.dynamics.get_spherical_wall_loss_coefficient_via_system_state(
            chamber_radius=(3*VOLUME/(4*np.pi))**(1/3), **kwargs)
    if shape == 'literature':
        diff=par.particles.get_diffusion_coefficient_via_system_state(radius,T,P)
        return 0.770 * np.sqrt(diff)
    assert shape in ('cylinder','height4','volume-box')
    if shape=='volume-box':
        width=np.sqrt(VOLUME/HEIGHT)
    else:
        width=DIAMETER
    height=4. if shape=='height4' else HEIGHT
    # Matching vertical wall area/volume and floor area/volume gives the native
    # rectangular law for a right circular cylinder in this orientation model.
    return par.dynamics.get_rectangle_wall_loss_coefficient_via_system_state(
        chamber_dimensions=(width,width,height), **kwargs)

def tag_inlet(hours,cfg):
    dose=cfg.get('dose',0.)
    if dose==0: return np.zeros_like(hours)
    half=cfg.get('width',12.)/2
    relative=(hours-TRANSITION)*60-cfg['offset']
    profile=np.maximum(0.,1-np.abs(relative)/half)
    if not np.any(profile):return np.zeros_like(hours)
    assert hours[0] < TRANSITION+(cfg['offset']-half)/60
    assert hours[-1] > TRANSITION+(cfg['offset']+half)/60
    return dose * profile/np.trapezoid(profile,hours*3600)
