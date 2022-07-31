""" diffusion coefficient
"""

# difp = bolk.* temp.* mobp; % diffusion of particle

from particula.util.input_handling import in_temperature, in_handling
from particula.constants import BOLTZMANN_CONSTANT
from particula.util.aerodynamic_mobility import pam


def pdc(
    temperature=298.15,
    bolk=BOLTZMANN_CONSTANT,
    pam_val=None,
    **kwargs
):
    """ particle diffusion coefficient
    """
    temperature = in_temperature(temperature)
    bolk = in_handling(bolk, BOLTZMANN_CONSTANT.u)
    pam_val = pam_val if pam_val is not None else pam(**kwargs)

    return (
        bolk * temperature * pam_val
    )
