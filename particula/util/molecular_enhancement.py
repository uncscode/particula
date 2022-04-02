""" calculating molecular enhancment of collision.

    mol_enh =
        ( (vapor_size + particle_size) / (particle_size) ) ** 2

"""

import numpy as np
from particula.util.input_handling import in_length


def mol_enh(vapor_size, particle_size):
    """ Returns the molecular enhancement.

        Parameters:
            vapor_size      (float)  [m]
            particle_size   (float)  [m]

        Returns:
                    (float)  [ ]

        TODO:
            - Add explanation of molecular enhancement.
    """

    a_q = in_length(vapor_size)
    b_q = in_length(particle_size)

    return np.transpose(
        (np.transpose([a_q.m])*a_q.u + b_q) / (b_q)
    )**2
