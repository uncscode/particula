""" calculating rms speed of molecules
"""

import numpy as np
from particula.constants import GAS_CONSTANT, MOLECULAR_WEIGHT_AIR
from particula.util.input_handling import (in_gas_constant,
                                           in_molecular_weight, in_temperature)


def cbar(
    temperature=298.15,
    molecular_weight=MOLECULAR_WEIGHT_AIR,
    gas_constant=GAS_CONSTANT,
):
    """ Returns the mean speed of molecules in an ideal gas.

        Parameters:
            temperature           (float) [K]      (default: 298.15)
            molecular_weight      (float) [kg/mol] (default: constants)

        Returns:
                                  (float) [m/s]

        Using particula.constants:
            GAS_CONSTANT            (float) [J/mol/K]
            MOLECULAR_WEIGHT_AIR    (float) [kg/mol]

    """

    temperature = in_temperature(temperature)
    molecular_weight = in_molecular_weight(molecular_weight)
    gas_constant = in_gas_constant(gas_constant)

    return (
        (8 * gas_constant * temperature/(np.pi * molecular_weight))**0.5
    ).to_base_units()
