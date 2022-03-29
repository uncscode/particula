""" calculating rms speed of molecules
"""

import numpy as np
from particula.constants import GAS_CONSTANT, MOLECULAR_WEIGHT_AIR
from particula.util.input_handling import in_molecular_weight, in_temperature


def cbar(**kwargs):
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

    temp = kwargs.get("temperature", 298.15)
    molec_wt = kwargs.get("molecular_weight", MOLECULAR_WEIGHT_AIR)

    temp = in_temperature(temp)
    molec_wt = in_molecular_weight(molec_wt)

    gas_con = GAS_CONSTANT

    return (
        (8 * gas_con * temp/(np.pi * molec_wt))**0.5
    ).to_base_units()
