""" calculating the mean free path of air

    The mean free path is the average distance
    traveled by a molecule between collisions
    with other molecules present in a medium (air).

    The expeected mean free path of air is approx.
    65 nm at 298 K and 101325 Pa.

"""

import numpy as np
from particula import u
from particula.constants import GAS_CONSTANT as GAS_CON
from particula.constants import MOLECULAR_WEIGHT_AIR as MOL_WT_AIR
from particula.util.dynamic_viscosity import dyn_vis


def mean_free_path(
    temperature=298,
    pressure=101325,
    molec_wt=MOL_WT_AIR,
):
    """ Returns the mean free path of in air.

        Examples:
        ```
        >>> # with no inputs, it defaults to 298 K and 101325 Pa
        >>> mean_free_path()
        >>> <Quantity(6.64373669e-08, 'meter')>
        >>> # specifying a temperature of 300 L
        >>> mean_free_path(temperature=300*u.K).magnitude
        >>> 6.700400687925813e-08
        >>> # specifying 300 K and  pressure of 1e5 Pa
        >>> mean_free_path(temperature=300*u.K, pressure=1e5*u.Pa)
        >>> <Quantity(6.789181e-08, 'meter')>
        >>> mean_free_path(temperature=300, pressure=1e5, molec_wt=0.03)
        >>> <Quantity(6.67097062e-08, 'meter')>
        ```

        Parameters:
            temperature (float) [K]      (default: 298)
            pressure    (float) [Pa]     (default: 101325)
            molec_wt    (float) [kg/mol] (default: MOL_WT_AIR)

        Returns:
                        (float) [m]

        The mean free path is the average distance
        traveled by a molecule between collisions
        with other molecules present in a medium (air).

        The expeected mean free path of air is approx.
        65 nm at 298 K and 101325 Pa.
    """

    if isinstance(temperature, u.Quantity):
        if temperature.to_base_units().u == "kelvin":
            temperature = temperature.to_base_units()
        else:
            raise ValueError(
                f"\n\t"
                f"Input {temperature} has unsupported units.\n\t"
                f"Input must have temperature units of\n\t"
                f"either 'kelvin' or 'degree_Celsius'.\n"
            )
    else:
        temperature = u.Quantity(temperature, u.K)

    if isinstance(pressure, u.Quantity):
        if pressure.to_base_units().u == (1*u.Pa).to_base_units().u:
            pressure = pressure.to_base_units()
        else:
            raise ValueError(
                f"\n\t"
                f"Input {pressure} has unsupported units.\n\t"
                f"Input must have pressure units of 'pascal'.\n"
            )
    else:
        pressure = u.Quantity(pressure, u.Pa)

    if isinstance(molec_wt, u.Quantity):
        if molec_wt.to_base_units().u == u.kg / u.mol:
            molec_wt = molec_wt.to_base_units()
        else:
            raise ValueError(
                f"\n\t"
                f"Input {molec_wt} has unsupported units.\n\t"
                f"Input must have molecular weight units of\n\t"
                f"'kg/mol'.\n"
            )
    else:
        molec_wt = u.Quantity(molec_wt, u.kg / u.mol)

    return (
        (2*dyn_vis(temperature)/pressure) /
        (8*molec_wt/(np.pi*GAS_CON*temperature))**0.5
    ).to_base_units()
