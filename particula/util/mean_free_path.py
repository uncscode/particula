""" calculating the mean free path of air

    The mean free path is the average distance
    traveled by a molecule between collisions
    with other molecules present in a medium (air).

    The expeected mean free path of air is approx.
    65 nm at 298 K and 101325 Pa.

"""

import numpy as np
from particula.constants import GAS_CONSTANT, MOLECULAR_WEIGHT_AIR
from particula.util.dynamic_viscosity import dyn_vis
from particula.util.input_handling import (in_molecular_weight, in_pressure,
                                           in_temperature, in_viscosity)


def mfp(**kwargs):
    """ Returns the mean free path of in air.

        The mean free path is the average distance
        traveled by a molecule between collisions
        with other molecules present in a medium (air).

        The expeected mean free path of air is approx.
        65 nm at 298 K and 101325 Pa.

        Examples:
        ```
        >>> from particula import u
        >>> from particula.util.mean_free_path import mfp
        >>> # with no inputs, it defaults to 298 K and 101325 Pa
        >>> mfp()
        <Quantity(6.64373669e-08, 'meter')>
        >>> # specifying a temperature of 300 K
        >>> mfp(temperature=300*u.K).magnitude
        6.700400687925813e-08
        >>> # specifying 300 K and  pressure of 1e5 Pa
        >>> mfp(temperature=300*u.K, pressure=1e5*u.Pa)
        <Quantity(6.789181e-08, 'meter')>
        >>> mfp(
        ... temperature=300,
        ... pressure=1e5,
        ... molecular_weight=0.03
        ... )
        <Quantity(6.67097062e-08, 'meter')>
        >>> # specifying explicit value for dynamic viscosity
        >>> mfp(dynamic_viscosity=1e-5)
        <Quantity(3.61864151e-08, 'meter')>
        >>> # specifying implicit value for dynamic viscosity
        >>> mfp(
        ... temperature=300,
        ... reference_viscosity=1e-5,
        ... reference_temperature=273.15
        ... )
        <Quantity(3.90466241e-08, 'meter')>
        ```

        Parameters: (either # or $)
            temperature           (float) [K]      (default: 298)
            pressure              (float) [Pa]     (default: 101325)
            molecular_weight      (float) [kg/mol] (default: constants)

        #   dynamic_viscosity     (float) [Pa*s]   (default: util)
        $   reference_viscosity   (float) [Pa*s]   (default: constants)
        $   reference_temperature (float) [K]      (default: constants)

        Returns:
                        (float) [m]

        Using particula.constants:
            GAS_CONSTANT            (float) [J/mol/K]
            MOLECULAR_WEIGHT_AIR    (float) [kg/mol]

            REF_VISCOSITY_AIR_STP   (float) [Pa*s]
            REF_TEMPERATURE_STP     (float) [K]
            SUTHERLAND_CONSTANT     (float) [K]

        Notes:
            dynamic_viscosity can be calculated independently via
            particula.util.dynamic_viscosity.dyn_vis(**kwargs), but
            if the value of dynamic_viscosity is provided directly,
            it overrides the calculated value.
    """

    temp = kwargs.get("temperature", 298.15)
    pres = kwargs.get("pressure", 101325)
    molec_wt = kwargs.get("molecular_weight", MOLECULAR_WEIGHT_AIR)
    dyn_vis_val = kwargs.get("dynamic_viscosity", dyn_vis(**kwargs))

    temp = in_temperature(temp)
    pres = in_pressure(pres)
    molec_wt = in_molecular_weight(molec_wt)
    dyn_vis_val = in_viscosity(dyn_vis_val)

    gas_con = GAS_CONSTANT

    return (
        (2 * dyn_vis_val / pres) /
        (8 * molec_wt / (np.pi * gas_con * temp))**0.5
    ).to_base_units()
