""" calculating the mean free path of air

    The mean free path is the average distance
    traveled by a molecule between collisions
    with other molecules present in a medium (air).

    The expeected mean free path of air is approx.
    65 nm at 298 K and 101325 Pa.

    TODO:
        add size checks for pressure--temperature pairs
        to ensure that they match; otherwise, an error will occur
        or use broadcast (though this is likely not a good idea)?
        perhaps allow for a height--temperature--pressure dependency
        somewhere? this could be import for @Gorkowski's parcels...
        (likely through a different utility function...)
"""

import numpy as np
from particula import u
from particula.constants import GAS_CONSTANT, MOLECULAR_WEIGHT_AIR
from particula.util.dynamic_viscosity import dyn_vis
from particula.util.input_handling import (in_gas_constant,
                                           in_molecular_weight, in_pressure,
                                           in_temperature, in_viscosity)


def mfp(
    temperature=298.15*u.K,
    pressure=101325*u.Pa,
    molecular_weight=MOLECULAR_WEIGHT_AIR,
    dynamic_viscosity=None,
    gas_constant=GAS_CONSTANT,
    **kwargs,
):
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
        >>> # specifying a list of temperatures
        >>> mfp(temperature=[200, 250, 300, 400]).m
        array([3.93734886e-08, 5.29859655e-08, 6.70040069e-08, 9.57800224e-08])
        >>> # specifying a list of pressures
        >>> mfp(pressure=[1.0e5, 1.1e5, 1.2e5, 1.3e5]).m
        array([6.73607078e-08, 6.12370071e-08, 5.61339232e-08, 5.18159291e-08])
        >>> # specifying a list of pressures and temperatures
        >>> mfp(temperature=[300,310], pressure=[1e5, 1.1e5])
        <Quantity([6.78918100e-08 6.43354325e-08], 'meter')>
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

    temp = in_temperature(temperature)
    pres = in_pressure(pressure)
    molec_wt = in_molecular_weight(molecular_weight)

    if dynamic_viscosity is None:
        dyn_vis_val = in_viscosity(
            dyn_vis(temperature=temp, **kwargs)
        )
    else:
        dyn_vis_val = in_viscosity(dynamic_viscosity)

    gas_con = in_gas_constant(gas_constant)

    return (
        (2 * dyn_vis_val / pres) /
        (8 * molec_wt / (np.pi * gas_con * temp))**0.5
    ).to_base_units()
