""" calculating the dynamic viscosity

    The dynamic viscocity is calculated using the Sutherland formula,
    assuming ideal gas behavior, as a function of temperature.

        "The dynamic viscosity equals the product of the sum of
        Sutherland's constant and the reference temperature divided by
        the sum of Sutherland's constant and the temperature,
        the reference viscosity and the ratio to the 3/2 power
        of the temperature to reference temperature."

        https://resources.wolframcloud.com/FormulaRepository/resources/Sutherlands-Formula

"""
from particula import u
from particula.constants import (REF_TEMPERATURE_STP, REF_VISCOSITY_AIR_STP,
                                 SUTHERLAND_CONSTANT)
from particula.util.input_handling import in_temperature, in_viscosity


def dyn_vis(
    temperature=298.15 * u.K,
    reference_viscosity=REF_VISCOSITY_AIR_STP,
    reference_temperature=REF_TEMPERATURE_STP,
    sutherland_constant=SUTHERLAND_CONSTANT,
    **kwargs,
):
    """ The dynamic viscosity of air via Sutherland formula.
        This formula depends on temperature (temp) and the reference
        temperature (t_ref) as well as the reference viscosity (mu_ref).

        Examples:
        ```
        >>> from particula import u
        >>> from particula.util.dynamic_viscosity import dyn_vis
        >>> # with units
        >>> dyn_vis(
        ... temperature=298.15*u.K,
        ... reference_viscosity=1.716e-5*u.Pa*u.s
        ... )
        <Quantity(1.83714937e-05, 'kilogram / meter / second')>
        >>> # without units and taking magnitude
        >>> dyn_vis(
        ... temperature=298.15,
        ... reference_viscosity=1.716e-5
        ... ).magnitude
        1.8371493734583912e-05
        >>> # without units, all keyword arguments
        >>> dyn_vis(
        ... temperature=298.15,
        ... reference_viscosity=1.716e-5,
        ... reference_temperature=273.15
        ... )
        <Quantity(1.83714937e-05, 'kilogram / meter / second')>
        >>> # for a list of temperatures
        >>> dyn_vis(temperature=[200, 250, 300, 400]).m
        array([1.32849751e-05, 1.59905239e-05, 1.84591625e-05, 2.28516090e-05])
        ```

        Inputs:
            temperature             (float) [K]     (default: 298.15)
            reference_viscosity     (float) [Pa*s]  (default: constants)
            reference_temperature   (float) [K]     (default: constants)
            sutherland_constant     (float) [K]     (default: constants)

        Returns:
                                    (float) [Pa*s]

        Using particula.constants:
            REF_VISCOSITY_AIR_STP   (float) [Pa*s]
            REF_TEMPERATURE_STP     (float) [K]
            SUTHERLAND_CONSTANT     (float) [K]

    """
    # trick to avoid triggering a linting error for kwargs
    temp = kwargs.get("temperature", temperature)
    temp = in_temperature(temperature)
    ref_vis = in_viscosity(reference_viscosity)
    ref_temp = in_temperature(reference_temperature)
    suth_const = in_temperature(sutherland_constant)

    return (
        ref_vis *
        (temp/ref_temp)**(3/2) *
        (ref_temp + suth_const) /
        (temp + suth_const)
    ).to_base_units()
