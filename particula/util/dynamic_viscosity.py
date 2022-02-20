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
from particula.constants import REF_TEMPERATURE_STP as REF_TEMP
from particula.constants import REF_VISCOSITY_AIR_STP as REF_VIS_AIR
from particula.constants import SUTHERLAND_CONSTANT as SUTH_CONST


def dyn_vis(
    temp=298.15 * u.K,
    mu_ref=REF_VIS_AIR,
    t_ref=REF_TEMP,
):
    """ The dynamic viscosity of air via Sutherland formula.
        This formula depends on temperature (temp) and the reference
        temperature (t_ref) as well as the reference viscosity (mu_ref).

        Examples:
        ```
        >>> # with units
        >>> dyn_vis(temp=298.15*u.K, mu_ref=1.716e-5*u.Pa*u.s)
        <Quantity(1.83714937e-05, 'kilogram / meter / second')>
        >>> # without units and taking magnitude
        >>> dyn_vis(temp=298.15, mu_ref=1.716e-5).magnitude
        1.8371493734583912e-05
        >>> # without units, all keyword arguments
        >>> dyn_vis(temp=298.15, mu_ref=1.716e-5, t_ref=273.15)
        <Quantity(1.83714937e-05, 'kilogram / meter / second')>
        ```

        Inputs:
            temp    (float) [K]     (default: 298.15 K)
            mu_ref  (float) [Pa*s]  (default: REF_VIS_AIR)
            t_ref   (float) [K]     (default: REF_TEMP)

        Returns:
                    (float) [Pa*s]

        Notes:
            * If inputs have no units, they get assigned units above
            * particula.constants has all relevant constants

    """

    if isinstance(temp, u.Quantity):
        if temp.to_base_units().u == "kelvin":
            temp = temp.to_base_units()
        else:
            raise ValueError(
                f"\n\t"
                f"Input {temp} has unsupported units.\n\t"
                f"Input must have temperature units of\n\t"
                f"either 'kelvin' or 'degree_celsius'.\n"
            )
    else:
        temp = u.Quantity(temp, u.K)

    if isinstance(mu_ref, u.Quantity):
        if mu_ref.to_base_units().u == REF_VIS_AIR.to_base_units().u:
            mu_ref = mu_ref.to_base_units()
        else:
            raise ValueError(
                f"\n\t"
                f"Input {mu_ref} has unsupported units.\n\t"
                f"Input must have units of\n\t"
                f"{REF_VIS_AIR.u}.\n"
            )
    else:
        mu_ref = u.Quantity(mu_ref, REF_VIS_AIR.to_base_units().u)

    if isinstance(t_ref, u.Quantity):
        if t_ref.to_base_units().u == REF_TEMP.to_base_units().u:
            t_ref = t_ref.to_base_units()
        else:
            raise ValueError(
                f"\n\t"
                f"Input {t_ref} has unsupported units.\n\t"
                f"Input must have units of\n\t"
                f"{REF_TEMP.u}.\n"
            )
    else:
        t_ref = u.Quantity(t_ref, u.K)

    suth_const = SUTH_CONST

    return (
        mu_ref * (temp/t_ref)**(3/2) * (t_ref + suth_const) /
        (temp + suth_const)
    )
