""" calculating the dynamic viscosity.

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
    gas="air",
    mu_ref=REF_VIS_AIR,
    t_ref=REF_TEMP,
):
    """ The dynamic viscosity of air via Sutherland formula.

        Inputs:
            temp    (float) [K]     (default: 298.15 K)
            gas     (str)   [ ]     (default: "air")
            mu_ref  (float) [Pa*s]  (default: constants.REF_VIS_AIR)
            t_ref   (float) [K]     (default: constants.REF_TEMP)

        Returns:
                    (float) [Pa*s]

        Notes:
            * particula.constants has all relevant constants
            * Most often, only gas="air" is used
    """

    if isinstance(temp, u.Quantity):
        temp = temp.to_base_units()
    else:
        temp = u.Quantity(temp, u.K)

    if isinstance(mu_ref, u.Quantity):
        mu_ref = mu_ref.to_base_units()
    else:
        mu_ref = u.Quantity(mu_ref, u.Pa * u.s)

    if isinstance(t_ref, u.Quantity):
        t_ref = t_ref.to_base_units()
    else:
        t_ref = u.Quantity(t_ref, u.K)

    suth_const = SUTH_CONST

    if gas != "air":
        print(
            f"\n"
            f"\tThis {gas} gas is not air!\n"
            f"\n"
            f"\tThe dynamic viscosity is defined by user-provided.\n"
            f"\treference and reference temperature constants.\n"
            f"\tthese are respectively\n"
            f"\t{mu_ref}\n"
            f"\tand\n"
            f"\t{t_ref}.\n"
        )

    return (
        mu_ref * (temp/t_ref)**(3/2) * (t_ref + suth_const) /
        (temp + suth_const)
    )
