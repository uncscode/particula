""" testing the dynamic viscosity function
"""

from particula.constants import REF_TEMPERATURE_STP as REF_TEMP
from particula.constants import REF_VISCOSITY_AIR_STP as REF_VIS_AIR
from particula.util.dynamic_viscosity import dyn_vis


def test_dyn_vis():
    """ Testing the dynamic viscosity calculation:
            * see if dyn_vis returns correct units of constant
            * see if defaults work properly
            * see if scaling is correct (dividing by 2)
            * see if user-provided constants work properly

    """

    assert dyn_vis().units == REF_VIS_AIR.to_base_units().units

    assert dyn_vis(temp=REF_TEMP) == REF_VIS_AIR

    assert dyn_vis(temp=REF_TEMP, gas="air") == REF_VIS_AIR

    assert dyn_vis(temp=REF_TEMP, gas="none") == REF_VIS_AIR

    assert dyn_vis(temp=REF_TEMP/2) <= REF_VIS_AIR

    assert dyn_vis(temp=REF_TEMP*2) >= REF_VIS_AIR

    assert (
        dyn_vis(
            temp=REF_TEMP,
            gas="made_up_gas",
            mu_ref=REF_VIS_AIR/2,
            t_ref=REF_TEMP,
        )
        ==
        REF_VIS_AIR/2
    )
