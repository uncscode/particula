""" Test get_constants.py
"""

from particula import u
from particula.utils import (
    BOLTZMANN_CONSTANT,
    AVOGADRO_NUMBER,
    GAS_CONSTANT,
)

def test_get_constants():

    """ testing get_constants

        * see if GAS_CONSTANT maniuplation is good
        * see if BOLTZMANN_CONSTANT units are good
    """

    assert (
        GAS_CONSTANT ==
            BOLTZMANN_CONSTANT * AVOGADRO_NUMBER
    )
    assert (
        GAS_CONSTANT.units ==
            BOLTZMANN_CONSTANT.units * AVOGADRO_NUMBER.units
    )
    assert (
        GAS_CONSTANT.magnitude ==
            BOLTZMANN_CONSTANT.magnitude * AVOGADRO_NUMBER.magnitude
    )
    assert (
        BOLTZMANN_CONSTANT.units ==
            u.m**2 * u.kg / u.s**2 / u.K
    )
