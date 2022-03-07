""" testing getting constants from the get_constants.py file
"""

from particula import u
from particula.constants import (AVOGADRO_NUMBER, BOLTZMANN_CONSTANT,
                                 GAS_CONSTANT)


def test_constants():
    """ simple tests are conducted as follows:

            * see if GAS_CONSTANT maniuplation is good
            * see if BOLTZMANN_CONSTANT units are good
    """

    assert (
        GAS_CONSTANT
        ==
        BOLTZMANN_CONSTANT * AVOGADRO_NUMBER
    )

    assert (
        GAS_CONSTANT.units
        ==
        BOLTZMANN_CONSTANT.units * AVOGADRO_NUMBER.units
    )

    assert (
        GAS_CONSTANT.magnitude
        ==
        BOLTZMANN_CONSTANT.magnitude * AVOGADRO_NUMBER.magnitude
    )

    assert (
        BOLTZMANN_CONSTANT.units
        ==
        u.m**2 * u.kg / u.s**2 / u.K
    )
