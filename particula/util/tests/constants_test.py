"""testing getting constants from the get_constants.py file."""

from particula.util.constants import (
    AVOGADRO_NUMBER,
    BOLTZMANN_CONSTANT,
    GAS_CONSTANT,
)


def test_constants():
    """Simple tests are conducted.

    * see if GAS_CONSTANT maniuplation is good
    """
    assert GAS_CONSTANT == BOLTZMANN_CONSTANT * AVOGADRO_NUMBER
