""" testing the input_handling function
"""

from particula.util.input_handling import convert_units


def test_convert_units_temperature():
    """ Testing the convert_units function with temperature units
    """
    result = 50 + convert_units('degC', 'degK')
    assert result == 323.15

    result = convert_units('degF', 'degK', value=50)
    assert result == 283.15000000000003

    result = 280 + convert_units('degK', 'degC')
    assert result == 6.850000000000023

    result = convert_units('K', 'degF', value=280)
    assert result == 44.32999999999998
