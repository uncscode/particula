""" test kelvin correct """

from particula import u

from particula.util.kelvin_correction import kelvin_radius, kelvin_term


def test_kelvin_radius():
    """ test: parcelona.kelvin_radius """
    assert kelvin_radius(
        surface_tension=0.072 * u.N/u.m,
        molecular_weight=0.01815 * u.kg/u.mol,
        density=1000 * u.kg/u.m**3,
        temperature=300
    ).u == u.m


def test_kelvin_term():
    """ test: parcelona.kelvin_term """
    assert kelvin_term(
        surface_tension=0.072 * u.N/u.m,
        molecular_weight=0.01815 * u.kg/u.mol,
        density=1000 * u.kg/u.m**3,
        temperature=300,
        radius=1 * u.m
    ).u == u.dimensionless
