""" testing the input_handling function
"""

import pytest
from particula import u
from particula.util.input_handling import (in_molecular_weight, in_pressure,
                                           in_temperature, in_viscosity)


def test_in_temp():
    """ Testing the in_temperature function
    """

    temp = in_temperature(5)
    assert temp.units == u.K
    assert temp.magnitude == 5

    temp = in_temperature(5*u.K)
    assert temp.units == u.K
    assert temp.magnitude == 5

    temp = in_temperature(u.Quantity(5, u.degC))
    assert temp.units == u.K
    assert temp.magnitude == u.Quantity(5, u.degC).m_as("degK")

    temp = in_temperature(u.Quantity(5, u.degF))
    assert temp.units == u.K
    assert temp.magnitude == u.Quantity(5, u.degF).m_as("kelvin")

    with pytest.raises(ValueError):
        temp = in_temperature(5*u.m)


def test_in_vis():
    """ Testing the in_viscosity function
    """

    vis = in_viscosity(5)
    assert vis.units == u.kg / u.m / u.s
    assert vis.magnitude == 5

    vis = in_viscosity(5 * u.kg / u.m / u.s**2 * u.s)
    assert vis.units == u.kg / u.m / u.s
    assert vis.magnitude == 5

    vis = in_viscosity(u.Quantity(5, u.mPa * u.s))
    assert vis.units == u.kg / u.m / u.s
    assert vis.magnitude == u.Quantity(5, u.mPa * u.s).m_as("Pa*s")

    vis = in_viscosity(u.Quantity(5, u.kg / u.m / u.s))
    assert vis.units == u.kg / u.m / u.s
    assert vis.magnitude == u.Quantity(5, u.kg / u.m / u.s).m_as("Pa*s")

    vis = in_viscosity(u.Quantity(5, u.kg / u.m / u.s))
    assert vis.units == u.kg / u.m / u.s
    assert vis.magnitude == u.Quantity(5, u.kg / u.m / u.s).m_as("Pa*s")

    vis = in_viscosity(u.Quantity(5, u.mPa * u.s))
    assert vis.units == u.kg / u.m / u.s
    assert vis.magnitude == u.Quantity(5, u.mPa * u.s).m_as("Pa*s")

    vis = in_viscosity(u.Quantity(5, u.mPa * u.s))
    assert vis.units == u.kg / u.m / u.s
    assert vis.magnitude == u.Quantity(5, u.mPa * u.s).m_as("Pa*s")

    with pytest.raises(ValueError):
        vis = in_viscosity(5*u.m)


def test_in_molec_wt():
    """ Testing the in_molecular_weight function
    """

    molec_wt = in_molecular_weight(5)
    assert molec_wt.units == u.kg / u.mol
    assert molec_wt.magnitude == 5

    molec_wt = in_molecular_weight(5 * u.kg / u.mol)
    assert molec_wt.units == u.kg / u.mol
    assert molec_wt.magnitude == 5

    molec_wt = in_molecular_weight(u.Quantity(5, u.g / u.mol))
    assert molec_wt.units == u.kg / u.mol
    assert molec_wt.magnitude == u.Quantity(5, u.g / u.mol).m_as("kg/mol")

    molec_wt = in_molecular_weight(u.Quantity(5, u.kg / u.mol))
    assert molec_wt.units == u.kg / u.mol
    assert molec_wt.magnitude == u.Quantity(5, u.kg / u.mol).m_as("kg/mol")

    molec_wt = in_molecular_weight(u.Quantity(5, u.g / u.mol))
    assert molec_wt.units == u.kg / u.mol
    assert molec_wt.magnitude == u.Quantity(5, u.g / u.mol).m_as("kg/mol")

    with pytest.raises(ValueError):
        molec_wt = in_molecular_weight(5*u.m)


def test_in_pres():
    """ Testing the in_pressure function
    """

    pres = in_pressure(5)
    assert pres.units == u.kg / u.m / u.s**2
    assert pres.magnitude == 5

    pres = in_pressure(5 * u.kg / u.m / u.s**2)
    assert pres.units == u.kg / u.m / u.s**2
    assert pres.magnitude == 5

    pres = in_pressure(u.Quantity(5, u.mPa))
    assert pres.units == u.kg / u.m / u.s**2
    assert pres.magnitude == u.Quantity(5, u.mPa).m_as("Pa")

    pres = in_pressure(u.Quantity(5,  u.kg / u.m / u.s**2))
    assert pres.units == u.kg / u.m / u.s**2
    assert pres.magnitude == u.Quantity(5,  u.kg / u.m / u.s**2).m_as("Pa")

    pres = in_pressure(u.Quantity(5, u.mmHg))
    assert pres.units == u.kg / u.m / u.s**2
    assert pres.magnitude == u.Quantity(5, u.mmHg).m_as("Pa")

    pres = in_pressure(u.Quantity(5, u.atm))
    assert pres.units == u.kg / u.m / u.s**2
    assert pres.magnitude == u.Quantity(5, u.atm).m_as("Pa")

    pres = in_pressure(u.Quantity(5, u.torr))
    assert pres.units == u.kg / u.m / u.s**2
    assert pres.magnitude == u.Quantity(5, u.torr).m_as("Pa")

    pres = in_pressure(u.Quantity(5, u.bar))
    assert pres.units == u.kg / u.m / u.s**2
    assert pres.magnitude == u.Quantity(5, u.bar).m_as("Pa")

    pres = in_pressure(u.Quantity(5, u.mbar))
    assert pres.units == u.kg / u.m / u.s**2
