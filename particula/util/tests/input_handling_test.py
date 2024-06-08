""" testing the input_handling function
"""

import pytest
from particula import u
from particula.util.input_handling import (in_density, in_length,
                                           in_molecular_weight, in_pressure,
                                           in_radius, in_scalar,
                                           in_temperature, in_viscosity,
                                           in_volume)


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


def test_in_radius():
    """ Testing the in_radius function
    """

    radius = in_radius(5)
    assert radius.units == u.m
    assert radius.magnitude == 5

    radius = in_radius(5 * u.m)
    assert radius.units == u.m
    assert radius.magnitude == 5

    radius = in_radius(u.Quantity(5, u.mm))
    assert radius.units == u.m
    assert radius.magnitude == u.Quantity(5, u.mm).m_as("m")

    radius = in_radius(u.Quantity(5, u.cm))
    assert radius.units == u.m
    assert radius.magnitude == u.Quantity(5, u.cm).m_as("m")

    radius = in_radius(u.Quantity(5, u.m))
    assert radius.units == u.m
    assert radius.magnitude == u.Quantity(5, u.m).m_as("m")

    radius = in_radius(u.Quantity(5, u.km))
    assert radius.units == u.m
    assert radius.magnitude == u.Quantity(5, u.km).m_as("m")

    radius = in_radius(u.Quantity(5, u.inch))
    assert radius.units == u.m
    assert radius.magnitude == u.Quantity(5, u.inch).m_as("m")

    radius = in_radius(u.Quantity(5, u.ft))
    assert radius.units == u.m
    assert radius.magnitude == u.Quantity(5, u.ft).m_as("m")

    radius = in_radius(u.Quantity(5, u.mi))
    assert radius.units == u.m
    assert radius.magnitude == u.Quantity(5, u.mi).m_as("m")

    radius = in_radius(u.Quantity(5, u.nmi))
    assert radius.units == u.m
    assert radius.magnitude == u.Quantity(5, u.nmi).m_as("m")


def test_in_density():
    """ Testing the in_density function
    """

    density = in_density(5)
    assert density.units == u.kg / u.m**3
    assert density.magnitude == 5

    density = in_density(5 * u.kg / u.m**3)
    assert density.units == u.kg / u.m**3
    assert density.magnitude == 5

    density = in_density(u.Quantity(5, u.g / u.cm**3))
    assert density.units == u.kg / u.m**3
    assert density.magnitude == u.Quantity(5, u.g / u.cm**3).m_as("kg/m^3")

    density = in_density(u.Quantity(5, u.kg / u.m**3))
    assert density.units == u.kg / u.m**3
    assert density.magnitude == u.Quantity(5, u.kg / u.m**3).m_as("kg/m^3")

    density = in_density(u.Quantity(5, u.kg / u.cm**3))
    assert density.units == u.kg / u.m**3
    assert density.magnitude == u.Quantity(5, u.kg / u.cm**3).m_as("kg/m^3")

    density = in_density(u.Quantity(5, u.kg / u.cm**3))
    assert density.units == u.kg / u.m**3
    assert density.magnitude == u.Quantity(5, u.kg / u.cm**3).m_as("kg/m^3")

    density = in_density(u.Quantity(5, u.g / u.cm**3))
    assert density.units == u.kg / u.m**3
    assert density.magnitude == u.Quantity(5, u.g / u.cm**3).m_as("kg/m^3")

    density = in_density(u.Quantity(5, u.lb / u.ft**3))
    assert density.units == u.kg / u.m**3
    assert density.magnitude == u.Quantity(5, u.lb / u.ft**3).m_as("kg/m^3")


def test_in_scalar():
    """ Testing the in_scalar function
    """

    scalar = in_scalar(5)
    assert scalar.units == u.dimensionless
    assert scalar.magnitude == 5

    scalar = in_scalar(5 * u.dimensionless)
    assert scalar.units == u.dimensionless
    assert scalar.magnitude == 5

    with pytest.raises(ValueError):
        in_scalar(u.Quantity(5, u.m))


def test_in_length():
    """ Testing the in_length function
    """

    length = in_length(5)
    assert length.units == u.m
    assert length.magnitude == 5

    length = in_length(5 * u.m)
    assert length.units == u.m
    assert length.magnitude == 5

    length = in_length(u.Quantity(5, u.mm))
    assert length.units == u.m
    assert length.magnitude == u.Quantity(5, u.mm).m_as("m")

    length = in_length(u.Quantity(5, u.cm))
    assert length.units == u.m
    assert length.magnitude == u.Quantity(5, u.cm).m_as("m")

    length = in_length(u.Quantity(5, u.m))
    assert length.units == u.m
    assert length.magnitude == u.Quantity(5, u.m).m_as("m")

    length = in_length(u.Quantity(5, u.km))
    assert length.units == u.m
    assert length.magnitude == u.Quantity(5, u.km).m_as("m")

    length = in_length(u.Quantity(5, u.inch))
    assert length.units == u.m
    assert length.magnitude == u.Quantity(5, u.inch).m_as("m")

    length = in_length(u.Quantity(5, u.ft))
    assert length.units == u.m
    assert length.magnitude == u.Quantity(5, u.ft).m_as("m")

    length = in_length(u.Quantity(5, u.mi))
    assert length.units == u.m
    assert length.magnitude == u.Quantity(5, u.mi).m_as("m")

    length = in_length(u.Quantity(5, u.nmi))
    assert length.units == u.m
    assert length.magnitude == u.Quantity(5, u.nmi).m_as("m")


def test_in_volume():
    """ Testing the in_volume function
    """

    volume = in_volume(5)
    assert volume.units == u.m**3
    assert volume.magnitude == 5

    volume = in_volume(5 * u.m**3)
    assert volume.units == u.m**3
    assert volume.magnitude == 5

    volume = in_volume(u.Quantity(5, u.mm**3))
    assert volume.units == u.m**3
    assert volume.magnitude == u.Quantity(5, u.mm**3).m_as("m^3")

    volume = in_volume(u.Quantity(5, u.cm**3))
    assert volume.units == u.m**3
    assert volume.magnitude == u.Quantity(5, u.cm**3).m_as("m^3")

    volume = in_volume(u.Quantity(5, u.m**3))
    assert volume.units == u.m**3
    assert volume.magnitude == u.Quantity(5, u.m**3).m_as("m^3")

    volume = in_volume(u.Quantity(5, u.km**3))
    assert volume.units == u.m**3
    assert volume.magnitude == u.Quantity(5, u.km**3).m_as("m^3")

    volume = in_volume(u.Quantity(5, u.inch**3))
    assert volume.units == u.m**3
    assert volume.magnitude == u.Quantity(5, u.inch**3).m_as("m^3")

    volume = in_volume(u.Quantity(5, u.ft**3))
    assert volume.units == u.m**3
    assert volume.magnitude == u.Quantity(5, u.ft**3).m_as("m^3")

    volume = in_volume(u.Quantity(5, u.mi**3))
    assert volume.units == u.m**3
    assert volume.magnitude == u.Quantity(5, u.mi**3).m_as("m^3")
