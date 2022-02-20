""" Testing the dimensionless coagulation module.
"""

import pytest
from particula import u
from particula.util.dimensionless_coagulation import full_coag, less_coag


def test_less():
    """ testing hardsphere dimensionless coagulation
    """

    assert (
        less_coag(radius=1e-9, other_radius=1e-8).units ==
        u.dimensionless
    )
    assert (
        less_coag(radius=1e-9, other_radius=1e-8).magnitude ==
        pytest.approx(18, rel=1e0)
    )

    assert (
        less_coag(radius=1e-9, other_radius=1e-8).magnitude <=
        less_coag(radius=1e-9, other_radius=1e-9).magnitude
    )

    assert (
        less_coag(
            radius=1e-9, other_radius=1e-8, charge=-1, other_charge=1
        ).magnitude >=
        less_coag(
            radius=1e-9, other_radius=1e-8, charge=-1, other_charge=0
        ).magnitude
    )


def test_full():
    """ testing hardsphere dimensioned coagulation
    """

    assert (
        full_coag(radius=1e-9, other_radius=1e-8).units ==
        u.m**3 / u.s
    )

    assert (
        full_coag(radius=1e-9, other_radius=1e-8).m_as(u.cm**3/u.s) ==
        pytest.approx(1.8e-8, rel=1e-1)
    )

    assert (
        full_coag(radius=1e-9, other_radius=1e-8).magnitude >=
        full_coag(radius=1e-9, other_radius=1e-9).magnitude
    )

    assert (
        full_coag(
            radius=1e-9, other_radius=1e-8, charge=-1, other_charge=1
        ).magnitude >=
        full_coag(
            radius=1e-9, other_radius=1e-8, charge=-1, other_charge=0
        ).magnitude
    )


def test_ipx():
    """ testing ion--particle coag
    """

    ion_rad = 0.5e-9
    ion_cha = -1
    ion_den = 1.84e3

    par_rad = 3e-9
    par_cha = 1
    par_den = 1.7e3

    coag = full_coag(
        radius=ion_rad, other_radius=par_rad, charge=ion_cha,
        other_charge=par_cha, density=ion_den, other_density=par_den,
        temperature=300, pressure=101325,
    )

    assert coag.m_as(u.cm**3/u.s) == pytest.approx(7.65e-8, rel=1e-1)

    # par_cha_new = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # coag_many = full_coag(
    #     radius=ion_rad, other_radius=par_rad, charge=ion_cha,
    #     other_charge=par_cha_new, density=ion_den, other_density=par_den,
    #     temperature=300, pressure=101325,
    # )

    # assert coag_many.m_as(u.cm**3/u.s).shape == (10,)
