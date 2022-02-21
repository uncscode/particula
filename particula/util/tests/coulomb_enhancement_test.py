""" test the coulomb ration utility
"""

import numpy as np
import pytest
from particula import u
from particula.util.coulomb_enhancement import CoulombEnhancement as CE
from particula.util.coulomb_enhancement import cecl, cekl


def test_coulomb_ratio():
    """ testing
    """

    a_ra = u.Quantity(1, u.m)
    b_ra = u.Quantity(1, u.m)

    ce_default = CE(radius=a_ra, other_radius=b_ra)
    ce_charged = CE(radius=a_ra, other_radius=b_ra, charge=-1, other_charge=1)
    ce_extreme = CE(radius=a_ra, other_radius=b_ra,
                    charge=-1000, other_charge=1000)
    ce_unlikes = CE(radius=a_ra, radiys=b_ra, charge=-1, other_charge=-1)
    ce_neutral_a = CE(radius=a_ra, other_radius=b_ra, charge=0, other_charge=0)
    ce_neutral_b = CE(radius=a_ra, other_radius=b_ra, charge=0, other_charge=1)
    ce_neutral_c = CE(radius=a_ra, other_radius=b_ra,
                      charge=-1, other_charge=0)

    assert (
        ce_default.coulomb_potential_ratio().to_base_units().units ==
        u.dimensionless
    )
    assert (
        ce_charged.coulomb_potential_ratio().to_base_units().units ==
        u.dimensionless
    )
    assert (
        ce_neutral_a.coulomb_potential_ratio().magnitude ==
        pytest.approx(0)
    )
    assert (
        ce_charged.coulomb_potential_ratio().magnitude >=
        ce_neutral_b.coulomb_potential_ratio().magnitude
    )
    assert (
        ce_neutral_c.coulomb_potential_ratio().magnitude >=
        ce_unlikes.coulomb_potential_ratio().magnitude
    )
    assert (
        ce_extreme.coulomb_potential_ratio().magnitude ==
        pytest.approx(2e-2, rel=1e0)
    )


def test_cekl():
    """ testing cekl
    """

    a_ra = u.Quantity(1, u.m)
    b_ra = u.Quantity(1, u.m)

    ce_default = cekl(radius=a_ra, other_radius=b_ra)
    ce_charged = cekl(radius=a_ra, other_radius=b_ra,
                      charge=-1, other_charge=1)
    ce_extreme = cekl(radius=a_ra, other_radius=b_ra,
                      charge=-1000, other_charge=1000)
    ce_unlikes = cekl(radius=a_ra, radiys=b_ra, charge=-1, other_charge=-1)
    ce_neutral_a = cekl(radius=a_ra, other_radius=b_ra,
                        charge=0, other_charge=0)
    ce_neutral_b = cekl(radius=a_ra, other_radius=b_ra,
                        charge=0, other_charge=1)
    ce_neutral_c = cekl(radius=a_ra, other_radius=b_ra,
                        charge=-1, other_charge=0)

    assert (
        ce_default.to_base_units().units ==
        u.dimensionless
    )
    assert (
        ce_charged.to_base_units().units ==
        u.dimensionless
    )
    assert (
        ce_neutral_a.to_base_units().magnitude ==
        pytest.approx(1)
    )
    assert (
        ce_charged.to_base_units().magnitude >=
        ce_neutral_b.to_base_units().magnitude
    )
    assert (
        ce_neutral_c.to_base_units().magnitude >=
        ce_unlikes.to_base_units().magnitude
    )
    assert (
        ce_extreme.to_base_units().magnitude ==
        pytest.approx(1, rel=1e0)
    )


def test_cecl():
    """ testing cecm
    """

    a_ra = u.Quantity(1, u.m)
    b_ra = u.Quantity(1, u.m)

    ce_default = cecl(radius=a_ra, other_radius=b_ra)
    ce_charged = cecl(radius=a_ra, other_radius=b_ra,
                      charge=-1, other_charge=1)
    ce_extreme = cecl(radius=a_ra, other_radius=b_ra,
                      charge=-1000, other_charge=1000)
    ce_unlikes = cecl(radius=a_ra, radiys=b_ra, charge=-1, other_charge=-1)
    ce_neutral_a = cecl(radius=a_ra, other_radius=b_ra,
                        charge=0, other_charge=0)
    ce_neutral_b = cecl(radius=a_ra, other_radius=b_ra,
                        charge=0, other_charge=1)
    ce_neutral_c = cecl(radius=a_ra, other_radius=b_ra,
                        charge=-1, other_charge=0)

    assert (
        ce_default.to_base_units().units ==
        u.dimensionless
    )
    assert (
        ce_charged.to_base_units().units ==
        u.dimensionless
    )
    assert (
        ce_neutral_a.to_base_units().magnitude ==
        pytest.approx(1)
    )
    assert (
        ce_charged.to_base_units().magnitude >=
        ce_neutral_b.to_base_units().magnitude
    )
    assert (
        ce_neutral_c.to_base_units().magnitude >=
        ce_unlikes.to_base_units().magnitude
    )
    assert (
        ce_extreme.to_base_units().magnitude ==
        pytest.approx(1, rel=1e0)
    )


def test_arrays():
    """ testing if enhancements accept arrays
    """

    assert CE(
        radius=np.array([1, 2, 3]), other_radius=np.array([1, 2])
    ).coulomb_potential_ratio().m.shape == (3, 2)

    with pytest.raises(ValueError):
        CE(
            radius=[1, 2, 3],
            temperature=[1, 2, 3, 4],
        ).coulomb_potential_ratio()

    with pytest.raises(ValueError):
        CE(
            radius=[1, 2, 3],
            charge=[1, 2, 3, 4],
        ).coulomb_potential_ratio()
