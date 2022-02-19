""" test the coulomb ration utility
"""

import pytest
from particula import u
from particula.util.coulomb_enhancement import CoulombEnhancement as CE


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
    assert (
        ce_neutral_a.coulomb_enhancement_continuum_limit()
    ).to_base_units().units == u.dimensionless
    assert (
        ce_extreme.coulomb_enhancement_continuum_limit()
    ).to_base_units().units == u.dimensionless
