""" testing the mol_enh calc
"""

import pytest
from particula import u
from particula.util.molecular_enhancement import mol_enh


def test_mol_enh():
    """ testing mol_enh
    """

    a_mol_enh = mol_enh(vapor_size=1, particle_size=1)
    b_mol_enh = mol_enh(vapor_size=1, particle_size=1)

    assert a_mol_enh == 4
    assert a_mol_enh == b_mol_enh
    assert a_mol_enh.units == u.dimensionless

    assert mol_enh(vapor_size=1, particle_size=[1, 2]).m.shape == (2,)
    assert mol_enh(vapor_size=[1, 2], particle_size=1).m.shape == (1, 2)
    assert mol_enh(
        vapor_size=[1, 2], particle_size=[1, 2, 3]
    ).m.shape == (3, 2)

    with pytest.raises(ValueError):
        mol_enh(vapor_size=5*u.m, particle_size=5*u.s)

    with pytest.raises(ValueError):
        mol_enh(vapor_size=1*u.kg, particle_size=5*u.m)
