""" test the coag approx utility for lf2013
"""

# import pytest
import numpy as np

from particula.util.lf2013_coagulation import lf2013_coag_full

# from particula import u


def test_approx_coag_less():
    """ testing
    """
    ret = np.nan_to_num(lf2013_coag_full(
        ion_type="air",
        particle_type="conductive",
        temperature_val=298.15,
        pressure_val=101325,
        charge_vals=[-1, -2],
        radius_vals=10e-9,)[0], 0)
    assert ret.max() >= 0
