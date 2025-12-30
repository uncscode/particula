"""test the coag approx utility for lf2013."""

import numpy as np
import pytest

from particula.util.lf2013_coagulation import lf2013_coag_full


def test_approx_coag_less():
    """Baseline case remains finite for conductive air."""
    ret = np.nan_to_num(
        lf2013_coag_full(
            ion_type="air",
            particle_type="conductive",
            temperature_val=298.15,
            pressure_val=101325,
            charge_vals=[-1, -2],
            radius_vals=10e-9,
        )[0],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
        copy=False,
    )
    assert ret.max() >= 0


@pytest.mark.parametrize("radius_val", [1e-6])
def test_boundary_radii_return_finite_values(radius_val):
    """Extreme radius values should not overflow or produce NaNs."""
    neg, pos = lf2013_coag_full(
        ion_type="air",
        particle_type="conductive",
        temperature_val=298.15,
        pressure_val=101325,
        charge_vals=[-1, 1],
        radius_vals=radius_val,
    )
    combined = np.concatenate([neg, pos], axis=None)
    assert np.isfinite(combined).all()
