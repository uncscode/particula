"""Test initialize"""


import numpy as np
import pytest
from particula.data.process import kappa_via_extinction


# Global parameters for the aerosol distribution and optics
GLOBAL_KAPPA = 0.8
GLOBAL_NUMBER_PER_CM3 = np.array([1000.0, 1500.0, 1000.0])
GLOBAL_DIAMETERS = np.array([100.0, 200.0, 300.0])
GLOBAL_WATER_ACTIVITY_SIZER = 0.2
GLOBAL_WATER_ACTIVITY_DRY = 0.3
GLOBAL_WATER_ACTIVITY_WET = 0.9
GLOBAL_REFRACTIVE_INDEX_DRY = 1.45
GLOBAL_WATER_REFRACTIVE_INDEX = 1.33
GLOBAL_WAVELENGTH = 450


def test_extinction_ratio_wet_dry():
    """Test the calculation of the extinction ratio between wet and
    dry aerosols."""
    exp_b_ext_wet = 1011.068479965262
    exp_b_ext_dry = 137.73807395245075
    exp_ratio = exp_b_ext_wet / exp_b_ext_dry

    ratio = kappa_via_extinction.extinction_ratio_wet_dry(
        kappa=GLOBAL_KAPPA,
        number_per_cm3=GLOBAL_NUMBER_PER_CM3,
        diameters=GLOBAL_DIAMETERS,
        water_activity_sizer=GLOBAL_WATER_ACTIVITY_SIZER,
        water_activity_dry=GLOBAL_WATER_ACTIVITY_DRY,
        water_activity_wet=GLOBAL_WATER_ACTIVITY_WET,
        refractive_index_dry=GLOBAL_REFRACTIVE_INDEX_DRY,
        water_refractive_index=GLOBAL_WATER_REFRACTIVE_INDEX,
        wavelength=GLOBAL_WAVELENGTH,
        discretize=False
    )
    # Assert ratio is correct
    assert ratio == pytest.approx(
        exp_ratio, abs=1e-6
    ), "Extiction ratio is incorrect"

    values = kappa_via_extinction.extinction_ratio_wet_dry(
        kappa=GLOBAL_KAPPA,
        number_per_cm3=GLOBAL_NUMBER_PER_CM3,
        diameters=GLOBAL_DIAMETERS,
        water_activity_sizer=GLOBAL_WATER_ACTIVITY_SIZER,
        water_activity_dry=GLOBAL_WATER_ACTIVITY_DRY,
        water_activity_wet=GLOBAL_WATER_ACTIVITY_WET,
        refractive_index_dry=GLOBAL_REFRACTIVE_INDEX_DRY,
        water_refractive_index=GLOBAL_WATER_REFRACTIVE_INDEX,
        wavelength=GLOBAL_WAVELENGTH,
        return_coefficients=True,
        discretize=False
    )
    # Assert values are correct
    assert values[0] == pytest.approx(
        exp_b_ext_wet, abs=1e-6
    ), "Wet extinction value is incorrect"
    assert values[1] == pytest.approx(
        exp_b_ext_dry, abs=1e-6
    ), "Dry extinction value is incorrect"


def test_fit_extinction_ratio_with_kappa():
    """Test fitting kappa based on simulated dry and wet aerosol extinction
    values."""
    # Simulate extinction values for dry and wet conditions
    b_ext_wet = 1011.068479965262
    b_ext_dry = 137.73807395245075
    expected_kappa = 0.8

    # Fit kappa using the function under test
    fitted_kappa = kappa_via_extinction.fit_extinction_ratio_with_kappa(
        b_ext_dry=b_ext_dry,
        b_ext_wet=b_ext_wet,
        number_per_cm3=GLOBAL_NUMBER_PER_CM3,
        diameters=GLOBAL_DIAMETERS,
        water_activity_sizer=GLOBAL_WATER_ACTIVITY_SIZER,
        water_activity_dry=GLOBAL_WATER_ACTIVITY_DRY,
        water_activity_wet=GLOBAL_WATER_ACTIVITY_WET,
        refractive_index_dry=GLOBAL_REFRACTIVE_INDEX_DRY,
        water_refractive_index=GLOBAL_WATER_REFRACTIVE_INDEX,
        wavelength=GLOBAL_WAVELENGTH,
        discretize=False,
        kappa_bounds=(0, 1),
        kappa_tolerance=1e-6,
        kappa_maxiter=250
    )
    # Assert that the fitted kappa is close to the expected kappa
    assert fitted_kappa == pytest.approx(
        expected_kappa, abs=1e-4), "Fitted kappa does not match expected value"


def test_kappa_from_extinction_looped():
    """Test the kappa fitting process looped over time-indexed measurements."""
    # Simulate arrays for inputs, with three time-indexed measurements
    extinction_dry = np.array([137.73, 137.73, 137.73])
    # scale extinction ratio to get different kappa fitted
    extinction_wet = np.array([1011.068, 1011.068*.9, 1011.068*1.1])
    number_per_cm3 = np.array([
        GLOBAL_NUMBER_PER_CM3,
        GLOBAL_NUMBER_PER_CM3,
        GLOBAL_NUMBER_PER_CM3
    ])
    diameter = np.array([100, 200, 300])
    water_activity_sizer = np.array([
        GLOBAL_WATER_ACTIVITY_SIZER,
        GLOBAL_WATER_ACTIVITY_SIZER,
        GLOBAL_WATER_ACTIVITY_SIZER])
    water_activity_sample_dry = np.array([
        GLOBAL_WATER_ACTIVITY_DRY,
        GLOBAL_WATER_ACTIVITY_DRY,
        GLOBAL_WATER_ACTIVITY_DRY])
    water_activity_sample_wet = np.array([
        GLOBAL_WATER_ACTIVITY_WET,
        GLOBAL_WATER_ACTIVITY_WET,
        GLOBAL_WATER_ACTIVITY_WET])
    refractive_index_dry = 1.45
    water_refractive_index = 1.33
    wavelength = 450

    # expected kappa fit
    exp_kappa_fit = np.array([[0.79874162, 1.04349672, 0.59162938],
                              [0.69864005, 0.88511056, 0.51936186],
                              [0.9279079, 1.17446761, 0.65957545]])

    # Execute the function under test
    kappa_fit = kappa_via_extinction.kappa_from_extinction_looped(
        extinction_dry=extinction_dry,
        extinction_wet=extinction_wet,
        number_per_cm3=number_per_cm3,
        diameter=diameter,
        water_activity_sizer=water_activity_sizer,
        water_activity_sample_dry=water_activity_sample_dry,
        water_activity_sample_wet=water_activity_sample_wet,
        refractive_index_dry=refractive_index_dry,
        water_refractive_index=water_refractive_index,
        wavelength=wavelength,
    )

    # Check the shape of the returned array
    assert kappa_fit.shape == (
        3, 3), "Returned kappa fit array has incorrect shape"

    # value checks
    assert kappa_fit == pytest.approx(
        exp_kappa_fit, abs=1e-4
    ), "Fitted kappa does not match expected value"
