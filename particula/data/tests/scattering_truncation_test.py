"""Test initialize"""

import numpy as np
import pytest

from particula.data.process import scattering_truncation

# Global parameters for the aerosol distribution and optics
KAPPA = 0.8
NUMBER_PER_CM3 = np.array([1000.0, 1500.0, 1000.0])
DIAMETERS = np.array([100.0, 200.0, 300.0])
WATER_ACTIVITY_SIZER = 0.2
WATER_ACTIVITY_DRY = 0.3
WATER_ACTIVITY_WET = 0.9
REFRACTIVE_INDEX_DRY = 1.45
WATER_REFRACTIVE_INDEX = 1.33
WAVELENGTH = 450


def test_get_truncated_scattering():
    """Test the truncation of scattering data based on specified
    angle bounds."""
    # Example scattering intensities and corresponding angles
    scattering_unpolarized = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    theta = np.linspace(0, np.pi, 10)  # Angles from 0 to pi radians

    # Define truncation angles
    theta1 = np.pi / 4  # Lower bound of angle range
    theta2 = 3 * np.pi / 4  # Upper bound of angle range

    # Call the function under test
    scattering_trunc, theta_trunc = \
        scattering_truncation.get_truncated_scattering(
            scattering_unpolarized, theta, theta1, theta2)

    # expected values
    exp_scattering_trunc = np.array([4, 5, 6, 7])
    exp_theta_trunc = np.array(
        [1.04719755, 1.3962634, 1.74532925, 2.0943951])

    # verify the content of the truncated arrays more explicitly
    assert np.allclose(
        scattering_trunc, exp_scattering_trunc, atol=1e-5
        ), "The scattering intensities do not match the expected values"
    assert np.allclose(
        theta_trunc, exp_theta_trunc, atol=1e-5
        ), "The theta angles do not match the expected values"


def test_trunc_mono_basic():
    """Test the basic functionality of trunc_mono without full output."""

    # No discretization
    # Execute the function under test with minimal parameters
    trunc_corr = scattering_truncation.trunc_mono(
        m_sphere=REFRACTIVE_INDEX_DRY,
        wavelength=WAVELENGTH,
        diameter=500.0,
        discretize=False,
        calibrated_trunc=False
    )
    exp_trunc_corr = 1.0984704282875792
    # Assert the type and a basic range check for the truncation correction
    # factor
    assert trunc_corr == pytest.approx(
        exp_trunc_corr, abs=1e-6
    ), "Truncation correction is the wrong value"

    # With discretization
    # Execute the function under test with minimal parameters
    trunc_corr = scattering_truncation.trunc_mono(
        m_sphere=REFRACTIVE_INDEX_DRY,
        wavelength=WAVELENGTH,
        diameter=500.0,
        discretize=True,
        calibrated_trunc=False
    )
    exp_trunc_corr = 1.0984704282875792
    # Assert the type and a basic range check for the truncation correction
    # factor
    assert trunc_corr == pytest.approx(
        exp_trunc_corr, abs=1e-6
    ), "Truncation correction is the wrong value"


def test_trunc_mono_full_output():
    """Test trunc_mono with full output enabled."""

    # Execute the function under test with full_output enabled
    trunc_corr, z_axis, qsca_trunc, qsca_ideal, theta1, theta2 = \
        scattering_truncation.trunc_mono(
            m_sphere=REFRACTIVE_INDEX_DRY,
            wavelength=WAVELENGTH,
            diameter=500.0,
            discretize=False,
            full_output=True,
            calibrated_trunc=False
        )  # type: ignore

    # expected values
    exp_trunc_corr = 1.0984704282875792

    # Assert the type and a basic range check for the truncation correction
    # factor
    assert trunc_corr == pytest.approx(
        exp_trunc_corr, abs=1e-6
    ), "Truncation correction is the wrong value"

    # Validate the returned data structures
    assert isinstance(
        trunc_corr, float), "Truncation correction factor should be a float"
    assert isinstance(z_axis, np.ndarray), "z_axis should be a numpy array"
    assert isinstance(
        qsca_trunc, float), "qsca_trunc should be a float"
    assert isinstance(
        qsca_ideal, float), "qsca_ideal should be a float"
    assert isinstance(theta1, np.ndarray), "theta1 should be a numpy array"
    assert isinstance(theta2, np.ndarray), "theta2 should be a numpy array"

    # Perform basic checks on array dimensions to ensure they match
    # expectations
    npos = 100  # Expected number of positions along the z-axis
    assert len(z_axis) == npos, "Length of z_axis not match expected value"
    assert len(theta1) == npos, "Length of theta1 not match expected value"
    assert len(theta2) == npos, "Length of theta2 not match expected value"


def test_truncation_for_diameters():
    """Test the calculation of truncations for an array of diameters."""
    # Call the function under test
    truncation_corrections = scattering_truncation.truncation_for_diameters(
        m_sphere=WATER_ACTIVITY_DRY,
        wavelength=WAVELENGTH,
        diameter_sizes=DIAMETERS,
        calibrated_trunc=False
    )
    # Expected values
    exp_truncation_corrections = np.array([1.02127865, 1.02514909, 1.03708443])
    # Check that the output is an NDArray of the same size as the input
    # diameter_sizes
    assert isinstance(truncation_corrections,
                      np.ndarray), "Output should be a numpy array"
    assert truncation_corrections == pytest.approx(
        exp_truncation_corrections, abs=1e-6
    )


def test_correction_for_distribution():
    """Test the calculation of the scattering correction factor for
    an aerosol size distribution."""

    # Execute the function under test
    correction_factor = \
        scattering_truncation.correction_for_distribution(
            m_sphere=REFRACTIVE_INDEX_DRY,
            wavelength=WAVELENGTH,
            diameter_sizes=DIAMETERS,
            number_per_cm3=NUMBER_PER_CM3
        )
    # Expected values
    exp_correction_factor = 1.0165280236871668

    # Assert the type of the return value
    assert isinstance(
        correction_factor, (float, np.float64)
        ), "Correction factor should be a float"

    # Perform basic validity checks on the correction factor
    # Assuming the correction factor should be positive and typically close to
    # 1, but might vary significantly based on the aerosol properties
    assert correction_factor == pytest.approx(
        exp_correction_factor, abs=1e-6
    ), "Scattering correction factor is the wrong value"


def test_correction_for_humidified():
    """Test the scattering correction calculation for humidified aerosols."""

    # Execute the function under test with global parameters
    bsca_correction = scattering_truncation.correction_for_humidified(
        kappa=KAPPA,
        number_per_cm3=NUMBER_PER_CM3,
        diameter=DIAMETERS,
        water_activity_sizer=WATER_ACTIVITY_SIZER,
        water_activity_sample=WATER_ACTIVITY_WET,
        refractive_index_dry=REFRACTIVE_INDEX_DRY,
        water_refractive_index=WATER_REFRACTIVE_INDEX,
        wavelength=WAVELENGTH
    )
    # Expected values
    exp_bsca_correction = 1.0751959691207493
    # test value
    assert bsca_correction == pytest.approx(
        exp_bsca_correction, abs=1e-6
    ), "Scattering correction factor is the wrong value"


def test_correction_for_humidified_looped():
    """Test the looped correction for humidified aerosol measurements."""
    # Simulate time-indexed input arrays with varying conditions
    kappa = np.array([0.1, 0.5, 0.95])  # Hygroscopicity parameter array
    number_per_cm3 = np.array([
        [1000.0, 1500.0, 1000.0],
        [1000.0, 1500.0, 1000.0],
        [1000.0, 1500.0, 1000.0]])  # Time-indexed number concentration
    diameter = DIAMETERS  # Use the global diameters for simplicity
    # Varying sizing instrument water activity
    water_activity_sizer = np.array([0.2, 0.25, 0.3])
    # Sample water activity under different conditions
    water_activity_sample = np.array([0.8, 0.8, 0.8])

    # Execute the function under test
    correction_factors = \
        scattering_truncation.correction_for_humidified_looped(
            kappa=kappa,
            number_per_cm3=number_per_cm3,
            diameter=diameter,
            water_activity_sizer=water_activity_sizer,
            water_activity_sample=water_activity_sample,
            refractive_index_dry=REFRACTIVE_INDEX_DRY,
            water_refractive_index=WATER_REFRACTIVE_INDEX,
            wavelength=WAVELENGTH
        )

    # Expected values
    exp_correction_factors = np.array(
        [1.0234830302, 1.0405837279854753, 1.04719491370677])

    # Assert the output is a numpy array with the correct length
    assert isinstance(correction_factors,
                      np.ndarray), "Output should be a numpy array"
    assert len(correction_factors) == len(
        kappa), "Output array length does not match the number of time indices"

    # Check values
    assert correction_factors == pytest.approx(
        exp_correction_factors, abs=1e-6
    ), "Scattering correction factors are the wrong values"
