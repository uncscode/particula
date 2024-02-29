"""Test the Mie Bulk module."""

import pytest
import numpy as np
from particula.data.process import mie_bulk


def test_mie_return_type():
    """Test that the function returns a tuple of floats."""
    result = mie_bulk.discretize_auto_mieq(
        m_sphere=1.5 + 0.1j,
        wavelength=550,
        diameter=100)
    assert isinstance(result, tuple), "Result should be a tuple"
    assert all(isinstance(x, float)
               for x in result), "All items in the tuple should be floats"


def test_mie_valid_inputs():
    """Test the function with valid inputs."""
    result = mie_bulk.discretize_auto_mieq(
        m_sphere=1.5 + 0.1j,
        wavelength=550,
        diameter=100)
    print(result)
    # Assuming specific expected values for these parameters, adjust as
    # necessary
    expected = (0.15762, 0.02551, 0.132118,
                0.0639714, 0.155997, 0.032652,
                1.27993)
    assert result == pytest.approx(
        expected, abs=1e-4), "The function output for valid inputs"


def test_mie_edge_cases():
    """Test the function with edge case inputs."""
    # Example edge case: very small diameter
    result = mie_bulk.discretize_auto_mieq(
        m_sphere=1.5 + 0.1j,
        wavelength=550,
        diameter=1)
    # You'll need to adjust the assertion based on what's expected in this
    # edge case
    assert result is not None, "Function should handle very small diameters"


def test_mie_default():
    """Test the function with default parameters."""
    # Default m_medium should be 1.0
    result_with_default = mie_bulk.discretize_auto_mieq(
        m_sphere=1.5, wavelength=550, diameter=100)
    result_explicit = mie_bulk.discretize_auto_mieq(
        m_sphere=1.5, wavelength=550, diameter=100, m_medium=1.0)
    assert result_with_default == result_explicit, "Function behavior differs"


def test_discretize_with_custom_bases():
    """Test function with custom base values."""
    m_sphere, wavelength, diameters = mie_bulk.discretize_mie_parameters(
        m_sphere=1.54321 + 0.01234j,
        wavelength=532.123,
        diameter=np.array([102.34, 158.76]),
        base_m_sphere=0.01,
        base_wavelength=5,
        base_diameter=10
    )
    # Example expected value, adjust based on your rounding strategy
    expected_m_sphere = 1.54 + 0.01j
    # Example expected value, adjust based on your rounding strategy
    expected_wavelength = 530
    # Example expected values, adjust based on your rounding strategy
    expected_diameters = np.array([100, 160])
    assert m_sphere == pytest.approx(
        expected_m_sphere), "Custom base m_sphere"
    assert wavelength == pytest.approx(
        expected_wavelength), "Custom base wavelength"
    assert all(d == pytest.approx(d_val)
               for d, d_val in zip(diameters, expected_diameters)
               ), "Custom base diameters were not applied correctly"


def test_discretize_diameter_float():
    """Test that a float diameter input remains a float after
    discretization."""
    m_sphere_input = 1.54321 + 0.01234j
    wavelength_input = 532.123
    diameter_input = 158.76  # Single float value for diameter
    base_m_sphere = 0.01
    base_wavelength = 5
    base_diameter = 10

    m_sphere, wavelength, diameter = mie_bulk.discretize_mie_parameters(
        m_sphere=m_sphere_input,
        wavelength=wavelength_input,
        diameter=diameter_input,  # Passing a single float value
        base_m_sphere=base_m_sphere,
        base_wavelength=base_wavelength,
        base_diameter=base_diameter
    )

    # Expected values, adjusted based on the discretization strategy
    expected_m_sphere = 1.54 + 0.01j
    expected_wavelength = 530
    expected_diameter = 160  # Expected discretized value for the diameter

    # Check if the discretized m_sphere matches the expected complex value
    assert m_sphere == pytest.approx(
        expected_m_sphere), "Custom base m_sphere was not applied correctly"

    # Check if the discretized wavelength matches the expected value
    assert wavelength == pytest.approx(
        expected_wavelength), "Custom base wavelength was not applied right"

    # Check if the discretized diameter remains a float and matches the
    # expected value
    assert isinstance(
        diameter, float
        ), "Diameter input was not maintained as a float after discretization"
    assert diameter == pytest.approx(
        expected_diameter), "Custom base diameter was not applied correctly"


def test_format_mie_results_as_dict():
    """Test that results are correctly formatted as a dictionary."""
    # Sample input arrays
    b_ext = np.array([1.0, 2.0])
    b_sca = np.array([0.5, 1.5])
    b_abs = np.array([0.5, 0.5])
    big_g = np.array([0.9, 0.8])
    b_pr = np.array([0.1, 0.2])
    b_back = np.array([0.05, 0.15])
    b_ratio = np.array([0.1, 0.1])

    # Call the function with as_dict=True
    results = mie_bulk.format_mie_results(
        b_ext,
        b_sca,
        b_abs,
        big_g,
        b_pr,
        b_back,
        b_ratio,
        as_dict=True)

    # Assertions to verify the dictionary format and values
    assert isinstance(results, dict), "Expected a dictionary"
    assert np.array_equal(results['b_ext'], b_ext), "b_ext values mismatch"
    assert np.array_equal(results['b_sca'], b_sca), "b_sca values mismatch"
    assert np.array_equal(results['b_abs'], b_abs), "b_abs values mismatch"
    assert np.array_equal(results['G'], big_g), "G values mismatch"
    assert np.array_equal(results['b_pr'], b_pr), "b_pr values mismatch"
    assert np.array_equal(results['b_back'], b_back), "b_back values mismatch"
    assert np.array_equal(
        results['b_ratio'], b_ratio), "b_ratio values mismatch"


def test_format_mie_results_as_tuple():
    """Test that results are correctly formatted as a tuple."""
    # Sample input arrays
    b_ext = np.array([1.0, 2.0])
    b_sca = np.array([0.5, 1.5])
    b_abs = np.array([0.5, 0.5])
    big_g = np.array([0.9, 0.8])
    b_pr = np.array([0.1, 0.2])
    b_back = np.array([0.05, 0.15])
    b_ratio = np.array([0.1, 0.1])

    # Call the function with as_dict=False
    results = mie_bulk.format_mie_results(
        b_ext,
        b_sca,
        b_abs,
        big_g,
        b_pr,
        b_back,
        b_ratio,
        as_dict=False)

    # Assertions to verify the tuple format and values
    assert isinstance(results, tuple), "Expected a tuple"
    assert np.array_equal(results[0], b_ext), "b_ext values mismatch in tuple"
    assert np.array_equal(results[1], b_sca), "b_sca values mismatch in tuple"
    assert np.array_equal(results[2], b_abs), "b_abs values mismatch in tuple"
    assert np.array_equal(results[3], big_g), "G values mismatch in tuple"
    assert np.array_equal(results[4], b_pr), "b_pr values mismatch in tuple"
    assert np.array_equal(
        results[5], b_back), "b_back values mismatch in tuple"
    assert np.array_equal(
        results[6], b_ratio), "b_ratio values mismatch in tuple"


def test_mie_size_distribution_extinction_only_tuple_output():
    """Test mie_size_distribution with extinction_only=True for a
    10-bin size distribution from 100 to 500 nm."""
    # Define inputs
    m_sphere = 1.5 + 0.01j  # Example refractive index
    wavelength = 532  # Example wavelength in nm
    # 10-bin size distribution from 100 to 500 nm
    diameter_bins = np.linspace(100, 500, 10)
    # Example number distribution, uniform for simplicity
    number_per_cm3 = np.ones(10) * 1e3

    # Call the function with extinction_only=True and expect a tuple output
    result_ext = mie_bulk.mie_size_distribution(
        m_sphere=m_sphere,
        wavelength=wavelength,
        diameter=diameter_bins,
        number_per_cm3=number_per_cm3,
        extinction_only=True,
        as_dict=False  # Ensure tuple output
    )
    print(result_ext)
    ext_expected = 1914.0383932892566
    # Assert that the result is a tuple
    assert isinstance(result_ext, float), "Expected result to be a float"

    # Assert that the tuple contains only the extinction coefficient
    assert result_ext == pytest.approx(
        ext_expected), "Expected extinction "

    result_all = mie_bulk.mie_size_distribution(
        m_sphere=m_sphere,
        wavelength=wavelength,
        diameter=diameter_bins,
        number_per_cm3=number_per_cm3,
        as_dict=False  # Ensure tuple output
    )
    all_expected = (
        1914.0383932892566,
        1827.8800860662814,
        86.15830722297528,
        0.6976095606315363,
        638.8917695614232,
        233.76303617887106,
        152.2227330199248)
    # Assert that the result is a tuple
    assert isinstance(result_all, tuple), "Expected result to be a tuple"
    # Assert that values are close
    assert result_all == pytest.approx(
        all_expected, abs=1e-8
    ), "Expected values compared"
