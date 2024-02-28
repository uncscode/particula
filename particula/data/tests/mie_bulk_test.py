"""Test the Mie Bulk module."""

import pytest
from particula.data.process import mie_bulk


def initialize_file():
    """just loads the file"""
    assert True


def discretize_mie_return_type():
    """Test that the function returns a tuple of floats."""
    result = mie_bulk.discretize_auto_mieq(
        m_sphere=1.5 + 0.1j,
        wavelength=550,
        diameter=100)
    assert isinstance(result, tuple), "Result should be a tuple"
    assert all(isinstance(x, float)
               for x in result), "All items in the tuple should be floats"


def discretize_mie_valid_inputs():
    """Test the function with valid inputs."""
    result = mie_bulk.discretize_auto_mieq(
        m_sphere=1.5 + 0.1j,
        wavelength=550,
        diameter=100)
    # Assuming specific expected values for these parameters, adjust as
    # necessary
    expected = (0.5, 0.2, 0.3, 0.05, 0.1, 0.15, 0.25)
    assert result == pytest.approx(
        expected), "The function output is not as expected for valid inputs"


def discretize_mie_edge_cases():
    """Test the function with edge case inputs."""
    # Example edge case: very small diameter
    result = mie_bulk.discretize_auto_mieq(
        m_sphere=1.5 + 0.1j,
        wavelength=550,
        diameter=1)
    # You'll need to adjust the assertion based on what's expected in this
    # edge case
    assert result is not None, "Function should handle very small diameters"


def discretize_mie_default_parameters():
    """Test the function with default parameters."""
    # Default m_medium should be 1.0
    result_with_default = mie_bulk.discretize_auto_mieq(
        m_sphere=1.5, wavelength=550, diameter=100)
    result_explicit = mie_bulk.discretize_auto_mieq(
        m_sphere=1.5, wavelength=550, diameter=100, m_medium=1.0)
    assert result_with_default == result_explicit, ["Function behavior differs with default parameters"]


@pytest.mark.xfail(raises=ValueError)
def discretize_mie_exception_for_invalid_input():
    """Test that the function raises an exception for invalid inputs."""
    # Example: invalid negative diameter
    mie_bulk.discretize_auto_mieq(m_sphere=1.5, wavelength=550, diameter=-100)
