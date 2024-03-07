"""Tests for the size_distribution_convert module."""
import numpy as np


from particula.util.size_distribution_convert import (
    SizerConverter,
    get_conversion_strategy,
    DNdlogDPtoPMSConversionStrategy,
    PMStoPDFConversionStrategy,
    DNdlogDPtoPDFConversionStrategy
)


def test_dndlogdp_to_pms_conversion_strategy():
    """Test the conversion from dn/dlogdp to PMS format."""
    strategy = DNdlogDPtoPMSConversionStrategy()
    diameters = np.array([1, 2, 3])
    concentration = np.array([0.1, 0.2, 0.3])
    expected = np.array([0.04771213, 0.04436975, 0.04383841])
    # call function
    converted_concentration = strategy.convert(diameters, concentration)
    assert np.allclose(converted_concentration, expected, atol=1e-6)
    # call inverse function
    converted_inverse = strategy.convert(
        diameters, converted_concentration, inverse=True)
    assert np.allclose(converted_inverse, concentration, atol=1e-6)


def test_pms_to_pdf_conversion_strategy():
    """Test the conversion from PMS to PDF format."""
    strategy = PMStoPDFConversionStrategy()
    diameters = np.array([1, 10, 30])
    concentration = np.array([0.1, 0.2, 0.3])
    expected = np.array([0.01111111, 0.01, 0.00681818])
    # call function
    converted_concentration = strategy.convert(diameters, concentration)
    assert np.allclose(converted_concentration, expected, atol=1e-6)
    # call inverse function
    converted_inverse = strategy.convert(
        diameters, converted_concentration, inverse=True)
    assert np.allclose(converted_inverse, concentration, atol=1e-6)


def test_dndlogdp_to_pdf_conversion_strategy():
    """Test the conversion from dn/dlogdp to PDF format."""
    strategy = DNdlogDPtoPDFConversionStrategy()
    diameters = np.array([1, 2, 3])
    concentration = np.array([0.1, 0.2, 0.3])
    expected = np.array([0.04771213, 0.04436975, 0.04383841])
    # call function
    converted_concentration = strategy.convert(diameters, concentration)
    assert np.allclose(converted_concentration, expected, atol=1e-6)
    # call inverse function
    converted_inverse = strategy.convert(
        diameters, converted_concentration, inverse=True)
    assert np.allclose(converted_inverse, concentration, atol=1e-6)


def test_converter():
    """Test the Converter class."""
    strategy = DNdlogDPtoPMSConversionStrategy()
    converter = SizerConverter(strategy)
    diameters = np.array([1, 2, 3])
    concentration = np.array([0.1, 0.2, 0.3])
    expected = np.array([0.04771213, 0.04436975, 0.04383841])
    # call function
    converted_concentration = converter.convert(diameters, concentration)
    assert np.allclose(converted_concentration, expected, atol=1e-6)
    # call inverse function
    converted_inverse = converter.convert(
        diameters, converted_concentration, inverse=True)
    assert np.allclose(converted_inverse, concentration, atol=1e-6)


def test_get_conversion_strategy():
    """Test the get_conversion_strategy function."""
    strategy = get_conversion_strategy('dn/dlogdp', 'pms')
    assert isinstance(strategy, DNdlogDPtoPMSConversionStrategy)

    strategy = get_conversion_strategy('pms', 'pdf')
    assert isinstance(strategy, PMStoPDFConversionStrategy)

    strategy = get_conversion_strategy('dn/dlogdp', 'pdf')
    assert isinstance(strategy, DNdlogDPtoPDFConversionStrategy)

    try:
        strategy = get_conversion_strategy('invalid', 'pms')
        assert False  # Should raise a ValueError
    except ValueError:
        assert True

    try:
        strategy = get_conversion_strategy('dn/dlogdp', 'invalid')
        assert False  # Should raise a ValueError
    except ValueError:
        assert True
