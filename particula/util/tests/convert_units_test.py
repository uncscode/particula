"""Pytest coverage for ``convert_units`` helper."""

import numpy as np
import pytest

import particula.util.convert_units as convert_units
from particula.util.convert_units import get_unit_conversion

pint = pytest.importorskip("pint")


def test_factor_unit_conversion_returns_conversion_factor():
    """Factor-only units return a multiplicative conversion factor."""
    assert get_unit_conversion("m", "cm") == pytest.approx(100.0)


def test_offset_unit_conversion_with_value():
    """Offset units convert numeric values correctly."""
    assert get_unit_conversion("degC", "degF", value=25.0) == pytest.approx(
        77.0
    )


def test_array_magnitude_conversion():
    """Array inputs convert elementwise and preserve shape."""
    meters = np.array([0.0, 1.0, 2.5])
    centimeters = get_unit_conversion("m", "cm", value=meters)

    assert np.allclose(centimeters, np.array([0.0, 100.0, 250.0]))


def test_conversion_factor_when_value_none():
    """When value is None, multiplicative units return the factor only."""
    factor = get_unit_conversion("ug/m^3", "kg/m^3")
    assert factor == pytest.approx(1e-9)


def test_invalid_unit_raises_undefined_unit_error():
    """Bad unit strings surface Pint's UndefinedUnitError."""
    with pytest.raises(pint.errors.UndefinedUnitError):
        get_unit_conversion("not_a_unit", "m")


def test_missing_pint_dependency_raises_importerror(monkeypatch):
    """If Pint is absent, an ImportError with guidance is raised."""
    monkeypatch.setattr(convert_units, "unit_registry", None)
    with pytest.raises(ImportError, match="Install pint"):
        get_unit_conversion("degC", "degF")
