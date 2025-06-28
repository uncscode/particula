"""Test for chemical_properties.py module."""

import numpy as np
import pytest

from particula.util.chemical.chemical_vapor_pressure import (
    get_chemical_vapor_pressure,
)
from particula.util.chemical.thermo_import import CHEMICALS_AVAILABLE


@pytest.mark.skipif(not CHEMICALS_AVAILABLE, reason="thermo not installed")
def test_get_vapor_pressure_scalar():
    """Test getting vapor pressure for a single temperature."""
    vp = get_chemical_vapor_pressure("water", 298.15)
    assert 2_000.0 < vp < 40_000.0  # ~3.2 kPa expected


@pytest.mark.skipif(not CHEMICALS_AVAILABLE, reason="thermo not installed")
def test_get_vapor_pressure_array():
    """Test getting vapor pressure for an array of temperatures."""
    temps = np.array([280.0, 298.15, 320.0])
    vp = get_chemical_vapor_pressure("water", temps)
    assert vp.shape == temps.shape
    # Vapor pressure increases with temperature
    assert np.all(vp[1:] > vp[:-1])


@pytest.mark.skipif(CHEMICALS_AVAILABLE, reason="thermo installed")
def test_get_vapor_pressure_importerror():
    """Test that ImportError is raised when thermo is not installed."""
    with pytest.raises(ImportError):
        get_chemical_vapor_pressure("water", 298.15)
