"""Test for chemical_properties.py module."""

import numpy as np
import pytest

from particula.util.chemical.chemical_surface_tension import (
    get_chemical_surface_tension,
)
from particula.util.chemical.thermo_import import CHEMICALS_AVAILABLE


@pytest.mark.skipif(not CHEMICALS_AVAILABLE, reason="thermo not installed")
def test_get_surface_tension_scalar():
    """Test getting surface tension for a single temperature."""
    st = get_chemical_surface_tension("water", 298.15)
    assert 0.05 < st < 0.08  # ~0.072 N/m expected


@pytest.mark.skipif(not CHEMICALS_AVAILABLE, reason="thermo not installed")
def test_get_surface_tension_array():
    """Test getting surface tension for an array of temperatures."""
    temps = np.array([280.0, 298.15, 320.0])
    st = get_chemical_surface_tension("water", temps)
    assert st.shape == temps.shape
    # Surface tension decreases with temperature
    assert np.all(st[1:] < st[:-1])


@pytest.mark.skipif(CHEMICALS_AVAILABLE, reason="thermo installed")
def test_get_surface_tension_importerror():
    """Test that ImportError is raised when thermo is not installed."""
    with pytest.raises(ImportError):
        get_chemical_surface_tension("water", 298.15)
