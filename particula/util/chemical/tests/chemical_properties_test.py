"""Test for chemical_properties.py module."""

import pytest

from particula.util.chemical.chemical_properties import (
    get_chemical_stp_properties,
)
from particula.util.chemical.thermo_import import CHEMICALS_AVAILABLE


# identifiers chosen so thermo/chemicals always resolve them
@pytest.mark.skipif(not CHEMICALS_AVAILABLE, reason="thermo not installed")
@pytest.mark.parametrize(
    ("identifier", "expected_mw"),
    [
        ("water", 0.01801528),  # H2O
        ("64-17-5", 0.04606844),  # ethanol CAS
    ],
)
def test_get_chemical_stp_properties(identifier, expected_mw):
    """Test getting STP properties for a chemical."""
    props = get_chemical_stp_properties(identifier)

    # dictionary must have the four required keys
    required = (
        "molar_mass",
        "density",
        "surface_tension",
        "pure_vapor_pressure",
    )
    for key in required:
        assert key in props, f"{key} missing"
        # every returned value should be a positive float
        assert isinstance(props[key], (float, int)), f"{key} not numeric"
        assert props[key] > 0, f"{key} not positive"

    # molar mass should agree with reference value within 5 %
    rel_err = abs(props["molar_mass"] - expected_mw) / expected_mw
    assert rel_err < 0.05, f"Molar mass off by {rel_err:.1%}"


@pytest.mark.skipif(CHEMICALS_AVAILABLE, reason="thermo installed")
def test_get_chemical_stp_properties_importerror():
    """Test that ImportError is raised when thermo is not installed."""
    with pytest.raises(ImportError):
        get_chemical_stp_properties("water")
